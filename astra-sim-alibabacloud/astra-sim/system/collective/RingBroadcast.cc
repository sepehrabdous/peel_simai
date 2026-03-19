// RingBroadcast.cc
//
// Implements a pipelined ring broadcast collective operation.
//
// Algorithm overview:
//   - One "root" node holds data to be sent to all other nodes in the ring.
//   - The data is split into `num_chunks` chunks (configurable via the
//     AS_RING_BCAST_CHUNKS environment variable).
//   - The root sends chunks one-at-a-time along the ring in one direction:
//       root → node1 → node2 → ... → last_node
//     where "last_node" is the node whose downstream neighbor is root.
//   - Each intermediate node relays each received chunk to the next node in
//     the ring (pipelining: it does not wait for all chunks before forwarding).
//   - The "last" node sends a completion ACK back to root after receiving all
//     chunks, signaling end-to-end broadcast completion.
//
// Key state per node:
//   - Root:    stages chunks, sends them, waits for completion ACK.
//   - Middle:  posts recv for each chunk, then relays it downstream.
//   - Last:    posts recv for each chunk, then sends completion ACK to root.
//
// Packet lifecycle:
//   1. stage_data_packet() creates a MyPacket and calls release_packets(),
//      which sends it through the memory bus (MA or NPU direction).
//   2. The memory bus fires an EventType::General callback when packetization
//      is complete, incrementing free_packets.
//   3. ready() picks up the staged packet and calls front_end_sim_send().
//   4. The recv side posts front_end_sim_recv() via post_data_recv(); when
//      the packet arrives EventType::PacketReceived fires on that node.

#include "RingBroadcast.hh"

#include <cstdlib>
#include <iostream>

#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"

namespace AstraSim {

RingBroadcast::RingBroadcast(
    ComType type,
    int id,
    int layer_num,
    RingTopology* ring_topology,
    uint64_t data_size,
    RingTopology::Direction direction,
    InjectionPolicy injection_policy,
    bool boost_mode,
    int root)
    : Algorithm(layer_num) {

  this->comType = type;
  this->id = id;

  // Derive the ring-local root node ID from the physical node ID.
  // The ring offset accounts for which slice of the global topology this
  // ring instance covers (e.g. a sub-ring within a multi-dimensional torus).
  // Keep your current ring-root behavior.
  // If later you want arbitrary roots, change this to:
  // this->root = root;
  this->root = id - ring_topology->index_in_ring * ring_topology->offset;

  this->logicalTopology = ring_topology;
  this->data_size = data_size;
  this->final_data_size = data_size;
  this->direction = direction;
  this->nodes_in_ring = ring_topology->get_nodes_in_ring();

  // current_receiver: the node this node sends data to (downstream neighbor).
  // current_sender:   the node this node receives data from (upstream neighbor).
  this->current_receiver = ring_topology->get_receiver_node(id, direction);
  this->current_sender = ring_topology->get_sender_node(id, direction);
  this->injection_policy = injection_policy;

  // free_packets tracks how many packets have been packetized by the memory
  // bus and are ready to be handed to the network simulator. Each
  // EventType::General callback increments this by 1.
  this->free_packets = 0;
  this->processed = false;
  this->send_back = false;
  this->send_from_npu = true;

  // Non-root nodes start with recv_done = false (they have not yet received
  // the broadcast data). Root already "has" the data, so recv_done = true.
  this->recv_done = (id == this->root);
  this->send_done = false;
  this->exited = false;

  // Allow the number of pipeline chunks to be overridden at runtime.
  const char* env_chunks = std::getenv("AS_RING_BCAST_CHUNKS");
  if (env_chunks != nullptr) {
    unsigned long long tmp = std::strtoull(env_chunks, nullptr, 10);
    if (tmp > 0) {
      AS_RING_BCAST_CHUNKS = static_cast<uint64_t>(tmp);
    }
  }

  this->num_chunks = static_cast<int>(AS_RING_BCAST_CHUNKS);
  if (this->num_chunks <= 0) {
    this->num_chunks = 1;
  }

  // Avoid creating zero-byte chunks: cap num_chunks at data_size.
  if (data_size > 0 && static_cast<uint64_t>(this->num_chunks) > data_size) {
    this->num_chunks = static_cast<int>(data_size);
  }
  if (this->num_chunks <= 0) {
    this->num_chunks = 1;
  }

  // chunk_tag() encodes the chunk index into the transport tag as:
  //   tag = stream_num * kChunkTagStride + chunk_idx
  // This limits chunk_idx to [0, kChunkTagStride-2]; the last slot is
  // reserved for the completion ACK tag.
  static constexpr int kChunkTagStride = 4096;
  if (this->num_chunks >= kChunkTagStride) {
    if (id == 0) {
      std::cout << "RingBroadcast: AS_RING_BCAST_CHUNKS=" << this->num_chunks
                << " is too large for current chunk tag encoding; "
                << "falling back to 1 chunk." << std::endl;
    }
    this->num_chunks = 1;
  }

  // Divide data_size as evenly as possible across chunks.
  // Chunks [0, remainder_bytes) get one extra byte (base_chunk_size + 1).
  this->base_chunk_size =
      (this->num_chunks == 0) ? 0 : (data_size / this->num_chunks);
  this->remainder_bytes =
      (this->num_chunks == 0) ? 0 : (data_size % this->num_chunks);

  // Kept for compatibility/debug prints. Actual send/recv size is computed
  // per chunk via chunk_size_bytes().
  this->msg_size = this->base_chunk_size;

  std::cout << "RingBroadcast being initialized:" << std::endl <<
      "\t type: " << comtype_to_string(type) << std::endl <<
      "\t id: " << id << std::endl <<
      "\t layer num: " << layer_num << std::endl <<
      "\t data_size: " << data_size << std::endl <<
      "\t AS_RING_BCAST_CHUNKS: " << AS_RING_BCAST_CHUNKS << std::endl <<
      "\t msg_size: " << msg_size << std::endl <<
      "\t direction: " << RingTopology::direction_to_string(direction) << std::endl <<
      "\t injection_policy: " << injection_policy_to_string(injection_policy) << std::endl << 
      "\t boost_mode: " << boost_mode << std::endl << 
      "\t root: " << this->root << std::endl << 
      "\t nodes_in_ring: " << this->nodes_in_ring << std::endl << 
      "\t current_sender: " << this->current_sender << std::endl << 
      "\t current_receiver: " << this->current_receiver << std::endl <<
      "---------------------------\n" << std::endl;

  // Pipeline progress counters:
  //   chunks_staged:   how many chunks root has pushed into the packet queue.
  //   chunks_sent:     how many chunks have been handed to front_end_sim_send.
  //   chunks_received: how many chunks this node has received (non-root only).
  //   next_chunk_to_post: the next chunk index to arm a recv for.
  //   recv_chunk_posted: the chunk index whose recv is currently outstanding
  //                      (-1 = no pending recv).
  this->chunks_staged = 0;
  this->chunks_sent = 0;
  this->chunks_received = 0;
  this->next_chunk_to_post = 0;
  this->recv_chunk_posted = -1;

  // Three-way handshake flags for the end-to-end completion ACK:
  //   completion_ack_posted:   root has armed its recv for the ACK.
  //   completion_ack_received: root's recv completed (ACK arrived).
  //   completion_ack_sent:     last node has sent the ACK to root.
  this->completion_ack_posted = false;
  this->completion_ack_received = false;
  this->completion_ack_sent = false;

  this->name = Name::Ring;
  this->enabled = true;
  if (boost_mode) {
    this->enabled = ring_topology->is_enabled();
  }

  // Local (intra-node) rings use the fast memory bus; remote rings use the
  // standard (slower) path.
  if (ring_topology->dimension == RingTopology::Dimension::Local) {
    transmition = MemBus::Transmition::Fast;
  } else {
    transmition = MemBus::Transmition::Usual;
  }
}

// Returns true if this node is the broadcast root (data origin).
bool RingBroadcast::is_root() const {
  return id == root;
}

// Returns true if this node is the "last" node in the broadcast chain,
// i.e. its downstream neighbor is root.  The last node is responsible for
// sending the completion ACK back to root.
bool RingBroadcast::is_last() const {
  return current_receiver == root;
}

// Encodes a chunk index into a unique transport tag for this stream.
// Tag layout: stream_num * kChunkTagStride + chunk_idx
// The stride leaves one reserved slot (kChunkTagStride - 1) for the
// completion ACK, so chunk_idx must be in [0, kChunkTagStride - 2).
int RingBroadcast::chunk_tag(int chunk_idx) const {
  static constexpr int kChunkTagStride = 4096;
  return stream->stream_num * kChunkTagStride + chunk_idx;
}

// Returns the byte size of chunk `chunk_idx`.
// Chunks [0, remainder_bytes) are one byte larger than base_chunk_size to
// absorb the remainder of data_size / num_chunks.
uint64_t RingBroadcast::chunk_size_bytes(int chunk_idx) const {
  return base_chunk_size +
         ((static_cast<uint64_t>(chunk_idx) < remainder_bytes) ? 1ULL : 0ULL);
}

// Returns the reserved transport tag used for the end-to-end completion ACK.
// Uses the last slot in this stream's tag range (kChunkTagStride - 1).
int RingBroadcast::completion_ack_tag() const {
  static constexpr int kChunkTagStride = 4096;
  return stream->stream_num * kChunkTagStride + (kChunkTagStride - 1);
}

// Arms the root's recv for the completion ACK.
// Called once during StreamInit on the root node.  When the last node in the
// ring has forwarded all chunks, it calls send_completion_ack(), which
// triggers a PacketReceived event on root with this tag.
void RingBroadcast::post_completion_ack_recv() {
  if (!enabled || !is_root() || completion_ack_posted) {
    return;
  }

  sim_request rcv_req;
  rcv_req.vnet = this->stream->current_queue_id;
  rcv_req.layerNum = layer_num;

  RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
      stream,
      stream->owner->id,
      EventType::PacketReceived,
      stream->current_queue_id,
      completion_ack_tag());

  // Root receives the completion ack from its sender in the ring,
  // which is the last node in the broadcast chain.
  stream->owner->front_end_sim_recv(
      0,
      Sys::dummy_data,
      1,  // tiny completion token (1 byte)
      UINT8,
      current_sender,
      completion_ack_tag(),
      &rcv_req,
      &Sys::handleEvent,
      ehd);

  completion_ack_posted = true;
}

// Sends a 1-byte completion ACK from the last node back to root.
// Called by the last node in run(PacketReceived) once it has received all
// num_chunks data chunks, signalling that the full broadcast is complete.
void RingBroadcast::send_completion_ack() {
  if (!enabled || !is_last() || completion_ack_sent) {
    return;
  }

  sim_request snd_req;
  snd_req.srcRank = id;
  snd_req.dstRank = current_receiver;  // root
  snd_req.tag = completion_ack_tag();
  snd_req.reqType = UINT8;
  snd_req.vnet = this->stream->current_queue_id;
  snd_req.layerNum = layer_num;

  stream->owner->front_end_sim_send(
      0,
      Sys::dummy_data,
      1,  // tiny completion token (1 byte)
      UINT8,
      current_receiver,
      completion_ack_tag(),
      &snd_req,
      &Sys::handleEvent,
      nullptr);

  completion_ack_sent = true;
}

// Arms a network recv for data chunk `chunk_idx` from this node's upstream
// sender.  Only one recv may be outstanding at a time (guarded by
// recv_chunk_posted != -1).  When the chunk arrives, the simulator fires
// EventType::PacketReceived on this node.
void RingBroadcast::post_data_recv(int chunk_idx) {
  if (!enabled || is_root()) {
    return;  // Root never receives broadcast data; it only receives the ACK.
  }
  if (chunk_idx < 0 || chunk_idx >= num_chunks) {
    return;
  }
  if (recv_chunk_posted != -1) {
    return;  // A recv is already outstanding; only one at a time.
  }

  const int tag = chunk_tag(chunk_idx);
  const uint64_t bytes = chunk_size_bytes(chunk_idx);

  sim_request rcv_req;
  rcv_req.vnet = this->stream->current_queue_id;
  rcv_req.layerNum = layer_num;

  RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
      stream,
      stream->owner->id,
      EventType::PacketReceived,
      stream->current_queue_id,
      tag);

  stream->owner->front_end_sim_recv(
      0,
      Sys::dummy_data,
      bytes,
      UINT8,
      current_sender,
      tag,
      &rcv_req,
      &Sys::handleEvent,
      ehd);

  recv_chunk_posted = chunk_idx;
}

// Enqueues a data packet for chunk `chunk_idx` into the send pipeline.
//   from_npu = true:  data originates from the NPU (root sending original data)
//                     → route through the memory bus to MA (network).
//   from_npu = false: data was just received from the network (relay node)
//                     → route back through MA → NPU direction (not used here;
//                        the flag selects the memory bus direction in
//                        release_packets()).
// The packet is added to `packets` and `locked_packets`; release_packets()
// immediately submits it to the memory bus for packetization.  When
// packetization completes, EventType::General fires and free_packets is
// incremented, unblocking ready().
void RingBroadcast::stage_data_packet(int chunk_idx, bool from_npu) {
  if (!enabled) {
    return;
  }
  if (chunk_idx < 0 || chunk_idx >= num_chunks) {
    return;
  }

  packets.push_back(
      MyPacket(stream->current_queue_id, id, current_receiver));
  packets.back().sender = nullptr;
  // Encode the chunk index in stream_num so the receiver can match the tag.
  packets.back().stream_num = chunk_tag(chunk_idx);

  locked_packets.push_back(&packets.back());
  packet_chunks.push_back(chunk_idx);

  processed = false;
  send_back = false;
  send_from_npu = from_npu;

  release_packets(chunk_size_bytes(chunk_idx));
  chunks_staged++;
}

// Submits all locked_packets to the memory bus for packetization.
//   send_from_npu = true  → send_to_MA()  (NPU → network memory bus path)
//   send_from_npu = false → send_to_NPU() (network → NPU memory bus path)
// After this call locked_packets is cleared; when the memory bus is done,
// it will fire EventType::General which increments free_packets and allows
// ready() to dispatch the packet to the network.
void RingBroadcast::release_packets(uint64_t packet_size) {
  for (auto packet : locked_packets) {
    packet->set_notifier(this);
  }

  if (send_from_npu == true) {
    (new PacketBundle(stream->owner,
                      stream,
                      locked_packets,
                      processed,
                      send_back,
                      packet_size,
                      transmition))
        ->send_to_MA();
  } else {
    (new PacketBundle(stream->owner,
                      stream,
                      locked_packets,
                      processed,
                      send_back,
                      packet_size,
                      transmition))
        ->send_to_NPU();
  }

  locked_packets.clear();
}

// Checks whether the front packet in the send queue is ready to be dispatched
// to the network.  Returns true if a packet was sent, false otherwise.
//
// A packet is "ready" when free_packets > 0 (the memory bus has finished
// packetizing it).  This function is called on every EventType::General event.
bool RingBroadcast::ready() {
  // Transition stream state to Executing on first activity.
  if (stream->state == StreamState::Created ||
      stream->state == StreamState::Ready) {
    stream->changeState(StreamState::Executing);
  }

  if (!enabled || packets.empty() || free_packets == 0) {
    return false;
  }

  // Dequeue the front packet and its corresponding chunk index.
  MyPacket packet = packets.front();
  int chunk_idx = packet_chunks.front();
  int tag = chunk_tag(chunk_idx);
  uint64_t bytes = chunk_size_bytes(chunk_idx);

  sim_request snd_req;
  snd_req.srcRank = id;
  snd_req.dstRank = packet.preferred_dest;
  snd_req.tag = tag;
  snd_req.reqType = UINT8;
  snd_req.vnet = this->stream->current_queue_id;
  snd_req.layerNum = layer_num;

  stream->owner->front_end_sim_send(
      0,
      Sys::dummy_data,
      bytes,
      UINT8,
      packet.preferred_dest,
      tag,
      &snd_req,
      &Sys::handleEvent,
      nullptr);

  packets.pop_front();
  packet_chunks.pop_front();
  free_packets--;
  chunks_sent++;
  // send_done becomes true once all chunks have been handed to the network.
  send_done = (chunks_sent == num_chunks);

  return true;
}

// Checks whether this node has completed all of its obligations and, if so,
// calls exit() to advance the stream.
//
// Root exit conditions (all must hold):
//   1. All chunks have been staged into the send queue.
//   2. All chunks have been dispatched to the network (chunks_sent).
//   3. No packets remain in flight locally (packets and locked_packets empty).
//   4. The end-to-end completion ACK has been received from the last node.
//
// Non-root exit conditions (all must hold):
//   1. All chunks have been received (chunks_received == num_chunks).
//   2. For intermediate nodes: all received chunks have been forwarded.
//   3. For the last node: the completion ACK has been sent to root.
//   4. No recv is currently outstanding (recv_chunk_posted == -1).
//   5. No packets remain in flight locally.
void RingBroadcast::maybe_exit() {
  if (!enabled || exited) {
    return;
  }

  if (is_root()) {
    if (chunks_staged < num_chunks) {
      return;
    }
    if (chunks_sent < num_chunks) {
      return;
    }
    if (!packets.empty() || !locked_packets.empty()) {
      return;
    }
    if (!completion_ack_received) {
      return;
    }
    exit();
    return;
  }

  // Non-root path:
  if (chunks_received < num_chunks) {
    return;
  }

  // Intermediate nodes must forward every chunk before exiting.
  if (!is_last() && chunks_sent < num_chunks) {
    return;
  }

  // Last node must have sent its completion ACK before exiting.
  if (is_last() && !completion_ack_sent) {
    return;
  }

  // Wait for any outstanding recv to complete.
  if (recv_chunk_posted != -1) {
    return;
  }

  if (!packets.empty() || !locked_packets.empty()) {
    return;
  }

  exit();
}

// Central event dispatcher.  Three event types are handled:
//
//   EventType::General
//     Fired by the memory bus when packetization of a staged packet is done.
//     Increments free_packets, then calls ready() to dispatch the packet to
//     the network.  Root also pipelines the next chunk here.
//
//   EventType::PacketReceived
//     Fired when a network recv completes (data chunk or completion ACK).
//     On root:  marks completion_ack_received = true and tries to exit.
//     On others: relays the received chunk downstream (or sends ACK if last),
//                then arms the next chunk's recv.
//
//   EventType::StreamInit
//     Fired once at stream startup to kick off the algorithm:
//     Root: arms the completion-ACK recv and stages the first data chunk.
//     Others: arm the recv for the first data chunk.
void RingBroadcast::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    // Memory bus finished packetizing; one more packet is now sendable.
    free_packets += 1;

    ready();

    // Root pipelines: stage the next chunk as soon as the previous one has
    // been handed off to the memory bus (chunks_staged tracks this).
    if (is_root() && chunks_staged < num_chunks) {
      stage_data_packet(chunks_staged, true);
    }

    maybe_exit();
    return;
  }

  if (event == EventType::PacketReceived) {
    // Root does not receive broadcast data in this algorithm.
    // The only receive posted on root is the final completion ack.
    if (is_root()) {
      completion_ack_received = true;
      maybe_exit();
      return;
    }

    // Determine which chunk just arrived.
    // recv_chunk_posted is the authoritative index; fall back to
    // chunks_received as a safety net if the bookkeeping is somehow off.
    int received_chunk = recv_chunk_posted;

    if (received_chunk < 0 || received_chunk >= num_chunks) {
      received_chunk = chunks_received;
    }

    // Clear the outstanding recv slot so the next post_data_recv() can proceed.
    recv_chunk_posted = -1;
    chunks_received++;
    recv_done = (chunks_received == num_chunks);

    // Eagerly arm the recv for the next chunk (pipelining: overlap network
    // recv with local forwarding work).
    if (next_chunk_to_post < num_chunks) {
      post_data_recv(next_chunk_to_post);
      if (recv_chunk_posted != -1) {
        next_chunk_to_post++;
      }
    }

    if (!is_last()) {
      // Intermediate node: relay the received chunk to the next node.
      stage_data_packet(received_chunk, false);
    } else if (chunks_received == num_chunks) {
      // Last node has received the final chunk: notify root that the
      // end-to-end broadcast is really complete.
      send_completion_ack();
    }

    maybe_exit();
    return;
  }

  if (event == EventType::StreamInit) {
    if (!enabled) {
      return;
    }

    // Trivial case: only one node, nothing to broadcast.
    if (nodes_in_ring <= 1) {
      exit();
      return;
    }

    if (is_root()) {
      // Arm the completion-ACK recv first so it is ready before any data
      // could possibly complete.  Then kick off chunk 0.
      post_completion_ack_recv();
      stage_data_packet(0, true);
    } else {
      // Non-root: arm the recv for chunk 0 and advance the posting pointer.
      post_data_recv(next_chunk_to_post);
      if (recv_chunk_posted != -1) {
        next_chunk_to_post++;
      }
    }

    return;
  }

}

// Marks this algorithm instance as done and advances the stream to the next
// virtual network baseline.  Clears all packet queues to release memory.
void RingBroadcast::exit() {
  if (exited) {
    return;
  }
  exited = true;

  if (!packets.empty()) {
    packets.clear();
  }

  if (!locked_packets.empty()) {
    locked_packets.clear();
  }

  packet_chunks.clear();

  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}

}  // namespace AstraSim

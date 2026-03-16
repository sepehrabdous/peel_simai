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

  // Keep your current ring-root behavior.
  // If later you want arbitrary roots, change this to:
  // this->root = root;
  this->root = id - ring_topology->index_in_ring * ring_topology->offset;

  this->logicalTopology = ring_topology;
  this->data_size = data_size;
  this->final_data_size = data_size;
  this->direction = direction;
  this->nodes_in_ring = ring_topology->get_nodes_in_ring();
  this->current_receiver = ring_topology->get_receiver_node(id, direction);
  this->current_sender = ring_topology->get_sender_node(id, direction);
  this->injection_policy = injection_policy;

  this->free_packets = 0;
  this->processed = false;
  this->send_back = false;
  this->send_from_npu = true;
  this->recv_done = (id == this->root);
  this->send_done = false;
  this->exited = false;

  const char* env_chunks = std::getenv("AS_RING_BCAST_CHUNKS");
  if (env_chunks != nullptr) {
    unsigned long long tmp = std::strtoull(env_chunks, nullptr, 10);
    if (tmp > 0) {
      AS_RING_BCAST_CHUNKS = static_cast<uint64_t>(tmp);
    }
  }

  if (id == 0) {
    std::cout << "AS_RING_BCAST_CHUNKS: " << AS_RING_BCAST_CHUNKS
              << std::endl;
  }

  this->num_chunks = static_cast<int>(AS_RING_BCAST_CHUNKS);
  if (this->num_chunks <= 0) {
    this->num_chunks = 1;
  }

  // Avoid creating zero-byte chunks.
  if (data_size > 0 && static_cast<uint64_t>(this->num_chunks) > data_size) {
    this->num_chunks = static_cast<int>(data_size);
  }
  if (this->num_chunks <= 0) {
    this->num_chunks = 1;
  }

  // We encode chunk_idx into the transport tag.
  static constexpr int kChunkTagStride = 4096;
  if (this->num_chunks >= kChunkTagStride) {
    if (id == 0) {
      std::cout << "RingBroadcast: AS_RING_BCAST_CHUNKS=" << this->num_chunks
                << " is too large for current chunk tag encoding; "
                << "falling back to 1 chunk." << std::endl;
    }
    this->num_chunks = 1;
  }

  this->base_chunk_size =
      (this->num_chunks == 0) ? 0 : (data_size / this->num_chunks);
  this->remainder_bytes =
      (this->num_chunks == 0) ? 0 : (data_size % this->num_chunks);

  // Kept for compatibility/debug prints. Actual send/recv size is computed
  // per chunk.
  this->msg_size = this->base_chunk_size;

  this->chunks_staged = 0;
  this->chunks_sent = 0;
  this->chunks_received = 0;
  this->next_chunk_to_post = 0;
  this->recv_chunk_posted = -1;

  this->name = Name::Ring;
  this->enabled = true;
  if (boost_mode) {
    this->enabled = ring_topology->is_enabled();
  }

  if (ring_topology->dimension == RingTopology::Dimension::Local) {
    transmition = MemBus::Transmition::Fast;
  } else {
    transmition = MemBus::Transmition::Usual;
  }
}

bool RingBroadcast::is_root() const {
  return id == root;
}

bool RingBroadcast::is_last() const {
  return current_receiver == root;
}

int RingBroadcast::chunk_tag(int chunk_idx) const {
  static constexpr int kChunkTagStride = 4096;
  return stream->stream_num * kChunkTagStride + chunk_idx;
}

uint64_t RingBroadcast::chunk_size_bytes(int chunk_idx) const {
  return base_chunk_size +
         ((static_cast<uint64_t>(chunk_idx) < remainder_bytes) ? 1ULL : 0ULL);
}

void RingBroadcast::post_data_recv(int chunk_idx) {
  if (!enabled || is_root()) {
    return;
  }
  if (chunk_idx < 0 || chunk_idx >= num_chunks) {
    return;
  }
  if (recv_chunk_posted != -1) {
    return;
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

void RingBroadcast::stage_data_packet(int chunk_idx, bool from_npu) {
  if (!enabled) {
    return;
  }
  if (chunk_idx < 0 || chunk_idx >= num_chunks) {
    return;
  }

  packets.push_back(
      MyPacket(stream->current_queue_id, current_sender, current_receiver));
  packets.back().sender = nullptr;
  packets.back().stream_num = chunk_tag(chunk_idx);

  locked_packets.push_back(&packets.back());
  packet_chunks.push_back(chunk_idx);

  processed = false;
  send_back = false;
  send_from_npu = from_npu;

  release_packets(chunk_size_bytes(chunk_idx));
  chunks_staged++;
}

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

bool RingBroadcast::ready() {
  if (stream->state == StreamState::Created ||
      stream->state == StreamState::Ready) {
    stream->changeState(StreamState::Executing);
  }

  if (!enabled || packets.empty() || free_packets == 0) {
    return false;
  }

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
  send_done = (chunks_sent == num_chunks);

  return true;
}

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
    exit();
    return;
  }

  if (chunks_received < num_chunks) {
    return;
  }

  if (!is_last() && chunks_sent < num_chunks) {
    return;
  }

  if (recv_chunk_posted != -1) {
    return;
  }

  if (!packets.empty() || !locked_packets.empty()) {
    return;
  }

  exit();
}

void RingBroadcast::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    free_packets += 1;

    ready();

    // Root stages the next chunk only after the previous local packetization
    // step has completed.
    if (is_root() && chunks_staged < num_chunks) {
      stage_data_packet(chunks_staged, true);
    }

    maybe_exit();
    return;
  }

  if (event == EventType::PacketReceived) {
    int received_chunk = recv_chunk_posted;

    // Defensive fallback in case bookkeeping and callback ordering ever drift.
    if (received_chunk < 0 || received_chunk >= num_chunks) {
      received_chunk = chunks_received;
    }

    recv_chunk_posted = -1;
    chunks_received++;
    recv_done = (chunks_received == num_chunks);

    // Post the next receive immediately so the incoming pipeline can continue.
    if (next_chunk_to_post < num_chunks) {
      post_data_recv(next_chunk_to_post);
      if (recv_chunk_posted != -1) {
        next_chunk_to_post++;
      }
    }

    if (!is_last()) {
      // Forward exactly the chunk that arrived.
      stage_data_packet(received_chunk, false);
    }

    maybe_exit();
    return;
  }

  if (event == EventType::StreamInit) {
    if (!enabled) {
      return;
    }

    if (nodes_in_ring <= 1) {
      exit();
      return;
    }

    if (is_root()) {
      stage_data_packet(0, true);
    } else {
      post_data_recv(next_chunk_to_post);
      if (recv_chunk_posted != -1) {
        next_chunk_to_post++;
      }
    }

    return;
  }
}

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
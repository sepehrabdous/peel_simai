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

  // Keep your current root behavior for now.
  // If you later want arbitrary broadcast roots, change this to:
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

  this->waiting_for_ack = false;
  this->ack_received = false;
  this->ack_sent = false;
  this->ack_recv_posted = false;

  this->ack_size = 1;

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

  // Keep equal-size chunks for now.
  if (data_size == 0 || (data_size % this->num_chunks) != 0) {
    if (id == 0 && data_size != 0 && (data_size % this->num_chunks) != 0) {
      std::cout << "RingBroadcast: data_size " << data_size
                << " is not divisible by AS_RING_BCAST_CHUNKS="
                << this->num_chunks
                << "; falling back to 1 chunk for correctness."
                << std::endl;
    }
    this->num_chunks = 1;
  }

  this->msg_size = data_size / this->num_chunks;

  this->chunks_staged = 0;
  this->chunks_sent = 0;
  this->chunks_received = 0;
  this->posted_data_recvs = 0;

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

void RingBroadcast::post_data_recv() {
  if (!enabled || is_root() || posted_data_recvs >= num_chunks) {
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
      stream->stream_num);

  stream->owner->front_end_sim_recv(
      0,
      Sys::dummy_data,
      msg_size,
      UINT8,
      current_sender,
      stream->stream_num,
      &rcv_req,
      &Sys::handleEvent,
      ehd);

  posted_data_recvs++;
}

void RingBroadcast::post_ack_recv() {
  if (!enabled || ack_recv_posted || is_last()) {
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
      stream->stream_num);

  // ACK travels backward, so each node receives it from current_receiver.
  stream->owner->front_end_sim_recv(
      0,
      Sys::dummy_data,
      ack_size,
      UINT8,
      current_receiver,
      stream->stream_num,
      &rcv_req,
      &Sys::handleEvent,
      ehd);

  ack_recv_posted = true;
}

void RingBroadcast::stage_data_packet(bool from_npu) {
  if (!enabled || chunks_staged >= num_chunks) {
    return;
  }

  packets.push_back(MyPacket(stream->current_queue_id, id, current_receiver));
  packets.back().sender = nullptr;
  locked_packets.push_back(&packets.back());
  packet_is_ack.push_back(false);

  processed = false;
  send_back = false;
  send_from_npu = from_npu;

  release_packets();
  chunks_staged++;
}

void RingBroadcast::stage_ack_packet(bool from_npu, int dest) {
  if (!enabled || ack_sent) {
    return;
  }

  packets.push_back(MyPacket(stream->current_queue_id, id, dest));
  packets.back().sender = nullptr;
  locked_packets.push_back(&packets.back());
  packet_is_ack.push_back(true);

  processed = false;
  send_back = false;
  send_from_npu = from_npu;

  release_packets();
}

void RingBroadcast::release_packets() {
  for (auto packet : locked_packets) {
    packet->set_notifier(this);
  }

  uint64_t bundle_size = msg_size;
  if (!packet_is_ack.empty() && packet_is_ack.back()) {
    bundle_size = ack_size;
  }

  if (send_from_npu == true) {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         bundle_size,
         transmition))
        ->send_to_MA();
  } else {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         bundle_size,
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
  bool is_ack_pkt = packet_is_ack.front();
  uint64_t send_size = is_ack_pkt ? ack_size : msg_size;

  sim_request snd_req;
  snd_req.srcRank = id;
  snd_req.dstRank = packet.preferred_dest;
  snd_req.tag = stream->stream_num;
  snd_req.reqType = UINT8;
  snd_req.vnet = this->stream->current_queue_id;
  snd_req.layerNum = layer_num;

  stream->owner->front_end_sim_send(
      0,
      Sys::dummy_data,
      send_size,
      UINT8,
      packet.preferred_dest,
      stream->stream_num,
      &snd_req,
      &Sys::handleEvent,
      nullptr);

  packets.pop_front();
  packet_is_ack.pop_front();
  free_packets--;

  if (is_ack_pkt) {
    ack_sent = true;
  } else {
    chunks_sent++;
    send_done = (chunks_sent == num_chunks);
  }

  return true;
}

void RingBroadcast::maybe_exit() {
  if (!enabled || exited) {
    return;
  }

  if (is_root()) {
    // Root must wait for the reverse ACK.
    if (!ack_received) {
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

  if (is_last()) {
    if (!ack_sent) {
      return;
    }
  } else {
    if (!ack_received || !ack_sent) {
      return;
    }
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

    // Root injects one new data chunk whenever local path frees up.
    if (is_root() && chunks_staged < num_chunks) {
      stage_data_packet(true);
    }

    maybe_exit();
    return;
  }

  if (event == EventType::PacketReceived) {
    // If this node is currently waiting for the reverse ACK,
    // this receive is the ACK, not a data chunk.
    if (waiting_for_ack) {
      ack_received = true;
      waiting_for_ack = false;
      ack_recv_posted = false;

      if (!is_root()) {
        // Forward ACK one hop backward toward the root.
        stage_ack_packet(false, current_sender);
      }

      maybe_exit();
      return;
    }

    // Otherwise this is a forward data chunk.
    chunks_received++;
    recv_done = (chunks_received == num_chunks);

    if (!is_last()) {
      stage_data_packet(false);
    }

    if (chunks_received == num_chunks) {
      if (is_last()) {
        // Last node originates the drain ACK.
        stage_ack_packet(true, current_sender);
      } else {
        waiting_for_ack = true;
        post_ack_recv();
      }
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
      // Root starts the data pipeline and also waits for final ACK
      // from its forward neighbor.
      stage_data_packet(true);
      waiting_for_ack = true;
      post_ack_recv();
    } else {
      for (int i = 0; i < num_chunks; i++) {
        post_data_recv();
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
  if (!packet_is_ack.empty()) {
    packet_is_ack.clear();
  }

  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}

} // namespace AstraSim
#include "RingBroadcast.hh"

#include <cstdlib>
#include <iostream>

#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"

namespace AstraSim {

std::unordered_map<std::string, RingBroadcast*> RingBroadcast::root_waiters;
std::recursive_mutex RingBroadcast::root_waiters_mutex;

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

  // Keep current behavior for choosing ring root.
  // If later you want arbitrary roots from the caller, change this to:
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
  this->drain_complete.store(false, std::memory_order_relaxed);

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

std::string RingBroadcast::completion_key() const {
  return std::to_string(root) + "|" +
         std::to_string(layer_num) + "|" +
         std::to_string(stream->stream_num) + "|" +
         std::to_string(stream->current_queue_id) + "|" +
         std::to_string(static_cast<int>(direction));
}

void RingBroadcast::notify_root_drain_complete() {
  std::lock_guard<std::recursive_mutex> lock(root_waiters_mutex);
  auto it = root_waiters.find(completion_key());
  if (it != root_waiters.end() && it->second != nullptr) {
    RingBroadcast* root_alg = it->second;
    root_alg->drain_complete.store(true, std::memory_order_release);
    root_alg->maybe_exit();
  }
}

void RingBroadcast::post_data_recv(int src, int vnet) {
  if (!enabled || is_root() || posted_data_recvs >= num_chunks) {
    return;
  }

  sim_request rcv_req;
  rcv_req.vnet = vnet;
  rcv_req.layerNum = layer_num;

  RecvPacketEventHadndlerData* ehd = new RecvPacketEventHadndlerData(
      stream,
      stream->owner->id,
      EventType::PacketReceived,
      vnet,
      stream->stream_num);

  stream->owner->front_end_sim_recv(
      0,
      Sys::dummy_data,
      msg_size,
      UINT8,
      src,
      stream->stream_num,
      &rcv_req,
      &Sys::handleEvent,
      ehd);

  posted_data_recvs++;
}

void RingBroadcast::stage_data_packet(bool from_npu) {
  if (!enabled || chunks_staged >= num_chunks) {
    return;
  }

  packets.push_back(
      MyPacket(stream->current_queue_id, current_sender, current_receiver));
  packets.back().sender = nullptr;

  // IMPORTANT: initialize packet metadata explicitly.
  packets.back().stream_num = stream->stream_num;

  locked_packets.push_back(&packets.back());

  processed = false;
  send_back = false;
  send_from_npu = from_npu;

  release_packets();
  chunks_staged++;
}

void RingBroadcast::release_packets() {
  for (auto packet : locked_packets) {
    packet->set_notifier(this);
  }

  if (send_from_npu == true) {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         msg_size,
         transmition))
        ->send_to_MA();
  } else {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         msg_size,
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
      msg_size,
      UINT8,
      packet.preferred_dest,
      stream->stream_num,
      &snd_req,
      &Sys::handleEvent,
      nullptr);

  if (!is_root() && !is_last() && posted_data_recvs < num_chunks) {
    post_data_recv(packet.preferred_src, packet.preferred_vnet);
  }

  packets.pop_front();
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
    if (!drain_complete.load(std::memory_order_acquire)) {
      return;
    }
    if (chunks_staged < num_chunks || chunks_sent < num_chunks) {
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

  if (!packets.empty() || !locked_packets.empty()) {
    return;
  }

  exit();
}

void RingBroadcast::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    free_packets += 1;
    ready();

    if (is_root() && chunks_staged < num_chunks) {
      stage_data_packet(true);
    }

    maybe_exit();
    return;
  }

  if (event == EventType::PacketReceived) {
    chunks_received++;
    recv_done = (chunks_received == num_chunks);

    if (is_last()) {
      if (chunks_received < num_chunks) {
        post_data_recv(current_sender, stream->current_queue_id);
      } else {
        notify_root_drain_complete();
      }
    } else {
      stage_data_packet(false);
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
      {
        std::lock_guard<std::recursive_mutex> lock(root_waiters_mutex);
        root_waiters[completion_key()] = this;
      }
      stage_data_packet(true);
    } else {
      post_data_recv(current_sender, stream->current_queue_id);
    }

    return;
  }
}

void RingBroadcast::exit() {
  if (exited) {
    return;
  }
  exited = true;

  if (is_root()) {
    std::lock_guard<std::recursive_mutex> lock(root_waiters_mutex);
    auto it = root_waiters.find(completion_key());
    if (it != root_waiters.end() && it->second == this) {
      root_waiters.erase(it);
    }
  }

  if (!packets.empty()) {
    packets.clear();
  }
  if (!locked_packets.empty()) {
    locked_packets.clear();
  }

  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}

}  // namespace AstraSim
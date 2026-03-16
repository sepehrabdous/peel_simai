#include "RingBroadcast.hh"

#include <cstdlib>
#include <iostream>

#include "astra-sim/system/PacketBundle.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"

namespace AstraSim {

std::unordered_map<std::string, RingBroadcast::RootCompletionState>
    RingBroadcast::root_waiters;
std::recursive_mutex RingBroadcast::root_waiters_mutex;

std::unordered_map<std::string, int> RingBroadcast::hop_credits;
std::recursive_mutex RingBroadcast::hop_credits_mutex;

std::unordered_map<std::string, RingBroadcast*> RingBroadcast::edge_senders;
std::recursive_mutex RingBroadcast::edge_senders_mutex;

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

  // Keeping your current ring-root behavior.
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

void RingBroadcast::notify_nonroot_exit() {
  std::lock_guard<std::recursive_mutex> lock(root_waiters_mutex);

  auto it = root_waiters.find(completion_key());
  if (it == root_waiters.end()) {
    return;
  }

  it->second.completed_nonroots++;

  if (it->second.root_alg != nullptr &&
      it->second.completed_nonroots >= it->second.expected_nonroots) {
    it->second.root_alg->maybe_exit();
  }
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

void RingBroadcast::stage_data_packet(bool from_npu) {
  if (!enabled || chunks_staged >= num_chunks) {
    return;
  }

  packets.push_back(
      MyPacket(stream->current_queue_id, current_sender, current_receiver));
  packets.back().sender = nullptr;
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

bool RingBroadcast::consume_outgoing_credit() {
  if (is_last()) {
    return true;
  }

  std::lock_guard<std::recursive_mutex> lock(hop_credits_mutex);
  auto key = outgoing_edge_key();
  auto it = hop_credits.find(key);
  if (it == hop_credits.end() || it->second == 0) {
    return false;
  }

  it->second--;
  return true;
}

bool RingBroadcast::ready() {
  if (stream->state == StreamState::Created ||
      stream->state == StreamState::Ready) {
    stream->changeState(StreamState::Executing);
  }

  if (!enabled || packets.empty() || free_packets == 0) {
    return false;
  }

  // Enforce depth-1 per outgoing link.
  if (!consume_outgoing_credit()) {
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

  packets.pop_front();
  free_packets--;
  chunks_sent++;
  send_done = (chunks_sent == num_chunks);

  return true;
}

bool RingBroadcast::try_progress_send() {
  bool sent = ready();

  // Root is allowed to stage the next chunk only after it actually
  // sent the current queued chunk.
  if (sent && is_root() && chunks_staged < num_chunks) {
    stage_data_packet(true);
  }

  maybe_exit();
  return sent;
}

void RingBroadcast::maybe_exit() {
  if (!enabled || exited) {
    return;
  }

  if (is_root()) {
    int completed_nonroots = 0;
    int expected_nonroots = nodes_in_ring - 1;

    {
      std::lock_guard<std::recursive_mutex> lock(root_waiters_mutex);
      auto it = root_waiters.find(completion_key());
      if (it != root_waiters.end()) {
        completed_nonroots = it->second.completed_nonroots;
        expected_nonroots = it->second.expected_nonroots;
      }
    }

    if (chunks_staged < num_chunks || chunks_sent < num_chunks) {
      return;
    }

    if (completed_nonroots < expected_nonroots) {
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

std::string RingBroadcast::incoming_edge_key() const {
  return std::to_string(current_sender) + "->" +
         std::to_string(id) + "|" +
         std::to_string(layer_num) + "|" +
         std::to_string(stream->stream_num) + "|" +
         std::to_string(stream->current_queue_id) + "|" +
         std::to_string(static_cast<int>(direction));
}

std::string RingBroadcast::outgoing_edge_key() const {
  return std::to_string(id) + "->" +
         std::to_string(current_receiver) + "|" +
         std::to_string(layer_num) + "|" +
         std::to_string(stream->stream_num) + "|" +
         std::to_string(stream->current_queue_id) + "|" +
         std::to_string(static_cast<int>(direction));
}

void RingBroadcast::register_as_edge_sender() {
  if (!enabled || is_last()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(edge_senders_mutex);
  edge_senders[outgoing_edge_key()] = this;
}

void RingBroadcast::unregister_as_edge_sender() {
  if (is_last()) {
    return;
  }

  std::lock_guard<std::recursive_mutex> lock(edge_senders_mutex);
  auto it = edge_senders.find(outgoing_edge_key());
  if (it != edge_senders.end() && it->second == this) {
    edge_senders.erase(it);
  }
}

void RingBroadcast::on_credit_available() {
  if (!enabled || exited) {
    return;
  }
  try_progress_send();
}

void RingBroadcast::grant_incoming_credit() {
  if (is_root()) {
    return;
  }

  RingBroadcast* sender = nullptr;
  std::string key = incoming_edge_key();

  {
    std::lock_guard<std::recursive_mutex> lock(hop_credits_mutex);
    hop_credits[key]++;
  }

  {
    std::lock_guard<std::recursive_mutex> lock(edge_senders_mutex);
    auto it = edge_senders.find(key);
    if (it != edge_senders.end()) {
      sender = it->second;
    }
  }

  // Wake predecessor after granting credit.
  if (sender != nullptr) {
    sender->on_credit_available();
  }
}

void RingBroadcast::cleanup_credit_state() {
  {
    std::lock_guard<std::recursive_mutex> lock(hop_credits_mutex);
    if (!is_root()) {
      hop_credits.erase(incoming_edge_key());
    }
    if (!is_last()) {
      hop_credits.erase(outgoing_edge_key());
    }
  }

  unregister_as_edge_sender();
}

void RingBroadcast::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    free_packets += 1;
    try_progress_send();
    return;
  }

  if (event == EventType::PacketReceived) {
    chunks_received++;
    recv_done = (chunks_received == num_chunks);

    // As soon as this node fully receives chunk k,
    // it should be ready to receive chunk k+1.
    if (!is_root() && chunks_received < num_chunks) {
      post_data_recv();
      grant_incoming_credit();
    }

    // Then forward the chunk that was just received.
    if (!is_last()) {
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

    if (!is_last()) {
      register_as_edge_sender();
    }

    if (is_root()) {
      {
        std::lock_guard<std::recursive_mutex> lock(root_waiters_mutex);
        RootCompletionState state;
        state.root_alg = this;
        state.completed_nonroots = 0;
        state.expected_nonroots = nodes_in_ring - 1;
        root_waiters[completion_key()] = state;
      }

      // Start with exactly one chunk staged.
      stage_data_packet(true);
    } else {
      // Initial receive post for chunk 0.
      post_data_recv();

      // Initial credit to predecessor so it may send chunk 0 on this edge.
      grant_incoming_credit();
    }

    return;
  }
}

void RingBroadcast::exit() {
  if (exited) {
    return;
  }
  exited = true;

  if (packets.size() != 0) {
    packets.clear();
  }

  if (locked_packets.size() != 0) {
    locked_packets.clear();
  }

  cleanup_credit_state();

  if (is_root()) {
    std::lock_guard<std::recursive_mutex> lock(root_waiters_mutex);
    auto it = root_waiters.find(completion_key());
    if (it != root_waiters.end() && it->second.root_alg == this) {
      root_waiters.erase(it);
    }
  } else {
    notify_nonroot_exit();
  }

  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
}

}  // namespace AstraSim
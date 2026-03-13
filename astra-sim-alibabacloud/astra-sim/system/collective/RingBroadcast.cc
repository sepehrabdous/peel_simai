/*

sudo AS_RING_BCAST_CHUNKS=8 AS_SEND_LAT=3 AS_NVLS_ENABLE=1 ./bin/SimAI_simulator -t 16 -w ./example/microBroadcast.txt -n ./DCN+SingleToR_512g_8gps_100Gbps_A100 -c astra-sim-alibabacloud/inputs/config/SimAI.conf

*/


#include "RingBroadcast.hh"

#include <algorithm>
#include <cstdlib>

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
    int /*root*/)
    : Algorithm(layer_num) {
  this->comType = type;
  this->id = id;

  // communicator-local root
  this->root = id - ring_topology->index_in_ring * ring_topology->offset;

  this->logicalTopology = ring_topology;
  this->data_size = data_size;
  this->final_data_size = data_size;
  this->msg_size = data_size;

  this->direction = direction;
  this->nodes_in_ring = ring_topology->get_nodes_in_ring();
  this->current_receiver = ring_topology->get_receiver_node(id, direction);
  this->current_sender = ring_topology->get_sender_node(id, direction);
  this->injection_policy = injection_policy;

  this->free_packets = 0;
  this->processed = false;
  this->send_back = false;
  this->send_from_npu = true;
  this->recv_posted = false;
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

  // Tunable chunk count:
  //   AS_RING_BCAST_CHUNKS=1  -> old whole-message behavior
  //   AS_RING_BCAST_CHUNKS=8  -> split into 8 chunks
  //   AS_RING_BCAST_CHUNKS=16 -> split into 16 chunks
  uint64_t parsed_chunks = 1;
  const char* env_chunks = std::getenv("AS_RING_BCAST_CHUNKS");
  if (env_chunks != nullptr) {
    unsigned long long tmp = std::strtoull(env_chunks, nullptr, 10);
    if (tmp > 0) {
      parsed_chunks = static_cast<uint64_t>(tmp);
    }
  }

  this->requested_chunks = parsed_chunks;

  if (this->msg_size == 0) {
    this->total_chunks = 0;
    this->nominal_chunk_size = 0;
  } else {
    // Do not create more chunks than bytes, otherwise zero-sized chunks appear.
    this->total_chunks = std::min<uint64_t>(this->requested_chunks, this->msg_size);
    if (this->total_chunks == 0) {
      this->total_chunks = 1;
    }
    this->nominal_chunk_size =
        (this->msg_size + this->total_chunks - 1) / this->total_chunks;
  }

  this->received_chunks = is_root() ? this->total_chunks : 0;
  this->sent_chunks = 0;

  this->recv_done = all_chunks_received();
  this->send_done = all_chunks_sent();
}

bool RingBroadcast::is_root() const {
  return id == root;
}

bool RingBroadcast::is_last() const {
  return current_receiver == root;
}

uint64_t RingBroadcast::chunk_size(uint64_t chunk_idx) const {
  if (total_chunks == 0 || chunk_idx >= total_chunks) {
    return 0;
  }

  uint64_t start = chunk_idx * nominal_chunk_size;
  uint64_t remaining = msg_size - start;
  return std::min<uint64_t>(nominal_chunk_size, remaining);
}

bool RingBroadcast::all_chunks_received() const {
  return received_chunks >= total_chunks;
}

bool RingBroadcast::all_chunks_sent() const {
  return sent_chunks >= total_chunks;
}

void RingBroadcast::post_recv() {
  if (!enabled || recv_posted || is_root() || all_chunks_received()) {
    return;
  }

  uint64_t next_chunk_bytes = chunk_size(received_chunks);
  if (next_chunk_bytes == 0) {
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
      next_chunk_bytes,
      UINT8,
      current_sender,
      stream->stream_num,
      &rcv_req,
      &Sys::handleEvent,
      ehd);

  recv_posted = true;
}

void RingBroadcast::stage_packet(bool from_npu, uint64_t chunk_bytes) {
  if (!enabled || send_done || chunk_bytes == 0) {
    return;
  }

  packets.push_back(MyPacket(stream->current_queue_id, id, current_receiver));
  packets.back().sender = nullptr;

  locked_packets.push_back(&packets.back());
  packet_sizes.push_back(chunk_bytes);

  processed = false;
  send_back = false;
  send_from_npu = from_npu;

  release_packets(chunk_bytes);
}

void RingBroadcast::release_packets(uint64_t chunk_bytes) {
  for (auto packet : locked_packets) {
    packet->set_notifier(this);
  }

  if (send_from_npu) {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         chunk_bytes,
         transmition))
        ->send_to_MA();
  } else {
    (new PacketBundle(
         stream->owner,
         stream,
         locked_packets,
         processed,
         send_back,
         chunk_bytes,
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

  if (!enabled || packets.empty() || packet_sizes.empty() || free_packets == 0 ||
      send_done) {
    return false;
  }

  MyPacket packet = packets.front();
  packets.pop_front();

  uint64_t chunk_bytes = packet_sizes.front();
  packet_sizes.pop_front();

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
      chunk_bytes,
      UINT8,
      packet.preferred_dest,
      stream->stream_num,
      &snd_req,
      &Sys::handleEvent,
      nullptr);

  free_packets--;
  sent_chunks++;
  send_done = all_chunks_sent();

  return true;
}

void RingBroadcast::maybe_exit() {
  if (!enabled) {
    return;
  }

  if (total_chunks == 0) {
    exit();
    return;
  }

  if (is_root()) {
    if (all_chunks_sent()) {
      exit();
    }
    return;
  }

  if (is_last()) {
    if (all_chunks_received()) {
      exit();
    }
    return;
  }

  if (all_chunks_received() && all_chunks_sent()) {
    exit();
  }
}

void RingBroadcast::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    free_packets += 1;

    while (ready()) {
      // keep sending while packets are queued and credits are available
    }

    maybe_exit();
    return;
  }

  if (event == EventType::PacketReceived) {
    if (!enabled) {
      return;
    }

    // One posted recv corresponds to exactly one chunk.
    recv_posted = false;

    uint64_t just_received_chunk_idx = received_chunks;
    uint64_t just_received_chunk_bytes = chunk_size(just_received_chunk_idx);

    received_chunks++;
    recv_done = all_chunks_received();

    // Post the next recv first so recv/send can overlap better.
    if (!all_chunks_received()) {
      post_recv();
    }

    // Forward the chunk that just arrived.
    if (!is_last()) {
      stage_packet(false, just_received_chunk_bytes);
    }

    maybe_exit();
    return;
  }

  if (event == EventType::StreamInit) {
    if (!enabled) {
      return;
    }

    if (nodes_in_ring <= 1 || total_chunks == 0) {
      exit();
      return;
    }

    if (is_root()) {
      for (uint64_t chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        stage_packet(true, chunk_size(chunk_idx));
      }
    } else {
      post_recv();
    }

    return;
  }
}

void RingBroadcast::exit() {
  if (!packets.empty()) {
    packets.clear();
  }
  if (!locked_packets.empty()) {
    locked_packets.clear();
  }
  if (!packet_sizes.empty()) {
    packet_sizes.clear();
  }

  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
  return;
}

}  // namespace AstraSim
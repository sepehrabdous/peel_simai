// sepehr

#include "RingBroadcast.hh"
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
  // the input root is not used anymore for now (default is zero)!
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

  this->recv_posted = false;
  this->recv_done = (id == this->root);
  this->send_done = false;

  const char* env_chunks = std::getenv("AS_RING_BCAST_CHUNKS");
  if (env_chunks != nullptr) {
    unsigned long long tmp = std::strtoull(env_chunks, nullptr, 10);
    if (tmp > 0) {
      AS_RING_BCAST_CHUNKS = static_cast<uint64_t>(tmp);
    }
  }

  if (id == 0)
    std::cout << "AS_RING_BCAST_CHUNKS: " << AS_RING_BCAST_CHUNKS << std::endl;

  this->num_chunks = AS_RING_BCAST_CHUNKS;
  this->msg_size = data_size / num_chunks;
  this->chunks_sent = 0;
  this->chunks_received = 0;

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

void RingBroadcast::post_recv() {
  if (!enabled || recv_posted || is_root()) {
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

  recv_posted = true;
}

void RingBroadcast::stage_packet(bool from_npu) {
  if (!enabled || send_done) {
    return;
  }

  packets.push_back(
      MyPacket(stream->current_queue_id, id, current_receiver));
  packets.back().sender = nullptr;
  locked_packets.push_back(&packets.back());

  processed = false;
  send_back = false;
  send_from_npu = from_npu;

  release_packets();
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

  if (!enabled || packets.size() == 0 || send_done || free_packets == 0) {
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

void RingBroadcast::maybe_exit() {
  if (!enabled) {
    return;
  }

  if (is_root()) {
    if (send_done) {
      exit();
    }
    return;
  }

  if (!recv_done) {
    return;
  }

  if (is_last() || send_done) {
    exit();
  }
}

void RingBroadcast::run(EventType event, CallData* data) {
  if (event == EventType::General) {
    free_packets += 1;
    ready();
    maybe_exit();
  } else if (event == EventType::PacketReceived) {
    chunks_received++;
    recv_done = (chunks_received == num_chunks);
    recv_posted = false;

    // Pipeline: post recv for the next chunk if more are expected.
    if (chunks_received < num_chunks) {
      post_recv();
    }

    if (!is_last()) {
      stage_packet(false);
    } else {
      maybe_exit();
    }
  } else if (event == EventType::StreamInit) {
    if (!enabled) {
      return;
    }

    if (nodes_in_ring <= 1) {
      exit();
      return;
    }

    if (is_root()) {
      // Pipeline: stage all chunks up front; each General event will send one.
      for (int i = 0; i < num_chunks; i++) {
        stage_packet(true);
      }
    } else {
      post_recv();
    }
  }
}

void RingBroadcast::exit() {
  if (packets.size() != 0) {
    packets.clear();
  }
  if (locked_packets.size() != 0) {
    locked_packets.clear();
  }
  stream->owner->proceed_to_next_vnet_baseline((StreamBaseline*)stream);
  return;
}

} // namespace AstraSim
#ifndef __RING_BROADCAST_HH__
#define __RING_BROADCAST_HH__

#include <assert.h>
#include <math.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <sstream>
#include <tuple>
#include <vector>

#include "Algorithm.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MemBus.hh"
#include "astra-sim/system/MyPacket.hh"
#include "astra-sim/system/topology/RingTopology.hh"

namespace AstraSim {

class RingBroadcast : public Algorithm {
 public:
  RingTopology::Direction direction;
  MemBus::Transmition transmition;

  int id;
  int root;
  int current_receiver;
  int current_sender;
  int nodes_in_ring;

  InjectionPolicy injection_policy;

  std::list<MyPacket> packets;
  std::list<MyPacket*> locked_packets;

  // Parallel list: false = data packet, true = ack packet
  std::list<bool> packet_is_ack;

  uint64_t msg_size;
  uint64_t ack_size;

  long free_packets;

  bool processed;
  bool send_back;
  bool send_from_npu;

  bool recv_done;
  bool send_done;
  bool exited;

  // ACK/drain state
  bool waiting_for_ack;
  bool ack_received;
  bool ack_sent;
  bool ack_recv_posted;

  int num_chunks;
  int chunks_staged;
  int chunks_sent;
  int chunks_received;
  int posted_data_recvs;

  uint64_t AS_RING_BCAST_CHUNKS = 1;

  RingBroadcast(
      ComType type,
      int id,
      int layer_num,
      RingTopology* ring_topology,
      uint64_t data_size,
      RingTopology::Direction direction,
      InjectionPolicy injection_policy,
      bool boost_mode,
      int root = 0);

  virtual void run(EventType event, CallData* data);

  bool is_root() const;
  bool is_last() const;

  void post_data_recv();
  void post_ack_recv();

  void stage_data_packet(bool from_npu);
  void stage_ack_packet(bool from_npu, int dest);

  void release_packets();
  bool ready();
  void maybe_exit();
  void exit();
};

} // namespace AstraSim

#endif
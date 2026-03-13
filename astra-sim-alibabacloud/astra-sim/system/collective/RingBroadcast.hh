// sepehr

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

// // Number of pipeline chunks for ring broadcast; change this to tune pipelining.
// #define AS_RING_BCAST_CHUNKS 8

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

  uint64_t msg_size;

  long free_packets;

  bool processed;
  bool send_back;
  bool send_from_npu;

  bool recv_posted;
  bool recv_done;
  bool send_done;

  int num_chunks;
  int chunks_sent;
  int chunks_received;

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

  void post_recv();
  void stage_packet(bool from_npu);
  void release_packets();
  bool ready();
  void maybe_exit();
  void exit();
};

} // namespace AstraSim

#endif
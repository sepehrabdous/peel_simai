#ifndef __RING_BROADCAST_HH__
#define __RING_BROADCAST_HH__

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <ctime>
#include <fstream>
#include <list>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
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

  uint64_t msg_size;

  long free_packets;

  bool processed;
  bool send_back;
  bool send_from_npu;

  bool recv_done;
  bool send_done;
  bool exited;
  std::atomic<bool> drain_complete;

  int num_chunks;
  int chunks_staged;
  int chunks_sent;
  int chunks_received;
  int posted_data_recvs;

  uint64_t AS_RING_BCAST_CHUNKS = 1;

  static std::unordered_map<std::string, RingBroadcast*> root_waiters;
  static std::recursive_mutex root_waiters_mutex;

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

  std::string completion_key() const;
  void notify_root_drain_complete();

  void post_data_recv(int src, int vnet, int stream_num);
  void stage_data_packet(bool from_npu);
  void release_packets();
  bool ready();
  void maybe_exit();
  void exit();
};

}  // namespace AstraSim

#endif
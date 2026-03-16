#ifndef __RING_BROADCAST_HH__
#define __RING_BROADCAST_HH__

#include <assert.h>
#include <math.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <deque>
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

  // For each packet staged in `packets`, keep the corresponding chunk id here.
  std::deque<int> packet_chunks;

  // Kept for compatibility with the rest of the code; actual per-chunk size is
  // computed by chunk_size_bytes().
  uint64_t msg_size;
  uint64_t base_chunk_size;
  uint64_t remainder_bytes;

  long free_packets;

  bool processed;
  bool send_back;
  bool send_from_npu;
  bool recv_done;
  bool send_done;
  bool exited;

  int num_chunks;
  int chunks_sent;
  int chunks_received;

  // Next chunk index that should be posted as a receive.
  int next_chunk_to_post;

  // Which chunk currently has an outstanding recv posted on this rank.
  // -1 means no recv is currently posted.
  int recv_chunk_posted;

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

  int chunk_tag(int chunk_idx) const;
  uint64_t chunk_size_bytes(int chunk_idx) const;

  void post_recv(int chunk_idx);
  void stage_packet(int chunk_idx, bool from_npu);
  void release_packets(uint64_t packet_size);

  bool ready();
  void maybe_exit();
  void exit();
};

}  // namespace AstraSim

#endif
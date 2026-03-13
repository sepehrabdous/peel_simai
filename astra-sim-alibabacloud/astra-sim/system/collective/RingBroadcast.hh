#ifndef __RING_BROADCAST_HH__
#define __RING_BROADCAST_HH__

#include <cstdint>
#include <list>

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
  std::list<uint64_t> packet_sizes;

  // Total broadcast size in bytes
  uint64_t msg_size;

  // Chunking state
  uint64_t requested_chunks;
  uint64_t total_chunks;
  uint64_t nominal_chunk_size;
  uint64_t received_chunks;
  uint64_t sent_chunks;

  long free_packets;

  bool processed;
  bool send_back;
  bool send_from_npu;
  bool recv_posted;
  bool recv_done;
  bool send_done;

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

  uint64_t chunk_size(uint64_t chunk_idx) const;
  bool all_chunks_received() const;
  bool all_chunks_sent() const;

  void post_recv();
  void stage_packet(bool from_npu, uint64_t chunk_bytes);
  void release_packets(uint64_t chunk_bytes);
  bool ready();
  void maybe_exit();
  void exit();
};

}  // namespace AstraSim

#endif
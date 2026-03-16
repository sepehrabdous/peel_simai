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
  struct RootCompletionState {
    RingBroadcast* root_alg;
    int completed_nonroots;
    int expected_nonroots;

    RootCompletionState()
        : root_alg(nullptr), completed_nonroots(0), expected_nonroots(0) {}
  };

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

  int num_chunks;
  int chunks_staged;
  int chunks_sent;
  int chunks_received;
  int posted_data_recvs;

  uint64_t AS_RING_BCAST_CHUNKS = 1;

  static std::unordered_map<std::string, RootCompletionState> root_waiters;
  static std::recursive_mutex root_waiters_mutex;

  // Per-hop credit state.
  static std::unordered_map<std::string, int> hop_credits;
  static std::recursive_mutex hop_credits_mutex;

  // Edge sender registry, so a receiver can wake its predecessor
  // when it grants new credit on that edge.
  static std::unordered_map<std::string, RingBroadcast*> edge_senders;
  static std::recursive_mutex edge_senders_mutex;

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
  void notify_nonroot_exit();

  void post_data_recv();
  void stage_data_packet(bool from_npu);
  void release_packets();
  bool ready();
  bool try_progress_send();
  void maybe_exit();
  void exit();

  std::string incoming_edge_key() const;
  std::string outgoing_edge_key() const;

  void grant_incoming_credit();
  bool consume_outgoing_credit();

  void register_as_edge_sender();
  void unregister_as_edge_sender();
  void on_credit_available();

  void cleanup_credit_state();

    struct PendingStage {
    MyPacket* packet;
    bool from_npu;
    };

    std::deque<PendingStage> pending_stage_queue;
    bool local_stage_busy;

    void kick_local_stage();
};

}  // namespace AstraSim

#endif
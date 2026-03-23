/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __COMMON_HH__
#define __COMMON_HH__
#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include "AstraNetworkAPI.hh"

enum class GPUType { A100, A800, H100, H800, NONE, H20};

namespace AstraSim {
#define CLOCK_PERIOD 1
#define FREQ (1000.0 / CLOCK_PERIOD)
#define GBps 1.0 / (1024 * 1024 * 1024)
typedef unsigned long long Tick;
enum class ComType {
  None,
  Reduce_Scatter,
  All_Gather,
  All_Reduce,
  All_to_All,
  All_Reduce_All_to_All,
  All_Reduce_NVLS,
  //sepehr
  Broadcast,
};
static constexpr std::string_view comtype_to_string(ComType type) {
  switch (type) {
    case ComType::None: return "None";
    case ComType::Reduce_Scatter: return "Reduce_Scatter";
    case ComType::All_Gather: return "All_Gather";
    case ComType::All_Reduce: return "All_Reduce";
    case ComType::All_to_All: return "All_to_All";
    case ComType::All_Reduce_All_to_All: return "All_Reduce_All_to_All";
    case ComType::All_Reduce_NVLS: return "All_Reduce_NVLS";
    case ComType::Broadcast: return "Broadcast";
  }
  return "Unknown ComType";
}

enum class CollectiveOptimization { Baseline, LocalBWAware };
static constexpr std::string_view collective_optimization_to_string(CollectiveOptimization opt) {
  switch (opt) {
    case CollectiveOptimization::Baseline: return "Baseline";
    case CollectiveOptimization::LocalBWAware: return "LocalBWAware";
  }
  return "Unknown CollectiveOptimization";
}

enum class CollectiveImplementationType {
  Ring, 
  OneRing,
  Direct, 
  OneDirect,
  AllToAll,
  DoubleBinaryTreeLocalAllToAll,
  LocalRingNodeA2AGlobalDBT,
  HierarchicalRing,
  DoubleBinaryTree,
  HalvingDoubling,  
  OneHalvingDoubling,
  NcclFlowModel,
  NcclTreeFlowModel,
  // sepehr
  // PeelMulticast,
};
static constexpr std::string_view collective_implementation_type_to_string(CollectiveImplementationType type) {
  switch (type) {
    case CollectiveImplementationType::Ring: return "Ring";
    case CollectiveImplementationType::OneRing: return "OneRing";
    case CollectiveImplementationType::Direct: return "Direct";
    case CollectiveImplementationType::OneDirect: return "OneDirect";
    case CollectiveImplementationType::AllToAll: return "AllToAll";
    case CollectiveImplementationType::DoubleBinaryTreeLocalAllToAll: return "DoubleBinaryTreeLocalAllToAll";
    case CollectiveImplementationType::LocalRingNodeA2AGlobalDBT: return "LocalRingNodeA2AGlobalDBT";
    case CollectiveImplementationType::HierarchicalRing: return "HierarchicalRing";
    case CollectiveImplementationType::DoubleBinaryTree: return "DoubleBinaryTree";
    case CollectiveImplementationType::HalvingDoubling: return "HalvingDoubling";
    case CollectiveImplementationType::OneHalvingDoubling: return "OneHalvingDoubling";
    case CollectiveImplementationType::NcclFlowModel: return "NcclFlowModel";
    case CollectiveImplementationType::NcclTreeFlowModel: return "NcclTreeFlowModel";
    // case CollectiveImplementationType::PeelMulticast: return "PeelMulticast";
  }
  return "Unknown CollectiveImplementationType";
}

enum class CollectiveBarrier { Blocking, Non_Blocking };
static constexpr std::string_view barrier_to_string(CollectiveBarrier barrier) {
  switch (barrier) {
    case CollectiveBarrier::Blocking: return "Blocking";
    case CollectiveBarrier::Non_Blocking: return "Non_Blocking";
  }
  return "Unknown CollectiveBarrier";
}

enum class SchedulingPolicy { LIFO, FIFO, HIGHEST, None };
static constexpr std::string_view scheduling_policy_to_string(SchedulingPolicy policy) {
  switch (policy) {
    case SchedulingPolicy::LIFO: return "LIFO";
    case SchedulingPolicy::FIFO: return "FIFO";
    case SchedulingPolicy::HIGHEST: return "HIGHEST";
    case SchedulingPolicy::None: return "None";
  }
  return "UnknownSchedulingPolicy";
}

enum class IntraDimensionScheduling {
  FIFO,
  RG,
  SmallestFirst,
  LessRemainingPhaseFirst
};
enum class InterDimensionScheduling {
  Ascending,
  OnlineGreedy,
  RoundRobin,
  OfflineGreedy,
  OfflineGreedyFlex
};
static constexpr std::string_view inter_dimension_scheduling_to_string(InterDimensionScheduling scheduling) {
  switch (scheduling) {
    case InterDimensionScheduling::Ascending: return "Ascending";
    case InterDimensionScheduling::OnlineGreedy: return "OnlineGreedy";
    case InterDimensionScheduling::RoundRobin: return "RoundRobin";
    case InterDimensionScheduling::OfflineGreedy: return "OfflineGreedy";
    case InterDimensionScheduling::OfflineGreedyFlex: return "OfflineGreedyFlex";
  }
  return "Unknown InterDimensionScheduling";
}

enum class InjectionPolicy {
  Infinite,
  Aggressive,
  SemiAggressive,
  ExtraAggressive,
  Normal
};
static constexpr std::string_view injection_policy_to_string(InjectionPolicy policy) {
  switch (policy) {
    case InjectionPolicy::Infinite: return "Infinite";
    case InjectionPolicy::Aggressive: return "Aggressive";
    case InjectionPolicy::SemiAggressive: return "SemiAggressive";
    case InjectionPolicy::ExtraAggressive: return "ExtraAggressive";
    case InjectionPolicy::Normal: return "Normal";
  }
  return "Unknown InjectionPolicy";
}

enum class PacketRouting { Hardware, Software };
enum class BusType { Both, Shared, Mem };
enum class StreamState {
  Created,
  Transferring,
  Ready,
  Executing,
  Zombie,
  Dead
};
static const char* stream_state_to_string(StreamState state) {
  switch (state) {
    case StreamState::Created:
      return "Created";
    case StreamState::Transferring:
      return "Transferring";
    case StreamState::Ready:
      return "Ready";
    case StreamState::Executing:
      return "Executing";
    case StreamState::Zombie:
      return "Zombie";
    case StreamState::Dead:
      return "Dead";
    default:
      return "Unknown";
  }
}

enum class EventType {
  NONE,
  RendezvousSend,
  RendezvousRecv,
  CallEvents,
  PacketReceived,
  PacketSent,
  PacketSentFinshed,
  WaitForVnetTurn,
  NCCL_General,
  General,
  TX_DMA,
  RX_DMA,
  Wight_Grad_Comm_Finished,
  Input_Grad_Comm_Finished,
  Fwd_Comm_Finished,
  Wight_Grad_Comm_Finished_After_Delay,
  Input_Grad_Comm_Finished_After_Delay,
  Fwd_Comm_Finished_After_Delay,
  Workload_Wait,
  Reduction_Ready,
  Rec_Finished,
  Send_Finished,
  Processing_Finished,
  Delivered,
  NPU_to_MA,
  MA_to_NPU,
  Read_Port_Free,
  Write_Port_Free,
  Apply_Boost,
  Stream_Transfer_Started,
  Stream_Ready,
  Consider_Process,
  Consider_Retire,
  Consider_Send_Back,
  StreamInit,
  StreamsFinishedIncrease,
  CommProcessingFinished,
  NotInitialized
};

static constexpr std::string_view event_to_string(EventType event) {
  switch (event) {
    case EventType::NONE: return "NONE";
    case EventType::RendezvousSend: return "RendezvousSend";
    case EventType::RendezvousRecv: return "RendezvousRecv";
    case EventType::CallEvents: return "CallEvents";
    case EventType::PacketReceived: return "PacketReceived";
    case EventType::PacketSent: return "PacketSent";
    case EventType::PacketSentFinshed: return "PacketSentFinshed";
    case EventType::WaitForVnetTurn: return "WaitForVnetTurn";
    case EventType::NCCL_General: return "NCCL_General";
    case EventType::General: return "General";
    case EventType::TX_DMA: return "TX_DMA";
    case EventType::RX_DMA: return "RX_DMA";
    case EventType::Wight_Grad_Comm_Finished: return "Wight_Grad_Comm_Finished";
    case EventType::Input_Grad_Comm_Finished: return "Input_Grad_Comm_Finished";
    case EventType::Fwd_Comm_Finished: return "Fwd_Comm_Finished";
    case EventType::Wight_Grad_Comm_Finished_After_Delay: return "Wight_Grad_Comm_Finished_After_Delay";
    case EventType::Input_Grad_Comm_Finished_After_Delay: return "Input_Grad_Comm_Finished_After_Delay";
    case EventType::Fwd_Comm_Finished_After_Delay: return "Fwd_Comm_Finished_After_Delay";
    case EventType::Workload_Wait: return "Workload_Wait";
    case EventType::Reduction_Ready: return "Reduction_Ready";
    case EventType::Rec_Finished: return "Rec_Finished";
    case EventType::Send_Finished: return "Send_Finished";
    case EventType::Processing_Finished: return "Processing_Finished";
    case EventType::Delivered: return "Delivered";
    case EventType::NPU_to_MA: return "NPU_to_MA";
    case EventType::MA_to_NPU: return "MA_to_NPU";
    case EventType::Read_Port_Free: return "Read_Port_Free";
    case EventType::Write_Port_Free: return "Write_Port_Free";
    case EventType::Apply_Boost: return "Apply_Boost";
    case EventType::Stream_Transfer_Started: return "Stream_Transfer_Started";
    case EventType::Stream_Ready: return "Stream_Ready";
    case EventType::Consider_Process: return "Consider_Process";
    case EventType::Consider_Retire: return "Consider_Retire";
    case EventType::Consider_Send_Back: return "Consider_Send_Back";
    case EventType::StreamInit: return "StreamInit";
    case EventType::StreamsFinishedIncrease: return "StreamsFinishedIncrease";
    case EventType::CommProcessingFinished: return "CommProcessingFinished";
    case EventType::NotInitialized: return "NotInitialized";
  }
  return "Unknown EventType";
}


class CloneInterface {
 public:
  virtual CloneInterface* clone() const = 0;
  virtual ~CloneInterface() = default;
};
class CollectiveImplementation : public CloneInterface {
 public:
  CollectiveImplementationType type;
  CollectiveImplementation(CollectiveImplementationType type) {
    this->type = type;
  };
  virtual CloneInterface* clone() const {
    return new CollectiveImplementation(*this);
  }
};
class DirectCollectiveImplementation : public CollectiveImplementation {
 public:
  int direct_collective_window;
  CloneInterface* clone() const {
    return new DirectCollectiveImplementation(*this);
  };
  DirectCollectiveImplementation(
      CollectiveImplementationType type,
      int direct_collective_window)
      : CollectiveImplementation(type) {
    this->direct_collective_window = direct_collective_window;
  }
};
} // namespace AstraSim
#endif

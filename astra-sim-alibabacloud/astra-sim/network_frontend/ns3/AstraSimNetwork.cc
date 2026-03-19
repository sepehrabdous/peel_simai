/* 
*Copyright (c) 2024, Alibaba Group;
*Licensed under the Apache License, Version 2.0 (the "License");
*you may not use this file except in compliance with the License.
*You may obtain a copy of the License at

*   http://www.apache.org/licenses/LICENSE-2.0

*Unless required by applicable law or agreed to in writing, software
*distributed under the License is distributed on an "AS IS" BASIS,
*WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
*See the License for the specific language governing permissions and
*limitations under the License.
*/

#include "astra-sim/system/AstraNetworkAPI.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/RecvPacketEventHadndlerData.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/csma-module.h"
#include "ns3/internet-module.h"
#include "ns3/network-module.h"
#include "entry.h"
#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <queue>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#ifdef NS3_MPI
#include "ns3/mpi-interface.h"
#include <mpi.h>
#endif

// Default output directory prefix for per-flow NCCL model result files.
#define RESULT_PATH "./ncclFlowModel_"

using namespace std;
using namespace ns3;

// Shared state populated by the NS3 topology setup (entry.h / main1()).
// receiver_pending_queue: maps (dst, src, tag) → ncclFlowTag for flows that
//   arrived before the matching sim_recv() was registered (early arrivals).
// node_num / switch_num / nvswitch_num / gpus_per_server: topology counts
//   read from the network config file.
// gpu_type: hardware GPU model (affects bandwidth/latency parameters).
// NVswitchs: list of node IDs that represent NVSwitch chips in the topology.
extern std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;
extern uint32_t node_num, switch_num, link_num, trace_num, nvswitch_num, gpus_per_server;
extern GPUType gpu_type;
extern std::vector<int>NVswitchs;

// Represents a pending network event queued for later processing.
// Currently kept for bookkeeping; the actual send/receive is handled
// directly through the task1 structure passed to SendFlow().
struct sim_event {
  void *buffer;    // payload buffer (unused in flow-model mode)
  uint64_t count;  // message size in bytes
  int type;        // 0 = send, 1 = receive
  int dst;         // destination node ID
  int tag;         // collective/flow tag for matching
  string fnType;   // identifier for the callback type
};

// ASTRASimNetwork bridges the AstraSim system layer and the NS3 network
// simulator.  It implements AstraNetworkAPI so that Sys objects can call
// sim_send / sim_recv / sim_schedule without knowing about NS3 internals.
//
// One ASTRASimNetwork instance is created per simulated GPU.  The 'rank'
// identifies this GPU's logical index within AstraSim, while 'npu_offset'
// is added to destination IDs so that the NS3 node numbering (which may
// include switch/NVSwitch nodes at the front) aligns with logical GPU IDs.
class ASTRASimNetwork : public AstraSim::AstraNetworkAPI {
private:
  // Added to every destination rank before passing to NS3 SendFlow().
  // Accounts for non-GPU nodes (switches, NVSwitches) that occupy lower
  // NS3 node IDs.
  int npu_offset;

public:
  // Queue of pending events; populated externally if deferred processing
  // is needed (not actively consumed in the current implementation).
  queue<sim_event> sim_event_queue;

  ASTRASimNetwork(int rank, int npu_offset) : AstraNetworkAPI(rank) {
    this->npu_offset = npu_offset;

    std::cout << "Initiating ASTRASimNetwork:" << std::endl <<
      "\t rank: " << this->rank << std::endl <<
      "\t npu offset: " << this->npu_offset << std::endl <<
      "------------------\n" << std::endl;

  }
  ~ASTRASimNetwork() {}
  // Not used in this back-end; communicator size queries are handled
  // at a higher level by AstraSim.
  int sim_comm_size(AstraSim::sim_comm comm, int *size) { return 0; }

  // Prints a summary of total bytes sent/received per node at simulation end.
  // nodeHash keys are (node_id, direction) where direction 0 = sent, 1 = received.
  int dump_sim_stats() {
    for (auto it = nodeHash.begin(); it != nodeHash.end(); it++) {
      std::pair<int, int> p = it->first;
      if (p.second == 0) {
        std::cout << "sim_finish on sent, "
                  << " Thread id: " << pthread_self() << std::endl;
        std::cout << "All data sent from node " << p.first << " is "
                  << it->second << "\n";
      } else {
        std::cout << "sim_finish on received, "
                  << " Thread id: " << pthread_self() << std::endl;
        std::cout << "All data received by node " << p.first << " is "
                  << it->second << "\n";
      }
    }
    return 0;
  }

  // Called when the simulation is complete; dumps stats and exits.
  int sim_finish() {
    dump_sim_stats();
    exit(0);
    return 0;
  }

  // Not implemented; NS3 has fixed nanosecond resolution.
  double sim_time_resolution() { return 0; }

  // Memory API initialization hook — not needed for network-only simulation.
  int sim_init(AstraSim::AstraMemoryAPI *MEM) { return 0; }

  // Returns the current NS3 simulation time in nanoseconds.
  AstraSim::timespec_t sim_get_time() {
    AstraSim::timespec_t timeSpec;
    timeSpec.time_val = Simulator::Now().GetNanoSeconds();
    return timeSpec;
  }

  // Schedules a callback 'delta' nanoseconds from now using NS3's event
  // scheduler.  Used by AstraSim to model compute delays and other timed events.
  virtual void sim_schedule(AstraSim::timespec_t delta,
                            void (*fun_ptr)(void *fun_arg), void *fun_arg) {
    task1 t;
    t.type = 2;
    t.fun_arg = fun_arg;
    t.msg_handler = fun_ptr;
    t.schTime = delta.time_val;
    Simulator::Schedule(NanoSeconds(t.schTime), t.msg_handler, t.fun_arg);
    return;
  }
  // Initiates a point-to-point flow from this node (rank) to 'dst'.
  // Steps:
  //   1. Translate logical dst rank to NS3 node ID by adding npu_offset.
  //   2. Record the send task in sentHash (keyed by tag and src/dst pair)
  //      so that a matching sim_recv can locate it when the flow completes.
  //      The hash update is protected by an MTP critical section when
  //      multi-threaded parallel simulation is enabled.
  //   3. Invoke SendFlow() which enqueues the NS3 OnOff/PacketSink flow.
  //      msg_handler is called back by the NS3 event loop when the flow
  //      finishes (i.e., all bytes have been delivered).
  virtual int sim_send(void *buffer,
                       uint64_t count,
                       int type,
                       int dst,
                       int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    dst += npu_offset;
    task1 t;
    t.src = rank;
    t.dest = dst;
    t.count = count;
    t.type = 0;      // type 0 = send
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    {
      #ifdef NS3_MTP
      MtpInterface::explicitCriticalSection cs;
      #endif
      sentHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
    }
    SendFlow(rank, dst, count, msg_handler, fun_arg, tag, request);
    return 0;
  }
  // Registers an expectation to receive 'count' bytes from 'src' on 'tag'.
  // Because NS3 flows can complete before the upper layer calls sim_recv(),
  // this function must handle two orderings:
  //
  //   Case A — data already arrived (recvHash hit):
  //     The NS3 receive callback has already deposited the byte count in
  //     recvHash.  We compare counts:
  //       - exact match or over-delivery: consume the entry, transfer any
  //         pending ncclFlowTag from receiver_pending_queue, fire the
  //         callback immediately and jump to sim_recv_end_section.
  //       - under-delivery (partial): reduce the outstanding count in
  //         expeRecvHash and wait for more data to arrive.
  //
  //   Case B — data not yet arrived (recvHash miss):
  //     Register the expected receive in expeRecvHash so that the NS3
  //     callback can find and satisfy it when the flow completes.
  //     If an entry already exists (duplicate registration), the count is
  //     accumulated.
  //
  // receiver_pending_queue holds flow-tag metadata for flows that have
  // fully arrived but whose NCCL flow context wasn't known at arrival time;
  // it is drained here when the matching sim_recv is finally called.
  //
  // The whole function body is protected by an MTP critical section to
  // prevent races between parallel NS3 threads.
  virtual int sim_recv(void *buffer, uint64_t count, int type, int src, int tag,
                       AstraSim::sim_request *request,
                       void (*msg_handler)(void *fun_arg), void *fun_arg) {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    AstraSim::ncclFlowTag flowTag = request->flowTag;
    src += npu_offset;  // translate logical rank to NS3 node ID
    task1 t;
    t.src = src;
    t.dest = rank;
    t.count = count;
    t.type = 1;  // type 1 = receive
    t.fun_arg = fun_arg;
    t.msg_handler = msg_handler;
    // Extract the actual tag from the handler data; the caller's 'tag'
    // parameter may be stale — the flow tag embedded in the handler is
    // authoritative.
    AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t.fun_arg;
    AstraSim::EventType event = ehd->event;
    tag = ehd->flowTag.tag_id;
    NcclLog->writeLog(NcclLogLevel::DEBUG,"[Receive event registration] src %d sim_recv on rank %d tag_id %d channdl id %d",src,rank,tag,ehd->flowTag.channel_id);

    if (recvHash.find(make_pair(tag, make_pair(t.src, t.dest))) !=
        recvHash.end()) {
      // Case A: flow data has already been received by NS3.
      uint64_t count = recvHash[make_pair(tag, make_pair(t.src, t.dest))];
      if (count == t.count) {
        // Exact match — consume entry and fire callback.
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
          // Restore the flow tag that arrived with the data.
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
          receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
          ehd->flowTag = pending_tag;
        }
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else if (count > t.count) {
        // Over-delivery: more bytes arrived than requested; reduce residual.
        recvHash[make_pair(tag, make_pair(t.src, t.dest))] = count - t.count;
        assert(ehd->flowTag.child_flow_id == -1 && ehd->flowTag.current_flow_id == -1);
        if(receiver_pending_queue.count(std::make_pair(std::make_pair(rank, src),tag))!= 0) {
          AstraSim::ncclFlowTag pending_tag = receiver_pending_queue[std::make_pair(std::make_pair(rank, src),tag)];
          receiver_pending_queue.erase(std::make_pair(std::make_pair(rank,src),tag));
          ehd->flowTag = pending_tag;
        }
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t.msg_handler(t.fun_arg);
        goto sim_recv_end_section;
      } else {
        // Under-delivery: only part of the data arrived; move remainder to
        // expeRecvHash so the next NS3 receive callback can satisfy it.
        recvHash.erase(make_pair(tag, make_pair(t.src, t.dest)));
        t.count -= count;
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
      }
    } else {
      // Case B: data has not arrived yet; register as an expected receive.
      if (expeRecvHash.find(make_pair(tag, make_pair(t.src, t.dest))) ==
          expeRecvHash.end()) {
        expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))] = t;
          NcclLog->writeLog(NcclLogLevel::DEBUG," [Packet arrived late, registering first] recvHash do not find expeRecvHash.new make src  %d dest  %d t.count:  %llu channel_id  %d current_flow_id  %d",t.src,t.dest,t.count,tag,flowTag.current_flow_id);
      } else {
        // Entry already exists — accumulate count (multiple small flows).
        uint64_t expecount =
            expeRecvHash[make_pair(tag, make_pair(t.src, t.dest))].count;
          NcclLog->writeLog(NcclLogLevel::DEBUG," [Packet arrived late, re-registering] recvHash do not find expeRecvHash.add make src  %d dest  %d expecount:  %d t.count:  %d tag_id  %d current_flow_id  %d",t.src,t.dest,expecount,t.count,tag,flowTag.current_flow_id);
      }
    }
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif

sim_recv_end_section:
    return 0;
  }

  // Placeholder event handler; not used in the current flow-model implementation.
  void handleEvent(int dst, int cnt) {
  }
};

// Command-line parameters passed to the simulator.
struct user_param {
  int thread;           // number of MTP threads (-t); default 1
  string workload;      // path to the workload description file (-w)
  string network_topo;  // path to the network topology file (-n)
  string network_conf;  // path to the network configuration file (-c)
  user_param() {
    thread = 1;
    workload = "";
    network_topo = "";
    network_conf = "";
  };
  ~user_param(){};
};

// Parses command-line arguments into 'user_param'.
// Returns 0 on success, 1 if -h (help) was requested or an unknown option
// was encountered (caller should exit cleanly in that case).
static int user_param_prase(int argc,char * argv[],struct user_param* user_param){
  int opt;
  while ((opt = getopt(argc,argv,"ht:w:g:s:n:c:"))!=-1){
    switch (opt)
    {
    case 'h':
      /* code */
      std::cout<<"-t    number of threads,default 1"<<std::endl;
      std::cout<<"-w    workloads default none "<<std::endl;
      std::cout<<"-n    network topo"<<std::endl;
      std::cout<<"-c    network_conf"<<std::endl;
      return 1;
      break;
    case 't':
      user_param->thread = stoi(optarg);
      break;
    case 'w':
      user_param->workload = optarg;
      break;
    case 'n':
      user_param->network_topo = optarg;
      break;
    case 'c':
      user_param->network_conf = optarg;
      break;
    default:
      std::cerr<<"-h    help message"<<std::endl;
      return 1;
    }
  }

  std::cout << "user_params parsed:" << std::endl << 
    "\t num_threads: " << user_param->thread << std::endl <<
    "\t workload: " << user_param->workload << std::endl <<
    "\t topology: " << user_param->network_topo << std::endl <<
    "\t config: " << user_param->network_conf << std::endl;

  return 0 ;
}

// Simulation entry point.
//
// Initialization sequence:
//   1. Parse CLI arguments → user_param.
//   2. Optionally enable MTP multi-threading (NS3_MTP build).
//   3. Call main1() to build the NS3 topology and populate the global
//      node/switch/link counts and NVSwitch lists.
//   4. Derive gpu_num by subtracting switch and NVSwitch nodes from the
//      total node count.
//   5. Build node2nvswitch: maps each GPU index to the NVSwitch chip that
//      serves its server (gpu_id / gpus_per_server gives the server index,
//      then offset by gpu_num to reach the NVSwitch NS3 node IDs).
//   6. Enable NS3 log components for flow-level visibility.
//   7. Create one ASTRASimNetwork + one AstraSim::Sys per GPU.
//      - npu_offset = 0 here because the NS3 node IDs for GPUs start at 0
//        in this single-process setup.
//      - Sys is configured with a single logical dimension of size gpu_num,
//        matching a flat all-reduce topology.
//   8. Fire each GPU's workload to inject the first events into the NS3
//      scheduler.
//   9. Run the NS3 discrete-event simulation until all events are processed
//      (or the 2 000 000 000 s hard stop is reached).
//  10. Tear down NS3 and, for MPI builds, disable the MPI interface.
int main(int argc, char *argv[]) {
  struct user_param user_param;
  MockNcclLog::set_log_name("SimAI.log");
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::INFO," init SimAI.log ");
  if(user_param_prase(argc,argv,&user_param)){
    return 0;
  }

  #ifdef NS3_MTP
  // Enable parallel simulation with the requested number of threads.
  MtpInterface::Enable(user_param.thread);
  #endif

  // Build the NS3 topology from the config files; populates node_num,
  // switch_num, nvswitch_num, gpus_per_server, gpu_type, etc.
  main1(user_param.network_topo,user_param.network_conf);
  int nodes_num = node_num - switch_num;           // GPUs + NVSwitches
  int gpu_num = node_num - nvswitch_num - switch_num;  // pure GPU count

  // Map each GPU to its NVSwitch node ID.
  // GPUs are numbered 0..gpu_num-1 in NS3; NVSwitches follow immediately.
  // Each group of gpus_per_server GPUs shares one NVSwitch.
  std::map<int, int> node2nvswitch;
  for(int i = 0; i < gpu_num; ++ i) {
    node2nvswitch[i] = gpu_num + i / gpus_per_server;
  }
  // NVSwitch nodes map to themselves; also populate the global NVSwitchs list
  // used by the flow model to identify intra-server switch hops.
  for(int i = gpu_num; i < gpu_num + nvswitch_num; ++ i){
    node2nvswitch[i] = i;
    NVswitchs.push_back(i);
  }

  LogComponentEnable("OnOffApplication", LOG_LEVEL_INFO);
  LogComponentEnable("PacketSink", LOG_LEVEL_INFO);
  LogComponentEnable("GENERIC_SIMULATION", LOG_LEVEL_INFO);

  std::cout << "nodes_num (num_gpus + num_nv_switches): " << nodes_num << std::endl;
  std::cout << "gpu_num: " << gpu_num << std::endl;

  std::vector<ASTRASimNetwork *> networks(gpu_num, nullptr);
  std::vector<AstraSim::Sys *> systems(gpu_num, nullptr);

  std::cout << "Initiating ASTRASimNetwork and AstraSim::Sys\n" << std::endl;

  for (int j = 0; j < gpu_num; j++) {
    // Each GPU gets its own network adapter (rank = j, npu_offset = 0).
    networks[j] = new ASTRASimNetwork(j, 0);
    // Sys constructor arguments (in order):
    //   network API, memory API (nullptr = none), node id, npu_offset,
    //   num_passes, dimension_sizes, queues_per_dim, scheduling_policy,
    //   workload_file, compute_scale, comm_scale, injection_scale,
    //   total_stat_rows, stat_row, result_path, run_name,
    //   separate_log, rendezvous_protocol, gpu_type,
    //   all_reduce_implementation_per_dimension, NVSwitchs, gpus_per_server
    systems[j] = new AstraSim::Sys(
        networks[j],
        nullptr,
        j,     // node id
        0,     // npu_offset
        1,     // num_passes (single-pass in this entry point)
        {gpu_num},
        {1},
        "",
        user_param.workload,
        1,     // compute_scale
        1,     // comm_scale
        1,     // injection_scale
        1,     // total_stat_rows
        0,     // stat_row
        RESULT_PATH,
        "test1",
        true,  // separate_log
        false, // rendezvous_protocol
        gpu_type,
        {gpu_num},
        NVswitchs,
        gpus_per_server
    );
    systems[j]->nvswitch_id = node2nvswitch[j];
    systems[j]->num_gpus = gpu_num;
  }

  // Inject the first workload event for each GPU into the NS3 scheduler.
  // Subsequent events are self-scheduled by AstraSim via sim_schedule().
  for (int i = 0; i < gpu_num; i++) {
    std::cout << "\nFiring the workload for gpu: " << i << std::endl <<
      "---------------------------" << std::endl;
    systems[i]->workload->fire();
  }

  // Hand control to the NS3 event loop.  The simulation runs until all
  // events are processed or the hard deadline (2e9 seconds) is hit.
  std::cout << "\nNow, let's run the simulation..." << std::endl;
  Simulator::Run();
  Simulator::Stop(Seconds(2000000000));
  Simulator::Destroy();

  #ifdef NS3_MPI
  MpiInterface::Disable();
  #endif
  return 0;
}

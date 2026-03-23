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

#ifndef __ENTRY_H__
#define __ENTRY_H__

#undef PGO_TRAINING
#define PATH_TO_PGO_CONFIG "path_to_pgo_config"
// Number of RDMA Queue Pairs (QPs) per logical connection between a src-dst pair
#define _QPS_PER_CONNECTION_  1
#include "common.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/error-model.h"
#include "ns3/global-route-manager.h"
#include "ns3/internet-module.h"
#include "ns3/ipv4-static-routing-helper.h"
#include "ns3/packet.h"
#include "ns3/point-to-point-helper.h"
#include "ns3/qbb-helper.h"
#include <fstream>
#include <iostream>
#include <ns3/rdma-client-helper.h>
#include <ns3/rdma-client.h>
#include <ns3/rdma-driver.h>
#include <ns3/rdma.h>
#include <ns3/sim-setting.h>
#include <ns3/switch-node.h>
#include <time.h>
#include <unordered_map>
#include <mutex>
#include <vector>
#ifdef NS3_MTP
#include "ns3/mtp-interface.h"
#endif
#include <map>
#include"astra-sim/system/MockNcclQps.h"
#include "astra-sim/system/MockNcclLog.h"
using namespace ns3;
using namespace std;


// Maps (dst, src, tag_id) -> flowTag for packets that have arrived at the receiver
// but whose corresponding RecvFlow() call hasn't been issued yet.
// Key: ((receiver_node, sender_node), tag_id)
std::map<std::pair<std::pair<int, int>,int>, AstraSim::ncclFlowTag> receiver_pending_queue;


// Maps (port, (src, dst)) -> flowTag so we can look up which flow a completed
// QP belongs to when its send/receive callback fires.
std::map<std::pair<int, std::pair<int, int>>, AstraSim::ncclFlowTag> sender_src_port_map;

// Descriptor for a pending send or receive operation registered by the AstraSim system layer.
struct task1 {
  int src;                          // Source node id
  int dest;                         // Destination node id
  int type;                         // Message type
  uint64_t count;                   // Remaining bytes expected / to send
  void *fun_arg;                    // Argument passed to the completion callback
  void (*msg_handler)(void *fun_arg); // Completion callback invoked when the transfer finishes
  double schTime;                   // Scheduled time for this task
};

// Pending receive tasks registered by RecvFlow() but not yet matched to arriving data.
// Key: (tag_id, (sender_node, receiver_node))
map<std::pair<int, std::pair<int, int>>, struct task1> expeRecvHash;

// Bytes already received for a (tag, src, dst) tuple that arrived before RecvFlow() was called.
// Key: (tag_id, (sender_node, receiver_node))
map<std::pair<int, std::pair<int, int>>, uint64_t> recvHash;

// Pending send tasks registered by SendFlow() waiting for the NIC to confirm transmission.
// Key: (tag_id, (sender_node, receiver_node))
map<std::pair<int, std::pair<int, int>>, struct task1> sentHash;

// Tracks cumulative bytes sent (flag=0) or received (flag=1) per node for accounting.
// Key: (node_id, direction)  where direction: 0=sent, 1=received
map<std::pair<int, int>, int64_t> nodeHash;

// Counts how many sub-QPs are still outstanding before the sender callback fires.
// Decremented in send_finish(); callback fires when it reaches 0.
// Key: (flow_id, (src, dst))
map<std::pair<int,std::pair<int,int>>,int> waiting_to_sent_callback;

// Counts how many sub-QPs are still outstanding before the receiver callback fires.
// Decremented in qp_finish(); callback fires when it reaches 0.
// Key: (flow_id, (src, dst))
map<std::pair<int,std::pair<int,int>>,int>waiting_to_notify_receiver;

// Accumulates total bytes received across all sub-QPs for a flow before the receiver callback fires.
// Key: (flow_id, (src, dst))
map<std::pair<int,std::pair<int,int>>,uint64_t>received_chunksize;

// Accumulates total bytes sent across all sub-QPs for a flow before the sender callback fires.
// Key: (flow_id, (src, dst))
map<std::pair<int,std::pair<int,int>>,uint64_t>sent_chunksize;

// Decrements the outstanding sub-QP counter for the sender side.
// Returns true (and removes the entry) once all sub-QPs for this flow have finished sending,
// indicating it's time to fire the send-completion callback.
bool is_sending_finished(int src,int dst,AstraSim::ncclFlowTag flowTag){
  int tag_id = flowTag.current_flow_id;
  if (waiting_to_sent_callback.count(
          std::make_pair(tag_id, std::make_pair(src, dst)))) {
    if (--waiting_to_sent_callback[std::make_pair(
            tag_id, std::make_pair(src, dst))] == 0) {
      waiting_to_sent_callback.erase(
          std::make_pair(tag_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

// Decrements the outstanding sub-QP counter for the receiver side.
// Returns true (and removes the entry) once all sub-QPs for this flow have been received,
// indicating it's time to fire the receive-completion callback.
bool is_receive_finished(int src,int dst,AstraSim::ncclFlowTag flowTag){
  int tag_id = flowTag.current_flow_id;
  map<std::pair<int,std::pair<int,int>>,int>::iterator it;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (waiting_to_notify_receiver.count(
          std::make_pair(tag_id, std::make_pair(src, dst)))) {
    NcclLog->writeLog(NcclLogLevel::DEBUG," is_receive_finished waiting_to_notify_receiver  tag_id  %d src  %d dst  %d count  %d",tag_id,src,dst,waiting_to_notify_receiver[std::make_pair(
                     tag_id, std::make_pair(src, dst))]);
    if (--waiting_to_notify_receiver[std::make_pair(
            tag_id, std::make_pair(src, dst))] == 0) {
      waiting_to_notify_receiver.erase(
          std::make_pair(tag_id, std::make_pair(src, dst)));
      return true;
    }
  }
  return false;
}

// Initiates an RDMA flow from src to dst of maxPacketCount bytes.
// Splits the transfer into _QPS_PER_CONNECTION_ sub-QPs and installs an
// RdmaClient application on the source node for each. The send-completion
// callback (msg_handler) will be invoked once all sub-QPs have finished.
// send_lat (default 6000 ns, overridable via AS_SEND_LAT env var) delays
// the application start to model NIC scheduling latency.
void SendFlow(int src, int dst, uint64_t maxPacketCount,
              void (*msg_handler)(void *fun_arg), void *fun_arg, int tag, AstraSim::sim_request *request) {

  std::cout << "SendFlow called:" << std::endl
          << "\t src=" << src << std::endl
          << "\t dst=" << dst << std::endl
          << "\t maxPacketCount=" << maxPacketCount << std::endl
          << "\t tag=" << tag << std::endl
          << "\t _QPS_PER_CONNECTION_=" << _QPS_PER_CONNECTION_ << std::endl;

  if (request != nullptr) {
    std::cout << "\t request->srcRank=" << request->srcRank << std::endl
              << "\t request->dstRank=" << request->dstRank << std::endl
              << "\t request->tag=" << request->tag << std::endl
              << "\t request->reqType=" << request->reqType << std::endl
              << "\t request->reqCount=" << request->reqCount << std::endl
              << "\t request->vnet=" << request->vnet << std::endl
              << "\t request->layerNum=" << request->layerNum << std::endl
              << "\t request->flowTag.channel_id=" << request->flowTag.channel_id << std::endl
              << "\t request->flowTag.chunk_id=" << request->flowTag.chunk_id << std::endl
              << "\t request->flowTag.current_flow_id=" << request->flowTag.current_flow_id << std::endl
              << "\t request->flowTag.child_flow_id=" << request->flowTag.child_flow_id << std::endl
              << "\t request->flowTag.sender_node=" << request->flowTag.sender_node << std::endl
              << "\t request->flowTag.receiver_node=" << request->flowTag.receiver_node << std::endl
              << "\t request->flowTag.flow_size=" << request->flowTag.flow_size << std::endl
              << "\t request->flowTag.pQps=" << request->flowTag.pQps << std::endl
              << "\t request->flowTag.tag_id=" << request->flowTag.tag_id << std::endl
              << "\t request->flowTag.nvls_on=" << request->flowTag.nvls_on << std::endl
              << "\t request->flowTag.tree_flow_list.size()="
              << request->flowTag.tree_flow_list.size() << std::endl;
  }

  MockNcclLog*NcclLog = MockNcclLog::getInstance();
  // Divide total bytes evenly across sub-QPs (ceiling division)
  uint64_t PacketCount=((maxPacketCount+_QPS_PER_CONNECTION_-1)/_QPS_PER_CONNECTION_);
  uint64_t leftPacketCount = maxPacketCount;
  for(int index = 0 ;index<_QPS_PER_CONNECTION_;index++){
    uint64_t real_PacketCount = min(PacketCount,leftPacketCount);
    leftPacketCount-=real_PacketCount;
    // Allocate a unique source port for this sub-QP so it can be looked up later
    uint32_t port = portNumber[src][dst]++;
      {
        #ifdef NS3_MTP
        MtpInterface::explicitCriticalSection cs;
        #endif
        // Register port -> flowTag mapping so qp_finish / send_finish can identify the flow
        sender_src_port_map[make_pair(port, make_pair(src, dst))] = request->flowTag;
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
      }
    int flow_id = request->flowTag.current_flow_id;
    bool nvls_on = request->flowTag.nvls_on;
    int pg = 3, dport = 100;  // Priority group and destination port for RDMA
    int send_lat = 6000;  // Default NIC scheduling latency in nanoseconds
    const char* send_lat_env = std::getenv("AS_SEND_LAT");
    if (send_lat_env) {
      try {
        send_lat = std::stoi(send_lat_env);
      } catch (const std::invalid_argument& e) {
        NcclLog->writeLog(NcclLogLevel::ERROR,"send_lat set error");
        exit(-1);
      }
    }
    send_lat *= 1000;  // Convert ns -> ps (NS-3 time step)
    flow_input.idx++;
    if(real_PacketCount == 0) 
      real_PacketCount = 1;  // Guard: always send at least 1 byte
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG," [Packet sending event]  %dSendFlow to  %d channelid:  %d flow_id  %d srcip  %d dstip  %d size:  %llu at the tick:  %d",src,dst,tag,flow_id,serverAddress[src],serverAddress[dst],maxPacketCount,AstraSim::Sys::boostedTick());
    NcclLog->writeLog(NcclLogLevel::DEBUG," request->flowTag [Packet sending event]  %dSendFlow to  %d tag_id:  %d flow_id  %d srcip  %d dstip  %d size:  %llu at the tick:  %d",request->flowTag.sender_node,request->flowTag.receiver_node,request->flowTag.tag_id,request->flowTag.current_flow_id,serverAddress[src],serverAddress[dst],maxPacketCount,AstraSim::Sys::boostedTick());
  
    RdmaClientHelper clientHelper(
      pg, serverAddress[src], serverAddress[dst], port, dport, real_PacketCount,
      has_win ? (global_t == 1 ? maxBdp : pairBdp[n.Get(src)][n.Get(dst)]) : 0,
      global_t == 1 ? maxRtt : pairRtt[src][dst], msg_handler, fun_arg, tag,
      src, dst);

    if(nvls_on) 
      clientHelper.SetAttribute("NVLS_enable", UintegerValue (1));
    {
      #ifdef NS3_MTP
      MtpInterface::explicitCriticalSection cs;
      #endif
      ApplicationContainer appCon = clientHelper.Install(n.Get(src));
      appCon.Start(Time(send_lat));
      // Track that one more sub-QP is in-flight for both sender and receiver sides
      waiting_to_sent_callback[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
      waiting_to_notify_receiver[std::make_pair(request->flowTag.current_flow_id,std::make_pair(src,dst))]++;
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG,"waiting_to_notify_receiver  current_flow_id  %d src  %d dst  %d count  %d",request->flowTag.current_flow_id,src,dst,waiting_to_notify_receiver[std::make_pair(request->flowTag.tag_id,std::make_pair(src,dst))]);
  }

  std::cout << "\n\n\n";

}

// Called by the NS-3 simulation when data from sender_node arrives at receiver_node.
// Tries to match this arriving data against a pending RecvFlow() task in expeRecvHash:
//   - Exact or over-delivery: remove the task and fire the receiver callback immediately.
//   - Under-delivery: reduce the expected count and wait for more data.
//   - No pending task yet: buffer the arrival in recvHash / receiver_pending_queue
//     so a future RecvFlow() can pick it up.
// Also updates nodeHash with the total bytes received by receiver_node.
void notify_receiver_receive_data(int sender_node, int receiver_node,
                                  uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG," %d notify recevier:  %d message size:  %llu",sender_node,receiver_node,message_size);
    int tag = flowTag.tag_id;
    if (expeRecvHash.find(make_pair(
            tag, make_pair(sender_node, receiver_node))) != expeRecvHash.end()) {
      // A matching RecvFlow() task exists — try to satisfy it
      task1 t2 =
          expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG," %d notify recevier:  %d message size:  %llu t2.count:  %llu channle id:  %d",sender_node,receiver_node,message_size,t2.count,flowTag.channel_id);
      AstraSim::RecvPacketEventHadndlerData* ehd = (AstraSim::RecvPacketEventHadndlerData*) t2.fun_arg;
      if (message_size == t2.count) {
        // Exact match: all expected bytes arrived — fire the callback
        NcclLog->writeLog(NcclLogLevel::DEBUG," message_size = t2.count expeRecvHash.erase  %d notify recevier:  %d message size:  %llu channel_id  %d",sender_node,receiver_node,message_size,tag);
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
        ehd->flowTag = flowTag;
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else if (message_size > t2.count) {
        // More data arrived than expected: satisfy the task and buffer the surplus in recvHash
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] =
            message_size - t2.count;
        NcclLog->writeLog(NcclLogLevel::DEBUG,"message_size > t2.count expeRecvHash.erase %d notify recevier:  %d message size:  %llu channel_id  %d",sender_node,receiver_node,message_size,tag);
        expeRecvHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        assert(ehd->flowTag.current_flow_id == -1 && ehd->flowTag.child_flow_id == -1);
        ehd->flowTag = flowTag;
        t2.msg_handler(t2.fun_arg);
        goto receiver_end_1st_section;
      } else {
        // Less data arrived than expected: decrement remaining count and keep waiting
        t2.count -= message_size;
        expeRecvHash[make_pair(tag, make_pair(sender_node, receiver_node))] = t2;
      }
    } else {
      // No RecvFlow() registered yet: store the flow tag and buffer the byte count
      receiver_pending_queue[std::make_pair(std::make_pair(receiver_node, sender_node),tag)] = flowTag;
      if (recvHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) ==
          recvHash.end()) {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] =
            message_size;
      } else {
        recvHash[make_pair(tag, make_pair(sender_node, receiver_node))] +=
            message_size;
      }
    }
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  receiver_end_1st_section:
    {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs2;
    #endif
    // Accumulate total bytes received by receiver_node for accounting / statistics
    if (nodeHash.find(make_pair(receiver_node, 1)) == nodeHash.end()) {
      nodeHash[make_pair(receiver_node, 1)] = message_size;
    } else {
      nodeHash[make_pair(receiver_node, 1)] += message_size;
    }
    #ifdef NS3_MTP
    cs2.ExitSection();
    #endif
    }
  }
}

// Called when the sender's NIC confirms all bytes of a flow have left the wire.
// Looks up the corresponding task in sentHash and fires the send-completion callback.
// Logs an error if the task is missing or the byte count doesn't match.
void notify_sender_sending_finished(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  {
    MockNcclLog * NcclLog = MockNcclLog::getInstance();
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    int tag = flowTag.tag_id;
    if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) !=
      sentHash.end()) {
      task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
      AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
      ehd->flowTag=flowTag;
      if (t2.count == message_size) {
        sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
        // Update sender-side byte accounting
        if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
          nodeHash[make_pair(sender_node, 0)] = message_size;
        } else {
          nodeHash[make_pair(sender_node, 0)] += message_size;
        }
        #ifdef NS3_MTP
        cs.ExitSection();
        #endif
        t2.msg_handler(t2.fun_arg);
        goto sender_end_1st_section;
      }else{
        NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash msg size != sender_node %d receiver_node %d message_size %lu flow_id ",sender_node,receiver_node,message_size);
      }
    }else{
      NcclLog->writeLog(NcclLogLevel::ERROR,"sentHash cann't find sender_node %d receiver_node %d message_size %lu",sender_node,receiver_node,message_size);
    }
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
sender_end_1st_section:
  return;
}


// Called when a packet from sender_node has been acknowledged at receiver_node
// (uses channel_id as the tag, unlike notify_sender_sending_finished which uses tag_id).
// Fires the send-completion callback once the full message has been acknowledged.
void notify_sender_packet_arrivered_receiver(int sender_node, int receiver_node,
                                    uint64_t message_size, AstraSim::ncclFlowTag flowTag) {
  int tag = flowTag.channel_id;
  if (sentHash.find(make_pair(tag, make_pair(sender_node, receiver_node))) !=
      sentHash.end()) {
    task1 t2 = sentHash[make_pair(tag, make_pair(sender_node, receiver_node))];
    AstraSim::SendPacketEventHandlerData* ehd = (AstraSim::SendPacketEventHandlerData*) t2.fun_arg;
    ehd->flowTag=flowTag;
    if (t2.count == message_size) {
      sentHash.erase(make_pair(tag, make_pair(sender_node, receiver_node)));
      // Update sender-side byte accounting
      if (nodeHash.find(make_pair(sender_node, 0)) == nodeHash.end()) {
        nodeHash[make_pair(sender_node, 0)] = message_size;
      } else {
        nodeHash[make_pair(sender_node, 0)] += message_size;
      }
      t2.msg_handler(t2.fun_arg);
    }
  }
}

// NS-3 callback invoked when an RDMA Queue Pair (QP) completes on the receiver side.
// Writes flow-completion statistics (addresses, ports, size, latency, standalone FCT) to fout.
// Accumulates received bytes across sub-QPs; once all sub-QPs for a flow are done,
// calls notify_receiver_receive_data() to inform the AstraSim system layer.
void qp_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  uint64_t base_rtt = pairRtt[sid][did], b = pairBw[sid][did];
  // total_bytes accounts for packet payload plus per-packet header overhead
  uint32_t total_bytes =
      q->m_size +
      ((q->m_size - 1) / packet_payload_size + 1) *
          (CustomHeader::GetStaticWholeHeaderSize() -
           IntHeader::GetStaticSize());
  // standalone_fct: ideal flow-completion time if this were the only flow on the link
  uint64_t standalone_fct = base_rtt + total_bytes * 8000000000lu / b;
  fprintf(fout, "%08x %08x %u %u %lu %lu %lu %lu\n", q->sip.Get(), q->dip.Get(),
          q->sport, q->dport, q->m_size, q->startTime.GetTimeStep(),
          (Simulator::Now() - q->startTime).GetTimeStep(), standalone_fct);
  fflush(fout);

  AstraSim::ncclFlowTag flowTag;
  uint64_t notify_size;
  {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    Ptr<Node> dstNode = n.Get(did);
    Ptr<RdmaDriver> rdma = dstNode->GetObject<RdmaDriver>();
    // Clean up the receive-side QP state at the destination
    rdma->m_rdma->DeleteRxQp(q->sip.Get(), q->m_pg, q->sport);
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
    NcclLog->writeLog(NcclLogLevel::DEBUG,"qp finish, src:  %d did:  %d port:  %d total bytes:  %llu at the tick:  %d",sid,did,q->sport,q->m_size,AstraSim::Sys::boostedTick());
    // Look up the flow tag registered when SendFlow() created this sub-QP
    if (sender_src_port_map.find(make_pair(q->sport, make_pair(sid, did))) ==
        sender_src_port_map.end()) {
      NcclLog->writeLog(NcclLogLevel::ERROR,"could not find the tag, there must be something wrong");
      exit(-1);
    }
    flowTag = sender_src_port_map[make_pair(q->sport, make_pair(sid, did))];
    sender_src_port_map.erase(make_pair(q->sport, make_pair(sid, did)));
    // Add this sub-QP's bytes to the running total for this flow
    received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))]+=q->m_size;
    // If more sub-QPs are still in-flight, wait for them before notifying
    if(!is_receive_finished(sid,did,flowTag)) {
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
      return;
    }
    // All sub-QPs received — collect the total and clean up
    notify_size = received_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))];
    received_chunksize.erase(std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did)));
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
  // Notify the AstraSim system layer that the full flow has arrived
  notify_receiver_receive_data(sid, did, notify_size, flowTag);
}

// NS-3 callback invoked when an RDMA Queue Pair (QP) finishes sending on the sender side.
// Accumulates sent bytes across sub-QPs; once all sub-QPs for a flow are done,
// calls notify_sender_sending_finished() to inform the AstraSim system layer.
void send_finish(FILE *fout, Ptr<RdmaQueuePair> q) {
  uint32_t sid = ip_to_node_id(q->sip), did = ip_to_node_id(q->dip);
  AstraSim::ncclFlowTag flowTag;
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"[Packet sent from NIC] send finish, src:  %d did:  %d port:  %d srcip  %d dstip  %d total bytes:  %llu at the tick:  %d",sid,did,q->sport,q->sip,q->dip,q->m_size,AstraSim::Sys::boostedTick());
  uint64_t all_sent_chunksize;
  {
    #ifdef NS3_MTP
    MtpInterface::explicitCriticalSection cs;
    #endif
    flowTag = sender_src_port_map[make_pair(q->sport, make_pair(sid, did))];
    // Add this sub-QP's bytes to the running total for this flow
    sent_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))]+=q->m_size;
    // If more sub-QPs are still in-flight, wait for them before notifying
    if(!is_sending_finished(sid,did,flowTag)) {
      #ifdef NS3_MTP
      cs.ExitSection();
      #endif
      return;
    }
    // All sub-QPs sent — collect the total and clean up
    all_sent_chunksize = sent_chunksize[std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did))];
    sent_chunksize.erase(std::make_pair(flowTag.current_flow_id,std::make_pair(sid,did)));
    #ifdef NS3_MTP
    cs.ExitSection();
    #endif
  }
  // Notify the AstraSim system layer that all bytes have left the sender's NIC
  notify_sender_sending_finished(sid, did, all_sent_chunksize, flowTag);
}

// Entry point for the NS-3 network simulation.
// Reads topology and configuration files, sets up the network, and starts the simulator.
int main1(string network_topo,string network_conf) {
  clock_t begint, endt;
  begint = clock();

  if (!ReadConf(network_topo,network_conf)) {
    std::cerr << "ReadConf returned false!" << endl;
    return -1;
  }
  SetConfig();
  // Build the simulated network and register the QP-finish / send-finish callbacks
  SetupNetwork(qp_finish,send_finish);

  std::cout << "Running Simulation." << std::endl;
  fflush(stdout);
  NS_LOG_INFO("Run Simulation.");

  endt = clock();
  return 0;
}
#endif

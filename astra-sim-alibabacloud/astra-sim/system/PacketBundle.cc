/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "PacketBundle.hh"
#include "astra-sim/system/MockNcclLog.h"
#include "PhyMultiThread.hh"
namespace AstraSim {

// Constructor for bundles with a pre-built list of locked packets but no
// explicit channel/flow assignment. channel_id is set to -1, meaning the
// bundle will use the generic EventType::General path when it fires.
PacketBundle::PacketBundle(
    Sys* generator,
    BaseStream* stream,
    std::list<MyPacket*> locked_packets,
    bool needs_processing,
    bool send_back,
    uint64_t size,
    MemBus::Transmition transmition) {
  this->generator = generator;
  this->locked_packets = locked_packets;
  this->needs_processing = needs_processing;
  this->send_back = send_back;
  this->size = size;
  this->stream = stream;
  this->transmition = transmition;
  creation_time = Sys::boostedTick();
  this->channel_id = -1;
}

// Constructor for bundles tied to a specific NCCL channel and flow.
// Used when the collective operation needs to track per-channel/flow progress
// (e.g. pipelined ring). Fires EventType::NCCL_General on completion.
PacketBundle::PacketBundle(
    Sys* generator,
    BaseStream* stream,
    std::list<MyPacket*> locked_packets,
    bool needs_processing,
    bool send_back,
    uint64_t size,
    MemBus::Transmition transmition,
    int channel_id,
    int flow_id) {
  this->generator = generator;
  this->locked_packets = locked_packets;
  this->needs_processing = needs_processing;
  this->send_back = send_back;
  this->size = size;
  this->stream = stream;
  this->transmition = transmition;
  this->channel_id = channel_id;
  this->flow_id = flow_id;
  creation_time = Sys::boostedTick();

  std::cout << "PacketBundle::PacketBundle called:" << std::endl
            << "\t node=" << stream->owner->id << std::endl
            << "\t size=" << size << std::endl
            << "\t needs_processing=" << needs_processing << std::endl
            << "\t send_back=" << send_back << std::endl
            << "\t channel_id=" << channel_id << std::endl
            << "\t flow_id=" << flow_id << std::endl
            << "\t creation_time=" << creation_time << std::endl
            << "\t locked_packets.size()=" << locked_packets.size() << std::endl;

}

// Minimal constructor — no locked_packets list. Used when the bundle only
// needs to model a memory-bus transfer without tracking individual packets.
PacketBundle::PacketBundle(
    Sys* generator,
    BaseStream* stream,
    bool needs_processing,
    bool send_back,
    uint64_t size,
    MemBus::Transmition transmition) {
  this->generator = generator;
  this->needs_processing = needs_processing;
  this->send_back = send_back;
  this->size = size;
  this->stream = stream;
  this->transmition = transmition;
  creation_time = Sys::boostedTick();
  this->channel_id = -1;
}

// Initiates a transfer from the NPU to Memory Attached (MA).
// Called when data produced by the NPU needs to be written to memory.
void PacketBundle::send_to_MA() {
  generator->memBus->send_from_NPU_to_MA(
      transmition, size, needs_processing, send_back, this);
}

// Initiates a transfer from Memory Attached (MA) back to the NPU.
// Called when data in memory is ready to be consumed by the NPU.
void PacketBundle::send_to_NPU() {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  generator->memBus->send_from_MA_to_NPU(
      transmition, size, needs_processing, send_back, this);
  NcclLog->writeLog(NcclLogLevel::DEBUG,"send_to_NPU done");
}

// Event callback — invoked by the simulator when the memory-bus transfer
// completes (or when CommProcessingFinished fires on the second pass).
void PacketBundle::call(EventType event, CallData* data) {
  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  NcclLog->writeLog(NcclLogLevel::DEBUG,"packet bundle call");

  Tick current = Sys::boostedTick();

  std::cout << "PacketBundle::call called:" << std::endl
            << "\t ID=" << generator->id << std::endl
            << "\t time=" << current << std::endl
            << "\t node=" << stream->owner->id << std::endl
            << "\t event=" << event_to_string(event) << std::endl
            << "\t size=" << size << std::endl
            << "\t needs_processing=" << needs_processing << std::endl
            << "\t send_back=" << send_back << std::endl
            << "\t channel_id=" << channel_id << std::endl
            << "\t flow_id=" << flow_id << std::endl
            << "\t creation_time=" << creation_time << std::endl
            << "\t current_tick=" << Sys::boostedTick() << std::endl
            << "\t locked_packets.size()=" << locked_packets.size() << std::endl;

  if (needs_processing == true) {
    // First pass: data has arrived but still needs NPU-side processing
    // (one write + two reads to model a reduce-copy operation).
    // Schedule a CommProcessingFinished event after the computed delay,
    // then return — call() will be invoked again when that event fires.
    needs_processing = false;
    this->delay = generator->mem_write(size) + generator->mem_read(size) +
        generator->mem_read(size);
    generator->try_register_event(
        this, EventType::CommProcessingFinished, data, this->delay);
    return;
  }

  // Processing is done (or was never needed). Stamp all locked packets with
  // the current simulation tick so downstream logic knows when they became
  // ready. Skipped under PHY_MTP because that mode manages packet timing
  // through a separate multi-threaded physical layer.
  #ifndef PHY_MTP
  for (auto& packet : locked_packets) {
    packet->ready_time = current;
  }
  #endif

  // Notify the owning stream that this bundle is complete.
  // If no channel was assigned (channel_id == -1) use the generic path;
  // otherwise use the NCCL-specific path and pass channel/flow metadata.
  BasicEventHandlerData* ehd = new BasicEventHandlerData(channel_id, flow_id);
  if(channel_id == -1) stream->call(EventType::General, data);
  else stream->call(EventType::NCCL_General, ehd);

  delete this; // Bundle is fully consumed; free its memory.
}
} // namespace AstraSim

/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "MemBus.hh"
#include "LogGP.hh"
#include "Sys.hh"
#include "astra-sim/system/MockNcclLog.h"
namespace AstraSim {

// Destructor: free the two LogGP endpoint objects created in the constructor.
MemBus::~MemBus() {
  delete NPU_side;
  delete MA_side;
}

// Constructor: creates a bidirectional memory bus between the NPU and
// Memory-Attached (MA) side using the LogGP model.
//
// Parameters:
//   side1/side2         - human-readable names for the two endpoints
//   generator           - owning Sys node (used for event scheduling)
//   L, o, g, G         - LogGP parameters (latency, overhead, gap, bandwidth)
//   model_shared_bus    - if true, transfers are serialized through LogGP
//                         queues; if false, a fixed delay is used instead
//   communication_delay - fixed delay (ticks) used when model_shared_bus=false
//   attach              - if true, also attach a secondary memory bus on the
//                         NPU side (models a shared-bus topology)
MemBus::MemBus(
    std::string side1,
    std::string side2,
    Sys* generator,
    Tick L,
    Tick o,
    Tick g,
    double G,
    bool model_shared_bus,
    int communication_delay,
    bool attach) {
  // NPU_side fires MA_to_NPU events; MA_side fires NPU_to_MA events.
  // The two endpoints are cross-linked as partners so LogGP can model
  // the shared-bus contention between the two directions.
  NPU_side = new LogGP(side1, generator, L, o, g, G, EventType::MA_to_NPU);
  MA_side = new LogGP(side2, generator, L, o, g, G, EventType::NPU_to_MA);
  NPU_side->partner = MA_side;
  MA_side->partner = NPU_side;
  this->generator = generator;
  this->model_shared_bus = model_shared_bus;
  this->communication_delay = communication_delay;
  if (attach) {
    // Attach a shared-bus model to the NPU side with a fixed G of 0.0038.
    // This models a second level of memory bus (e.g. HBM shared bandwidth).
    NPU_side->attach_mem_bus(
        generator, L, o, g, 0.0038, model_shared_bus, communication_delay);
  }
}

// Initiates a transfer from the NPU to Memory-Attached (MA) side.
//
// If model_shared_bus is enabled and the transfer is Usual, it goes through
// the LogGP queue on NPU_side (serialized, contention-aware).
// Otherwise a direct event is scheduled with a fixed delay:
//   - Transmition::Fast uses a minimal 10-tick delay (models zero-copy / RDMA)
//   - Transmition::Usual (non-shared-bus) uses communication_delay ticks
void MemBus::send_from_NPU_to_MA(
    MemBus::Transmition transmition,
    uint64_t bytes,
    bool processed,
    bool send_back,
    Callable* callable) {
  if (model_shared_bus && transmition == Transmition::Usual) {
    NPU_side->request_read(bytes, processed, send_back, callable);
  } else {
    if (transmition == Transmition::Fast) {
      MockNcclLog* NcclLog = MockNcclLog::getInstance();
      generator->register_event(
          callable,
          EventType::NPU_to_MA,
          new SharedBusStat(BusType::Shared, 0, 10, 0, 0),
          10);
    } else {
      generator->register_event(
          callable,
          EventType::NPU_to_MA,
          new SharedBusStat(BusType::Shared, 0, communication_delay, 0, 0),
          communication_delay);
    }
  }
}

// Initiates a transfer from Memory-Attached (MA) side back to the NPU.
// Mirror of send_from_NPU_to_MA: same Transmition logic, but fires
// MA_to_NPU events and uses the MA_side LogGP queue when contention is modeled.
void MemBus::send_from_MA_to_NPU(
    MemBus::Transmition transmition,
    uint64_t bytes,
    bool processed,
    bool send_back,
    Callable* callable) {
    MockNcclLog* NcclLog = MockNcclLog::getInstance();
  if (model_shared_bus && transmition == Transmition::Usual) {
    MA_side->request_read(bytes, processed, send_back, callable);
  } else {
    if (transmition == Transmition::Fast) {
      generator->register_event(
          callable,
          EventType::MA_to_NPU,
          new SharedBusStat(BusType::Shared, 0, 10, 0, 0),
          10);
    } else {
      generator->register_event(
          callable,
          EventType::MA_to_NPU,
          new SharedBusStat(BusType::Shared, 0, communication_delay, 0, 0),
          communication_delay);
    }
  }
}
} // namespace AstraSim

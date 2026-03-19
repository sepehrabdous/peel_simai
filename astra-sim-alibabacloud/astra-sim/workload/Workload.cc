/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "Workload.hh"
#include "CSVWriter.hh"
#include "Layer.hh"
#include "astra-sim/system/MockNcclLog.h"

namespace AstraSim {

// Destructor: release all heap-allocated resources.
// CSV writers (end_to_end, detailed, dimension_utilization) and the Layer
// array are only allocated on node 0 / when seprate_log is set, so each
// pointer is guarded by a nullptr check before deletion.
Workload::~Workload() {
  if (end_to_end != nullptr) {
    delete end_to_end;
  }
  if (detailed != nullptr) {
    delete detailed;
  }
  if (dimension_utilization != nullptr) {
    delete dimension_utilization;
  }
  for (int i = 0; i < SIZE; i++) {
    delete layers[i];
  }
  if (layers != nullptr) {
    delete[] layers;
  }
}
// Constructor: initializes member variables and calls initialize_workload()
// to parse the workload description file.  Only node 0 creates CSV log
// writers; all other nodes skip the I/O setup entirely.
//
// Parameters:
//   run_name    – identifier string used in output file names
//   generator   – owning Sys node; used for scheduling and id checks
//   name        – path to the workload input file
//   TOTAL_PASS  – number of training/inference passes to simulate
//   total_rows  – total CSV rows pre-allocated for statistics
//   stat_row    – row offset at which this node writes its stats
//   path        – directory where CSV output files are created
//   seprate_log – when true, node 0 writes per-run CSV log files
Workload::Workload(
    std::string run_name,
    Sys* generator,
    std::string name,
    int TOTAL_PASS,
    int total_rows,
    int stat_row,
    std::string path,
    bool seprate_log) {

  if (generator->id == 0) {
    std::cout
        << "\nInitiating Workload with inputs:\n"
        << "\t run_name: " << run_name << "\n"
        << "\t name: " << name << "\n"
        << "\t TOTAL_PASS: " << TOTAL_PASS << "\n"
        << "\t total_rows: " << total_rows << "\n"
        << "\t stat_row: " << stat_row << "\n"
        << "\t path: " << path << "\n"
        << "\t seprate_log: " << seprate_log << "\n"
        << "--------------------\n" << std::endl;
  }   

  this->initialized = false;
  this->layers = nullptr;
  this->SIZE = 0;
  this->counter = 0;
  this->delay_loaded = false;
  this->checkpoint_initiated = false;
  this->collective_issued = false;
  this->current_state = LoopState::Forward_Pass;
  this->generator = generator;
  this->TOTAL_PASS = TOTAL_PASS;
  this->pass_counter = 0;
  this->index = 0;
  this->waiting_for_comm = 0;
  end_to_end = nullptr;
  detailed = nullptr;
  dimension_utilization = nullptr;
  this->path = path;
  this->stat_row = stat_row;
  this->seprate_log = seprate_log;
  this->initialized = initialize_workload(name);
  if (this->initialized == false) {
    return;
  }
  this->total_rows = total_rows;
  this->run_name = run_name;
  this->registered_for_finished_streams = false;

  #ifndef PHY_MTP
  if (generator->id == 0 && seprate_log) {
    std::cout << "stat path: " << path << " ,total rows: " << total_rows
              << " ,stat row: " << stat_row << std::endl;
    detailed = new CSVWriter(path, "detailed_"+std::to_string(generator->total_nodes)+".csv");
    end_to_end = new CSVWriter(path, "EndToEnd.csv");
    dimension_utilization =
        new CSVWriter(path, run_name + "_dimension_utilization_"+std::to_string(generator->npu_offset)+".csv");
    if (stat_row == 0) {
      initialize_stat_files();
    }
  }
  #endif
}

// Pre-allocates rows in the CSV writers so the simulation can write stats
// without repeated resizing.  The +20 margin guards against minor over-runs.
// The detailed CSV is only pre-allocated under NS3 MPI/MTP builds because
// those back-ends require up-front reservation.
void Workload::initialize_stat_files() {
  #ifdef NS3_MPI
  detailed->initialize_csv(SIZE * total_rows + 20, 50);
  #endif
  #ifdef NS3_MTP
  detailed->initialize_csv(SIZE * total_rows + 20, 50);
  #endif
  end_to_end->initialize_csv(SIZE * total_rows + 20, 50);
}

// Entry point called by the event scheduler each time the workload needs
// to advance.  Acts as a dispatcher: it first handles any residual compute
// delay (counter > 0 means we are still "busy" for that many ticks), then
// delegates to the parallelism-specific iterator.
void Workload::call(EventType event, CallData* data) {

  std::cout << "Workload::call called:" << std::endl <<
    "\t id: " << generator->id << std::endl <<
    "\t event type: " << event_to_string(event) << std::endl << 
    "\t counter: " << counter << std::endl <<
    "-----------------------\n" << std::endl;
    
  // If there is outstanding compute time, re-register and yield.
  // The scheduler will call us again after 'counter' ticks.
  if (counter > 0) {
    std::cout << "Counter > 0 ... waiting!" << std::endl;
    generator->try_register_event(
        this, EventType::Workload_Wait, NULL, counter);
    return;
  }

  if (parallelismPolicy == ParallelismPolicy::Data) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::Data" << std::endl;
    iterate_data_parallel();
  } else if (parallelismPolicy == ParallelismPolicy::Transformer) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::Transformer" << std::endl;
    iterate_hybrid_parallel_Transformer();
  } else if (
      parallelismPolicy == ParallelismPolicy::DLRM ||
      parallelismPolicy == ParallelismPolicy::DLRMEnhanced) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::DLRM || ParallelismPolicy::DLRMEnhanced" << std::endl;
    iterate_hybrid_parallel_DLRM();
  } else if (parallelismPolicy == ParallelismPolicy::MicroBenchmark) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::MicroBenchmark" << std::endl;
    iterate_micro_benchmark();
  } else if (parallelismPolicy == ParallelismPolicy::Model) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::Model" << std::endl;
    iterate_model_parallel();
  } else if (parallelismPolicy == ParallelismPolicy::HybridDataModel) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::HybridDataModel" << std::endl;
    iterate_hybrid_parallel_data_model();
  } else if (parallelismPolicy == ParallelismPolicy::HybridModelData) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::HybridModelData" << std::endl;
    iterate_hybrid_parallel_model_data();
  } else if (parallelismPolicy == ParallelismPolicy::DistributedInference) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::DistributedInference" << std::endl;
    iterate_distributed_inference();
  } else if (parallelismPolicy == ParallelismPolicy::TransformerFwdInBckwd) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::TransformerFwdInBckwd" << std::endl;
    iterate_hybrid_parallel_Transformer_fwd_in_bckwd();
  } else if (parallelismPolicy == ParallelismPolicy::HybridCustomized) {
    if (generator->id == 0)
      std::cout << "parallelismPolicy == ParallelismPolicy::HybridCustomized" << std::endl;
    iterate_hybrid_parallel_customized();
  } else {
    Sys::sys_panic("No known parallelism!");
  }

}

// Collects per-layer statistics and reports them via the network interface.
// Called once all passes are complete (on node 0 only under non-PHY_MTP builds).
// Aggregates total compute time, exposed communication time, and per-phase
// (fwd/wg/ig) times, then writes dimension utilization CSVs for NS3 builds.
void Workload::report() {
  double total_compute = 0;
  double total_exposed = 0;
  // #ifdef ANALYTI
  double pre_bubble_time = 0;
  double DP_comm = 0;
  double DP_EP_comm = 0;
  double Expose_TP_comm = 0;
  double Expose_EP_comm = 0;
  // #endif
  std::vector<double> total_fwd_time = {0, 0, 0};
  std::vector<double> total_wg_time = {0, 0, 0};
  std::vector<double> total_ig_time = {0, 0, 0};
  AstraSimDataAPI astraSimDataAPI;
  astraSimDataAPI.run_name = run_name;
  astraSimDataAPI.workload_finished_time = ((double)Sys::boostedTick()) / FREQ;
  std::cout<<"workload stats for the job scheduled at NPU offset: "
            <<generator->npu_offset<<std::endl;
  for (int i = 0; i < SIZE; i++) {
    #ifdef ANALYTI
    astraSimDataAPI.layers_stats.push_back(layers[i]->report(
        run_name,
        i,
        total_rows,
        stat_row,
        detailed,
        end_to_end,
        total_compute,
        total_exposed,
        pre_bubble_time,
        DP_comm,
        DP_EP_comm,
        Expose_TP_comm,
        Expose_EP_comm,
        this->seprate_log));
    #else
    astraSimDataAPI.layers_stats.push_back(layers[i]->report(
        run_name,
        i,
        total_rows,
        stat_row,
        detailed,
        end_to_end,
        total_compute,
        total_exposed,
        this->seprate_log,
        total_fwd_time,
        total_wg_time,
        total_ig_time,
        pre_bubble_time,
        DP_comm,
        DP_EP_comm,
        Expose_TP_comm,
        Expose_EP_comm));
    #endif
  }
  astraSimDataAPI.total_compute = total_compute;
  astraSimDataAPI.total_exposed_comm = total_exposed;
  astraSimDataAPI.avg_chunk_latency_per_logical_dimension =
      generator->scheduler_unit->get_average_latency_per_dimension();
  for (auto& latency :
       astraSimDataAPI.avg_chunk_latency_per_logical_dimension) {
    latency /= FREQ;
  }
  std::cout << "*************************" << std::endl;
  std::cout << "all passes finished at time: " << Sys::boostedTick()
            << ", id of first layer: " << layers[0]->id << std::endl;
  generator->NI->pass_front_end_report(astraSimDataAPI);
  #ifdef NS3_MTP 
  if (this->seprate_log) {
    std::list<std::list<std::pair<uint64_t, double>>> dims;
    for (int i = 0; i < generator->scheduler_unit->usage.size(); i++) {
      dims.push_back(
          generator->scheduler_unit->usage[i].report_percentage(10000));
    }
    dimension_utilization->finalize_csv(dims);
  }
  #endif
  #ifdef NS3_MPI 
  if (this->seprate_log) {
    std::list<std::list<std::pair<uint64_t, double>>> dims;
    for (int i = 0; i < generator->scheduler_unit->usage.size(); i++) {
      dims.push_back(
          generator->scheduler_unit->usage[i].report_percentage(10000));
    }
    dimension_utilization->finalize_csv(dims);
  }
  #endif
}
// Called at the top of every iterate_* function to see whether all training
// passes have been completed and in-flight collective streams have drained.
//
// Flow:
//  1. If pass_counter has reached TOTAL_PASS, transition to Wait_For_Sim_Finish.
//  2. If there are still outstanding streams, register for a callback and wait.
//     Blocking on layer[0]'s weight-grad comm ensures we don't exit prematurely.
//  3. Once all streams are finished, call report() (node 0 only) and signal
//     the system that the workload is done.
void Workload::check_for_sim_end() {
  if (pass_counter == TOTAL_PASS) {
    current_state = LoopState::Wait_For_Sim_Finish;
    if (generator->streams_finished != generator->streams_injected &&
        registered_for_finished_streams == false) {
      generator->register_for_finished_stream(this);
      registered_for_finished_streams = true;
      layers[0]->is_weight_grad_comm_finished_blocking();
      return;
    }
    if (generator->streams_finished == generator->streams_injected) {
      #ifndef PHY_MTP
      if (generator->id == 0) {
        report();
      }
      #endif
      generator->workload_finished();
      return;
    }
  }
  return;
}
// Micro-benchmark mode: issues weight-gradient collectives on a single layer
// (identified by 'index') for every requested pass without any compute delay.
// This is used to stress-test the collective communication subsystem in
// isolation, bypassing the normal fwd/ig/wg training loop.
void Workload::iterate_micro_benchmark() {

  if (generator->id == 0)
    std::cout << "iterate_micro_benchmark called!" << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  if (current_state != LoopState::Wait_For_Sim_Finish) {
    for (pass_counter = 0; pass_counter < TOTAL_PASS; pass_counter++) {
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::None, CollectiveBarrier::Non_Blocking);
    }
  }
  check_for_sim_end();
}

// Data-parallel training loop.  All NPUs hold a full model replica; gradients
// are synchronized via weight-grad AllReduce after the backward pass.
//
// Loop structure per pass:
//   Forward_Pass  (layer 0 → SIZE-1): wait for any prior wg comm to finish,
//                 simulate compute, then advance to next layer.
//   Weight_Gradient (layer SIZE-1 → 0): simulate wg compute, issue wg comm
//                 (Non_Blocking — overlaps with input-grad compute of next
//                 layer), then move to Input_Gradient (or restart if at layer 0).
//   Input_Gradient (same layer as Weight_Gradient): simulate ig compute,
//                 decrement layer index, loop back to Weight_Gradient.
//
// The 'delay_loaded' flag prevents reloading the compute delay on re-entry
// after a Workload_Wait event.
void Workload::iterate_data_parallel() {

  if (generator->id == 0)
    std::cout << "iterate_data_parallel called!" << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    // Block until the previous pass's wg comm for this layer is done.
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    // Load compute delay once, then wait for it to expire.
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    index++;
    delay_loaded = false;
    if (index >= SIZE) {
      // Reached the last layer; switch to backward pass.
      current_state = LoopState::Weight_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    delay_loaded = false;
    // Issue AllReduce for weight gradients (non-blocking: overlaps with ig).
    layers[index]->issue_weight_grad_comm(
        SchedulingPolicy::None, CollectiveBarrier::Non_Blocking);
    if (index == 0) {
      // Completed all layers in the backward pass — one full pass done.
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    delay_loaded = false;
    index--;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
// Hybrid-customized parallelism: each layer in the workload file specifies
// its own parallelism strategy, allowing mixed fwd/ig/wg collective types
// within a single model.  The iteration order mirrors the Transformer hybrid
// policy (fwd → ig → wg interleaved backward), but dimension assignments
// are per-layer rather than global.
void Workload::iterate_hybrid_parallel_customized() {

  if (generator->id == 0)
    std::cout << "iterate_hybrid_parallel_customized called:" << std::endl << 
       "\t current_state: " << static_cast<int>(current_state)  << std::endl << 
       "\t delay_loaded: " << delay_loaded  << std::endl << 
       "\t counter: " << counter  << std::endl << 
       "\t collective_issued: " << collective_issued << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}
// HybridDataModel: dimension 0 is used for model-parallel (tensor) comm
// (fwd/ig), while dimensions 1+ handle data-parallel (wg AllReduce).
// The iteration pattern is identical to HybridCustomized/Transformer:
// blocking fwd comm → non-blocking wg comm overlapped with ig → repeat.
void Workload::iterate_hybrid_parallel_data_model() {

  if (generator->id == 0)
    std::cout << "iterate_hybrid_parallel_data_model called:" << std::endl << 
       "\t current_state: " << static_cast<int>(current_state)  << std::endl << 
       "\t delay_loaded: " << delay_loaded  << std::endl << 
       "\t counter: " << counter  << std::endl << 
       "\t collective_issued: " << collective_issued << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}

// HybridModelData: mirror of HybridDataModel with the dimension assignment
// swapped — dimension 0 carries data-parallel wg comm, dimensions 1+ carry
// model-parallel fwd/ig comm.  The scheduling logic is identical.
void Workload::iterate_hybrid_parallel_model_data() {

  if (generator->id == 0)
    std::cout << "iterate_hybrid_parallel_model_data called:" << std::endl << 
       "\t current_state: " << static_cast<int>(current_state)  << std::endl << 
       "\t delay_loaded: " << delay_loaded  << std::endl << 
       "\t counter: " << counter  << std::endl << 
       "\t collective_issued: " << collective_issued << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}

// Distributed-inference mode: forward-pass only, no backward phases.
// Each layer issues a blocking forward-pass collective (e.g., tensor-parallel
// AllReduce/AllGather), then advances to the next layer.  When the last layer
// is processed, the index wraps back to 0 and pass_counter is incremented,
// allowing multiple inference batches to be simulated.
void Workload::iterate_distributed_inference() {

  if (generator->id == 0)
    std::cout << "iterate_distributed_inference called:" << std::endl << 
       "\t current_state: " << static_cast<int>(current_state)  << std::endl << 
       "\t delay_loaded: " << delay_loaded  << std::endl << 
       "\t counter: " << counter  << std::endl << 
       "\t collective_issued: " << collective_issued << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      index = 0;
      pass_counter++;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}

// Pure model-parallel training: all dimensions participate in fwd/ig comms
// (tensor parallelism); wg comms are omitted (no data-parallel AllReduce).
// The backward pass processes layers in reverse: ig compute → ig comm
// (LIFO, non-blocking) → wg compute → wg comm (blocking on prior ig) →
// move to next lower layer or restart forward pass if at layer 0.
void Workload::iterate_model_parallel() {

  if (generator->id == 0)
    std::cout << "iterate_model_parallel called:" << std::endl << 
       "\t current_state: " << static_cast<int>(current_state)  << std::endl << 
       "\t delay_loaded: " << delay_loaded  << std::endl << 
       "\t counter: " << counter  << std::endl << 
       "\t collective_issued: " << collective_issued << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      std::vector<bool> involved_dimensions{true, true, true};
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued && index > 0) {
      collective_issued = true;
      std::vector<bool> involved_dimensions{true, true, true};
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Non_Blocking);
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}

// Transformer hybrid-parallel training (tensor + data parallelism).
// Forward dimensions (up to model_parallel_boundary) handle tensor-parallel
// comm (fwd/ig blocking), while outer dimensions handle data-parallel wg
// AllReduce (FIFO, non-blocking — overlaps with ig of the next layer).
//
// Pass structure:
//   Forward_Pass: fwd compute → blocking fwd comm → advance layer
//   Input_Gradient: ig compute → blocking ig comm → switch to Weight_Gradient
//   Weight_Gradient: wg compute → non-blocking wg comm → wait for prior ig
//                    comm → move to next layer or restart
void Workload::iterate_hybrid_parallel_Transformer() {

  if (generator->id == 0)
    std::cout << "iterate_hybrid_parallel_Transformer called:" << std::endl << 
       "\t current_state: " << static_cast<int>(current_state)  << std::endl << 
       "\t delay_loaded: " << delay_loaded  << std::endl << 
       "\t counter: " << counter  << std::endl << 
       "\t collective_issued: " << collective_issued << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Blocking);
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}

// Transformer hybrid-parallel with gradient checkpointing ("forward-in-backward").
// Identical to iterate_hybrid_parallel_Transformer except that certain layers
// are marked as checkpoints: when the backward pass reaches a layer whose
// 'needs_fwd_in_bckwd_initiation' flag is set, the state machine switches to
// Forward_In_BackPass, re-executes forward computation from the preceding
// checkpoint layer up to the current one, then resumes the backward pass.
// This trades re-computation for reduced activation memory.
//
// Additional state: Forward_In_BackPass — re-runs fwd compute and comm for
// checkpoint layers; 'checkpoint_initiated' prevents double-initiation.
// fwd_pass_comm_size is clamped to 4 KB minimum to avoid degenerate small
// messages on the network.
void Workload::iterate_hybrid_parallel_Transformer_fwd_in_bckwd() {

  if (generator->id == 0)
    std::cout << "iterate_hybrid_parallel_Transformer_fwd_in_bckwd called!" << std::endl << 
       "\t current_state: " << loopstate_to_string(current_state) << std::endl << 
       "\t delay_loaded: " << delay_loaded  << std::endl << 
       "\t counter: " << counter  << std::endl << 
       "\t collective_issued: " << collective_issued << std::endl;

  MockNcclLog* NcclLog = MockNcclLog::getInstance();
  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      if (generator->id == 0)
        std::cout << "is_weight_grad_comm_finished_blocking returned false!" << std::endl;
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      if (generator->id == 0)
        std::cout << "counter updated to " << counter << " which is the fwd_pass_compute_time" << std::endl;
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      if(layers[index]->fwd_pass_comm_size < 4096 && layers[index]->fwd_pass_comm_size >0){
        layers[index]->fwd_pass_comm_size = 4096;
      }
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Input_Gradient;
      index--;
    }
    NcclLog->writeLog(NcclLogLevel::DEBUG,"workload::call fwd_pass register_event EventType::General ");
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::FIFO, CollectiveBarrier::Non_Blocking);
    }
    if (!layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    collective_issued = false;
    delay_loaded = false;
    if (index >= 0) {
      index--;
    }
    if (index == -1) {
      index = 0;
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Input_Gradient) {
    // If this layer triggers a checkpoint re-computation, scan backward to
    // find the nearest checkpoint layer and switch to Forward_In_BackPass.
    if (layers[index]->needs_fwd_in_bckwd_initiation && !checkpoint_initiated) {
      int tmp = index;
      while (!layers[index--]->is_checkpoint)
        ;
      index++;
      current_state = LoopState::Forward_In_BackPass;
      checkpoint_initiated = true;
      generator->register_event(this, EventType::General, NULL, 1);
      if (generator->id == 0) {
        std::cout << "***** info, initiating fwd_in_bkwd starting from layer:"
                  << index << " to layer: " << tmp
                  << " ,at time: " << Sys::boostedTick() << std::endl;
      }
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_input_grad_comm(
          SchedulingPolicy::LIFO, CollectiveBarrier::Blocking);
      return;
    }
    checkpoint_initiated = false;
    collective_issued = false;
    delay_loaded = false;
    current_state = LoopState::Weight_Gradient;
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Forward_In_BackPass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::None, CollectiveBarrier::Blocking);
      return;
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (layers[index]->needs_fwd_in_bckwd_initiation) {
      current_state = LoopState::Input_Gradient;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  }
}

// DLRM / DLRMEnhanced hybrid-parallel training.
// DLRM has a distinct "bottom" MLP + embedding layers followed by "top" MLP
// layers.  The bottom embedding layers (indices 0..DLRM_LAST_BOTTOM_LAYER)
// use All-to-All for forward communication (embedding look-up redistribution),
// while all layers use Non-Blocking wg AllReduce.
//
// Key scheduling difference from Transformer:
//   - Bottom-layer fwd All-to-All is issued non-blocking on first encounter.
//   - The top section (layer DLRM_LAST_BOTTOM_LAYER+1) stalls in the fwd
//     pass until layer 0's All-to-All has completed.
//   - During input-gradient, layer DLRM_LAST_BOTTOM_LAYER+1 kicks off
//     layer 0's ig All-to-All (non-blocking, HIGHEST priority).
//   - Under DLRMEnhanced policy, the blocking-on-ig-comm check is skipped,
//     allowing further overlap.
void Workload::iterate_hybrid_parallel_DLRM() {

  if (generator->id == 0)
    std::cout << "iterate_hybrid_parallel_DLRM called:" << std::endl << 
       "\t current_state: " << static_cast<int>(current_state)  << std::endl;

  assert(index >= 0);
  assert(index < SIZE);
  check_for_sim_end();
  if (current_state == LoopState::Forward_Pass) {
    if (!layers[index]->is_weight_grad_comm_finished_blocking()) {
      return;
    }
    if (delay_loaded == false) {
      counter = layers[index]->get_fwd_pass_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued &&
        layers[index]->fwd_pass_comm_type == ComType::All_to_All) {
      collective_issued = true;
      layers[index]->issue_forward_pass_comm(
          SchedulingPolicy::HIGHEST, CollectiveBarrier::Non_Blocking);

    } else if (index == DLRM_LAST_BOTTOM_LAYER) {
      if (!layers[0]->is_fwd_pass_comm_finished_blocking()) {
        return;
      }
    }
    index++;
    delay_loaded = false;
    collective_issued = false;
    if (index >= SIZE) {
      current_state = LoopState::Weight_Gradient;
      index--;
    }
    if (generator->id == 0) {
      std::cout << "*************************layer changed to: " << index
                << std::endl;
    }
    generator->register_event(this, EventType::General, NULL, 1);
    return;
  } else if (current_state == LoopState::Weight_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_weight_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (!collective_issued) {
      collective_issued = true;
      layers[index]->issue_weight_grad_comm(
          SchedulingPolicy::None, CollectiveBarrier::Non_Blocking);
    }
    if (parallelismPolicy == ParallelismPolicy::DLRM &&
        !layers[index]->is_input_grad_comm_finished_blocking()) {
      return;
    }
    if (index == 0) {
      if (generator->id == 0) {
        std::cout << "pass: " << pass_counter
                  << " finished at time: " << Sys::boostedTick() << std::endl;
      }
      pass_counter++;
      current_state = LoopState::Forward_Pass;
    } else {
      current_state = LoopState::Input_Gradient;
    }
    delay_loaded = false;
    collective_issued = false;
    generator->register_event(this, EventType::General, NULL, 1);
  } else if (current_state == LoopState::Input_Gradient) {
    if (delay_loaded == false) {
      counter = layers[index]->get_input_grad_compute();
      delay_loaded = true;
    }
    if (counter > 0) {
      generator->try_register_event(
          this, EventType::Workload_Wait, NULL, counter);
      return;
    }
    if (index == DLRM_LAST_BOTTOM_LAYER + 1) {
      layers[0]->issue_input_grad_comm(
          SchedulingPolicy::HIGHEST, CollectiveBarrier::Non_Blocking);
    }
    index--;
    if (generator->id == 0) {
      std::cout << "*************************layer changed to: " << index
                << " in ig" << std::endl;
    }
    current_state = LoopState::Weight_Gradient;
    collective_issued = false;
    delay_loaded = false;
    generator->register_event(this, EventType::General, NULL, 1);
  }
}
// Reads and returns the number of layers declared in a workload file.
// Opens the file under the "workload_inputs/" prefix, skips the header
// line, and reads the integer on the second line.  Used externally to
// pre-size data structures before constructing the full Workload object.
int Workload::get_layer_numbers(std::string workload_input) {
  std::ifstream inFile;
  inFile.open("workload_inputs/" + workload_input);
  if (!inFile) {
    std::cerr << "Unable to open file: " << workload_input << std::endl;
    std::cerr << "This error is fatal. Please check your path and filename."
              << std::endl;
    exit(1);
  } else {
    std::cout << "Success in opening workload file" << std::endl;
  }
  std::string dummyLine;
  std::getline(inFile, dummyLine);
  int layers;
  inFile >> layers;
  inFile.close();
  return layers;
}

// Converts a parallelism strategy string (as it appears in the workload file
// header) to the corresponding ParallelismPolicy enum value.
// Returns ParallelismPolicy::None if the string is unrecognized, which will
// cause initialize_workload() to abort (except under PHY_MTP builds).
ParallelismPolicy Workload::decode_parallelsim(std::string parallelism) {

  if (generator->id == 0)
    std::cout << "decode_parallelsim called. input parallelism: " << parallelism << std::endl;

  if (parallelism == "DATA")
    return ParallelismPolicy::Data;
  else if (parallelism == "HYBRID_TRANSFORMER")
    return ParallelismPolicy::Transformer;
  else if (parallelism == "HYBRID_TRANSFORMER_FWD_IN_BCKWD")
    return ParallelismPolicy::TransformerFwdInBckwd;
  else if (parallelism == "HYBRID_DLRM")
    return ParallelismPolicy::DLRM;
  else if (parallelism == "HYBRID_DLRM_ENHANCED")
    return ParallelismPolicy ::DLRMEnhanced;
  else if (parallelism == "MODEL")
    return ParallelismPolicy::Model;
  else if (parallelism == "HYBRID_DATA_MODEL")
    return ParallelismPolicy::HybridDataModel;
  else if (parallelism == "HYBRID_MODEL_DATA")
    return ParallelismPolicy::HybridModelData;
  else if (parallelism == "HYBRID_CUSTOMIZED")
    return ParallelismPolicy::HybridCustomized;
  else if (parallelism == "MICRO")
    return ParallelismPolicy::MicroBenchmark;
  else if (parallelism == "DISTRIBUTED_INFERENCE")
    return ParallelismPolicy::DistributedInference;
  else
    return ParallelismPolicy::None;
}

// Maps a parallelism policy to per-phase dimension masks.
// Returns a map with keys "fwd", "ig", "wg"; each value is a 10-element
// boolean vector indicating which logical network dimensions participate in
// that phase's collective.
//
// Rules:
//   Data / DLRM / MicroBenchmark: only wg uses all dims (data-parallel only).
//   Model / DistributedInference: fwd and ig use all dims (tensor-parallel).
//   HybridModelData: dim 0 = data (wg), dims 1-9 = model (fwd/ig).
//   HybridDataModel: dim 0 = model (fwd/ig), dims 1-9 = data (wg).
//   Transformer / TransformerFwdInBckwd: dims 0..model_parallel_boundary
//       are model-parallel (fwd/ig); remaining dims are data-parallel (wg).
//       The boundary is derived from model_parallel_npu_group via
//       generator->break_dimension().
std::map<std::string, std::vector<bool>> Workload::decode_involved_dimensions(
    ParallelismPolicy policy,
    int model_parallel_npu_group) {
  std::map<std::string, std::vector<bool>> result;
  std::vector<bool> none{
      false, false, false, false, false, false, false, false, false, false};
  std::vector<bool> all{
      true, true, true, true, true, true, true, true, true, true};
  if (policy == ParallelismPolicy::All) {
    result["fwd"] = all;
    result["ig"] = all;
    result["wg"] = all;
  } else if (
      policy == ParallelismPolicy::Data || policy == ParallelismPolicy::DLRM ||
      policy == ParallelismPolicy::DLRMEnhanced ||
      policy == ParallelismPolicy::MicroBenchmark) {
    result["fwd"] = none;
    result["ig"] = none;
    result["wg"] = all;
  } else if (
      policy == ParallelismPolicy::Model ||
      policy == ParallelismPolicy::DistributedInference) {
    result["fwd"] = all;
    result["ig"] = all;
    result["wg"] = none;
  } else if (policy == ParallelismPolicy::HybridModelData) {
    std::vector<bool> data{
        true, false, false, false, false, false, false, false, false, false};
    std::vector<bool> model{
        false, true, true, true, true, true, true, true, true, true};
    result["fwd"] = model;
    result["ig"] = model;
    result["wg"] = data;
  } else if (policy == ParallelismPolicy::HybridDataModel) {
    std::vector<bool> model{
        true, false, false, false, false, false, false, false, false, false};
    std::vector<bool> data{
        false, true, true, true, true, true, true, true, true, true};
    result["fwd"] = model;
    result["ig"] = model;
    result["wg"] = data;
  } else if (
      policy == ParallelismPolicy::TransformerFwdInBckwd ||
      policy == ParallelismPolicy::Transformer) {
    int model_parallel_boundary =
        generator->break_dimension(model_parallel_npu_group);
    std::vector<bool> model;
    std::vector<bool> data;
    for (int i = 0; i <= model_parallel_boundary; i++) {
      model.push_back(true);
      data.push_back(false);
    }
    for (int i = model_parallel_boundary + 1; i < 10; i++) {
      model.push_back(false);
      data.push_back(true);
    }
    result["fwd"] = model;
    result["ig"] = model;
    result["wg"] = data;
  }
  return result;
}

// Parses the workload description file and populates the layers[] array.
//
// File format (text):
//   Line 1: <PARALLELISM_STRATEGY> [key: value ...]
//           e.g. "HYBRID_TRANSFORMER model_parallel_NPU_group: 8 pp: 2 ..."
//   Line 2: <number_of_layers>
//   Lines 3+: one layer per line with fields:
//     id  dependency  fp_compute  fp_comm_type  fp_comm_size
//                     ig_compute  ig_comm_type  ig_comm_size
//                     wg_compute  wg_comm_type  wg_comm_size  wg_update_time
//     [specific_parallelism]  (only for HybridCustomized policy)
//
// Communication type strings encode both the collective type (e.g. ALLREDUCE,
// ALLTOALL, ALLGATHER, REDUCESCATTER, BROADCAST) and the process group
// (DP, EP, DP_EP suffix, or none for the default group).
//
// Compute times are scaled by generator->compute_scale; comm sizes are scaled
// by generator->comm_scale, allowing global slow-down/speed-up factors.
//
// Checkpoint and checkpoint-initiation layers (used in TransformerFwdInBckwd)
// are read from the header line and applied to the corresponding Layer objects.
//
// Returns true on success; calls exit(1) on file-open or parse errors.
bool Workload::initialize_workload(std::string name) {
  std::map<int, bool> chekpoints;
  std::map<int, bool> need_checkpoint_initiation;
  std::ifstream inFile;
  inFile.open(name);
  if (!inFile) {
    std::cerr << "Unable to open file: " << name << std::endl;
    std::cerr << "######### Exiting because unable to open the workload input "
                 "file #########"
              << std::endl;
    std::cerr << "This error is fatal. Please check your path and filename."
              << std::endl;
    exit(1);
  } else {
    if (generator->id == 0) {
      std::cout << "Success in opening workload file" << std::endl;
    }
  }
 std::string firstline;
  std::getline(inFile,firstline);
  if (generator->id == 0)
    std::cout << "First line is : '" << firstline << "'" << std::endl;
  std::istringstream iss(firstline);
  std:string token;
  std::vector<std::string> tokens;
  // bool findparallesimPolcy = false;
  
  while (iss >> token) {
      tokens.push_back(token);
      // if (generator->id == 0)
      //   std::cout << "Token is : '" << token << "'" << std::endl;
  }

  if(!tokens.empty()){
    parallelismPolicy = decode_parallelsim(tokens[0]);
  }

  if (parallelismPolicy == ParallelismPolicy::TransformerFwdInBckwd ||
      parallelismPolicy == ParallelismPolicy::Transformer) {
    for (size_t i = 1; i < tokens.size(); i = i+1) {
      if(tokens[i]=="model_parallel_NPU_group:"){
        model_parallel_npu_group = std::stoi(tokens[i+1]);
        if (generator->id == 0) {
          std::cout <<"model_parallel_NPU_group is " << model_parallel_npu_group << std::endl;
        }
      }else if(tokens[i]=="ep:"){
        expert_parallel_npu_group = std::stoi(tokens[i+1]);
      }else if(tokens[i]== "pp:"){
        pipeline_model_parallelism = std::stoi(tokens[i+1]);
      }else if(tokens[i]=="vpp:"){
        vpp = std::stoi(tokens[i+1]);
      }else if(tokens[i]=="ga:"){
        GA = std::stoi(tokens[i+1]);
      }else if(tokens[i]=="all_gpus:"){
        all_gpus = std::stoi(tokens[i+1]);
      }
    }

    if(parallelismPolicy == ParallelismPolicy::TransformerFwdInBckwd) {
      if (generator->id == 0) {
        std::cout << "checkpoints layers are: ";
      }
      for(size_t i = 1; i < tokens.size(); i = i+1) {
        if(tokens[i]=="checkpoints:"){
          int account = std::stoi(tokens[i+1]);
          while(account-- >0){
            int j = 2;
            int layer = std::stoi(tokens[i+j]);
            chekpoints[layer] = true;
            if (generator->id == 0) {
              std::cout << layer << ", ";
            }
            j++;
          }
            
        } else if(tokens[i]=="checkpoint_initiates:") {
          if (generator->id == 0) {
            std::cout << std::endl;
            std::cout << "layers initiating fwd_in_bckwd are: ";
          }
          int account = std::stoi(tokens[i+1]);
          while(account-- > 0) {
            int j = 2;
            int layer = std::stoi(tokens[i+j]);
            need_checkpoint_initiation[layer] = true;
            if (generator->id == 0) {
              std::cout << layer << ", ";
            }
            j++;
          }
          if (generator->id == 0) {
            std::cout << std::endl;
          }
        }
      }
    }
  } else if(parallelismPolicy == ParallelismPolicy::DLRM ||
            parallelismPolicy == ParallelismPolicy::DLRMEnhanced) {
    for (size_t i = 1; i < tokens.size(); i = i+1) {
      if(tokens[i]=="DLRM_LAST_BOTTOM_LAYER:"){
        DLRM_LAST_BOTTOM_LAYER = std::stoi(tokens[i+1]);
      }
    }
    if (generator->id == 0) {
      std::cout
      << "****************** info: DLRM workload last bottom layer is: "
      << DLRM_LAST_BOTTOM_LAYER << std::endl;
    }
  } else if (parallelismPolicy == ParallelismPolicy::None) {
      #ifndef PHY_MTP
      std::cerr << "######### Exiting because unable to decode the workload "
              "parallelization strategy #########"
              << std::endl;
      inFile.close();
      exit(1);
      #else
      parallelismPolicy = ParallelismPolicy::TransformerFwdInBckwd;
      #endif
  }
  
  std::map<std::string, std::vector<bool>> general_involved_dimensions =
      decode_involved_dimensions(parallelismPolicy, model_parallel_npu_group);
      
  pp_commsize = 0;
  for (size_t i = 1; i < tokens.size(); i = i+1){
    if(tokens[i]=="pp_comm"||tokens[i]=="pp_comm:"){
      pp_commsize = std::stoi(tokens[i+1]);
    }
  }
  if (generator->id == 0) {
      std::cout <<"pp_commsize:"<< pp_commsize << std::endl;
  }
  if(generator->id == 0){
    if (model_parallel_npu_group == 0 || expert_parallel_npu_group == 0 || pipeline_model_parallelism == 0 
        || vpp==0 || GA == 0 || all_gpus == 0 ||(pipeline_model_parallelism !=1 && pp_commsize ==0)||(pipeline_model_parallelism == 1 && pp_commsize !=0)){
          std::cerr << "*****Warining: Input workload format mismatch. It may cause simulation error. Pleased use the latest AICB to generate.*****" << std::endl;
      }
  }        
  run_type = tokens[0];
  std::string secondline;
  std::getline(inFile,secondline);

  int lines;
  // std::cout << "Second line content: '" << secondline << "'" << std::endl;
  lines = std::stoi(secondline);


  SIZE = lines;
  layers = new Layer*[SIZE];
  for (int i = 0; i < lines; i++) {
    std::string id;
    inFile >> id;
    int depen;
    inFile >> depen;

    Tick fp_compute_time;
    inFile >> fp_compute_time;
    std::string fp_comm_type_s;
    inFile >> fp_comm_type_s;
    uint64_t fp_comm_size;
    inFile >> fp_comm_size;

    Tick ig_compute_time;
    inFile >> ig_compute_time;
    std::string ig_comm_type_s;
    inFile >> ig_comm_type_s;
    uint64_t ig_comm_size;
    inFile >> ig_comm_size;

    Tick wg_compute_time;
    inFile >> wg_compute_time;
    std::string wg_comm_type_s;
    inFile >> wg_comm_type_s;
    uint64_t wg_comm_size;
    inFile >> wg_comm_size;
    Tick wg_update_time;
    inFile >> wg_update_time;

    if (generator->id == 0) {
      std::cout << "\nRead workload info from file line " << i << ":\n" <<
        "\t id: " << id << "\n" <<
        "\t depen: " << depen << "\n" <<
        "\t fp_compute_time: " << fp_compute_time << "\n" <<
        "\t fp_comm_type_s: " << fp_comm_type_s << "\n" <<
        "\t fp_comm_size: " << fp_comm_size << "\n" <<
        "\t ig_compute_time: " << ig_compute_time << "\n" <<
        "\t g_comm_type_s: " << ig_comm_type_s << "\n" <<
        "\t ig_comm_size: " << ig_comm_size << "\n" <<
        "\t wg_compute_time: " << wg_compute_time << "\n" <<
        "\t wg_comm_type_s: " << wg_comm_type_s << "\n" <<
        "\t wg_comm_size: " << wg_comm_size << "\n" <<
        "\t wg_update_time: " << wg_update_time << "\n" << 
        "-------------------------\n" << std::endl;
    }

    // Decode collective type and process-group from the string fields read
    // from the workload file.  Each phase (wg, ig, fp) is decoded independently.
    // The prefix determines ComType; an optional suffix (_EP, _DP_EP) selects
    // the MockNccl group (EP = expert-parallel, DP_EP = data+expert-parallel,
    // no suffix = default data/tensor-parallel group for that phase).
    ParallelismPolicy specific_policy = ParallelismPolicy::None;
    std::map<std::string, std::vector<bool>> selected_involved_dimensions;
    ComType fp_type = ComType::None;
    ComType ig_type = ComType::None;
    ComType wg_type = ComType::None;
    MockNccl::GroupType fp_group_type = MockNccl::GroupType::NONE;
    MockNccl::GroupType ig_group_type = MockNccl::GroupType::NONE;
    MockNccl::GroupType wg_group_type = MockNccl::GroupType::NONE;
    if (wg_comm_type_s.substr(0,9) == "ALLREDUCE") {
      wg_type = ComType::All_Reduce;
      if(wg_comm_type_s == "ALLREDUCE"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLREDUCE_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLREDUCE_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,8) == "ALLTOALL") {
      wg_type = ComType::All_to_All;
      if(wg_comm_type_s == "ALLTOALL"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLTOALL_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLTOALL_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,17) == "ALLREDUCEALLTOALL") {
      wg_type = ComType::All_Reduce_All_to_All;
      if(wg_comm_type_s == "ALLREDUCEALLTOALL"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLREDUCEALLTOALL_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLREDUCEALLTOALL_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,9) == "ALLGATHER") {
      wg_type = ComType::All_Gather;
      if(wg_comm_type_s == "ALLGATHER"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "ALLGATHER_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "ALLGATHER_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,13) == "REDUCESCATTER") {
      wg_type = ComType::Reduce_Scatter;
      if(wg_comm_type_s == "REDUCESCATTER"){
        wg_group_type = MockNccl::GroupType::DP;
      } else if(wg_comm_type_s == "REDUCESCATTER_EP"){
        wg_group_type = MockNccl::GroupType::EP;
      } else if(wg_comm_type_s == "REDUCESCATTER_DP_EP"){
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else{
        wg_group_type = MockNccl::GroupType::NONE;
      }
    } else if (wg_comm_type_s.substr(0,9) == "BROADCAST") {
      // sepehr
      wg_type = ComType::Broadcast;
      if (wg_comm_type_s == "BROADCAST") {
        wg_group_type = MockNccl::GroupType::DP;
      } else if (wg_comm_type_s == "BROADCAST_EP") {
        wg_group_type = MockNccl::GroupType::EP;
      } else if (wg_comm_type_s == "BROADCAST_DP_EP") {
        wg_group_type = MockNccl::GroupType::DP_EP;
      } else {
        wg_group_type = MockNccl::GroupType::NONE;
      }
    }

    // generate flow model

    if (ig_comm_type_s.substr(0,9) == "ALLREDUCE") {
      ig_type = ComType::All_Reduce;
      if(ig_comm_type_s == "ALLREDUCE"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLREDUCE_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLREDUCE_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,8) == "ALLTOALL") {
      ig_type = ComType::All_to_All;
      if(ig_comm_type_s == "ALLTOALL"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLTOALL_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLTOALL_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,17) == "ALLREDUCEALLTOALL") {
      ig_type = ComType::All_Reduce_All_to_All;
      if(ig_comm_type_s == "ALLREDUCEALLTOALL"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLREDUCEALLTOALL_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLREDUCEALLTOALL_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,9) == "ALLGATHER") {
      ig_type = ComType::All_Gather;
      if(ig_comm_type_s == "ALLGATHER"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "ALLGATHER_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "ALLGATHER_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,13) == "REDUCESCATTER") {
      ig_type = ComType::Reduce_Scatter;
      if(ig_comm_type_s == "REDUCESCATTER"){
        ig_group_type = MockNccl::GroupType::TP;
      } else if(ig_comm_type_s == "REDUCESCATTER_EP"){
        ig_group_type = MockNccl::GroupType::EP;
      } else if(ig_comm_type_s == "REDUCESCATTER_DP_EP"){
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else{
        ig_group_type = MockNccl::GroupType::NONE;
      }
    } else if (ig_comm_type_s.substr(0,9) == "BROADCAST") {
      // sepehr
      ig_type = ComType::Broadcast;
      if (ig_comm_type_s == "BROADCAST") {
        ig_group_type = MockNccl::GroupType::TP;
      } else if (ig_comm_type_s == "BROADCAST_EP") {
        ig_group_type = MockNccl::GroupType::EP;
      } else if (ig_comm_type_s == "BROADCAST_DP_EP") {
        ig_group_type = MockNccl::GroupType::DP_EP;
      } else {
        ig_group_type = MockNccl::GroupType::NONE;
      }
    }

    if (fp_comm_type_s.substr(0,9) == "ALLREDUCE") {
      fp_type = ComType::All_Reduce;
      if(fp_comm_type_s == "ALLREDUCE"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLREDUCE_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLREDUCE_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,8) == "ALLTOALL") {
      fp_type = ComType::All_to_All;
      if(fp_comm_type_s == "ALLTOALL"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLTOALL_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLTOALL_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,17) == "ALLREDUCEALLTOALL") {
      fp_type = ComType::All_Reduce_All_to_All;
      if(fp_comm_type_s == "ALLREDUCEALLTOALL"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLREDUCEALLTOALL_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLREDUCEALLTOALL_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,9) == "ALLGATHER") {
      fp_type = ComType::All_Gather;
      if(fp_comm_type_s == "ALLGATHER"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "ALLGATHER_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "ALLGATHER_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,13) == "REDUCESCATTER") {
      fp_type = ComType::Reduce_Scatter;
      if(fp_comm_type_s == "REDUCESCATTER"){
        fp_group_type = MockNccl::GroupType::TP;
      } else if(fp_comm_type_s == "REDUCESCATTER_EP"){
        fp_group_type = MockNccl::GroupType::EP;
      } else if(fp_comm_type_s == "REDUCESCATTER_DP_EP"){
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else{
        fp_group_type = MockNccl::GroupType::NONE;
      }
    } else if (fp_comm_type_s.substr(0,9) == "BROADCAST") {
      // sepehr
      fp_type = ComType::Broadcast;
      if (fp_comm_type_s == "BROADCAST") {
        fp_group_type = MockNccl::GroupType::TP;
      } else if (fp_comm_type_s == "BROADCAST_EP") {
        fp_group_type = MockNccl::GroupType::EP;
      } else if (fp_comm_type_s == "BROADCAST_DP_EP") {
        fp_group_type = MockNccl::GroupType::DP_EP;
      } else {
        fp_group_type = MockNccl::GroupType::NONE;
      }
    }
    
    if (parallelismPolicy == ParallelismPolicy::HybridCustomized) {
      std::string specific_parallelsim;
      inFile >> specific_parallelsim;
      specific_policy = decode_parallelsim(specific_parallelsim);
    }
    if ((parallelismPolicy == ParallelismPolicy::DLRM ||
         parallelismPolicy == ParallelismPolicy::DLRMEnhanced) &&
        i == 0) {
      specific_policy = ParallelismPolicy::All;
    }
    if (specific_policy != ParallelismPolicy::None) {
      selected_involved_dimensions =
          decode_involved_dimensions(specific_policy, model_parallel_npu_group);
    } else {
      selected_involved_dimensions = general_involved_dimensions;
    }
    Layer* l = new Layer(
        id,
        i,
        generator,
        this,
        fp_compute_time * generator->compute_scale,
        fp_type,
        fp_group_type,
        fp_comm_size * generator->comm_scale,
        selected_involved_dimensions["fwd"],
        ig_compute_time * generator->compute_scale,
        ig_type,
        ig_group_type,
        ig_comm_size * generator->comm_scale,
        selected_involved_dimensions["ig"],
        wg_compute_time * generator->compute_scale,
        wg_type,
        wg_group_type,
        wg_comm_size * generator->comm_scale,
        selected_involved_dimensions["wg"],
        wg_update_time,
        specific_policy);
    if (chekpoints.find(i) != chekpoints.end()) {
      l->is_checkpoint = true;
    }
    if (need_checkpoint_initiation.find(i) !=
        need_checkpoint_initiation.end()) {
      l->needs_fwd_in_bckwd_initiation = true;
    }
    layers[i] = l;
  }
  if (generator->id == 0) {
    std::cout << "type: " << run_type << ", num passes: " << TOTAL_PASS
              << ", lines: " << lines
              << ", compute scale: " << generator->compute_scale
              << ", comm scale: " << generator->comm_scale << std::endl;
  }
  inFile.close();
  return true;
}
// Convenience wrapper used by the scheduler to kick off a workload event.
// Equivalent to calling call() directly with a General event.
void Workload::fire() {
  call(EventType::General, NULL);
}
} // namespace AstraSim

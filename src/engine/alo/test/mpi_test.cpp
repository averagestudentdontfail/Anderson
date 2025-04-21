#include "../mpi_wrapper.h"
#include "../hybrid_parallel.h"
#include "../parallel_diagnostics.h"
#include "../alodistribute.h"
#include "../aloengine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>

using namespace engine::alo;

class Timer {
public:
    Timer() : start_(std::chrono::high_resolution_clock::now()) {}
    
    double elapsed() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }
    
    void reset() {
        start_ = std::chrono::high_resolution_clock::now();
    }

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

void testBasicMPI() {
    int rank = mpi::MPIWrapper::rank();
    int size = mpi::MPIWrapper::size();
    
    if (rank == 0) {
        std::cout << "\n=== Basic MPI Test ===\n";
        std::cout << "Number of processes: " << size << "\n";
    }
    
    // Barrier to ensure all processes reach this point
    mpi::MPIWrapper::barrier();
    
    // Each process prints its rank
    for (int i = 0; i < size; ++i) {
        if (rank == i) {
            std::cout << "Hello from process " << rank << "\n";
        }
        mpi::MPIWrapper::barrier(); // Ensure ordered output
    }
    
    if (rank == 0) {
        std::cout << "Basic MPI test completed.\n";
    }
}

void testDistributedPricing() {
    int rank = mpi::MPIWrapper::rank();
    int size = mpi::MPIWrapper::size();
    
    if (rank == 0) {
        std::cout << "\n=== Distributed Option Pricing Test ===\n";
    }
    
    // Create a task dispatcher
    auto dispatcher = dist::createTaskDispatcher(ACCURATE, 100);
    
    // Option parameters
    double S = 100.0;
    double r = 0.05;
    double q = 0.02;
    double vol = 0.2;
    double T = 1.0;
    
    // Create a large array of strikes
    std::vector<double> strikes;
    const size_t numStrikes = 10000;
    for (size_t i = 0; i < numStrikes; ++i) {
        strikes.push_back(50.0 + 100.0 * static_cast<double>(i) / numStrikes);
    }
    
    // Time the distributed calculation
    Timer timer;
    auto results = dispatcher->distributedBatchCalculatePut(S, strikes, r, q, vol, T);
    double distributedTime = timer.elapsed();
    
    if (rank == 0) {
        std::cout << "Pricing " << numStrikes << " options across " << size << " processes\n";
        std::cout << "Total time: " << distributedTime << " ms\n";
        std::cout << "Time per option: " << distributedTime / numStrikes << " ms\n";
        
        // Compare with sequential calculation (time a small subset)
        timer.reset();
        ALOEngine engine(ACCURATE);
        auto subset = std::vector<double>(strikes.begin(), strikes.begin() + 100);
        auto seqResults = engine.batchCalculatePut(S, subset, r, q, vol, T);
        double seqTime = timer.elapsed();
        
        double estimatedSeqTime = (seqTime / 100.0) * numStrikes;
        std::cout << "Estimated sequential time: " << estimatedSeqTime << " ms\n";
        std::cout << "Speedup: " << estimatedSeqTime / distributedTime << "x\n";
        
        // Verify results for first few options
        bool resultsMatch = true;
        for (size_t i = 0; i < subset.size(); ++i) {
            if (std::abs(results[i] - seqResults[i]) > 1e-10) {
                resultsMatch = false;
                break;
            }
        }
        std::cout << "Results verification: " << (resultsMatch ? "PASSED" : "FAILED") << "\n";
    }
}

void testHybridParallel() {
    int rank = mpi::MPIWrapper::rank();
    int size = mpi::MPIWrapper::size();
    
    if (rank == 0) {
        std::cout << "\n=== Hybrid OpenMP-MPI Test ===\n";
    }
    
    // Test data
    const size_t dataSize = 1000000;
    std::vector<double> data(dataSize);
    
    // Initialize data
    for (size_t i = 0; i < dataSize; ++i) {
        data[i] = static_cast<double>(i) / dataSize;
    }
    
    // Simple computation - square each element
    Timer timer;
    auto results = hybrid::HybridExecutor::processDistributed(
        data,
        [](double x) { return x * x; },
        1000
    );
    double hybridTime = timer.elapsed();
    
    if (rank == 0) {
        std::cout << "Processing " << dataSize << " elements across " << size << " processes\n";
        std::cout << "Total time: " << hybridTime << " ms\n";
        
        // Verify results
        bool correct = true;
        for (size_t i = 0; i < std::min(size_t(10), results.size()); ++i) {
            double expected = data[i] * data[i];
            if (std::abs(results[i] - expected) > 1e-10) {
                correct = false;
                break;
            }
        }
        std::cout << "Results verification: " << (correct ? "PASSED" : "FAILED") << "\n";
    }
}

int main(int argc, char** argv) {
    try {
        // Initialize MPI
        mpi::MPIWrapper::init(&argc, &argv);
        
        int rank = mpi::MPIWrapper::rank();
        
        if (rank == 0) {
            std::cout << "=== MPI Test Program ===\n";
        }
        
        // Run diagnostics
        diag::ParallelDiagnostics::diagnose();
        
        // Run tests
        testBasicMPI();
        testDistributedPricing();
        testHybridParallel();
        
        // Finalize MPI
        mpi::MPIWrapper::finalize();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
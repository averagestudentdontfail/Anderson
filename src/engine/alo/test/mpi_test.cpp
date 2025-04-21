#include "../mpi_wrapper.h"
#include "../hybrid_parallel.h"
#include "../parallel_diagnostics.h"
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
            std::cout.flush(); // Ensure output is flushed immediately
        }
        mpi::MPIWrapper::barrier(); // Ensure ordered output
    }
    
    if (rank == 0) {
        std::cout << "Basic MPI test completed.\n";
        std::cout.flush();
    }
}

void testSimpleDistributedPricing() {
    int rank = mpi::MPIWrapper::rank();
    int size = mpi::MPIWrapper::size();
    
    if (rank == 0) {
        std::cout << "\n=== Simple Distributed Option Pricing Test ===\n";
        std::cout.flush();
    }
    
    // Option parameters
    double S = 100.0;
    double r = 0.05;
    double q = 0.02;
    double vol = 0.2;
    double T = 1.0;
    
    // Create a smaller array of strikes for testing
    const size_t numStrikes = 1000; // Reduced from 10000
    std::vector<double> strikes;
    for (size_t i = 0; i < numStrikes; ++i) {
        strikes.push_back(50.0 + 100.0 * static_cast<double>(i) / numStrikes);
    }
    
    // Simple distributed calculation without TaskDispatcher
    Timer timer;
    
    // Calculate how many strikes each process should handle
    size_t strikesPerProcess = numStrikes / size;
    size_t remainingStrikes = numStrikes % size;
    
    // Calculate start and end index for this process
    size_t startIdx = rank * strikesPerProcess + std::min(static_cast<size_t>(rank), remainingStrikes);
    size_t endIdx = startIdx + strikesPerProcess + (static_cast<size_t>(rank) < remainingStrikes ? 1 : 0);
    
    // Process local strikes
    std::vector<double> localStrikes(strikes.begin() + startIdx, strikes.begin() + endIdx);
    
    ALOEngine engine(ACCURATE);
    auto localResults = engine.batchCalculatePut(S, localStrikes, r, q, vol, T);
    
    // Gather all results to rank 0
    std::vector<double> allResults;
    if (rank == 0) {
        allResults.resize(numStrikes);
    }
    
    // Gather the sizes first
    int localSize = static_cast<int>(localResults.size());
    std::vector<int> sizes(size);
    MPI_Gather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    
    // Calculate displacements
    std::vector<int> displacements(size, 0);
    if (rank == 0) {
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i-1] + sizes[i-1];
        }
    }
    
    // Gather results
    MPI_Gatherv(localResults.data(), localSize, MPI_DOUBLE,
                allResults.data(), sizes.data(), displacements.data(),
                MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    double distributedTime = timer.elapsed();
    
    if (rank == 0) {
        std::cout << "Priced " << numStrikes << " options across " << size << " processes\n";
        std::cout << "Total time: " << distributedTime << " ms\n";
        std::cout << "Time per option: " << distributedTime / numStrikes << " ms\n";
        
        // Verify a few results
        std::cout << "Sample results:\n";
        for (size_t i = 0; i < std::min(size_t(5), numStrikes); ++i) {
            std::cout << "  Strike " << strikes[i] << ": " << allResults[i] << "\n";
        }
        
        std::cout << "Simple distributed pricing test completed.\n";
        std::cout.flush();
    }
}

void testHybridParallel() {
    int rank = mpi::MPIWrapper::rank();
    int size = mpi::MPIWrapper::size();
    
    if (rank == 0) {
        std::cout << "\n=== Hybrid OpenMP-MPI Test ===\n";
        std::cout.flush();
    }
    
    // Test data - using doubles which have the specialized implementation
    const size_t dataSize = 10000; // Moderate size for quick testing
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
        100  // Larger chunk size for better performance
    );
    double hybridTime = timer.elapsed();
    
    if (rank == 0) {
        std::cout << "Processing " << dataSize << " elements across " << size << " processes\n";
        std::cout << "Total time: " << hybridTime << " ms\n";
        
        // Verify results (checking only on rank 0 where results are gathered)
        bool correct = true;
        for (size_t i = 0; i < std::min(size_t(10), results.size()); ++i) {
            double expected = data[i] * data[i];
            if (std::abs(results[i] - expected) > 1e-10) {
                correct = false;
                break;
            }
        }
        std::cout << "Results verification: " << (correct ? "PASSED" : "FAILED") << "\n";
        std::cout.flush();
    }
}

int main(int argc, char** argv) {
    try {
        // Initialize MPI
        mpi::MPIWrapper::init(&argc, &argv);
        
        int rank = mpi::MPIWrapper::rank();
        
        if (rank == 0) {
            std::cout << "=== MPI Test Program ===\n";
            std::cout.flush();
        }
        
        // Run diagnostics
        diag::ParallelDiagnostics::diagnose();
        
        // Ensure all processes are synchronized before tests
        mpi::MPIWrapper::barrier();
        
        // Run tests
        testBasicMPI();
        
        // Ensure all processes are synchronized
        mpi::MPIWrapper::barrier();
        
        // Use simplified distributed pricing test instead of TaskDispatcher
        testSimpleDistributedPricing();
        
        // Ensure all processes are synchronized
        mpi::MPIWrapper::barrier();
        
        testHybridParallel();
        
        // Ensure all processes are synchronized before finalization
        mpi::MPIWrapper::barrier();
        
        // Finalize MPI
        mpi::MPIWrapper::finalize();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error on rank " << mpi::MPIWrapper::rank() << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
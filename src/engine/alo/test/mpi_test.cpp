#include "../mpiwrapper.h"
#include "../hybridparallel.h"
#include "../paralleldiagnose.h"
#include "../aloengine.h"
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>

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

// Test result structure
struct TestResult {
    bool passed;
    double executionTime;
    std::string message;
};

// Performance metrics structure
struct PerformanceMetrics {
    double averageTimePerOption;
    double totalTime;
    double speedup;
    size_t optionsProcessed;
};

// Utility function to generate random option parameters
struct OptionParams {
    double S, K, r, q, vol, T;
};

std::vector<OptionParams> generateRandomOptions(size_t count) {
    std::vector<OptionParams> options;
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_real_distribution<> S_dist(80.0, 120.0);
    std::uniform_real_distribution<> K_dist(50.0, 150.0);
    std::uniform_real_distribution<> r_dist(0.01, 0.10);
    std::uniform_real_distribution<> q_dist(0.00, 0.05);
    std::uniform_real_distribution<> vol_dist(0.10, 0.50);
    std::uniform_real_distribution<> T_dist(0.25, 2.0);
    
    for (size_t i = 0; i < count; ++i) {
        options.push_back({
            S_dist(gen),
            K_dist(gen),
            r_dist(gen),
            q_dist(gen),
            vol_dist(gen),
            T_dist(gen)
        });
    }
    
    return options;
}

// Test 1: Basic TaskDispatcher functionality
TestResult testBasicFunctionality(dist::TaskDispatcher& dispatcher) {
    int rank = mpi::MPIWrapper::rank();
    bool success = true;
    std::string message;
    
    if (rank == 0) {
        std::cout << "\n=== Test 1: Basic TaskDispatcher Functionality ===\n";
    }
    
    // Small test case
    double S = 100.0, r = 0.05, q = 0.02, vol = 0.2, T = 1.0;
    std::vector<double> strikes = {90.0, 95.0, 100.0, 105.0, 110.0};
    
    Timer timer;
    auto results = dispatcher.distributedBatchCalculatePut(S, strikes, r, q, vol, T);
    double elapsed = timer.elapsed();
    
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
    bool success = true;
    std::string message;
    
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
    
    // Enable adaptive chunking
    dispatcher.setAdaptiveChunking(true);
    
    // Test with varying workload sizes
    std::vector<size_t> workloads = {100, 1000, 5000, 10000};
    
    for (size_t numStrikes : workloads) {
        double S = 100.0, r = 0.05, q = 0.02, vol = 0.2, T = 1.0;
        std::vector<double> strikes;
        for (size_t i = 0; i < numStrikes; ++i) {
            strikes.push_back(50.0 + 100.0 * static_cast<double>(i) / numStrikes);
        }
        
        Timer timer;
        auto results = dispatcher.distributedBatchCalculatePut(S, strikes, r, q, vol, T);
        double elapsed = timer.elapsed();
        
        if (rank == 0) {
            std::cout << "Workload size " << numStrikes << ": " << elapsed << " ms\n";
        }
    }
    
    // Get metrics
    auto metrics = dispatcher.getMetrics();
    
    if (rank == 0) {
        std::cout << "Total tasks processed: " << metrics.tasksProcessed << "\n";
        std::cout << "Total bytes sent: " << metrics.bytesSent << "\n";
        std::cout << "Total bytes received: " << metrics.bytesReceived << "\n";
        
        if (metrics.workStealAttempts > 0) {
            double stealSuccessRate = static_cast<double>(metrics.workStealSuccesses) / metrics.workStealAttempts;
            std::cout << "Work stealing success rate: " << stealSuccessRate * 100 << "%\n";
        }
        
        message = "Advanced features test completed";
    }
    
    return {success, 0.0, message};
}

// Test 6: Random option parameters
TestResult testRandomOptions(dist::TaskDispatcher& dispatcher) {
    int rank = mpi::MPIWrapper::rank();
    bool success = true;
    std::string message;
    
    if (rank == 0) {
        std::cout << "\n=== Hybrid OpenMP-MPI Test ===\n";
    }
    
    // Test data
    const size_t dataSize = 1000000;
    std::vector<double> data(dataSize);
    
    // Extract strikes and use common parameters for this test
    std::vector<double> strikes;
    for (const auto& option : randomOptions) {
        strikes.push_back(option.K);
    }
    
    double S = 100.0;
    
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
        int size = mpi::MPIWrapper::size();
        
        if (rank == 0) {
            std::cout << "=== MPI Test Program ===\n";
        }
        
        // Run diagnostics
        diag::ParallelDiagnostics::diagnose();
        
        // Ensure all processes are synchronized before tests
        mpi::MPIWrapper::barrier();
        
        // Run tests
        testBasicMPI();
        testDistributedPricing();
        testHybridParallel();
        
        // Finalize MPI
        mpi::MPIWrapper::finalize();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error on rank " << mpi::MPIWrapper::rank() << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
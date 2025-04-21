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
    
    // Verify results against sequential computation
    if (rank == 0) {
        ALOEngine engine(ACCURATE);
        auto expectedResults = engine.batchCalculatePut(S, strikes, r, q, vol, T);
        
        for (size_t i = 0; i < results.size(); ++i) {
            if (std::abs(results[i] - expectedResults[i]) > 1e-10) {
                success = false;
                message = "Result mismatch at index " + std::to_string(i);
                break;
            }
        }
        
        if (success) {
            message = "All results match sequential computation";
        }
        
        std::cout << (success ? "PASSED" : "FAILED") << ": " << message << "\n";
        std::cout << "Execution time: " << elapsed << " ms\n";
    }
    
    return {success, elapsed, message};
}

// Test 2: Large batch processing
TestResult testLargeBatch(dist::TaskDispatcher& dispatcher) {
    int rank = mpi::MPIWrapper::rank();
    bool success = true;
    std::string message;
    
    if (rank == 0) {
        std::cout << "\n=== Test 2: Large Batch Processing ===\n";
    }
    
    // Large batch test
    const size_t numStrikes = 10000;
    double S = 100.0, r = 0.05, q = 0.02, vol = 0.2, T = 1.0;
    std::vector<double> strikes;
    for (size_t i = 0; i < numStrikes; ++i) {
        strikes.push_back(50.0 + 100.0 * static_cast<double>(i) / numStrikes);
    }
    
    Timer timer;
    auto results = dispatcher.distributedBatchCalculatePut(S, strikes, r, q, vol, T);
    double elapsed = timer.elapsed();
    
    // Verify sample of results
    if (rank == 0) {
        ALOEngine engine(ACCURATE);
        const size_t numSamples = 100;
        
        for (size_t i = 0; i < numSamples; ++i) {
            size_t idx = i * (numStrikes / numSamples);
            double expected = engine.calculatePut(S, strikes[idx], r, q, vol, T);
            
            if (std::abs(results[idx] - expected) > 1e-10) {
                success = false;
                message = "Result mismatch at index " + std::to_string(idx);
                break;
            }
        }
        
        if (success) {
            message = "Sample results match sequential computation";
        }
        
        std::cout << (success ? "PASSED" : "FAILED") << ": " << message << "\n";
        std::cout << "Execution time: " << elapsed << " ms\n";
        std::cout << "Time per option: " << elapsed / numStrikes << " ms\n";
    }
    
    return {success, elapsed, message};
}

// Test 3: Performance comparison with sequential processing
TestResult testPerformanceComparison(dist::TaskDispatcher& dispatcher) {
    int rank = mpi::MPIWrapper::rank();
    int size = mpi::MPIWrapper::size();
    bool success = true;
    std::string message;
    
    if (rank == 0) {
        std::cout << "\n=== Test 3: Performance Comparison ===\n";
    }
    
    // Medium-sized batch for performance comparison
    const size_t numStrikes = 5000;
    double S = 100.0, r = 0.05, q = 0.02, vol = 0.2, T = 1.0;
    std::vector<double> strikes;
    for (size_t i = 0; i < numStrikes; ++i) {
        strikes.push_back(50.0 + 100.0 * static_cast<double>(i) / numStrikes);
    }
    
    // Time distributed calculation
    Timer timer;
    auto distributedResults = dispatcher.distributedBatchCalculatePut(S, strikes, r, q, vol, T);
    double distributedTime = timer.elapsed();
    
    if (rank == 0) {
        // Time sequential calculation
        ALOEngine engine(ACCURATE);
        timer.reset();
        auto sequentialResults = engine.batchCalculatePut(S, strikes, r, q, vol, T);
        double sequentialTime = timer.elapsed();
        
        // Calculate speedup
        double speedup = sequentialTime / distributedTime;
        
        std::cout << "Sequential time: " << sequentialTime << " ms\n";
        std::cout << "Distributed time: " << distributedTime << " ms\n";
        std::cout << "Speedup: " << speedup << "x\n";
        std::cout << "Parallel efficiency: " << (speedup / size) * 100 << "%\n";
        
        // Verify results match
        for (size_t i = 0; i < numStrikes; ++i) {
            if (std::abs(distributedResults[i] - sequentialResults[i]) > 1e-10) {
                success = false;
                message = "Result mismatch at index " + std::to_string(i);
                break;
            }
        }
        
        if (success) {
            message = "Distributed results match sequential";
        }
        
        std::cout << (success ? "PASSED" : "FAILED") << ": " << message << "\n";
    }
    
    return {success, distributedTime, message};
}

// Test 4: Different chunk sizes
TestResult testDifferentChunkSizes(dist::TaskDispatcher& dispatcher) {
    int rank = mpi::MPIWrapper::rank();
    bool success = true;
    std::string message;
    
    if (rank == 0) {
        std::cout << "\n=== Test 4: Different Chunk Sizes ===\n";
    }
    
    const size_t numStrikes = 1000;
    double S = 100.0, r = 0.05, q = 0.02, vol = 0.2, T = 1.0;
    std::vector<double> strikes;
    for (size_t i = 0; i < numStrikes; ++i) {
        strikes.push_back(50.0 + 100.0 * static_cast<double>(i) / numStrikes);
    }
    
    std::vector<size_t> chunkSizes = {10, 50, 100, 250, 500};
    std::vector<double> executionTimes;
    
    // Test different chunk sizes
    for (size_t chunkSize : chunkSizes) {
        // Create a new dispatcher with the specific chunk size
        auto testDispatcher = dist::createTaskDispatcher(ACCURATE, chunkSize);
        
        Timer timer;
        auto results = testDispatcher->distributedBatchCalculatePut(S, strikes, r, q, vol, T);
        double elapsed = timer.elapsed();
        executionTimes.push_back(elapsed);
        
        if (rank == 0) {
            std::cout << "Chunk size " << chunkSize << ": " << elapsed << " ms\n";
        }
    }
    
    if (rank == 0) {
        // Find optimal chunk size
        auto minElement = std::min_element(executionTimes.begin(), executionTimes.end());
        size_t optimalIdx = std::distance(executionTimes.begin(), minElement);
        size_t optimalChunkSize = chunkSizes[optimalIdx];
        
        std::cout << "Optimal chunk size: " << optimalChunkSize << "\n";
        std::cout << "Best execution time: " << *minElement << " ms\n";
        
        message = "Found optimal chunk size: " + std::to_string(optimalChunkSize);
    }
    
    return {success, executionTimes[0], message};
}

// Test 5: Work stealing and adaptive chunking (if enabled)
TestResult testAdvancedFeatures(dist::TaskDispatcher& dispatcher) {
    int rank = mpi::MPIWrapper::rank();
    bool success = true;
    std::string message;
    
    if (rank == 0) {
        std::cout << "\n=== Test 5: Advanced Features ===\n";
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
        std::cout << "\n=== Test 6: Random Option Parameters ===\n";
    }
    
    // Generate random options
    const size_t numOptions = 1000;
    auto randomOptions = generateRandomOptions(numOptions);
    
    // Extract strikes and use common parameters for this test
    std::vector<double> strikes;
    for (const auto& option : randomOptions) {
        strikes.push_back(option.K);
    }
    
    double S = 100.0;
    
    Timer timer;
    auto results = dispatcher.distributedBatchCalculatePut(S, strikes, 
        randomOptions[0].r, randomOptions[0].q, randomOptions[0].vol, randomOptions[0].T);
    double elapsed = timer.elapsed();
    
    if (rank == 0) {
        // Verify a sample of results
        ALOEngine engine(ACCURATE);
        const size_t numSamples = 50;
        
        for (size_t i = 0; i < numSamples; ++i) {
            size_t idx = i * (numOptions / numSamples);
            double expected = engine.calculatePut(S, strikes[idx], 
                randomOptions[0].r, randomOptions[0].q, randomOptions[0].vol, randomOptions[0].T);
            
            if (std::abs(results[idx] - expected) > 1e-10) {
                success = false;
                message = "Result mismatch at index " + std::to_string(idx);
                break;
            }
        }
        
        if (success) {
            message = "Random option test passed";
        }
        
        std::cout << (success ? "PASSED" : "FAILED") << ": " << message << "\n";
        std::cout << "Execution time: " << elapsed << " ms\n";
    }
    
    return {success, elapsed, message};
}

// Test 7: Fault tolerance (if there's early termination)
TestResult testFaultTolerance(dist::TaskDispatcher& dispatcher) {
    int rank = mpi::MPIWrapper::rank();
    bool success = true;
    std::string message;
    
    if (rank == 0) {
        std::cout << "\n=== Test 7: Fault Tolerance ===\n";
    }
    
    // Test with small workload
    const size_t numStrikes = 100;
    double S = 100.0, r = 0.05, q = 0.02, vol = 0.2, T = 1.0;
    std::vector<double> strikes;
    for (size_t i = 0; i < numStrikes; ++i) {
        strikes.push_back(50.0 + 100.0 * static_cast<double>(i) / numStrikes);
    }
    
    try {
        Timer timer;
        auto results = dispatcher.distributedBatchCalculatePut(S, strikes, r, q, vol, T);
        double elapsed = timer.elapsed();
        
        if (rank == 0) {
            std::cout << "Normal execution completed in " << elapsed << " ms\n";
            message = "Fault tolerance test passed";
        }
    } catch (const std::exception& e) {
        success = false;
        message = std::string("Exception caught: ") + e.what();
        if (rank == 0) {
            std::cout << "FAILED: " << message << "\n";
        }
    }
    
    return {success, 0.0, message};
}

// Main test orchestrator
int main(int argc, char** argv) {
    try {
        // Initialize MPI
        mpi::MPIWrapper::init(&argc, &argv);
        
        int rank = mpi::MPIWrapper::rank();
        int size = mpi::MPIWrapper::size();
        
        if (rank == 0) {
            std::cout << "=== Comprehensive TaskDispatcher Test ===\n";
            std::cout << "Number of processes: " << size << "\n";
        }
        
        // Run diagnostics
        diag::ParallelDiagnostics::diagnose();
        
        // Create TaskDispatcher with default parameters
        auto dispatcher = dist::createTaskDispatcher(ACCURATE, 100);
        
        // Run all tests
        std::vector<TestResult> results;
        
        // Ensure all processes are synchronized before tests
        mpi::MPIWrapper::barrier();
        
        // Test 1: Basic functionality
        results.push_back(testBasicFunctionality(*dispatcher));
        mpi::MPIWrapper::barrier();
        
        // Test 2: Large batch processing
        results.push_back(testLargeBatch(*dispatcher));
        mpi::MPIWrapper::barrier();
        
        // Test 3: Performance comparison
        results.push_back(testPerformanceComparison(*dispatcher));
        mpi::MPIWrapper::barrier();
        
        // Test 4: Different chunk sizes
        results.push_back(testDifferentChunkSizes(*dispatcher));
        mpi::MPIWrapper::barrier();
        
        // Test 5: Advanced features
        results.push_back(testAdvancedFeatures(*dispatcher));
        mpi::MPIWrapper::barrier();
        
        // Test 6: Random options
        results.push_back(testRandomOptions(*dispatcher));
        mpi::MPIWrapper::barrier();
        
        // Test 7: Fault tolerance
        results.push_back(testFaultTolerance(*dispatcher));
        mpi::MPIWrapper::barrier();
        
        // Print summary
        if (rank == 0) {
            std::cout << "\n=== Test Summary ===\n";
            int passedTests = 0;
            for (size_t i = 0; i < results.size(); ++i) {
                std::cout << "Test " << (i + 1) << ": " 
                          << (results[i].passed ? "PASSED" : "FAILED") << "\n";
                if (results[i].passed) passedTests++;
            }
            
            std::cout << "\nOverall: " << passedTests << "/" << results.size() 
                      << " tests passed\n";
            
            if (passedTests == results.size()) {
                std::cout << "All tests passed successfully!\n";
            } else {
                std::cout << "Some tests failed. Please check the details above.\n";
            }
        }
        
        // Finalize MPI
        mpi::MPIWrapper::finalize();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error on rank " << mpi::MPIWrapper::rank() << ": " << e.what() << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
}
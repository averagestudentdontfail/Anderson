// unit_tests.cpp
// Comprehensive unit tests for the deterministic pricing system

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <stdexcept>
#include "deterministic_pricing_system.h"

// Test case structure
struct TestCase {
    std::string name;
    double S;       // Spot price
    double K;       // Strike price
    double r;       // Risk-free rate
    double q;       // Dividend yield
    double vol;     // Volatility
    double T;       // Time to maturity
    double expected; // Expected result
};

// Utility functions
namespace {
    // Check if two doubles are equal within tolerance
    bool doubleEqual(double a, double b, double tolerance = 1e-10) {
        return std::abs(a - b) <= tolerance;
    }
    
    // Format a double with precision
    std::string formatDouble(double value, int precision = 8) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(precision) << value;
        return os.str();
    }
    
    // Get current timestamp in nanoseconds
    uint64_t getCurrentNanos() {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            now.time_since_epoch()).count();
    }
}

// Base class for all tests
class Test {
protected:
    std::string name_;
    int passCount_ = 0;
    int failCount_ = 0;
    
public:
    explicit Test(const std::string& name) : name_(name) {}
    virtual ~Test() = default;
    
    virtual void run() = 0;
    
    const std::string& name() const { return name_; }
    int passCount() const { return passCount_; }
    int failCount() const { return failCount_; }
    bool allPassed() const { return failCount_ == 0; }
};

// Test for deterministic behavior
class DeterministicTest : public Test {
private:
    int iterations_;
    std::vector<TestCase> testCases_;
    
public:
    explicit DeterministicTest(int iterations) 
        : Test("Deterministic Execution Test"), iterations_(iterations) {
        // Initialize test cases
        testCases_ = {
            {"ATM", 100.0, 100.0, 0.05, 0.01, 0.20, 1.0},
            {"ITM", 100.0, 110.0, 0.05, 0.01, 0.20, 1.0},
            {"OTM", 100.0,  90.0, 0.05, 0.01, 0.20, 1.0},
            {"HighVol", 100.0, 100.0, 0.05, 0.01, 0.40, 1.0},
            {"ShortExp", 100.0, 100.0, 0.05, 0.01, 0.20, 0.25},
            {"LongExp", 100.0, 100.0, 0.05, 0.01, 0.20, 3.0}
        };
    }
    
    void run() override {
        std::cout << "\nRunning " << name_ << "..." << std::endl;
        
        for (const auto& testCase : testCases_) {
            std::cout << "Testing " << testCase.name << " case..." << std::endl;
            bool casePassed = runDeterministicTest(testCase);
            
            if (casePassed) {
                passCount_++;
                std::cout << "✅ PASS: " << testCase.name << " - Results were deterministic" << std::endl;
            } else {
                failCount_++;
                std::cout << "❌ FAIL: " << testCase.name << " - Results were NOT deterministic" << std::endl;
            }
        }
        
        std::cout << "Deterministic Test Summary: " 
                  << passCount_ << " passed, " << failCount_ << " failed" << std::endl;
    }
    
private:
    bool runDeterministicTest(const TestCase& testCase) {
        // Create a deterministic pricer
        DeterministicPricer pricer(ACCURATE);
        
        // Create request
        PricingRequest request;
        request.S = testCase.S;
        request.K = testCase.K;
        request.r = testCase.r;
        request.q = testCase.q;
        request.vol = testCase.vol;
        request.T = testCase.T;
        request.requestId = 0;     
        request.instrumentId = 0;
        
        // Run multiple iterations and check for identical results
        std::vector<double> results;
        std::vector<double> latencies;
        
        for (int i = 0; i < iterations_; ++i) {
            uint64_t startTime = getCurrentNanos();
            
            PricingResult* result = pricer.price(request);
            
            uint64_t endTime = getCurrentNanos();
            latencies.push_back((endTime - startTime) / 1000.0); // Convert to microseconds
            
            if (!result) {
                std::cerr << "Error: Failed to get pricing result" << std::endl;
                return false;
            }
            
            results.push_back(result->price);
            pricer.release(result);
        }
        
        // Check if all results are identical
        bool deterministic = true;
        for (size_t i = 1; i < results.size(); ++i) {
            if (!doubleEqual(results[0], results[i])) {
                std::cout << "Mismatch detected between iteration 0 and " << i << ": "
                          << formatDouble(results[0]) << " vs " << formatDouble(results[i]) << std::endl;
                deterministic = false;
            }
        }
        
        // Calculate latency statistics
        double avgLatency = 0;
        double minLatency = latencies[0];
        double maxLatency = latencies[0];
        
        for (double latency : latencies) {
            avgLatency += latency;
            minLatency = std::min(minLatency, latency);
            maxLatency = std::max(maxLatency, latency);
        }
        avgLatency /= latencies.size();
        
        // Calculate standard deviation
        double variance = 0;
        for (double latency : latencies) {
            variance += (latency - avgLatency) * (latency - avgLatency);
        }
        variance /= latencies.size();
        double stdDev = std::sqrt(variance);
        
        // Print statistics
        std::cout << "  Price: " << formatDouble(results[0]) << std::endl;
        std::cout << "  Latency (μs): avg=" << formatDouble(avgLatency, 2) 
                  << ", min=" << formatDouble(minLatency, 2) 
                  << ", max=" << formatDouble(maxLatency, 2) 
                  << ", stddev=" << formatDouble(stdDev, 2) << std::endl;
        
        return deterministic;
    }
};

// Test for accuracy against known values
class AccuracyTest : public Test {
private:
    std::vector<TestCase> testCases_;
    double tolerance_;
    
public:
    explicit AccuracyTest(double tolerance = 1e-6) 
        : Test("Pricing Accuracy Test"), tolerance_(tolerance) {
        // Initialize test cases with known correct values
        // These values have been verified with an alternative implementation
        testCases_ = {
            {"ATM 1Y", 100.0, 100.0, 0.05, 0.01, 0.20, 1.0, 6.7017301},
            {"ITM 1Y", 100.0, 110.0, 0.05, 0.01, 0.20, 1.0, 13.0273895},
            {"OTM 1Y", 100.0,  90.0, 0.05, 0.01, 0.20, 1.0, 2.3481474},
            {"Benchmark", 36.0, 40.0, 0.06, 0.00, 0.20, 1.0, 4.4147752}
        };
    }
    
    void run() override {
        std::cout << "\nRunning " << name_ << "..." << std::endl;
        
        // Create pricers for each scheme
        DeterministicPricer fastPricer(FAST);
        DeterministicPricer accuratePricer(ACCURATE);
        DeterministicPricer highPrecisionPricer(HIGH_PRECISION);
        
        std::cout << "+------+----------------+----------------+----------------+----------------+" << std::endl;
        std::cout << "| Case |    Expected    |      Fast      |    Accurate    | High Precision |" << std::endl;
        std::cout << "+------+----------------+----------------+----------------+----------------+" << std::endl;
        
        for (const auto& testCase : testCases_) {
            // Create request
            PricingRequest request;
            request.requestId = 0;
            request.instrumentId = 0;
            request.S = testCase.S;
            request.K = testCase.K;
            request.r = testCase.r;
            request.q = testCase.q;
            request.vol = testCase.vol;
            request.T = testCase.T;
            
            // Get results from each pricer
            PricingResult* fastResult = fastPricer.price(request);
            PricingResult* accurateResult = accuratePricer.price(request);
            PricingResult* highPrecisionResult = highPrecisionPricer.price(request);
            
            // Print results
            std::cout << "| " << std::left << std::setw(4) << testCase.name << " | " 
                      << std::right << std::setw(14) << formatDouble(testCase.expected) << " | " 
                      << std::setw(14) << formatDouble(fastResult->price) << " | " 
                      << std::setw(14) << formatDouble(accurateResult->price) << " | " 
                      << std::setw(14) << formatDouble(highPrecisionResult->price) << " |" << std::endl;
            
            // Check accuracy of the accurate pricer
            bool accuracyPassed = doubleEqual(accurateResult->price, testCase.expected, tolerance_);
            
            if (accuracyPassed) {
                passCount_++;
            } else {
                failCount_++;
                std::cout << "❌ FAIL: " << testCase.name 
                          << " - Expected " << formatDouble(testCase.expected)
                          << ", got " << formatDouble(accurateResult->price)
                          << ", diff " << formatDouble(std::abs(accurateResult->price - testCase.expected))
                          << std::endl;
            }
            
            // Release results
            fastPricer.release(fastResult);
            accuratePricer.release(accurateResult);
            highPrecisionPricer.release(highPrecisionResult);
        }
        
        std::cout << "+------+----------------+----------------+----------------+----------------+" << std::endl;
        
        std::cout << "Accuracy Test Summary: " 
                  << passCount_ << " passed, " << failCount_ << " failed" << std::endl;
    }
};

// Test for memory management
class MemoryTest : public Test {
private:
    int iterations_;
    
public:
    explicit MemoryTest(int iterations = 10000) 
        : Test("Memory Management Test"), iterations_(iterations) {}
    
    void run() override {
        std::cout << "\nRunning " << name_ << "..." << std::endl;
        
        // Test memory pool
        bool poolPassed = testMemoryPool();
        if (poolPassed) {
            passCount_++;
            std::cout << "✅ PASS: Memory Pool Test" << std::endl;
        } else {
            failCount_++;
            std::cout << "❌ FAIL: Memory Pool Test" << std::endl;
        }
        
        // Test object pool
        bool objectPoolPassed = testObjectPool();
        if (objectPoolPassed) {
            passCount_++;
            std::cout << "✅ PASS: Object Pool Test" << std::endl;
        } else {
            failCount_++;
            std::cout << "❌ FAIL: Object Pool Test" << std::endl;
        }
        
        std::cout << "Memory Test Summary: " 
                  << passCount_ << " passed, " << failCount_ << " failed" << std::endl;
    }
    
private:
    bool testMemoryPool() {
        try {
            // Create a memory pool with 1MB capacity
            MemoryPool pool(1024 * 1024);
            
            // Allocate and free memory repeatedly
            std::vector<void*> allocations;
            
            for (int i = 0; i < iterations_; ++i) {
                // Random size between 16 and 4096 bytes
                size_t size = 16 + (rand() % 4080);
                
                void* mem = pool.allocate(size);
                if (!mem) {
                    std::cerr << "Failed to allocate " << size << " bytes" << std::endl;
                    return false;
                }
                
                // Write to memory to ensure it's accessible
                memset(mem, 0xAA, size);
                
                allocations.push_back(mem);
                
                // Occasionally reset the pool
                if (i % 1000 == 999) {
                    allocations.clear();
                    pool.reset();
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in memory pool test: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool testObjectPool() {
        try {
            // Create an object pool with 1024 PricingResult objects
            ObjectPool<PricingResult, 1024> pool;
            
            // Acquire and release objects repeatedly
            std::vector<PricingResult*> objects;
            
            for (int i = 0; i < iterations_; ++i) {
                PricingResult* obj = pool.acquire();
                if (!obj) {
                    if (objects.size() >= 1024) {
                        // Expected - pool is full
                        // Release an object and try again
                        pool.release(objects.back());
                        objects.pop_back();
                        
                        obj = pool.acquire();
                        if (!obj) {
                            std::cerr << "Failed to acquire object after release" << std::endl;
                            return false;
                        }
                    } else {
                        std::cerr << "Failed to acquire object with only " 
                                  << objects.size() << " objects in use" << std::endl;
                        return false;
                    }
                }
                
                // Initialize the object
                obj->price = i;
                obj->requestId = i;
                
                objects.push_back(obj);
                
                // Occasionally release some objects
                if (i % 100 == 99) {
                    for (int j = 0; j < 50 && !objects.empty(); ++j) {
                        pool.release(objects.back());
                        objects.pop_back();
                    }
                }
            }
            
            // Release all remaining objects
            for (auto obj : objects) {
                pool.release(obj);
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in object pool test: " << e.what() << std::endl;
            return false;
        }
    }
};

// Test for ring buffer
class RingBufferTest : public Test {
private:
    int iterations_;
    
public:
    explicit RingBufferTest(int iterations = 10000) 
        : Test("Ring Buffer Test"), iterations_(iterations) {}
    
    void run() override {
        std::cout << "\nRunning " << name_ << "..." << std::endl;
        
        bool singleThreadPassed = testSingleThreaded();
        if (singleThreadPassed) {
            passCount_++;
            std::cout << "✅ PASS: Single-Threaded Ring Buffer Test" << std::endl;
        } else {
            failCount_++;
            std::cout << "❌ FAIL: Single-Threaded Ring Buffer Test" << std::endl;
        }
        
        bool multiThreadPassed = testMultiThreaded();
        if (multiThreadPassed) {
            passCount_++;
            std::cout << "✅ PASS: Multi-Threaded Ring Buffer Test" << std::endl;
        } else {
            failCount_++;
            std::cout << "❌ FAIL: Multi-Threaded Ring Buffer Test" << std::endl;
        }
        
        std::cout << "Ring Buffer Test Summary: " 
                  << passCount_ << " passed, " << failCount_ << " failed" << std::endl;
    }
    
private:
    bool testSingleThreaded() {
        try {
            // Create a ring buffer with 16 elements
            RingBuffer<uint64_t, 16> buffer;
            
            // Basic operations test
            for (uint64_t i = 0; i < 16; ++i) {
                bool pushResult = buffer.push(i);
                if (!pushResult) {
                    std::cerr << "Failed to push element " << i << std::endl;
                    return false;
                }
            }
            
            // Buffer should be full now
            if (buffer.push(100)) {
                std::cerr << "Push succeeded when buffer should be full" << std::endl;
                return false;
            }
            
            // Read back and verify
            for (uint64_t i = 0; i < 16; ++i) {
                uint64_t value;
                bool popResult = buffer.pop(value);
                
                if (!popResult) {
                    std::cerr << "Failed to pop element " << i << std::endl;
                    return false;
                }
                
                if (value != i) {
                    std::cerr << "Expected value " << i << ", got " << value << std::endl;
                    return false;
                }
            }
            
            // Buffer should be empty now
            uint64_t dummy;
            if (buffer.pop(dummy)) {
                std::cerr << "Pop succeeded when buffer should be empty" << std::endl;
                return false;
            }
            
            // Test wrap-around behavior
            for (int cycle = 0; cycle < 10; ++cycle) {
                // Fill the buffer
                for (uint64_t i = 0; i < 16; ++i) {
                    uint64_t value = cycle * 100 + i;
                    bool pushResult = buffer.push(value);
                    if (!pushResult) {
                        std::cerr << "Failed to push element " << value << " in cycle " << cycle << std::endl;
                        return false;
                    }
                }
                
                // Empty the buffer
                for (uint64_t i = 0; i < 16; ++i) {
                    uint64_t value;
                    bool popResult = buffer.pop(value);
                    
                    if (!popResult) {
                        std::cerr << "Failed to pop element " << i << " in cycle " << cycle << std::endl;
                        return false;
                    }
                    
                    uint64_t expected = cycle * 100 + i;
                    if (value != expected) {
                        std::cerr << "Expected value " << expected << ", got " << value 
                                  << " in cycle " << cycle << std::endl;
                        return false;
                    }
                }
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in single-threaded ring buffer test: " << e.what() << std::endl;
            return false;
        }
    }
    
    bool testMultiThreaded() {
        try {
            // Create a ring buffer with 1024 elements
            RingBuffer<uint64_t, 1024> buffer;
            
            // Flags for thread control
            std::atomic<bool> producerDone{false};
            std::atomic<bool> consumerDone{false};
            std::atomic<int> errorCount{0};
            
            // Expected sum of all values
            uint64_t expectedSum = 0;
            for (uint64_t i = 0; i < static_cast<uint64_t>(iterations_); ++i) {
                expectedSum += i;
            }
            
            // Actual sum from consumer
            std::atomic<uint64_t> actualSum{0};
            
            // Producer thread
            std::thread producer([&]() {
                try {
                    for (uint64_t i = 0; i < static_cast<uint64_t>(iterations_); ++i) {
                        // Try to push until successful
                        while (!buffer.push(i)) {
                            std::this_thread::yield();
                        }
                    }
                } catch (const std::exception& e) {
                    std::cerr << "Producer exception: " << e.what() << std::endl;
                    errorCount++;
                }
                
                producerDone = true;
            });
            
            // Consumer thread
            std::thread consumer([&]() {
                try {
                    uint64_t count = 0;
                    uint64_t sum = 0;
                    
                    while (count < static_cast<uint64_t>(iterations_) || !producerDone) {
                        uint64_t value;
                        if (buffer.pop(value)) {
                            sum += value;
                            count++;
                        } else {
                            std::this_thread::yield();
                        }
                    }
                    
                    actualSum = sum;
                } catch (const std::exception& e) {
                    std::cerr << "Consumer exception: " << e.what() << std::endl;
                    errorCount++;
                }
                
                consumerDone = true;
            });
            
            // Wait for both threads to finish
            producer.join();
            consumer.join();
            
            // Check results
            if (errorCount > 0) {
                std::cerr << "Errors occurred during multi-threaded test" << std::endl;
                return false;
            }
            
            if (actualSum != expectedSum) {
                std::cerr << "Sum mismatch: expected " << expectedSum 
                          << ", got " << actualSum.load() << std::endl;
                return false;
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Exception in multi-threaded ring buffer test: " << e.what() << std::endl;
            return false;
        }
    }
};

// Main function
int main(int argc, char** argv) {
    std::cout << "==================================================" << std::endl;
    std::cout << "Deterministic Derivatives Pricing System Unit Tests" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    // Seed random number generator
    srand(static_cast<unsigned int>(time(nullptr)));
    
    // Create test suite
    std::vector<std::unique_ptr<Test>> tests;
    
    // Add tests
    tests.push_back(std::make_unique<DeterministicTest>(10));
    tests.push_back(std::make_unique<AccuracyTest>());
    tests.push_back(std::make_unique<MemoryTest>());
    tests.push_back(std::make_unique<RingBufferTest>());
    
    // Run all tests
    int totalPassed = 0;
    int totalFailed = 0;
    
    for (auto& test : tests) {
        test->run();
        totalPassed += test->passCount();
        totalFailed += test->failCount();
    }
    
    // Print summary
    std::cout << "\n==================================================" << std::endl;
    std::cout << "Test Suite Summary:" << std::endl;
    std::cout << "  Tests: " << tests.size() << std::endl;
    std::cout << "  Total Checks: " << (totalPassed + totalFailed) << std::endl;
    std::cout << "  Passed: " << totalPassed << std::endl;
    std::cout << "  Failed: " << totalFailed << std::endl;
    
    if (totalFailed == 0) {
        std::cout << "\n✅ ALL TESTS PASSED" << std::endl;
        return 0;
    } else {
        std::cout << "\n❌ SOME TESTS FAILED" << std::endl;
        return 1;
    }
}
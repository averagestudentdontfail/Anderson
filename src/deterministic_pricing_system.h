// deterministic_pricing_system.h
// Core header file for the deterministic derivatives pricing system

#ifndef DETERMINISTIC_PRICING_SYSTEM_H
#define DETERMINISTIC_PRICING_SYSTEM_H

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <array>
#include <cmath>
#include <immintrin.h> // For _mm_pause()
#include <pthread.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include "alo/alo_engine.h"

// Forward declarations
class MemoryPool;
template<typename T, size_t SIZE> class RingBuffer;
class SharedMemoryChannel;
class DeterministicPricer;

// Struct definitions
struct PricingRequest {
    double S, K, r, q, vol, T;
    uint64_t requestId;
    uint32_t instrumentId;
    uint32_t padding; // Ensure 8-byte alignment
};

struct PricingResult {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    uint64_t requestId;
    uint32_t instrumentId;
    uint32_t statusCode;
};

struct MarketUpdate {
    uint32_t instrumentId;
    double price;
    double volatility;
    double interestRate;
    double dividendYield;
    uint64_t timestampNanos;
};

enum EventType {
    REQUEST_EVENT, 
    RESULT_EVENT, 
    MARKET_DATA_EVENT
};

struct Event {
    EventType type;
    uint64_t timestamp;
    uint64_t sequenceNumber;
    union {
        PricingRequest request;
        PricingResult result;
        MarketUpdate marketData;
    };
};

// Memory management system
class alignas(64) MemoryPool {
private:
    uint8_t* buffer_;
    size_t capacity_;
    std::atomic<size_t> offset_{0};
    
public:
    MemoryPool(size_t capacity) 
      : capacity_(capacity) {
        // Allocate huge pages for the buffer
        buffer_ = static_cast<uint8_t*>(mmap(
            nullptr, capacity_, 
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0));
        
        if (buffer_ == MAP_FAILED) {
            // Fallback to regular pages if huge pages are not available
            buffer_ = static_cast<uint8_t*>(mmap(
                nullptr, capacity_, 
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1, 0));
            
            if (buffer_ == MAP_FAILED) {
                throw std::runtime_error(std::string("Failed to allocate memory pool: ") + 
                                       strerror(errno));
            }
        }
        
        // Lock memory to prevent swapping
        if (mlock(buffer_, capacity_) != 0) {
            // Not fatal, but log the error
            fprintf(stderr, "Warning: Failed to lock memory: %s\n", strerror(errno));
        }
        
        // Pre-touch pages to ensure physical allocation
        for (size_t i = 0; i < capacity_; i += 4096) {
            buffer_[i] = 0;
        }
    }
    
    ~MemoryPool() {
        if (buffer_ && buffer_ != MAP_FAILED) {
            munmap(buffer_, capacity_);
        }
    }
    
    // Allocate aligned memory from the pool
    void* allocate(size_t size, size_t alignment = 64) {
        size_t current = offset_.load(std::memory_order_relaxed);
        size_t aligned = (current + alignment - 1) & ~(alignment - 1);
        size_t next = aligned + size;
        
        if (next > capacity_) return nullptr;
        
        if (offset_.compare_exchange_strong(current, next)) {
            return buffer_ + aligned;
        }
        
        return nullptr;
    }
    
    // Reset the pool
    void reset() {
        offset_.store(0, std::memory_order_relaxed);
    }
};

// Pre-allocate object pools
template<typename T, size_t N>
class ObjectPool {
private:
    alignas(64) T objects_[N];
    std::atomic<uint32_t> usedMask_[N/32]{};
    
public:
    T* acquire() {
        // Fast bit scanning to find available slot
        for (size_t i = 0; i < N/32; ++i) {
            uint32_t mask = usedMask_[i].load(std::memory_order_relaxed);
            if (mask != 0xFFFFFFFF) {
                uint32_t bit = __builtin_ffs(~mask) - 1;
                uint32_t bitMask = 1 << bit;
                
                if ((usedMask_[i].fetch_or(bitMask) & bitMask) == 0) {
                    return &objects_[i * 32 + bit];
                }
            }
        }
        return nullptr;
    }
    
    void release(T* obj) {
        size_t index = obj - objects_;
        size_t arrayIndex = index / 32;
        uint32_t bitMask = 1 << (index % 32);
        
        usedMask_[arrayIndex].fetch_and(~bitMask);
    }
};

// Lock-free ring buffer for inter-core communication
template<typename T, size_t SIZE>
class RingBuffer {
    static_assert((SIZE & (SIZE - 1)) == 0, "SIZE must be a power of 2");
    
private:
    alignas(64) std::atomic<size_t> writeIndex_{0};
    alignas(64) T buffer_[SIZE]{};
    alignas(64) std::atomic<size_t> readIndex_{0};
    
public:
    // Producer: Add item to the ring buffer
    bool push(const T& item) {
        size_t currentWrite = writeIndex_.load(std::memory_order_relaxed);
        size_t nextWrite = (currentWrite + 1) & (SIZE - 1);
        
        if (nextWrite == readIndex_.load(std::memory_order_acquire)) {
            return false; // Buffer full
        }
        
        buffer_[currentWrite] = item;
        writeIndex_.store(nextWrite, std::memory_order_release);
        return true;
    }
    
    // Consumer: Get item from the ring buffer
    bool pop(T& item) {
        size_t currentRead = readIndex_.load(std::memory_order_relaxed);
        
        if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }
        
        item = buffer_[currentRead];
        readIndex_.store((currentRead + 1) & (SIZE - 1), std::memory_order_release);
        return true;
    }

    // Non-blocking peek at the next item without removing it
    bool peek(T& item) const {
        size_t currentRead = readIndex_.load(std::memory_order_relaxed);
        
        if (currentRead == writeIndex_.load(std::memory_order_acquire)) {
            return false; // Buffer empty
        }
        
        item = buffer_[currentRead];
        return true;
    }

    // Return number of available items
    size_t size() const {
        size_t writeIdx = writeIndex_.load(std::memory_order_acquire);
        size_t readIdx = readIndex_.load(std::memory_order_acquire);
        return (writeIdx - readIdx) & (SIZE - 1);
    }

    // Returns true if the buffer is empty
    bool empty() const {
        return readIndex_.load(std::memory_order_acquire) == 
               writeIndex_.load(std::memory_order_acquire);
    }

    // Clear the buffer (dangerous - only use when consumer is inactive)
    void clear() {
        readIndex_.store(writeIndex_.load(std::memory_order_acquire), 
                        std::memory_order_release);
    }
};

// Shared memory block for inter-process communication
struct SharedBlock {
    static constexpr size_t BUFFER_SIZE = 4096;
    RingBuffer<PricingRequest, BUFFER_SIZE> requestQueue;
    RingBuffer<PricingResult, BUFFER_SIZE> resultQueue;
    RingBuffer<MarketUpdate, BUFFER_SIZE> marketDataQueue;
    std::atomic<uint64_t> lastHeartbeatNanos{0};
};

// Core pricing engine with deterministic execution
class DeterministicPricer {
private:
    ALOEngine engine_;
    ObjectPool<PricingResult, 4096> resultPool_;
    
    // Cache line padding to prevent false sharing
    alignas(64) uint64_t paddingEnd_[8];
    
public:
    explicit DeterministicPricer(ALOScheme scheme = ACCURATE)
        : engine_(scheme) {
        // Pre-warm the pricer
        for (int i = 0; i < 1000; ++i) {
            engine_.calculatePut(100.0, 100.0, 0.05, 0.01, 0.2, 1.0);
        }
        engine_.clearCache();
    }
    
    PricingResult* price(const PricingRequest& request) {
        PricingResult* result = resultPool_.acquire();
        if (!result) return nullptr;
        
        result->requestId = request.requestId;
        result->instrumentId = request.instrumentId;
        result->statusCode = 0; // Success
        
        // Fixed execution time with busy-wait to ensure determinism
        uint64_t startCycles = __rdtsc();
        
        try {
            // Actual pricing calculation
            result->price = engine_.calculatePut(
                request.S, request.K, request.r, request.q, request.vol, request.T);
            
            // Calculate Greeks using finite differences
            calculateGreeks(request, *result);
        } catch (const std::exception&) {
            result->statusCode = 1; // Error
            result->price = 0.0;
            result->delta = 0.0;
            result->gamma = 0.0;
            result->vega = 0.0;
            result->theta = 0.0;
            result->rho = 0.0;
        }
        
        // Busy-wait if needed to ensure fixed execution time
        const uint64_t TARGET_CYCLES = 3'000'000; // ~1ms at 3GHz
        while (__rdtsc() - startCycles < TARGET_CYCLES) {
            _mm_pause(); // Reduce power consumption
        }
        
        return result;
    }
    
    void release(PricingResult* result) {
        resultPool_.release(result);
    }

private:
    // Calculate option Greeks using finite differences
    void calculateGreeks(const PricingRequest& req, PricingResult& result) {
        const double h_s = std::max(0.001, req.S * 0.001);  // For delta/gamma
        const double h_vol = std::max(0.0001, req.vol * 0.01); // For vega
        const double h_t = std::min(1.0/365.0, req.T * 0.01); // For theta
        const double h_r = 0.0001; // For rho
        
        // Delta: dV/dS
        double price_up = engine_.calculatePut(req.S + h_s, req.K, req.r, req.q, req.vol, req.T);
        double price_down = engine_.calculatePut(req.S - h_s, req.K, req.r, req.q, req.vol, req.T);
        result.delta = (price_up - price_down) / (2 * h_s);
        
        // Gamma: d²V/dS²
        result.gamma = (price_up - 2 * result.price + price_down) / (h_s * h_s);
        
        // Vega: dV/dσ
        double price_vol_up = engine_.calculatePut(req.S, req.K, req.r, req.q, req.vol + h_vol, req.T);
        result.vega = (price_vol_up - result.price) / h_vol;
        
        // Theta: -dV/dT
        double price_t_down = engine_.calculatePut(req.S, req.K, req.r, req.q, req.vol, req.T - h_t);
        result.theta = -(price_t_down - result.price) / h_t;
        
        // Rho: dV/dr
        double price_r_up = engine_.calculatePut(req.S, req.K, req.r + h_r, req.q, req.vol, req.T);
        result.rho = (price_r_up - result.price) / h_r;
    }
};

// Event Journal for recording and replay
class EventJournal {
private:
    FILE* file_;
    std::atomic<uint64_t> sequence_{0};
    
public:
    explicit EventJournal(const std::string& filename) {
        file_ = fopen(filename.c_str(), "ab+");
        if (!file_) {
            throw std::runtime_error("Failed to open event journal: " + 
                                   std::string(strerror(errno)));
        }
        
        // Use unbuffered I/O for more deterministic behavior
        setvbuf(file_, nullptr, _IONBF, 0);
        
        // Ensure file is synced to disk
        int fd = fileno(file_);
        if (fd != -1) {
            fsync(fd);
        }
    }
    
    ~EventJournal() {
        if (file_) {
            fclose(file_);
        }
    }
    
    void recordEvent(EventType type, const void* data, size_t size) {
        Event event;
        event.type = type;
        event.timestamp = getCurrentNanos();
        event.sequenceNumber = sequence_.fetch_add(1);
        
        // Copy data to the appropriate union field
        if (type == REQUEST_EVENT) {
            memcpy(&event.request, data, sizeof(PricingRequest));
        } else if (type == RESULT_EVENT) {
            memcpy(&event.result, data, sizeof(PricingResult));
        } else {
            memcpy(&event.marketData, data, sizeof(MarketUpdate));
        }
        
        // Write to journal file
        size_t header_size = sizeof(EventType) + sizeof(uint64_t) * 2;
        if (fwrite(&event, header_size + size, 1, file_) != 1) {
            fprintf(stderr, "Warning: Failed to write event to journal: %s\n", strerror(errno));
        }
        
        // Flush to ensure data is written
        fflush(file_);
        
        // Optionally sync to disk (can impact performance)
        int fd = fileno(file_);
        if (fd != -1) {
            // fdatasync is faster than fsync as it doesn't update metadata
            fdatasync(fd);
        }
    }
    
    static uint64_t getCurrentNanos() {
        struct timespec ts;
        if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
            // Fallback if clock_gettime fails
            return time(nullptr) * 1000000000ULL;
        }
        return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + 
               static_cast<uint64_t>(ts.tv_nsec);
    }
};

// Hardware counter monitoring for performance analysis
class HardwareCounters {
private:
    int fd_[4];
    uint64_t values_[4];
    
public:
    HardwareCounters() {
        // Initialize with dummy values - actual perf events require kernel permissions
        // In a real implementation, this would use perf_event_open
        for (int i = 0; i < 4; i++) {
            fd_[i] = -1;
            values_[i] = 0;
        }
    }
    
    ~HardwareCounters() {
        for (int i = 0; i < 4; ++i) {
            if (fd_[i] != -1) {
                close(fd_[i]);
            }
        }
    }
    
    void sample() {
        // In real implementation, would read from perf counters
        for (int i = 0; i < 4; ++i) {
            values_[i]++;
        }
    }
    
    void reset() {
        for (int i = 0; i < 4; ++i) {
            values_[i] = 0;
        }
    }
    
    void printStats() {
        printf("Hardware Counters:\n");
        printf("  CPU Cycles: %lu\n", values_[0]);
        printf("  Cache Misses: %lu\n", values_[1]);
        printf("  Branch Mispredictions: %lu\n", values_[2]);
        printf("  Instructions: %lu\n", values_[3]);
        if (values_[3] > 0) {
            printf("  IPC: %.3f\n", (double)values_[0] / values_[3]);
        }
    }
};

// CPU pinning for core isolation
void pinToCore(int coreId) {
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(coreId, &cpuset);
    
    int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
    if (result != 0) {
        throw std::runtime_error("Failed to pin thread to core: " + 
                               std::string(strerror(result)));
    }
    
    // Verify that we were actually pinned to the requested core
    cpu_set_t check_cpuset;
    CPU_ZERO(&check_cpuset);
    if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &check_cpuset) == 0) {
        if (!CPU_ISSET(coreId, &check_cpuset)) {
            fprintf(stderr, "Warning: Failed to verify thread pinning to core %d\n", coreId);
        }
    }
}

// Latency monitoring for performance tracking
class LatencyMonitor {
private:
    struct alignas(64) Bucket {
        std::atomic<uint64_t> count{0};
        std::atomic<uint64_t> sum{0};
        std::atomic<uint64_t> min{UINT64_MAX};
        std::atomic<uint64_t> max{0};
    };
    
    static constexpr size_t BUCKET_COUNT = 10; // Multiple latency ranges
    std::array<Bucket, BUCKET_COUNT> buckets_{};
    
public:
    void recordLatency(uint64_t nanos) {
        size_t bucketIdx = std::min(
            (size_t)(std::log2(nanos) - 10), // Start at ~1µs
            BUCKET_COUNT - 1
        );
        
        Bucket& bucket = buckets_[bucketIdx];
        bucket.count.fetch_add(1);
        bucket.sum.fetch_add(nanos);
        
        // Update min/max
        uint64_t current_min = bucket.min.load();
        while (nanos < current_min && 
              !bucket.min.compare_exchange_weak(current_min, nanos)) {}
        
        uint64_t current_max = bucket.max.load();
        while (nanos > current_max && 
              !bucket.max.compare_exchange_weak(current_max, nanos)) {}
    }
    
    void printStats() {
        printf("+------+-------------+-------------+-------------+-------------+------------+\n");
        printf("| Bucket |    Count    |     Min     |     Avg     |     Max     |  Range(ns) |\n");
        printf("+------+-------------+-------------+-------------+-------------+------------+\n");
        
        for (size_t i = 0; i < BUCKET_COUNT; ++i) {
            const Bucket& b = buckets_[i];
            uint64_t count = b.count.load();
            if (count > 0) {
                double avg = (double)b.sum.load() / count;
                printf("| %4zu | %11lu | %11lu | %11.1f | %11lu | %8lu+ |\n", 
                       i, count, b.min.load(), avg, b.max.load(), (1UL << (i + 10)));
            }
        }
        
        printf("+------+-------------+-------------+-------------+-------------+------------+\n");
    }
    
    void reset() {
        for (auto& b : buckets_) {
            b.count.store(0);
            b.sum.store(0);
            b.min.store(UINT64_MAX);
            b.max.store(0);
        }
    }
};

// Deterministic processing loop for pricing engine core
void processingLoop(SharedBlock* shared, bool& running, LatencyMonitor& latencyMonitor) {
    // Initialize pricer
    DeterministicPricer pricer(ACCURATE);
    
    // Per-cycle memory pool for temporary allocations
    MemoryPool perCyclePool(1024 * 1024); // 1MB per cycle
    PricingRequest request;
    
    // Event journal for recording requests and results
    EventJournal journal("pricing_events.bin");
    
    // Hardware performance counters
    HardwareCounters hwCounters;
    
    while (running) {
        // Reset per-cycle allocations
        perCyclePool.reset();
        
        // Fixed-time processing cycle
        uint64_t cycleStart = __rdtsc();
        const uint64_t CYCLE_TICKS = 10'000'000; // ~3.3ms at 3GHz
        
        hwCounters.reset();
        hwCounters.sample();
        
        // Process all pending requests in this cycle
        int processedCount = 0;
        
        while (shared->requestQueue.pop(request)) {
            journal.recordEvent(REQUEST_EVENT, &request, sizeof(request));
            
            uint64_t startNanos = EventJournal::getCurrentNanos();
            PricingResult* result = pricer.price(request);
            uint64_t endNanos = EventJournal::getCurrentNanos();
            
            if (result) {
                shared->resultQueue.push(*result);
                journal.recordEvent(RESULT_EVENT, result, sizeof(*result));
                latencyMonitor.recordLatency(endNanos - startNanos);
                pricer.release(result);
                processedCount++;
            }
            
            // Check cycle time budget
            if (__rdtsc() - cycleStart > CYCLE_TICKS * 0.9) {
                break; // Approaching end of cycle
            }
        }
        
        hwCounters.sample();
        
        // Update heartbeat timestamp
        shared->lastHeartbeatNanos.store(EventJournal::getCurrentNanos());
        
        // Wait until the end of this cycle for deterministic timing
        while (__rdtsc() - cycleStart < CYCLE_TICKS) {
            _mm_pause();
        }
        
        // Every 1000 cycles, print statistics
        static int cycleCount = 0;
        if (++cycleCount % 1000 == 0) {
            printf("Processed %d requests in last 1000 cycles\n", processedCount);
            latencyMonitor.printStats();
            latencyMonitor.reset();
            hwCounters.printStats();
        }
    }
}

// Shared memory implementation using shmget/shmat
class SharedMemoryChannel {
private:
    int shmId_;
    SharedBlock* sharedBlock_;
    key_t key_;
    bool isCreator_;
    
public:
    SharedMemoryChannel(const char* keyFile, bool create) : isCreator_(create) {
        // Generate IPC key from file path
        key_ = ftok(keyFile, 'R');
        if (key_ == -1) {
            // If the key file doesn't exist, create it
            if (errno == ENOENT && create) {
                int fd = open(keyFile, O_CREAT | O_WRONLY, 0666);
                if (fd != -1) {
                    close(fd);
                    key_ = ftok(keyFile, 'R');
                }
            }
            
            if (key_ == -1) {
                throw std::runtime_error("Failed to generate IPC key: " + 
                                       std::string(strerror(errno)));
            }
        }
        
        // Create or get shared memory segment
        if (create) {
            shmId_ = shmget(key_, sizeof(SharedBlock), IPC_CREAT | IPC_EXCL | 0666);
            if (shmId_ == -1 && errno == EEXIST) {
                // Segment exists, try to get it
                shmId_ = shmget(key_, sizeof(SharedBlock), 0666);
            }
        } else {
            shmId_ = shmget(key_, sizeof(SharedBlock), 0666);
        }
        
        if (shmId_ == -1) {
            throw std::runtime_error("Failed to create shared memory segment: " + 
                                   std::string(strerror(errno)));
        }
        
        // Attach the shared memory segment
        sharedBlock_ = static_cast<SharedBlock*>(shmat(shmId_, nullptr, 0));
        
        if (sharedBlock_ == (void*)-1) {
            throw std::runtime_error("Failed to attach shared memory segment: " + 
                                   std::string(strerror(errno)));
        }
        
        // Initialize if creating the segment
        if (create) {
            // Use placement new to initialize the shared block
            new (sharedBlock_) SharedBlock();
        }
    }
    
    ~SharedMemoryChannel() {
        if (sharedBlock_ != nullptr && sharedBlock_ != (void*)-1) {
            // Detach from shared memory
            shmdt(sharedBlock_);
        }
        
        // If we created the segment, mark it for deletion when all processes detach
        if (isCreator_ && shmId_ != -1) {
            shmctl(shmId_, IPC_RMID, nullptr);
        }
    }
    
    SharedBlock* get() { return sharedBlock_; }
};

// System initialization and setup
int initializeSystem(int argc, char** argv) {
    // Configure system and process
    
    // Disable stdio buffering for more deterministic output
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    
    // In a real implementation, would lock memory with mlockall
    // Try to lock memory to prevent swapping
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        fprintf(stderr, "Warning: Failed to lock memory: %s\n", strerror(errno));
        fprintf(stderr, "Performance may be degraded due to swapping.\n");
    }
    
    // Create shared memory channel
    try {
        SharedMemoryChannel channel("/tmp/pricing.key", true);
        
        // Initialize shared state
        bool running = true;
        
        // Initialize latency monitor
        LatencyMonitor latencyMonitor;
        
        // Start processing loop in this thread (simplified example)
        processingLoop(channel.get(), running, latencyMonitor);
        
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Fatal error: %s\n", e.what());
        return 1;
    }
}

#endif // DETERMINISTIC_PRICING_SYSTEM_H
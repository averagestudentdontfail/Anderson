#ifndef ENGINE_DETERMINE_SHMEM_H
#define ENGINE_DETERMINE_SHMEM_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <fcntl.h>

// Forward declarations
template<typename T, size_t SIZE> class RingBuffer;

namespace engine {
namespace determine {

// Core data structures for shared memory
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

/**
 * @brief Shared memory channel for inter-process communication
 */
class SharedMemoryChannel {
public:
    /**
     * @brief Create or attach to a shared memory channel
     * @param keyFile File path used to generate the IPC key
     * @param create Whether to create the segment or just attach to it
     */
    SharedMemoryChannel(const std::string& keyFile, bool create = false);
    
    /**
     * @brief Cleanup and detach from shared memory
     */
    ~SharedMemoryChannel();
    
    /**
     * @brief Get the shared memory block
     * @return Pointer to the shared block
     */
    SharedBlock* get() { return sharedBlock_; }
    
    /**
     * @brief Get the shared memory block (const version)
     * @return Const pointer to the shared block
     */
    const SharedBlock* get() const { return sharedBlock_; }
    
    /**
     * @brief Check if the shared memory channel is valid
     * @return True if the channel is valid
     */
    bool isValid() const { return sharedBlock_ != nullptr && sharedBlock_ != (void*)-1; }
    
    /**
     * @brief Get the timestamp of the last heartbeat
     * @return Timestamp in nanoseconds
     */
    uint64_t getLastHeartbeatNanos() const;
    
    /**
     * @brief Update the heartbeat timestamp
     */
    void updateHeartbeat();
    
    /**
     * @brief Generate a unique key for a shared memory segment
     * @param keyPrefix Prefix for the key
     * @return Unique key as a string
     */
    static std::string generateUniqueKey(const std::string& keyPrefix);

private:
    int shmId_;
    bool isCreator_;
    SharedBlock* sharedBlock_;
    std::string keyFile_;
    
    /**
     * @brief Initialize the shared memory segment
     */
    void initialize();
};

/**
 * @brief Manager for shared memory channels
 * 
 * Provides centralized management of multiple shared memory channels
 */
class SharedMemoryManager {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the shared memory manager
     */
    static SharedMemoryManager& getInstance();
    
    /**
     * @brief Get or create a shared memory channel
     * @param keyFile File path used to generate the IPC key
     * @param create Whether to create the segment or just attach to it
     * @return Shared pointer to the channel
     */
    std::shared_ptr<SharedMemoryChannel> getChannel(const std::string& keyFile, bool create = false);
    
    /**
     * @brief Remove a channel from the manager
     * @param keyFile The key file for the channel to remove
     */
    void removeChannel(const std::string& keyFile);
    
    /**
     * @brief Update heartbeats for all channels
     */
    void updateAllHeartbeats();

private:
    SharedMemoryManager() = default;
    ~SharedMemoryManager() = default;
    
    // Disable copy and move
    SharedMemoryManager(const SharedMemoryManager&) = delete;
    SharedMemoryManager& operator=(const SharedMemoryManager&) = delete;
    SharedMemoryManager(SharedMemoryManager&&) = delete;
    SharedMemoryManager& operator=(SharedMemoryManager&&) = delete;
    
    std::unordered_map<std::string, std::shared_ptr<SharedMemoryChannel>> channels_;
};

} 
} 

#endif
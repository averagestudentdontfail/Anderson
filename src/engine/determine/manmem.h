#ifndef ENGINE_DETERMINE_MANMEM_H
#define ENGINE_DETERMINE_MANMEM_H

#include <atomic>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <errno.h>
#include <string.h>

namespace engine {
namespace determine {

// Forward declarations
class MemoryPool;
class PerCycleAllocator;

/**
 * @brief Memory Management System for deterministic execution.
 * 
 * Provides facilities for deterministic memory allocation including:
 * - Global memory pools for long-lived objects
 * - Per-cycle memory pools for transient allocations
 * - Huge page support for better performance
 * - Memory locking to prevent swapping
 */
class MemoryManager {
public:
    /**
     * @brief Initialize the memory manager
     * @param globalPoolSize Size of the global memory pool in bytes
     * @param perCyclePoolSize Size of the per-cycle memory pool in bytes
     * @param numCyclePools Number of per-cycle pools to maintain
     */
    MemoryManager(size_t globalPoolSize = 64 * 1024 * 1024,
                  size_t perCyclePoolSize = 1 * 1024 * 1024,
                  size_t numCyclePools = 4);
    
    /**
     * @brief Cleanup and release all memory
     */
    ~MemoryManager();
    
    /**
     * @brief Allocate memory from the global pool
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment requirement (default: 64 bytes for cache line alignment)
     * @return Pointer to allocated memory or nullptr if allocation failed
     */
    void* allocateGlobal(size_t size, size_t alignment = 64);
    
    /**
     * @brief Get the current per-cycle allocator
     * @return Reference to the current per-cycle allocator
     */
    PerCycleAllocator& getCurrentAllocator();
    
    /**
     * @brief Advance to the next cycle (rotates the per-cycle allocator)
     */
    void advanceCycle();

    /**
     * @brief Lock all current and future pages in memory
     * @return True if successful, false otherwise
     */
    static bool lockMemory();

    /**
     * @brief Check if huge pages are available
     * @return True if huge pages are available and configured
     */
    static bool areHugePagesAvailable();

private:
    std::unique_ptr<MemoryPool> globalPool_;
    std::vector<std::unique_ptr<MemoryPool>> cyclePools_;
    std::atomic<size_t> currentCycleIndex_{0};
    size_t numCyclePools_;
};

/**
 * @brief Memory pool with aligned allocations
 */
class MemoryPool {
public:
    /**
     * @brief Create a memory pool of the specified capacity
     * @param capacity Size of the pool in bytes
     * @param useHugePages Whether to attempt to use huge pages for the allocation
     */
    explicit MemoryPool(size_t capacity, bool useHugePages = true);
    
    /**
     * @brief Release the memory pool
     */
    ~MemoryPool();
    
    /**
     * @brief Allocate aligned memory from the pool
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment requirement
     * @return Pointer to allocated memory or nullptr if allocation failed
     */
    void* allocate(size_t size, size_t alignment = 64);
    
    /**
     * @brief Reset the pool to its initial state
     */
    void reset();
    
    /**
     * @brief Get the total capacity of the pool
     * @return Capacity in bytes
     */
    size_t capacity() const { return capacity_; }
    
    /**
     * @brief Get the currently used size of the pool
     * @return Used size in bytes
     */
    size_t used() const { return offset_.load(std::memory_order_relaxed); }
    
    /**
     * @brief Get the available space in the pool
     * @return Available size in bytes
     */
    size_t available() const { return capacity_ - used(); }

private:
    uint8_t* buffer_;
    size_t capacity_;
    std::atomic<size_t> offset_{0};
    bool usesHugePages_;
};

/**
 * @brief Per-cycle allocator for transient objects
 * 
 * Provides a scoped memory allocation scheme that gets reset every cycle,
 * allowing for efficient memory management without costly deallocations.
 */
class PerCycleAllocator {
public:
    /**
     * @brief Create a per-cycle allocator wrapping a memory pool
     * @param pool The underlying memory pool
     */
    explicit PerCycleAllocator(MemoryPool& pool);
    
    /**
     * @brief Allocate memory for the current cycle
     * @param size Size in bytes to allocate
     * @param alignment Memory alignment requirement
     * @return Pointer to allocated memory or nullptr if allocation failed
     */
    void* allocate(size_t size, size_t alignment = 64);
    
    /**
     * @brief Reset the allocator for a new cycle
     */
    void reset();
    
    /**
     * @brief Get the underlying memory pool
     * @return Reference to the memory pool
     */
    MemoryPool& getPool() { return pool_; }
    
private:
    MemoryPool& pool_;
};

/**
 * @brief Template class for object allocation with type safety
 */
template<typename T>
class TypedAllocator {
public:
    explicit TypedAllocator(PerCycleAllocator& allocator) : allocator_(allocator) {}
    
    /**
     * @brief Allocate and construct an object of type T
     * @param args Constructor arguments for T
     * @return Pointer to the newly constructed object
     */
    template<typename... Args>
    T* create(Args&&... args) {
        void* memory = allocator_.allocate(sizeof(T), alignof(T));
        if (!memory) {
            return nullptr;
        }
        return new (memory) T(std::forward<Args>(args)...);
    }
    
private:
    PerCycleAllocator& allocator_;
};

/**
 * @brief RAII wrapper for PerCycleAllocator to ensure reset at scope exit
 */
class ScopedCycleAllocator {
public:
    explicit ScopedCycleAllocator(MemoryManager& manager)
        : manager_(manager) {}
    
    ~ScopedCycleAllocator() {
        manager_.advanceCycle();
    }
    
    PerCycleAllocator& getAllocator() {
        return manager_.getCurrentAllocator();
    }
    
private:
    MemoryManager& manager_;
};

} 
} 

#endif 
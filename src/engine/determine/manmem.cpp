#include "manmem.h"
#include <sys/mman.h>
#include <unistd.h>
#include <fcntl.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>  // Add this include for std::numeric_limits

namespace engine {
namespace determine {

// MemoryManager implementation
MemoryManager::MemoryManager(size_t globalPoolSize, size_t perCyclePoolSize, size_t numCyclePools)
    : numCyclePools_(numCyclePools) {
    
    // Create the global memory pool
    try {
        globalPool_ = std::make_unique<MemoryPool>(globalPoolSize, true);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to create global memory pool: " + std::string(e.what()));
    }
    
    // Create per-cycle pools
    cyclePools_.reserve(numCyclePools);
    for (size_t i = 0; i < numCyclePools; ++i) {
        try {
            cyclePools_.push_back(std::make_unique<MemoryPool>(perCyclePoolSize, false));
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to create cycle memory pool: " + std::string(e.what()));
        }
    }
}

MemoryManager::~MemoryManager() {
    // Memory pools will be cleaned up automatically
    // through unique_ptr destructors
}

void* MemoryManager::allocateGlobal(size_t size, size_t alignment) {
    if (!globalPool_) {
        return nullptr;
    }
    return globalPool_->allocate(size, alignment);
}

PerCycleAllocator& MemoryManager::getCurrentAllocator() {
    static thread_local PerCycleAllocator* currentAllocator = nullptr;
    static thread_local size_t lastCycleIndex = std::numeric_limits<size_t>::max();
    
    size_t cycleIndex = currentCycleIndex_.load(std::memory_order_acquire) % numCyclePools_;
    
    // Check if we need to update the allocator reference
    if (!currentAllocator || cycleIndex != lastCycleIndex) {
        currentAllocator = new PerCycleAllocator(*cyclePools_[cycleIndex]);
        lastCycleIndex = cycleIndex;
    }
    
    return *currentAllocator;
}

void MemoryManager::advanceCycle() {
    size_t current = currentCycleIndex_.load(std::memory_order_relaxed);
    size_t next = (current + 1) % numCyclePools_;
    
    // Reset the next pool before switching to it
    cyclePools_[next]->reset();
    
    // Advance the cycle
    currentCycleIndex_.store(next, std::memory_order_release);
}

bool MemoryManager::lockMemory() {
    // Lock current and future memory pages into RAM
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
        std::cerr << "Warning: Failed to lock memory: " << strerror(errno) << std::endl;
        std::cerr << "Performance may be degraded due to swapping." << std::endl;
        return false;
    }
    return true;
}

bool MemoryManager::areHugePagesAvailable() {
    // Try to open the hugepage control file
    int fd = open("/proc/sys/vm/nr_hugepages", O_RDONLY);
    if (fd == -1) {
        return false;
    }
    
    char buffer[32] = {0};
    ssize_t bytesRead = read(fd, buffer, sizeof(buffer) - 1);
    close(fd);
    
    if (bytesRead <= 0) {
        return false;
    }
    
    // Check if the value is greater than 0
    return std::stoi(std::string(buffer)) > 0;
}

// MemoryPool implementation
MemoryPool::MemoryPool(size_t capacity, bool useHugePages)
    : capacity_(capacity), usesHugePages_(false) {
    
    // Try to allocate using huge pages if requested
    if (useHugePages) {
        buffer_ = static_cast<uint8_t*>(mmap(
            nullptr, capacity_, 
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB,
            -1, 0));
        
        if (buffer_ != static_cast<uint8_t*>(MAP_FAILED)) {
            usesHugePages_ = true;
        } else {
            // Fallback to regular pages if huge pages failed
            buffer_ = static_cast<uint8_t*>(MAP_FAILED);
        }
    } else {
        buffer_ = static_cast<uint8_t*>(MAP_FAILED);
    }
    
    // If huge pages failed or weren't requested, use regular pages
    if (buffer_ == static_cast<uint8_t*>(MAP_FAILED)) {
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
    
    // Lock memory to prevent swapping if possible
    if (mlock(buffer_, capacity_) != 0) {
        // Not fatal, but log the error
        std::cerr << "Warning: Failed to lock memory pool: " << strerror(errno) << std::endl;
    }
    
    // Pre-touch pages to ensure physical allocation
    const size_t pageSize = 4096; // Default page size
    for (size_t i = 0; i < capacity_; i += pageSize) {
        buffer_[i] = 0;
    }
}

MemoryPool::~MemoryPool() {
    if (buffer_ && buffer_ != static_cast<uint8_t*>(MAP_FAILED)) {
        // Release lock if we previously locked it
        munlock(buffer_, capacity_);
        
        // Unmap the memory
        munmap(buffer_, capacity_);
    }
}

void* MemoryPool::allocate(size_t size, size_t alignment) {
    if (size == 0) {
        return nullptr;
    }
    
    // Calculate aligned offset
    size_t current = offset_.load(std::memory_order_relaxed);
    size_t aligned = (current + alignment - 1) & ~(alignment - 1);
    size_t next = aligned + size;
    
    // Check if we have enough space
    if (next > capacity_) {
        return nullptr;
    }
    
    // Try to update the offset atomically
    if (offset_.compare_exchange_strong(current, next, std::memory_order_release, std::memory_order_relaxed)) {
        return buffer_ + aligned;
    }
    
    // If the CAS failed, someone else updated the offset concurrently
    // We could retry, but for simplicity, we'll just fail the allocation
    return nullptr;
}

void MemoryPool::reset() {
    offset_.store(0, std::memory_order_relaxed);
    
    // Optional: zero the memory for security or debugging
    // memset(buffer_, 0, capacity_);
}

// PerCycleAllocator implementation
PerCycleAllocator::PerCycleAllocator(MemoryPool& pool)
    : pool_(pool) {
}

void* PerCycleAllocator::allocate(size_t size, size_t alignment) {
    return pool_.allocate(size, alignment);
}

void PerCycleAllocator::reset() {
    pool_.reset();
}

} 
}
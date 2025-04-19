#ifndef MEMORY_POOL_H
#define MEMORY_POOL_H

#include <array>
#include <bitset>
#include <memory>
#include <mutex>
#include <vector>
#include <cassert>
#include <atomic>

/**
 * @brief High-performance memory pool for frequent allocations of fixed-size objects
 * 
 * This memory pool is optimized for cache-line alignment and thread safety.
 * It allocates memory in blocks and manages the allocation within those blocks.
 * 
 * @tparam T Type of objects to allocate
 * @tparam BlockSize Size of each memory block in bytes
 * @tparam Alignment Memory alignment requirement (default is 64 bytes for cache line)
 */
template <typename T, size_t BlockSize = 4096, size_t Alignment = 64>
class MemPool {
private:
    // Make sure BlockSize is a multiple of sizeof(T)
    static constexpr size_t AdjustedBlockSize = 
        (BlockSize / sizeof(T)) * sizeof(T);
    
    // Number of objects per block
    static constexpr size_t ObjectsPerBlock = AdjustedBlockSize / sizeof(T);
    
    // Structure to represent a block of memory
    struct alignas(Alignment) Block {
        std::array<std::byte, AdjustedBlockSize> storage;
        std::bitset<ObjectsPerBlock> used;
        Block* next = nullptr;
        
        Block() : used() { }
        ~Block() = default;
    };
    
    // Mutex for thread safety
    mutable std::mutex allocation_mutex_;
    
    // Head of the block list
    Block* head_ = nullptr;
    
    // Statistics
    std::atomic<size_t> allocated_count_{0};
    std::atomic<size_t> deallocated_count_{0};
    std::atomic<size_t> block_count_{0};
    
public:
    /**
     * @brief Constructor
     */
    MemPool() = default;
    
    /**
     * @brief Destructor
     * 
     * Frees all allocated blocks
     */
    ~MemPool() {
        // Free all blocks
        while (head_) {
            Block* next = head_->next;
            delete head_;
            head_ = next;
        }
    }
    
    // Prevent copying
    MemPool(const MemPool&) = delete;
    MemPool& operator=(const MemPool&) = delete;
    
    // Allow moving
    MemPool(MemPool&& other) noexcept {
        std::lock_guard<std::mutex> lock(other.allocation_mutex_);
        head_ = other.head_;
        allocated_count_.store(other.allocated_count_.load());
        deallocated_count_.store(other.deallocated_count_.load());
        block_count_.store(other.block_count_.load());
        other.head_ = nullptr;
    }
    
    MemPool& operator=(MemPool&& other) noexcept {
        if (this != &other) {
            std::lock_guard<std::mutex> lock_this(allocation_mutex_);
            std::lock_guard<std::mutex> lock_other(other.allocation_mutex_);
            
            // Free current blocks
            while (head_) {
                Block* next = head_->next;
                delete head_;
                head_ = next;
            }
            
            // Take ownership of other's blocks
            head_ = other.head_;
            allocated_count_.store(other.allocated_count_.load());
            deallocated_count_.store(other.deallocated_count_.load());
            block_count_.store(other.block_count_.load());
            other.head_ = nullptr;
        }
        return *this;
    }
    
    /**
     * @brief Allocate a new object
     * 
     * @return Pointer to allocated object
     */
    T* allocate() {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        
        // If no blocks or current block is full, allocate a new one
        if (!head_ || head_->used.all()) {
            auto new_block = new Block();
            new_block->next = head_;
            head_ = new_block;
            block_count_++;
        }
        
        // Find first available slot in the current block
        size_t index = 0;
        for (; index < ObjectsPerBlock; ++index) {
            if (!head_->used[index]) {
                break;
            }
        }
        
        assert(index < ObjectsPerBlock && "Block should have available space");
        
        // Mark as used
        head_->used.set(index);
        
        // Calculate pointer
        T* ptr = reinterpret_cast<T*>(&head_->storage[index * sizeof(T)]);
        
        // Placement new to construct the object
        new (ptr) T();
        
        // Update statistics
        allocated_count_++;
        
        return ptr;
    }
    
    /**
     * @brief Deallocate an object
     * 
     * @param ptr Pointer to the object to deallocate
     */
    void deallocate(T* ptr) noexcept {
        if (!ptr) return;
        
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        
        // Call destructor
        ptr->~T();
        
        // Find which block contains this pointer
        for (Block* block = head_; block; block = block->next) {
            // Calculate block address range
            std::byte* block_start = &block->storage[0];
            std::byte* block_end = block_start + AdjustedBlockSize;
            
            // Check if pointer is in this block
            std::byte* ptr_byte = reinterpret_cast<std::byte*>(ptr);
            if (ptr_byte >= block_start && ptr_byte < block_end) {
                // Calculate the index
                size_t index = (ptr_byte - block_start) / sizeof(T);
                
                // Check if it was actually allocated
                assert(block->used[index] && "Attempting to deallocate memory that wasn't allocated");
                
                // Mark as available
                block->used.reset(index);
                
                // Update statistics
                deallocated_count_++;
                
                return;
            }
        }
        
        // If we get here, the pointer is not from this pool
        assert(false && "Attempting to deallocate memory that wasn't allocated from this pool");
    }
    
    /**
     * @brief Create an object using the memory pool
     * 
     * @tparam Args Constructor argument types
     * @param args Constructor arguments
     * @return Smart pointer to the object
     */
    template <typename... Args>
    std::shared_ptr<T> create(Args&&... args) {
        // Allocate memory
        T* ptr = allocate();
        
        // Destroy the default-constructed object
        ptr->~T();
        
        // Construct with provided arguments
        new (ptr) T(std::forward<Args>(args)...);
        
        // Create a shared_ptr with custom deleter
        return std::shared_ptr<T>(ptr, [this](T* p) { this->deallocate(p); });
    }
    
    /**
     * @brief Get the number of allocated objects
     * 
     * @return Number of allocated objects
     */
    size_t getAllocatedCount() const {
        return allocated_count_.load();
    }
    
    /**
     * @brief Get the number of deallocated objects
     * 
     * @return Number of deallocated objects
     */
    size_t getDeallocatedCount() const {
        return deallocated_count_.load();
    }
    
    /**
     * @brief Get the number of active objects
     * 
     * @return Number of active objects
     */
    size_t getActiveCount() const {
        return allocated_count_.load() - deallocated_count_.load();
    }
    
    /**
     * @brief Get the number of blocks
     * 
     * @return Number of blocks
     */
    size_t getBlockCount() const {
        return block_count_.load();
    }
    
    /**
     * @brief Clear all allocated memory
     * 
     * Note: This will invalidate any pointers obtained from this pool
     */
    void clear() {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        
        // Free all blocks
        while (head_) {
            Block* next = head_->next;
            delete head_;
            head_ = next;
        }
        
        // Reset statistics
        allocated_count_.store(0);
        deallocated_count_.store(0);
        block_count_.store(0);
    }
    
    /**
     * @brief Get the memory utilization percentage
     * 
     * @return Percentage of memory used (0-100)
     */
    double getUtilization() const {
        std::lock_guard<std::mutex> lock(allocation_mutex_);
        
        if (block_count_.load() == 0) {
            return 0.0;
        }
        
        size_t total_capacity = block_count_.load() * ObjectsPerBlock;
        return 100.0 * getActiveCount() / total_capacity;
    }
};

#endif
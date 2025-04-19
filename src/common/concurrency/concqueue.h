#ifndef CONC_QUEUE_H
#define CONC_QUEUE_H

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>
#include <chrono>
#include <atomic>
#include <memory>

namespace concurrency {

/**
 * @brief Thread-safe concurrent queue implementation
 * 
 * This queue supports multiple producers and multiple consumers
 * with both blocking and non-blocking operations.
 * 
 * @tparam T Type of elements in the queue
 */
template <typename T>
class ConcQueue {
public:
    /**
     * @brief Constructor
     * 
     * @param max_size Maximum size of the queue (0 for unlimited)
     */
    explicit ConcQueue(size_t max_size = 0) 
        : max_size_(max_size), closed_(false), size_(0) {}
    
    /**
     * @brief Destructor
     */
    ~ConcQueue() {
        close();
    }
    
    /**
     * @brief Push an element into the queue (blocking)
     * 
     * If the queue has a size limit and is full, this function
     * will block until space is available or the queue is closed.
     * 
     * @param value Value to push
     * @return true if the element was pushed, false if the queue is closed
     */
    bool push(const T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait until there's space or the queue is closed
        if (max_size_ > 0) {
            not_full_.wait(lock, [this] { 
                return size_ < max_size_ || closed_; 
            });
        }
        
        // Check if the queue was closed while waiting
        if (closed_) {
            return false;
        }
        
        // Add the element
        queue_.push(value);
        size_++;
        
        // Notify one waiting consumer
        not_empty_.notify_one();
        
        return true;
    }
    
    /**
     * @brief Push an element into the queue (blocking)
     * 
     * Move version of push.
     * 
     * @param value Value to push
     * @return true if the element was pushed, false if the queue is closed
     */
    bool push(T&& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait until there's space or the queue is closed
        if (max_size_ > 0) {
            not_full_.wait(lock, [this] { 
                return size_ < max_size_ || closed_; 
            });
        }
        
        // Check if the queue was closed while waiting
        if (closed_) {
            return false;
        }
        
        // Add the element
        queue_.push(std::move(value));
        size_++;
        
        // Notify one waiting consumer
        not_empty_.notify_one();
        
        return true;
    }
    
    /**
     * @brief Try to push an element without blocking
     * 
     * @param value Value to push
     * @return true if the element was pushed, false if the queue is full or closed
     */
    bool try_push(const T& value) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if the queue is closed or full
        if (closed_ || (max_size_ > 0 && size_ >= max_size_)) {
            return false;
        }
        
        // Add the element
        queue_.push(value);
        size_++;
        
        // Notify one waiting consumer
        not_empty_.notify_one();
        
        return true;
    }
    
    /**
     * @brief Try to push an element with a timeout
     * 
     * @param value Value to push
     * @param timeout Maximum time to wait
     * @return true if the element was pushed, false if timeout or queue closed
     */
    template <typename Rep, typename Period>
    bool try_push_for(const T& value, 
                     const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait until there's space or the queue is closed or timeout
        if (max_size_ > 0) {
            bool success = not_full_.wait_for(lock, timeout, [this] { 
                return size_ < max_size_ || closed_; 
            });
            
            if (!success) {
                return false;  // Timeout
            }
        }
        
        // Check if the queue was closed while waiting
        if (closed_) {
            return false;
        }
        
        // Add the element
        queue_.push(value);
        size_++;
        
        // Notify one waiting consumer
        not_empty_.notify_one();
        
        return true;
    }
    
    /**
     * @brief Pop an element from the queue (blocking)
     * 
     * This function blocks until an element is available or the queue is closed.
     * 
     * @return Element if available, nullopt if the queue is closed and empty
     */
    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait until there's an element or the queue is closed
        not_empty_.wait(lock, [this] { 
            return size_ > 0 || closed_; 
        });
        
        // Check if the queue is empty (could happen if closed while waiting)
        if (size_ == 0) {
            return std::nullopt;
        }
        
        // Get the element
        T value = std::move(queue_.front());
        queue_.pop();
        size_--;
        
        // Notify one waiting producer if there's a size limit
        if (max_size_ > 0) {
            not_full_.notify_one();
        }
        
        return value;
    }
    
    /**
     * @brief Try to pop an element without blocking
     * 
     * @return Element if available, nullopt if the queue is empty
     */
    std::optional<T> try_pop() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Check if the queue is empty
        if (size_ == 0) {
            return std::nullopt;
        }
        
        // Get the element
        T value = std::move(queue_.front());
        queue_.pop();
        size_--;
        
        // Notify one waiting producer if there's a size limit
        if (max_size_ > 0) {
            not_full_.notify_one();
        }
        
        return value;
    }
    
    /**
     * @brief Try to pop an element with a timeout
     * 
     * @param timeout Maximum time to wait
     * @return Element if available, nullopt if timeout or queue empty
     */
    template <typename Rep, typename Period>
    std::optional<T> try_pop_for(const std::chrono::duration<Rep, Period>& timeout) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        // Wait until there's an element or the queue is closed or timeout
        bool success = not_empty_.wait_for(lock, timeout, [this] { 
            return size_ > 0 || closed_; 
        });
        
        if (!success || size_ == 0) {
            return std::nullopt;  // Timeout or queue empty
        }
        
        // Get the element
        T value = std::move(queue_.front());
        queue_.pop();
        size_--;
        
        // Notify one waiting producer if there's a size limit
        if (max_size_ > 0) {
            not_full_.notify_one();
        }
        
        return value;
    }
    
    /**
     * @brief Check if the queue is empty
     * 
     * @return true if the queue is empty, false otherwise
     */
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_ == 0;
    }
    
    /**
     * @brief Get the number of elements in the queue
     * 
     * @return Number of elements
     */
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return size_;
    }
    
    /**
     * @brief Close the queue
     * 
     * After this call, no new elements can be pushed.
     * Consumers can still pop existing elements.
     */
    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        
        // Wake up all waiting threads
        not_empty_.notify_all();
        not_full_.notify_all();
    }
    
    /**
     * @brief Check if the queue is closed
     * 
     * @return true if the queue is closed, false otherwise
     */
    bool is_closed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return closed_;
    }
    
    /**
     * @brief Clear all elements from the queue
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Empty the queue
        std::queue<T> empty;
        std::swap(queue_, empty);
        size_ = 0;
        
        // Notify all waiting producers if there's a size limit
        if (max_size_ > 0) {
            not_full_.notify_all();
        }
    }
    
private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t max_size_;
    bool closed_;
    size_t size_;  // Cache size to avoid queue_.size() calls, which are O(1) but can be slow
    
    // Prevent copying and moving
    ConcQueue(const ConcQueue&) = delete;
    ConcQueue& operator=(const ConcQueue&) = delete;
    ConcQueue(ConcQueue&&) = delete;
    ConcQueue& operator=(ConcQueue&&) = delete;
};

/**
 * @brief Thread-safe concurrent bounded queue with wait-free operations
 * 
 * This queue is optimized for high-performance scenarios where
 * blocking operations are not desirable. It uses atomics for synchronization.
 * 
 * @tparam T Type of elements in the queue (must be trivially copyable)
 * @tparam Capacity Fixed capacity of the queue (must be a power of 2)
 */
template <typename T, size_t Capacity>
class LockFreeBoundedQueue {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(std::is_trivially_copyable<T>::value, "T must be trivially copyable");
    
public:
    /**
     * @brief Constructor
     */
    LockFreeBoundedQueue() : head_(0), tail_(0) {}
    
    /**
     * @brief Try to push an element
     * 
     * @param value Value to push
     * @return true if the element was pushed, false if the queue is full
     */
    bool try_push(const T& value) {
        size_t head = head_.load(std::memory_order_relaxed);
        size_t tail = tail_.load(std::memory_order_acquire);
        
        // Check if the queue is full
        if ((head + 1) % (Capacity * 2) == tail) {
            return false;
        }
        
        // Store the value
        buffer_[head & (Capacity - 1)] = value;
        
        // Update head
        head_.store((head + 1) % (Capacity * 2), std::memory_order_release);
        
        return true;
    }
    
    /**
     * @brief Try to pop an element
     * 
     * @param[out] value Reference to store the popped value
     * @return true if an element was popped, false if the queue is empty
     */
    bool try_pop(T& value) {
        size_t tail = tail_.load(std::memory_order_relaxed);
        size_t head = head_.load(std::memory_order_acquire);
        
        // Check if the queue is empty
        if (tail == head) {
            return false;
        }
        
        // Get the value
        value = buffer_[tail & (Capacity - 1)];
        
        // Update tail
        tail_.store((tail + 1) % (Capacity * 2), std::memory_order_release);
        
        return true;
    }
    
    /**
     * @brief Check if the queue is empty
     * 
     * @return true if the queue is empty, false otherwise
     */
    bool empty() const {
        return tail_.load(std::memory_order_acquire) == 
               head_.load(std::memory_order_acquire);
    }
    
    /**
     * @brief Get the approximate number of elements in the queue
     * 
     * This is an approximate count due to the lock-free nature of the queue.
     * 
     * @return Approximate number of elements
     */
    size_t size() const {
        size_t head = head_.load(std::memory_order_acquire);
        size_t tail = tail_.load(std::memory_order_acquire);
        
        if (head >= tail) {
            return head - tail;
        } else {
            return Capacity * 2 - tail + head;
        }
    }
    
    /**
     * @brief Get the capacity of the queue
     * 
     * @return Capacity
     */
    constexpr size_t capacity() const {
        return Capacity;
    }
    
private:
    // Buffer with padding to avoid false sharing
    alignas(64) T buffer_[Capacity];
    
    // Head index (producer writes here)
    alignas(64) std::atomic<size_t> head_;
    
    // Tail index (consumer reads here)
    alignas(64) std::atomic<size_t> tail_;
    
    // Prevent copying and moving
    LockFreeBoundedQueue(const LockFreeBoundedQueue&) = delete;
    LockFreeBoundedQueue& operator=(const LockFreeBoundedQueue&) = delete;
    LockFreeBoundedQueue(LockFreeBoundedQueue&&) = delete;
    LockFreeBoundedQueue& operator=(LockFreeBoundedQueue&&) = delete;
};

} // namespace concurrency

#endif // CONC_QUEUE_H
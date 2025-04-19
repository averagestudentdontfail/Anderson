#ifndef SPINLOCK_H
#define SPINLOCK_H

#include <atomic>
#include <thread>
#include <immintrin.h>

namespace concurrency {

/**
 * @brief A lightweight spinlock implementation
 * 
 * Spinlocks are best used for very short critical sections
 * where contention is expected to be low. For longer operations,
 * std::mutex is usually more appropriate.
 */
class SpinLock {
public:
    /**
     * @brief Constructor
     */
    SpinLock() noexcept : flag_(false) {}
    
    /**
     * @brief Acquire the lock
     * 
     * This function spins until the lock is acquired.
     * Uses _mm_pause() to reduce CPU consumption and pipeline contention.
     */
    void lock() noexcept {
        while (true) {
            // Try to acquire the lock
            if (!flag_.exchange(true, std::memory_order_acquire)) {
                // Successfully acquired the lock
                return;
            }
            
            // Spin without constantly trying to acquire
            while (flag_.load(std::memory_order_relaxed)) {
                // Pause instruction to reduce power consumption and improve performance
                _mm_pause();
            }
        }
    }
    
    /**
     * @brief Try to acquire the lock without blocking
     * 
     * @return true if the lock was acquired, false otherwise
     */
    bool try_lock() noexcept {
        // Try to set the flag to true only if it's currently false
        return !flag_.exchange(true, std::memory_order_acquire);
    }
    
    /**
     * @brief Release the lock
     */
    void unlock() noexcept {
        flag_.store(false, std::memory_order_release);
    }
    
    /**
     * @brief Check if the lock is currently held
     * 
     * Note: This is only useful for debugging, as the lock state may
     * change immediately after this function returns.
     * 
     * @return true if the lock is held, false otherwise
     */
    bool is_locked() const noexcept {
        return flag_.load(std::memory_order_relaxed);
    }
    
private:
    std::atomic<bool> flag_;
    
    // Prevent copying and moving
    SpinLock(const SpinLock&) = delete;
    SpinLock& operator=(const SpinLock&) = delete;
    SpinLock(SpinLock&&) = delete;
    SpinLock& operator=(SpinLock&&) = delete;
};

/**
 * @brief RAII wrapper for SpinLock
 * 
 * This class provides a convenient RAII-style wrapper for SpinLock,
 * automatically releasing the lock when it goes out of scope.
 */
class SpinLockGuard {
public:
    /**
     * @brief Constructor
     * 
     * @param lock The SpinLock to acquire
     */
    explicit SpinLockGuard(SpinLock& lock) noexcept : lock_(lock) {
        lock_.lock();
    }
    
    /**
     * @brief Destructor
     * 
     * Releases the lock automatically when the guard goes out of scope.
     */
    ~SpinLockGuard() noexcept {
        lock_.unlock();
    }
    
private:
    SpinLock& lock_;
    
    // Prevent copying and moving
    SpinLockGuard(const SpinLockGuard&) = delete;
    SpinLockGuard& operator=(const SpinLockGuard&) = delete;
    SpinLockGuard(SpinLockGuard&&) = delete;
    SpinLockGuard& operator=(SpinLockGuard&&) = delete;
};

} // namespace concurrency

#endif // SPINLOCK_H
#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include "concqueue.h"
#include <thread>
#include <functional>
#include <vector>
#include <future>
#include <memory>
#include <type_traits>
#include <atomic>
#include <stdexcept>

namespace concurrency {

/**
 * @brief Thread pool for parallel task execution
 * 
 * This thread pool provides a fixed number of worker threads
 * that can execute tasks asynchronously. Tasks can be submitted
 * as functions or lambdas, and results can be retrieved via futures.
 */
class ThreadPool {
public:
    /**
     * @brief Constructor
     * 
     * @param num_threads Number of worker threads (default is hardware concurrency)
     * @param queue_size Maximum size of the task queue (0 for unlimited)
     * @param thread_names Optional name prefix for worker threads (for debugging)
     */
    explicit ThreadPool(
        size_t num_threads = std::thread::hardware_concurrency(),
        size_t queue_size = 0,
        const std::string& thread_names = "Worker"
    ) : task_queue_(queue_size), running_(true), active_tasks_(0) {
        // Ensure at least one thread
        if (num_threads == 0) {
            num_threads = 1;
        }
        
        // Create worker threads
        workers_.reserve(num_threads);
        for (size_t i = 0; i < num_threads; ++i) {
            workers_.emplace_back(&ThreadPool::worker_loop, this, thread_names + "-" + std::to_string(i));
        }
    }
    
    /**
     * @brief Destructor
     * 
     * Waits for all tasks to complete and shuts down the pool.
     */
    ~ThreadPool() {
        shutdown();
    }
    
    /**
     * @brief Submit a task to the pool
     * 
     * This method enqueues a task for execution and returns a future
     * that can be used to retrieve the result.
     * 
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return Future for the result
     * @throws std::runtime_error if the pool is stopped or queue is full
     */
    template <typename F, typename... Args>
    auto submit(F&& f, Args&&... args) -> std::future<typename std::invoke_result<F, Args...>::type> {
        using ResultType = typename std::invoke_result<F, Args...>::type;
        
        // Check if the pool is running
        if (!running_) {
            throw std::runtime_error("ThreadPool: Cannot submit task to stopped pool");
        }
        
        // Create a packaged task to wrap the function
        auto task = std::make_shared<std::packaged_task<ResultType()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        // Get the future before pushing the task
        std::future<ResultType> result = task->get_future();
        
        // Wrap the task in a void function for the queue
        auto task_wrapper = [task, this]() {
            // Track active tasks
            active_tasks_.fetch_add(1, std::memory_order_relaxed);
            
            // Execute the task
            try {
                (*task)();
            } catch (...) {
                // Just catch here to prevent thread termination
            }
            
            // Decrement active tasks
            active_tasks_.fetch_sub(1, std::memory_order_relaxed);
        };
        
        // Push the task to the queue
        if (!task_queue_.push(std::move(task_wrapper))) {
            throw std::runtime_error("ThreadPool: Task queue is full or closed");
        }
        
        return result;
    }
    
    /**
     * @brief Submit a task without waiting for the result
     * 
     * This method is useful for fire-and-forget tasks.
     * 
     * @tparam F Function type
     * @tparam Args Argument types
     * @param f Function to execute
     * @param args Arguments to pass to the function
     * @return true if the task was submitted, false otherwise
     */
    template <typename F, typename... Args>
    bool execute(F&& f, Args&&... args) {
        // Check if the pool is running
        if (!running_) {
            return false;
        }
        
        // Create a task wrapper
        auto task = [f = std::forward<F>(f), args = std::make_tuple(std::forward<Args>(args)...), this]() {
            // Track active tasks
            active_tasks_.fetch_add(1, std::memory_order_relaxed);
            
            // Execute the task
            try {
                std::apply(f, args);
            } catch (...) {
                // Just catch here to prevent thread termination
            }
            
            // Decrement active tasks
            active_tasks_.fetch_sub(1, std::memory_order_relaxed);
        };
        
        // Push the task to the queue
        return task_queue_.push(std::move(task));
    }
    
    /**
     * @brief Wait for all tasks to complete
     * 
     * This method blocks until all submitted tasks have been completed.
     * The pool remains active after this call.
     */
    void wait_all() {
        while (active_tasks_.load(std::memory_order_relaxed) > 0 || !task_queue_.empty()) {
            std::this_thread::yield();
        }
    }
    
    /**
     * @brief Shutdown the thread pool
     * 
     * This method stops accepting new tasks, waits for all
     * submitted tasks to complete, and then joins all worker threads.
     * 
     * @param wait_for_tasks Whether to wait for all tasks to complete
     */
    void shutdown(bool wait_for_tasks = true) {
        if (!running_) {
            return;  // Already stopped
        }
        
        // Mark as not running
        running_ = false;
        
        // Close the task queue to prevent new tasks
        task_queue_.close();
        
        // Optionally wait for tasks to complete
        if (wait_for_tasks) {
            wait_all();
        }
        
        // Join all worker threads
        for (auto& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
        
        // Clear the workers
        workers_.clear();
    }
    
    /**
     * @brief Get the number of worker threads
     * 
     * @return Number of worker threads
     */
    size_t num_threads() const {
        return workers_.size();
    }
    
    /**
     * @brief Get the number of tasks waiting in the queue
     * 
     * @return Queue size
     */
    size_t queue_size() const {
        return task_queue_.size();
    }
    
    /**
     * @brief Get the number of tasks currently being processed
     * 
     * @return Number of active tasks
     */
    size_t active_tasks() const {
        return active_tasks_.load(std::memory_order_relaxed);
    }
    
    /**
     * @brief Check if the pool is running
     * 
     * @return true if the pool is running, false if stopped
     */
    bool is_running() const {
        return running_;
    }
    
private:
    /**
     * @brief Worker thread loop
     * 
     * This method runs on each worker thread and continuously
     * processes tasks from the queue until the pool is shutdown.
     * 
     * @param thread_name Name of the thread (for debugging)
     */
    void worker_loop(const std::string& thread_name) {
        // Set thread name (platform-specific implementation)
        set_thread_name(thread_name);
        
        while (running_ || !task_queue_.empty()) {
            // Try to get a task
            auto task_opt = task_queue_.pop();
            
            // Execute the task if we got one
            if (task_opt) {
                (*task_opt)();
            } else if (!running_) {
                // If no task and not running, exit
                break;
            }
        }
    }
    
    /**
     * @brief Set the name of the current thread (platform-specific)
     * 
     * @param name Thread name
     */
    void set_thread_name(const std::string& name) {
#if defined(__linux__)
        pthread_setname_np(pthread_self(), name.substr(0, 15).c_str());
#elif defined(__APPLE__)
        pthread_setname_np(name.substr(0, 63).c_str());
#endif
        // Windows requires a different approach using SetThreadDescription
        // Omitted for brevity
    }
    
    using Task = std::function<void()>;
    ConcQueue<Task> task_queue_;
    std::vector<std::thread> workers_;
    std::atomic<bool> running_;
    std::atomic<size_t> active_tasks_;
    
    // Prevent copying and moving
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;
    ThreadPool(ThreadPool&&) = delete;
    ThreadPool& operator=(ThreadPool&&) = delete;
};

/**
 * @brief Specialized thread pool with thread pinning capabilities
 * 
 * This thread pool allows pinning threads to specific CPU cores
 * for improved determinism and performance in latency-critical paths.
 */
class PinnedThreadPool : public ThreadPool {
public:
    /**
     * @brief Constructor with core affinity specification
     * 
     * @param core_ids Vector of CPU core IDs to pin threads to
     * @param queue_size Maximum size of the task queue (0 for unlimited)
     * @param thread_names Optional name prefix for worker threads
     */
    explicit PinnedThreadPool(
        const std::vector<int>& core_ids,
        size_t queue_size = 0,
        const std::string& thread_names = "PinnedWorker"
    ) : ThreadPool(core_ids.size(), queue_size, thread_names), core_ids_(core_ids) {
        // Pin threads to cores
        pin_threads();
    }
    
    /**
     * @brief Constructor with core range specification
     * 
     * @param start_core Starting CPU core ID
     * @param end_core Ending CPU core ID (inclusive)
     * @param queue_size Maximum size of the task queue (0 for unlimited)
     * @param thread_names Optional name prefix for worker threads
     */
    PinnedThreadPool(
        int start_core,
        int end_core,
        size_t queue_size = 0,
        const std::string& thread_names = "PinnedWorker"
    ) : ThreadPool(end_core - start_core + 1, queue_size, thread_names) {
        // Generate core IDs
        core_ids_.reserve(end_core - start_core + 1);
        for (int i = start_core; i <= end_core; ++i) {
            core_ids_.push_back(i);
        }
        
        // Pin threads to cores
        pin_threads();
    }
    
private:
    /**
     * @brief Pin worker threads to specified CPU cores
     */
    void pin_threads() {
        // Implementation depends on the platform
        // This is a placeholder - actual implementation would use:
        // - Linux: pthread_setaffinity_np
        // - Windows: SetThreadAffinityMask
        // - macOS: thread_policy_set
        
        // For now, just log that we would pin threads
        // (Implementation would go here in a real system)
    }
    
    std::vector<int> core_ids_;
};

} // namespace concurrency

#endif // THREAD_POOL_H
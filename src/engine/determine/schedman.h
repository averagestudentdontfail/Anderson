// schedman.h
// Scheduler Manager for the deterministic execution framework

#ifndef ENGINE_DETERMINE_SCHEDMAN_H
#define ENGINE_DETERMINE_SCHEDMAN_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>
#include <string>
#include <unordered_map>
#include <future>
#include <immintrin.h>  // For _mm_pause()

namespace engine {
namespace determine {

// Forward declarations
class SchedulingTask;
class DeterministicScheduler;
class SchedulerManager;

/**
 * @brief Execution mode for tasks
 */
enum class ExecutionMode {
    DETERMINISTIC,   // Task executes with deterministic timing
    ASYNCHRONOUS,    // Task executes asynchronously
    PRIORITY_HIGH,   // High priority task
    PRIORITY_NORMAL, // Normal priority task
    PRIORITY_LOW     // Low priority task
};

/**
 * @brief Task state
 */
enum class TaskState {
    CREATED,    // Task has been created but not scheduled
    SCHEDULED,  // Task has been scheduled but not started
    RUNNING,    // Task is currently running
    COMPLETED,  // Task has completed successfully
    FAILED,     // Task has failed
    CANCELLED   // Task has been cancelled
};

/**
 * @brief Task status
 */
struct TaskStatus {
    uint64_t taskId;
    TaskState state;
    uint64_t scheduledTime;   // When the task was scheduled (in cycles)
    uint64_t startTime;       // When the task started executing (in cycles)
    uint64_t completionTime;  // When the task completed (in cycles)
    int32_t resultCode;       // Result code (0 = success)
    double executionTimeMs;   // Execution time in milliseconds
};

/**
 * @brief Interface for schedulable tasks
 */
class SchedulingTask {
public:
    virtual ~SchedulingTask() = default;
    
    /**
     * @brief Execute the task
     * @return Result code (0 = success)
     */
    virtual int32_t execute() = 0;
    
    /**
     * @brief Cancel the task
     * @return True if cancellation was successful
     */
    virtual bool cancel() { return false; }
    
    /**
     * @brief Get the task ID
     * @return Task ID
     */
    uint64_t getTaskId() const { return taskId_; }
    
    /**
     * @brief Get the task name
     * @return Task name
     */
    const std::string& getName() const { return name_; }
    
    /**
     * @brief Get the task priority
     * @return Task priority
     */
    ExecutionMode getMode() const { return mode_; }
    
    /**
     * @brief Set the task ID
     * @param id Task ID
     */
    void setTaskId(uint64_t id) { taskId_ = id; }
    
    /**
     * @brief Set the task name
     * @param name Task name
     */
    void setName(const std::string& name) { name_ = name; }
    
    /**
     * @brief Set the task mode
     * @param mode Task mode
     */
    void setMode(ExecutionMode mode) { mode_ = mode; }
    
    /**
     * @brief Wait for the task to complete
     * @param timeoutMs Timeout in milliseconds (0 = wait forever)
     * @return True if the task completed, false if timed out
     */
    bool waitForCompletion(uint64_t timeoutMs = 0);
    
    /**
     * @brief Set the task status
     * @param state New task state
     */
    void setState(TaskState state) { state_ = state; }
    
    /**
     * @brief Get the task state
     * @return Task state
     */
    TaskState getState() const { return state_; }
    
    /**
     * @brief Set the result code
     * @param code Result code
     */
    void setResultCode(int32_t code) { resultCode_ = code; }
    
    /**
     * @brief Get the result code
     * @return Result code
     */
    int32_t getResultCode() const { return resultCode_; }
    
    /**
     * @brief Get the task status
     * @return Task status
     */
    TaskStatus getStatus() const;
    
protected:
    uint64_t taskId_ = 0;
    std::string name_;
    ExecutionMode mode_ = ExecutionMode::DETERMINISTIC;
    std::atomic<TaskState> state_{TaskState::CREATED};
    int32_t resultCode_ = 0;
    std::promise<void> completion_;
    uint64_t scheduledCycle_ = 0;
    uint64_t startCycle_ = 0;
    uint64_t completionCycle_ = 0;
    double executionTimeMs_ = 0.0;
};

/**
 * @brief Function-based task implementation
 */
class FunctionTask : public SchedulingTask {
public:
    /**
     * @brief Create a task from a function
     * @param func Function to execute
     */
    explicit FunctionTask(std::function<int32_t()> func) : func_(func) {}
    
    /**
     * @brief Execute the task
     * @return Result code from the function
     */
    int32_t execute() override {
        if (func_) {
            return func_();
        }
        return -1;
    }
    
private:
    std::function<int32_t()> func_;
};

/**
 * @brief Deterministic scheduler for timing-critical tasks
 * 
 * Executes tasks with predictable timing and consistent execution patterns.
 */
class DeterministicScheduler {
public:
    /**
     * @brief Create a deterministic scheduler
     * @param cycleTimeNs Duration of a single execution cycle in nanoseconds
     * @param maxTasksPerCycle Maximum number of tasks to execute in a single cycle
     */
    DeterministicScheduler(uint64_t cycleTimeNs = 1000000, uint32_t maxTasksPerCycle = 16);
    
    /**
     * @brief Destructor - stops the scheduler if still running
     */
    ~DeterministicScheduler();
    
    /**
     * @brief Start the scheduler
     * @param pinToCore Core ID to pin the scheduler thread to (-1 = don't pin)
     * @return True if started successfully
     */
    bool start(int pinToCore = -1);
    
    /**
     * @brief Stop the scheduler
     */
    void stop();
    
    /**
     * @brief Schedule a task for execution
     * @param task Task to schedule
     * @param targetCycle Target cycle to execute the task (0 = next available)
     * @return Task ID if scheduled successfully, 0 otherwise
     */
    uint64_t scheduleTask(std::shared_ptr<SchedulingTask> task, uint64_t targetCycle = 0);
    
    /**
     * @brief Cancel a scheduled task
     * @param taskId ID of the task to cancel
     * @return True if cancelled successfully
     */
    bool cancelTask(uint64_t taskId);
    
    /**
     * @brief Get the status of a task
     * @param taskId ID of the task
     * @return Task status
     */
    TaskStatus getTaskStatus(uint64_t taskId);
    
    /**
     * @brief Get the current cycle count
     * @return Current cycle count
     */
    uint64_t getCurrentCycle() const { return currentCycle_.load(std::memory_order_acquire); }
    
    /**
     * @brief Get the scheduler statistics
     * @return String containing scheduler statistics
     */
    std::string getStatistics() const;
    
    /**
     * @brief Check if the scheduler is running
     * @return True if the scheduler is running
     */
    bool isRunning() const { return running_.load(std::memory_order_acquire); }
    
    /**
     * @brief Wait for the scheduler to complete all tasks
     * @param timeoutMs Timeout in milliseconds (0 = wait forever)
     * @return True if all tasks completed, false if timed out
     */
    bool waitForCompletion(uint64_t timeoutMs = 0);

private:
    // Task queue entry for scheduling
    struct TaskQueueEntry {
        std::shared_ptr<SchedulingTask> task;
        uint64_t targetCycle;
        
        // Comparison operator for priority queue (earlier cycle = higher priority)
        bool operator>(const TaskQueueEntry& other) const {
            return targetCycle > other.targetCycle;
        }
    };
    
    // Scheduler state and control
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};
    std::atomic<uint64_t> currentCycle_{0};
    std::atomic<uint64_t> nextTaskId_{1};
    
    // Scheduler configuration
    uint64_t cycleTimeNs_;
    uint32_t maxTasksPerCycle_;
    
    // Task management
    std::priority_queue<TaskQueueEntry, std::vector<TaskQueueEntry>, std::greater<>> taskQueue_;
    std::unordered_map<uint64_t, std::shared_ptr<SchedulingTask>> activeTasks_;
    std::mutex taskMutex_;
    std::condition_variable taskCondition_;
    
    // Scheduler thread
    std::thread schedulerThread_;
    
    // Statistics
    struct Statistics {
        uint64_t totalTasksExecuted{0};
        uint64_t totalCyclesExecuted{0};
        uint64_t overrunCycles{0};
        uint64_t idleCycles{0};
        uint64_t peakTasksInCycle{0};
        double avgTasksPerCycle{0.0};
        double avgCycleTimeNs{0.0};
    };
    mutable std::mutex statsMutex_;
    Statistics stats_;
    
    /**
     * @brief Main scheduler thread function
     */
    void schedulerThreadFunc();
    
    /**
     * @brief Execute a single cycle of the scheduler
     * @return Number of tasks executed in this cycle
     */
    uint32_t executeCycle();
    
    /**
     * @brief Pin the current thread to a specific CPU core
     * @param coreId Core ID to pin to
     * @return True if pinned successfully
     */
    bool pinThreadToCore(int coreId);
    
    /**
     * @brief Get the current time in nanoseconds
     * @return Current time in nanoseconds
     */
    static uint64_t getCurrentTimeNs();
};

/**
 * @brief Manager for multiple schedulers
 * 
 * Provides centralized management of multiple deterministic schedulers
 * for different execution domains.
 */
class SchedulerManager {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the scheduler manager
     */
    static SchedulerManager& getInstance();
    
    /**
     * @brief Initialize the scheduler manager
     * @param numSchedulers Number of schedulers to create
     * @param cycleTimeNs Duration of a single execution cycle in nanoseconds
     * @param maxTasksPerCycle Maximum number of tasks to execute in a single cycle
     */
    void initialize(size_t numSchedulers = 1, 
                   uint64_t cycleTimeNs = 1000000, 
                   uint32_t maxTasksPerCycle = 16);
    
    /**
     * @brief Start all schedulers
     * @param pinToCores Vector of core IDs to pin schedulers to (-1 = don't pin)
     * @return True if all schedulers started successfully
     */
    bool startAll(const std::vector<int>& pinToCores = {});
    
    /**
     * @brief Stop all schedulers
     */
    void stopAll();
    
    /**
     * @brief Get a scheduler by index
     * @param index Scheduler index
     * @return Shared pointer to the scheduler
     */
    std::shared_ptr<DeterministicScheduler> getScheduler(size_t index = 0);
    
    /**
     * @brief Schedule a task on a specific scheduler
     * @param task Task to schedule
     * @param schedulerIndex Index of the scheduler to use
     * @param targetCycle Target cycle to execute the task (0 = next available)
     * @return Task ID if scheduled successfully, 0 otherwise
     */
    uint64_t scheduleTask(std::shared_ptr<SchedulingTask> task, 
                        size_t schedulerIndex = 0,
                        uint64_t targetCycle = 0);
    
    /**
     * @brief Schedule a function as a task
     * @param func Function to execute
     * @param name Task name
     * @param mode Execution mode
     * @param schedulerIndex Index of the scheduler to use
     * @param targetCycle Target cycle to execute the task (0 = next available)
     * @return Task ID if scheduled successfully, 0 otherwise
     */
    uint64_t scheduleFunction(std::function<int32_t()> func,
                            const std::string& name,
                            ExecutionMode mode = ExecutionMode::DETERMINISTIC,
                            size_t schedulerIndex = 0,
                            uint64_t targetCycle = 0);
    
    /**
     * @brief Cancel a scheduled task
     * @param taskId ID of the task to cancel
     * @return True if cancelled successfully
     */
    bool cancelTask(uint64_t taskId);
    
    /**
     * @brief Get the status of a task
     * @param taskId ID of the task
     * @return Task status
     */
    TaskStatus getTaskStatus(uint64_t taskId);
    
    /**
     * @brief Wait for a task to complete
     * @param taskId ID of the task
     * @param timeoutMs Timeout in milliseconds (0 = wait forever)
     * @return True if the task completed, false if timed out
     */
    bool waitForTask(uint64_t taskId, uint64_t timeoutMs = 0);
    
    /**
     * @brief Wait for all schedulers to complete all tasks
     * @param timeoutMs Timeout in milliseconds (0 = wait forever)
     * @return True if all tasks completed, false if timed out
     */
    bool waitForAll(uint64_t timeoutMs = 0);
    
    /**
     * @brief Get scheduler statistics for all schedulers
     * @return String containing scheduler statistics
     */
    std::string getStatistics() const;

private:
    SchedulerManager() = default;
    ~SchedulerManager() = default;
    
    // Disable copy and move
    SchedulerManager(const SchedulerManager&) = delete;
    SchedulerManager& operator=(const SchedulerManager&) = delete;
    SchedulerManager(SchedulerManager&&) = delete;
    SchedulerManager& operator=(SchedulerManager&&) = delete;
    
    // Scheduler management
    std::vector<std::shared_ptr<DeterministicScheduler>> schedulers_;
    
    // Task ID to scheduler index mapping
    std::unordered_map<uint64_t, size_t> taskSchedulerMap_;
    std::mutex managerMutex_;
};

/**
 * @brief Timer for measuring execution time with high precision
 */
class HighResolutionTimer {
public:
    /**
     * @brief Start the timer
     */
    void start() {
        startTime_ = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief Stop the timer
     */
    void stop() {
        endTime_ = std::chrono::high_resolution_clock::now();
    }
    
    /**
     * @brief Get the elapsed time in nanoseconds
     * @return Elapsed time in nanoseconds
     */
    uint64_t elapsedNanos() const {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            endTime_ - startTime_).count();
    }
    
    /**
     * @brief Get the elapsed time in microseconds
     * @return Elapsed time in microseconds
     */
    uint64_t elapsedMicros() const {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            endTime_ - startTime_).count();
    }
    
    /**
     * @brief Get the elapsed time in milliseconds
     * @return Elapsed time in milliseconds
     */
    double elapsedMillis() const {
        return std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(
            endTime_ - startTime_).count();
    }
    
    /**
     * @brief Get the elapsed time in seconds
     * @return Elapsed time in seconds
     */
    double elapsedSeconds() const {
        return std::chrono::duration_cast<std::chrono::duration<double>>(
            endTime_ - startTime_).count();
    }
    
private:
    std::chrono::high_resolution_clock::time_point startTime_;
    std::chrono::high_resolution_clock::time_point endTime_;
};

/**
 * @brief Cycle controller for deterministic execution
 * 
 * Provides a fixed-time execution cycle with busy-wait for precise timing.
 */
class CycleController {
public:
    /**
     * @brief Create a cycle controller
     * @param cycleTimeNs Duration of a single execution cycle in nanoseconds
     */
    explicit CycleController(uint64_t cycleTimeNs = 1000000)
        : cycleTimeNs_(cycleTimeNs), cycleCount_(0) {}
    
    /**
     * @brief Start a new cycle
     */
    void startCycle() {
        cycleStart_ = __rdtsc();
        timer_.start();
    }
    
    /**
     * @brief Wait until the end of the current cycle
     * @return Actual cycle time in nanoseconds
     */
    uint64_t waitForCycleEnd() {
        // Busy-wait until the end of this cycle for deterministic timing
        const uint64_t TARGET_CYCLES = cycleTimeNs_ * 0.3; // Approximate CPU cycles
        while (__rdtsc() - cycleStart_ < TARGET_CYCLES) {
            _mm_pause(); // Reduce power consumption
        }
        
        // Fine-grained timing using sleep and busy-wait
        timer_.stop();
        uint64_t elapsed = timer_.elapsedNanos();
        
        if (elapsed < cycleTimeNs_) {
            uint64_t remainingNs = cycleTimeNs_ - elapsed;
            
            // For longer waits, use sleep
            if (remainingNs > 50000) { // 50 microseconds
                struct timespec ts;
                ts.tv_sec = 0;
                ts.tv_nsec = remainingNs - 50000; // Leave some margin
                nanosleep(&ts, nullptr);
            }
            
            // Busy-wait for the final microseconds
            auto start = std::chrono::high_resolution_clock::now();
            while (true) {
                auto now = std::chrono::high_resolution_clock::now();
                uint64_t busyElapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    now - start).count();
                if (busyElapsed + elapsed >= cycleTimeNs_) {
                    break;
                }
                _mm_pause();
            }
        }
        
        cycleCount_++;
        timer_.stop();
        return timer_.elapsedNanos();
    }
    
    /**
     * @brief Get the current cycle count
     * @return Current cycle count
     */
    uint64_t getCycleCount() const { return cycleCount_; }
    
    /**
     * @brief Set the cycle time
     * @param cycleTimeNs New cycle time in nanoseconds
     */
    void setCycleTimeNs(uint64_t cycleTimeNs) { cycleTimeNs_ = cycleTimeNs; }
    
private:
    uint64_t cycleTimeNs_;
    uint64_t cycleCount_;
    uint64_t cycleStart_;
    HighResolutionTimer timer_;
};

} 
} 

#endif 
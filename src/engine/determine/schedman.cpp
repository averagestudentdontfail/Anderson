#include "schedman.h"
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <pthread.h>
#include <sstream>
#include <iomanip>
#include <sys/time.h>
#include <unistd.h>

namespace engine {
namespace determine {

// SchedulingTask implementation
bool SchedulingTask::waitForCompletion(uint64_t timeoutMs) {
if (state_ == TaskState::COMPLETED || state_ == TaskState::FAILED || state_ == TaskState::CANCELLED) {
return true;
}

if (timeoutMs == 0) {
// Wait indefinitely
completion_.get_future().wait();
return true;
} else {
// Wait with timeout
auto status = completion_.get_future().wait_for(std::chrono::milliseconds(timeoutMs));
return status == std::future_status::ready;
}
}

TaskStatus SchedulingTask::getStatus() const {
TaskStatus status;
status.taskId = taskId_;
status.state = state_;
status.scheduledTime = scheduledCycle_;
status.startTime = startCycle_;
status.completionTime = completionCycle_;
status.resultCode = resultCode_;
status.executionTimeMs = executionTimeMs_;
return status;
}

// DeterministicScheduler implementation
DeterministicScheduler::DeterministicScheduler(uint64_t cycleTimeNs, uint32_t maxTasksPerCycle)
: cycleTimeNs_(cycleTimeNs), maxTasksPerCycle_(maxTasksPerCycle) {
}

DeterministicScheduler::~DeterministicScheduler() {
stop();
}

bool DeterministicScheduler::start(int pinToCore) {
if (running_.exchange(true)) {
// Already running
return true;
}

// Reset statistics
std::lock_guard<std::mutex> statsLock(statsMutex_);
stats_ = Statistics();

// Start the scheduler thread
try {
schedulerThread_ = std::thread(&DeterministicScheduler::schedulerThreadFunc, this);

// Pin the thread to the specified core if requested
if (pinToCore >= 0) {
if (!pinThreadToCore(pinToCore)) {
std::cerr << "Warning: Failed to pin scheduler thread to core " << pinToCore << std::endl;
}
}

return true;
} catch (const std::exception& e) {
std::cerr << "Error starting scheduler thread: " << e.what() << std::endl;
running_.store(false);
return false;
}
}

void DeterministicScheduler::stop() {
if (!running_.exchange(false)) {
// Already stopped
return;
}

// Wake up the scheduler thread
{
std::lock_guard<std::mutex> lock(taskMutex_);
taskCondition_.notify_all();
}

// Wait for the thread to exit
if (schedulerThread_.joinable()) {
schedulerThread_.join();
}
}

uint64_t DeterministicScheduler::scheduleTask(std::shared_ptr<SchedulingTask> task, uint64_t targetCycle) {
if (!task || !running_.load()) {
return 0;
}

std::lock_guard<std::mutex> lock(taskMutex_);

// Assign a task ID
uint64_t taskId = nextTaskId_.fetch_add(1);
task->setTaskId(taskId);

// Set the target cycle (0 means next available)
uint64_t currentCycle = currentCycle_.load();
uint64_t schedCycle = (targetCycle > 0) ? targetCycle : currentCycle + 1;

// Update task state
task->setState(TaskState::SCHEDULED);

// Add to the task queue
TaskQueueEntry entry{task, schedCycle};
taskQueue_.push(entry);

// Store the task in active tasks map
activeTasks_[taskId] = task;

// Signal the scheduler thread
taskCondition_.notify_one();

return taskId;
}

bool DeterministicScheduler::cancelTask(uint64_t taskId) {
std::lock_guard<std::mutex> lock(taskMutex_);

auto it = activeTasks_.find(taskId);
if (it == activeTasks_.end()) {
return false;
}

auto task = it->second;
if (task->getState() == TaskState::RUNNING) {
// Can't cancel a running task
return false;
}

// Mark as cancelled
task->setState(TaskState::CANCELLED);

// Resolve the task's future
try {
task->completion_.set_value();
} catch (...) {
// Ignore if promise already satisfied
}

// Remove from active tasks
activeTasks_.erase(it);

return true;
}

TaskStatus DeterministicScheduler::getTaskStatus(uint64_t taskId) {
std::lock_guard<std::mutex> lock(taskMutex_);

auto it = activeTasks_.find(taskId);
if (it == activeTasks_.end()) {
TaskStatus emptyStatus;
emptyStatus.taskId = taskId;
emptyStatus.state = TaskState::CANCELLED;  // Default for unknown tasks
return emptyStatus;
}

return it->second->getStatus();
}

std::string DeterministicScheduler::getStatistics() const {
std::lock_guard<std::mutex> lock(statsMutex_);

std::ostringstream oss;
oss << "Scheduler Statistics:" << std::endl;
oss << "  Total tasks executed: " << stats_.totalTasksExecuted << std::endl;
oss << "  Total cycles executed: " << stats_.totalCyclesExecuted << std::endl;
oss << "  Overrun cycles: " << stats_.overrunCycles << std::endl;
oss << "  Idle cycles: " << stats_.idleCycles << std::endl;
oss << "  Peak tasks in cycle: " << stats_.peakTasksInCycle << std::endl;

if (stats_.totalCyclesExecuted > 0) {
oss << "  Average tasks per cycle: " << std::fixed << std::setprecision(2)
<< stats_.avgTasksPerCycle << std::endl;
oss << "  Average cycle time: " << std::fixed << std::setprecision(2)
<< stats_.avgCycleTimeNs / 1000.0 << " µs" << std::endl;
}

return oss.str();
}

bool DeterministicScheduler::waitForCompletion(uint64_t timeoutMs) {
auto startTime = std::chrono::steady_clock::now();

while (running_.load()) {
// Check if task queue is empty
{
std::lock_guard<std::mutex> lock(taskMutex_);
if (taskQueue_.empty() && activeTasks_.empty()) {
return true;
}
}

// Check timeout
if (timeoutMs > 0) {
auto now = std::chrono::steady_clock::now();
auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
if (elapsed >= timeoutMs) {
return false;
}
}

// Sleep briefly to avoid spinning
std::this_thread::sleep_for(std::chrono::milliseconds(1));
}

return false;
}

void DeterministicScheduler::schedulerThreadFunc() {
CycleController cycleController(cycleTimeNs_);

while (running_.load()) {
// Start a new cycle
cycleController.startCycle();

// Update current cycle count
currentCycle_.store(cycleController.getCycleCount(), std::memory_order_release);

// Execute tasks for this cycle
uint32_t tasksExecuted = executeCycle();

// Wait until the end of the cycle
uint64_t actualCycleTimeNs = cycleController.waitForCycleEnd();

// Update statistics
{
std::lock_guard<std::mutex> statsLock(statsMutex_);
stats_.totalTasksExecuted += tasksExecuted;
stats_.totalCyclesExecuted++;
stats_.peakTasksInCycle = std::max(stats_.peakTasksInCycle, (uint64_t)tasksExecuted);

if (tasksExecuted == 0) {
stats_.idleCycles++;
}

if (actualCycleTimeNs > cycleTimeNs_) {
stats_.overrunCycles++;
}

// Update averages
stats_.avgTasksPerCycle = (double)stats_.totalTasksExecuted / stats_.totalCyclesExecuted;
stats_.avgCycleTimeNs = ((stats_.avgCycleTimeNs * (stats_.totalCyclesExecuted - 1)) + 
      actualCycleTimeNs) / stats_.totalCyclesExecuted;
}

// If no tasks and not running anymore, exit
if (tasksExecuted == 0 && !running_.load()) {
break;
}
}
}

uint32_t DeterministicScheduler::executeCycle() {
uint32_t executedCount = 0;
uint64_t cycle = currentCycle_.load();
std::vector<std::shared_ptr<SchedulingTask>> tasksToExecute;

// Collect tasks that are due for this cycle
{
std::unique_lock<std::mutex> lock(taskMutex_);

// Check if we have any tasks in the queue
if (taskQueue_.empty()) {
// Wait for new tasks or timeout
taskCondition_.wait_for(lock, std::chrono::milliseconds(1), 
[this]{ return !taskQueue_.empty() || !running_.load(); });

// Check if we're still running
if (!running_.load()) {
return 0;
}

// Recheck if we have any tasks
if (taskQueue_.empty()) {
return 0;
}
}

// Get tasks for this cycle
while (!taskQueue_.empty() && executedCount < maxTasksPerCycle_) {
const TaskQueueEntry& entry = taskQueue_.top();

// Check if this task is due for this cycle
if (entry.targetCycle > cycle) {
break;  // Not yet due
}

// Move the task to the execution list
if (entry.task && entry.task->getState() == TaskState::SCHEDULED) {
tasksToExecute.push_back(entry.task);
executedCount++;
}

// Remove from queue
taskQueue_.pop();
}
}

// Execute tasks
for (auto& task : tasksToExecute) {
// Update task state and timestamps
task->setState(TaskState::RUNNING);
task->startCycle_ = cycle;

// Measure execution time
auto startTime = std::chrono::high_resolution_clock::now();

// Execute the task
int32_t result = task->execute();

// Calculate execution time
auto endTime = std::chrono::high_resolution_clock::now();
task->executionTimeMs_ = std::chrono::duration<double, std::milli>(endTime - startTime).count();

// Update task state
task->setResultCode(result);
task->setState(result == 0 ? TaskState::COMPLETED : TaskState::FAILED);
task->completionCycle_ = cycle;

// Signal completion
try {
task->completion_.set_value();
} catch (...) {
// Ignore if promise already satisfied
}

// Remove from active tasks if completed
{
std::lock_guard<std::mutex> lock(taskMutex_);
auto it = activeTasks_.find(task->getTaskId());
if (it != activeTasks_.end()) {
activeTasks_.erase(it);
}
}
}

return executedCount;
}

bool DeterministicScheduler::pinThreadToCore(int coreId) {
cpu_set_t cpuset;
CPU_ZERO(&cpuset);
CPU_SET(coreId, &cpuset);

int result = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
if (result != 0) {
std::cerr << "Failed to pin thread to core: " << strerror(result) << std::endl;
return false;
}

// Verify that we were actually pinned to the requested core
cpu_set_t check_cpuset;
CPU_ZERO(&check_cpuset);
if (pthread_getaffinity_np(pthread_self(), sizeof(cpu_set_t), &check_cpuset) == 0) {
if (!CPU_ISSET(coreId, &check_cpuset)) {
std::cerr << "Warning: Failed to verify thread pinning to core " << coreId << std::endl;
return false;
}
}

return true;
}

uint64_t DeterministicScheduler::getCurrentTimeNs() {
struct timespec ts;
if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
// Fallback if clock_gettime fails
return time(nullptr) * 1000000000ULL;
}
return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + 
static_cast<uint64_t>(ts.tv_nsec);
}

// SchedulerManager implementation
SchedulerManager& SchedulerManager::getInstance() {
static SchedulerManager instance;
return instance;
}

void SchedulerManager::initialize(size_t numSchedulers, uint64_t cycleTimeNs, uint32_t maxTasksPerCycle) {
std::lock_guard<std::mutex> lock(managerMutex_);

// Stop any existing schedulers
stopAll();

// Clear existing schedulers
schedulers_.clear();
taskSchedulerMap_.clear();

// Create new schedulers
for (size_t i = 0; i < numSchedulers; ++i) {
schedulers_.push_back(std::make_shared<DeterministicScheduler>(cycleTimeNs, maxTasksPerCycle));
}
}

bool SchedulerManager::startAll(const std::vector<int>& pinToCores) {
std::lock_guard<std::mutex> lock(managerMutex_);

bool allStarted = true;

for (size_t i = 0; i < schedulers_.size(); ++i) {
int coreId = -1;
if (i < pinToCores.size()) {
coreId = pinToCores[i];
}

if (!schedulers_[i]->start(coreId)) {
allStarted = false;
}
}

return allStarted;
}

void SchedulerManager::stopAll() {
std::lock_guard<std::mutex> lock(managerMutex_);

for (auto& scheduler : schedulers_) {
scheduler->stop();
}
}

std::shared_ptr<DeterministicScheduler> SchedulerManager::getScheduler(size_t index) {
std::lock_guard<std::mutex> lock(managerMutex_);

if (index >= schedulers_.size()) {
return nullptr;
}

return schedulers_[index];
}

uint64_t SchedulerManager::scheduleTask(std::shared_ptr<SchedulingTask> task, 
    size_t schedulerIndex,
    uint64_t targetCycle) {
if (!task) {
return 0;
}

std::lock_guard<std::mutex> lock(managerMutex_);

if (schedulerIndex >= schedulers_.size() || !schedulers_[schedulerIndex]) {
return 0;
}

// Schedule the task
uint64_t taskId = schedulers_[schedulerIndex]->scheduleTask(task, targetCycle);

if (taskId > 0) {
// Record which scheduler this task is on
taskSchedulerMap_[taskId] = schedulerIndex;
}

return taskId;
}

uint64_t SchedulerManager::scheduleFunction(std::function<int32_t()> func,
            const std::string& name,
            ExecutionMode mode,
            size_t schedulerIndex,
            uint64_t targetCycle) {
if (!func) {
return 0;
}

// Create a function task
auto task = std::make_shared<FunctionTask>(func);
task->setName(name);
task->setMode(mode);

// Schedule it
return scheduleTask(task, schedulerIndex, targetCycle);
}

bool SchedulerManager::cancelTask(uint64_t taskId) {
std::lock_guard<std::mutex> lock(managerMutex_);

// Find which scheduler has this task
auto it = taskSchedulerMap_.find(taskId);
if (it == taskSchedulerMap_.end()) {
return false;
}

size_t schedulerIndex = it->second;
if (schedulerIndex >= schedulers_.size() || !schedulers_[schedulerIndex]) {
return false;
}

// Cancel the task
bool result = schedulers_[schedulerIndex]->cancelTask(taskId);

if (result) {
// Remove from our map
taskSchedulerMap_.erase(it);
}

return result;
}

TaskStatus SchedulerManager::getTaskStatus(uint64_t taskId) {
std::lock_guard<std::mutex> lock(managerMutex_);

// Find which scheduler has this task
auto it = taskSchedulerMap_.find(taskId);
if (it == taskSchedulerMap_.end()) {
// Task not found
TaskStatus emptyStatus;
emptyStatus.taskId = taskId;
emptyStatus.state = TaskState::CANCELLED;  // Default for unknown tasks
return emptyStatus;
}

size_t schedulerIndex = it->second;
if (schedulerIndex >= schedulers_.size() || !schedulers_[schedulerIndex]) {
// Invalid scheduler
TaskStatus emptyStatus;
emptyStatus.taskId = taskId;
emptyStatus.state = TaskState::CANCELLED;
return emptyStatus;
}

// Get the task status
return schedulers_[schedulerIndex]->getTaskStatus(taskId);
}

bool SchedulerManager::waitForTask(uint64_t taskId, uint64_t timeoutMs) {
std::shared_ptr<DeterministicScheduler> scheduler;

// Find which scheduler has this task
{
std::lock_guard<std::mutex> lock(managerMutex_);

auto it = taskSchedulerMap_.find(taskId);
if (it == taskSchedulerMap_.end()) {
return false;
}

size_t schedulerIndex = it->second;
if (schedulerIndex >= schedulers_.size()) {
return false;
}

scheduler = schedulers_[schedulerIndex];
}

if (!scheduler) {
return false;
}

// Get the task status
TaskStatus status = scheduler->getTaskStatus(taskId);

// Check if already completed
if (status.state == TaskState::COMPLETED || 
status.state == TaskState::FAILED || 
status.state == TaskState::CANCELLED) {
return true;
}

// Wait for completion
auto startTime = std::chrono::steady_clock::now();

while (true) {
// Check if task completed
status = scheduler->getTaskStatus(taskId);
if (status.state == TaskState::COMPLETED || 
status.state == TaskState::FAILED || 
status.state == TaskState::CANCELLED) {
return true;
}

// Check timeout
if (timeoutMs > 0) {
auto now = std::chrono::steady_clock::now();
auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
now - startTime).count();

if (elapsed >= timeoutMs) {
return false;
}
}

// Sleep briefly to avoid spinning
std::this_thread::sleep_for(std::chrono::milliseconds(1));
}
}

bool SchedulerManager::waitForAll(uint64_t timeoutMs) {
auto startTime = std::chrono::steady_clock::now();
std::vector<std::shared_ptr<DeterministicScheduler>> schedulersCopy;

// Make a copy of the schedulers to avoid holding the lock
{
std::lock_guard<std::mutex> lock(managerMutex_);
schedulersCopy = schedulers_;
}

for (auto& scheduler : schedulersCopy) {
if (!scheduler) {
continue;
}

// Calculate remaining timeout
uint64_t remainingMs = timeoutMs;
if (timeoutMs > 0) {
auto now = std::chrono::steady_clock::now();
auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
now - startTime).count();

if (elapsed >= timeoutMs) {
return false;
}

remainingMs = timeoutMs - elapsed;
}

// Wait for this scheduler
if (!scheduler->waitForCompletion(remainingMs)) {
return false;
}
}

return true;
}

std::string SchedulerManager::getStatistics() const {
std::vector<std::shared_ptr<DeterministicScheduler>> schedulersCopy;

// Make a copy of the schedulers to avoid holding the lock
{
std::lock_guard<std::mutex> lock(managerMutex_);
schedulersCopy = schedulers_;
}

std::ostringstream oss;
oss << "Scheduler Manager Statistics:" << std::endl;
oss << "Number of schedulers: " << schedulersCopy.size() << std::endl;

for (size_t i = 0; i < schedulersCopy.size(); ++i) {
if (!schedulersCopy[i]) {
continue;
}

oss << std::endl << "Scheduler " << i << ":" << std::endl;
oss << schedulersCopy[i]->getStatistics();
}

return oss.str();
}

} 
} 
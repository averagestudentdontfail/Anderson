#include "harcount.h"
#include <iostream>
#include <algorithm>
#include <cstring>
#include <unordered_map>

namespace engine {
namespace system {

// Static mapping of counter types to human-readable names
static const std::unordered_map<CounterType, std::string> counterTypeNames = {
    {CounterType::CYCLES, "CPU Cycles"},
    {CounterType::INSTRUCTIONS, "Instructions"},
    {CounterType::CACHE_REFERENCES, "Cache References"},
    {CounterType::CACHE_MISSES, "Cache Misses"},
    {CounterType::BRANCH_INSTRUCTIONS, "Branch Instructions"},
    {CounterType::BRANCH_MISSES, "Branch Mispredictions"},
    {CounterType::BUS_CYCLES, "Bus Cycles"},
    {CounterType::L1D_LOADS, "L1 Data Cache Loads"},
    {CounterType::L1D_LOAD_MISSES, "L1 Data Cache Load Misses"},
    {CounterType::L1D_STORES, "L1 Data Cache Stores"},
    {CounterType::L1D_STORE_MISSES, "L1 Data Cache Store Misses"},
    {CounterType::L1I_LOAD_MISSES, "L1 Instruction Cache Load Misses"},
    {CounterType::LLC_LOADS, "Last Level Cache Loads"},
    {CounterType::LLC_LOAD_MISSES, "Last Level Cache Load Misses"},
    {CounterType::LLC_STORES, "Last Level Cache Stores"},
    {CounterType::LLC_STORE_MISSES, "Last Level Cache Store Misses"},
    {CounterType::DTLB_LOAD_MISSES, "Data TLB Load Misses"},
    {CounterType::ITLB_LOAD_MISSES, "Instruction TLB Load Misses"},
    {CounterType::PAGE_FAULTS, "Page Faults"},
    {CounterType::CONTEXT_SWITCHES, "Context Switches"},
    {CounterType::CPU_MIGRATIONS, "CPU Migrations"},
    {CounterType::ALIGNMENT_FAULTS, "Alignment Faults"},
    {CounterType::EMULATION_FAULTS, "Emulation Faults"}
};

// Static mapping of counter types to perf_event_attr config values for Linux
#ifdef __linux__
static const std::unordered_map<CounterType, uint64_t> counterConfigs = {
    {CounterType::CYCLES, PERF_COUNT_HW_CPU_CYCLES},
    {CounterType::INSTRUCTIONS, PERF_COUNT_HW_INSTRUCTIONS},
    {CounterType::CACHE_REFERENCES, PERF_COUNT_HW_CACHE_REFERENCES},
    {CounterType::CACHE_MISSES, PERF_COUNT_HW_CACHE_MISSES},
    {CounterType::BRANCH_INSTRUCTIONS, PERF_COUNT_HW_BRANCH_INSTRUCTIONS},
    {CounterType::BRANCH_MISSES, PERF_COUNT_HW_BRANCH_MISSES},
    {CounterType::BUS_CYCLES, PERF_COUNT_HW_BUS_CYCLES},
    {CounterType::L1D_LOADS, 
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {CounterType::L1D_LOAD_MISSES, 
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {CounterType::L1D_STORES, 
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {CounterType::L1D_STORE_MISSES, 
        PERF_COUNT_HW_CACHE_L1D | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {CounterType::L1I_LOAD_MISSES, 
        PERF_COUNT_HW_CACHE_L1I | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {CounterType::LLC_LOADS, 
        PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {CounterType::LLC_LOAD_MISSES, 
        PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {CounterType::LLC_STORES, 
        PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_ACCESS << 16)},
    {CounterType::LLC_STORE_MISSES, 
        PERF_COUNT_HW_CACHE_LL | (PERF_COUNT_HW_CACHE_OP_WRITE << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {CounterType::DTLB_LOAD_MISSES, 
        PERF_COUNT_HW_CACHE_DTLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {CounterType::ITLB_LOAD_MISSES, 
        PERF_COUNT_HW_CACHE_ITLB | (PERF_COUNT_HW_CACHE_OP_READ << 8) | 
        (PERF_COUNT_HW_CACHE_RESULT_MISS << 16)},
    {CounterType::PAGE_FAULTS, PERF_COUNT_SW_PAGE_FAULTS},
    {CounterType::CONTEXT_SWITCHES, PERF_COUNT_SW_CONTEXT_SWITCHES},
    {CounterType::CPU_MIGRATIONS, PERF_COUNT_SW_CPU_MIGRATIONS},
    {CounterType::ALIGNMENT_FAULTS, PERF_COUNT_SW_ALIGNMENT_FAULTS},
    {CounterType::EMULATION_FAULTS, PERF_COUNT_SW_EMULATION_FAULTS}
};
#endif

HardwareCounter::HardwareCounter(CounterType type, const std::string& name)
    : type_(type),
      name_(name.empty() ? getCounterName(type) : name),
      running_(false),
      start_value_(0),
      file_descriptor_(-1) {
#ifdef __linux__
    // Initialize the perf_event_attr structure
    memset(&attr_, 0, sizeof(attr_));
    attr_.type = PERF_TYPE_HARDWARE;
    attr_.size = sizeof(attr_);
    attr_.config = getCounterConfig(type);
    attr_.disabled = 1;  // Start in disabled state
    attr_.exclude_kernel = 1;  // Don't count kernel events
    attr_.exclude_hv = 1;      // Don't count hypervisor events
    
    // For some types, we need to adjust the type attribute
    if (type == CounterType::PAGE_FAULTS || 
        type == CounterType::CONTEXT_SWITCHES ||
        type == CounterType::CPU_MIGRATIONS ||
        type == CounterType::ALIGNMENT_FAULTS ||
        type == CounterType::EMULATION_FAULTS) {
        attr_.type = PERF_TYPE_SOFTWARE;
    } else if (type == CounterType::L1D_LOADS ||
               type == CounterType::L1D_LOAD_MISSES ||
               type == CounterType::L1D_STORES ||
               type == CounterType::L1D_STORE_MISSES ||
               type == CounterType::L1I_LOAD_MISSES ||
               type == CounterType::LLC_LOADS ||
               type == CounterType::LLC_LOAD_MISSES ||
               type == CounterType::LLC_STORES ||
               type == CounterType::LLC_STORE_MISSES ||
               type == CounterType::DTLB_LOAD_MISSES ||
               type == CounterType::ITLB_LOAD_MISSES) {
        attr_.type = PERF_TYPE_HW_CACHE;
    }
#endif
}

HardwareCounter::~HardwareCounter() {
    // Make sure to close the file descriptor
    if (file_descriptor_ >= 0) {
#ifdef __linux__
        close(file_descriptor_);
#endif
        file_descriptor_ = -1;
    }
}

bool HardwareCounter::start(pid_t pid, int cpu) {
    if (running_) {
        return true;  // Already running
    }
    
    // Setup the counter if not already done
    if (file_descriptor_ < 0) {
        if (!setupCounter(pid, cpu)) {
            return false;
        }
    }
    
    // Start the counter
#ifdef __linux__
    if (file_descriptor_ >= 0) {
        if (ioctl(file_descriptor_, PERF_EVENT_IOC_RESET, 0) < 0) {
            std::cerr << "Failed to reset counter: " << strerror(errno) << std::endl;
            return false;
        }
        
        if (ioctl(file_descriptor_, PERF_EVENT_IOC_ENABLE, 0) < 0) {
            std::cerr << "Failed to enable counter: " << strerror(errno) << std::endl;
            return false;
        }
        
        // Read the initial value
        CounterResult initial = read();
        start_value_ = initial.value;
        
        running_ = true;
        return true;
    }
#endif
    
    // Fallback for non-Linux platforms
    running_ = true;
    start_value_ = 0;
    return true;
}

bool HardwareCounter::stop() {
    if (!running_) {
        return true;  // Already stopped
    }
    
#ifdef __linux__
    if (file_descriptor_ >= 0) {
        if (ioctl(file_descriptor_, PERF_EVENT_IOC_DISABLE, 0) < 0) {
            std::cerr << "Failed to disable counter: " << strerror(errno) << std::endl;
            return false;
        }
        
        running_ = false;
        return true;
    }
#endif
    
    // Fallback for non-Linux platforms
    running_ = false;
    return true;
}

bool HardwareCounter::reset() {
    if (running_) {
        stop();
    }
    
#ifdef __linux__
    if (file_descriptor_ >= 0) {
        if (ioctl(file_descriptor_, PERF_EVENT_IOC_RESET, 0) < 0) {
            std::cerr << "Failed to reset counter: " << strerror(errno) << std::endl;
            return false;
        }
        
        start_value_ = 0;
        return true;
    }
#endif
    
    // Fallback for non-Linux platforms
    start_value_ = 0;
    return true;
}

CounterResult HardwareCounter::read() const {
    CounterResult result;
    result.value = 0;
    result.time_enabled = 0;
    result.time_running = 0;
    result.scaled_value = 0.0;
    
#ifdef __linux__
    if (file_descriptor_ >= 0) {
        struct read_format {
            uint64_t value;
            uint64_t time_enabled;
            uint64_t time_running;
            uint64_t id;
        };
        
        read_format data;
        if (::read(file_descriptor_, &data, sizeof(data)) == sizeof(data)) {
            result.value = data.value;
            result.time_enabled = data.time_enabled;
            result.time_running = data.time_running;
            
            // Calculate scaled value if counter was multiplexed
            if (data.time_enabled > 0 && data.time_running > 0 && data.time_running < data.time_enabled) {
                double scale = (double)data.time_enabled / data.time_running;
                result.scaled_value = data.value * scale;
            } else {
                result.scaled_value = data.value;
            }
        }
    } else {
        // Fallback for platforms without counter support
        // Generate some dummy values for testing
        using namespace std::chrono;
        static auto start_time = high_resolution_clock::now();
        auto now = high_resolution_clock::now();
        auto elapsed = duration_cast<nanoseconds>(now - start_time).count();
        
        // Simulate different counter types
        switch (type_) {
            case CounterType::CYCLES:
                result.value = elapsed * 3; // Simulate ~3GHz CPU
                break;
            case CounterType::INSTRUCTIONS:
                result.value = elapsed * 2; // Simulate ~2 IPC
                break;
            case CounterType::CACHE_REFERENCES:
                result.value = elapsed / 10;
                break;
            case CounterType::CACHE_MISSES:
                result.value = elapsed / 100;
                break;
            default:
                result.value = elapsed / 1000;
                break;
        }
        
        result.time_enabled = elapsed;
        result.time_running = elapsed;
        result.scaled_value = result.value;
    }
#endif
    
    // Adjust for relative measurement if we're tracking from start
    if (running_ && start_value_ > 0) {
        result.value -= start_value_;
        result.scaled_value -= start_value_;
    }
    
    return result;
}

bool HardwareCounter::isAvailable() {
#ifdef __linux__
    // Try to create a simple cycle counter to test availability
    struct perf_event_attr attr;
    memset(&attr, 0, sizeof(attr));
    attr.type = PERF_TYPE_HARDWARE;
    attr.size = sizeof(attr);
    attr.config = PERF_COUNT_HW_CPU_CYCLES;
    attr.disabled = 1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    
    int fd = syscall(__NR_perf_event_open, &attr, 0, -1, -1, 0);
    if (fd >= 0) {
        close(fd);
        return true;
    }
    return false;
#else
    // On non-Linux platforms, rely on platform-specific detection
    // For now, assume not available
    return false;
#endif
}

std::string HardwareCounter::getCounterName(CounterType type) {
    auto it = counterTypeNames.find(type);
    if (it != counterTypeNames.end()) {
        return it->second;
    }
    return "Unknown Counter";
}

bool HardwareCounter::setupCounter(pid_t pid, int cpu) {
#ifdef __linux__
    // Open the counter using perf_event_open system call
    file_descriptor_ = syscall(__NR_perf_event_open, &attr_, pid, cpu, -1, 0);
    if (file_descriptor_ < 0) {
        std::cerr << "Failed to open performance counter: " << strerror(errno) << std::endl;
        return false;
    }
    return true;
#else
    // On non-Linux platforms, implement platform-specific counter setup
    // For now, return true to allow simulated values
    return true;
#endif
}

uint64_t HardwareCounter::getCounterConfig(CounterType type) {
#ifdef __linux__
    auto it = counterConfigs.find(type);
    if (it != counterConfigs.end()) {
        return it->second;
    }
    // Default to CPU cycles if the type is not found
    return PERF_COUNT_HW_CPU_CYCLES;
#else
    // On non-Linux platforms, return a placeholder value
    return 0;
#endif
}

HardwareCounterManager& HardwareCounterManager::getInstance() {
    static HardwareCounterManager instance;
    return instance;
}

std::shared_ptr<HardwareCounter> HardwareCounterManager::createCounter(
    CounterType type, const std::string& name) {
    
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Generate a unique name if not provided
    std::string counter_name = name.empty() ? 
        HardwareCounter::getCounterName(type) + "_" + std::to_string(counters_.size()) : 
        name;
    
    // Create the counter
    auto counter = std::make_shared<HardwareCounter>(type, counter_name);
    
    // Store it
    counters_[counter_name] = counter;
    
    return counter;
}

std::shared_ptr<HardwareCounter> HardwareCounterManager::getCounter(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = counters_.find(name);
    if (it != counters_.end()) {
        return it->second;
    }
    
    return nullptr;
}

void HardwareCounterManager::startAll(pid_t pid, int cpu) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : counters_) {
        pair.second->start(pid, cpu);
    }
}

void HardwareCounterManager::stopAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : counters_) {
        pair.second->stop();
    }
}

void HardwareCounterManager::resetAll() {
    std::lock_guard<std::mutex> lock(mutex_);
    
    for (auto& pair : counters_) {
        pair.second->reset();
    }
}

std::map<std::string, CounterResult> HardwareCounterManager::readAll() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::map<std::string, CounterResult> results;
    for (const auto& pair : counters_) {
        results[pair.first] = pair.second->read();
    }
    
    return results;
}

ScopedHardwareCounter::ScopedHardwareCounter(std::shared_ptr<HardwareCounter> counter)
    : counter_(counter) {
    
    if (counter_) {
        counter_->reset();
        counter_->start();
    }
}

ScopedHardwareCounter::~ScopedHardwareCounter() {
    if (counter_) {
        counter_->stop();
        result_ = counter_->read();
    }
}

CounterResult ScopedHardwareCounter::getResult() const {
    if (counter_) {
        return counter_->read();
    }
    return result_;
}

} // namespace system
} // namespace engine
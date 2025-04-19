#include "procore.h"
#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

#ifdef __linux__
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <sys/sysinfo.h>
#include <sys/types.h>
#include <unistd.h>
#endif

#ifdef _WIN32
#include <processthreadsapi.h>
#include <windows.h>
#endif

namespace engine {
namespace system {

// Helper function to trim whitespace
static std::string trim(const std::string &str) {
  auto start = std::find_if_not(str.begin(), str.end(),
                                [](int c) { return std::isspace(c); });
  auto end = std::find_if_not(str.rbegin(), str.rend(), [](int c) {
               return std::isspace(c);
             }).base();
  return (start < end) ? std::string(start, end) : std::string();
}

// -----------------------------------------------------------------------------
// ProcessorCoreManager Implementation
// -----------------------------------------------------------------------------

ProcessorCoreManager &ProcessorCoreManager::getInstance() {
  static ProcessorCoreManager instance;
  return instance;
}

bool ProcessorCoreManager::initialize() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Clear any existing data
  cores_.clear();
  socket_to_cores_.clear();
  allocations_.clear();

  // Detect cores
  if (!detectCoresImpl()) {
    return false;
  }

  // Organize cores by socket
  organizeBySocket();

  // Detect core types
  detectCoreTypes();

  // Print core information for debugging
  // printCoreInfo();

  return true;
}

const CoreInfo *ProcessorCoreManager::getCoreInfo(int core_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  for (const auto &core : cores_) {
    if (core.id == core_id) {
      return &core;
    }
  }

  return nullptr;
}

bool ProcessorCoreManager::pinCurrentThread(int core_id) {
  // Get native handle for current thread
#ifdef __linux__
  pthread_t thread = pthread_self();
  return pinThreadImpl(thread, core_id);
#elif defined(_WIN32)
  HANDLE thread = GetCurrentThread();
  return pinThreadImpl(thread, core_id);
#else
  // Unsupported platform
  return false;
#endif
}

bool ProcessorCoreManager::pinThread(std::thread &thread, int core_id) {
  return pinThreadImpl(thread.native_handle(), core_id);
}

bool ProcessorCoreManager::pinThreadToMultipleCores(
    std::thread &thread, const std::vector<int> &core_ids) {
  return pinThreadToMultipleCoresImpl(thread.native_handle(), core_ids);
}

std::vector<int> ProcessorCoreManager::getPerformanceCoreIds() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int> performance_cores;
  for (const auto &core : cores_) {
    if (core.type == CoreType::PERFORMANCE) {
      performance_cores.push_back(core.id);
    }
  }

  return performance_cores;
}

std::vector<int> ProcessorCoreManager::getEfficiencyCoreIds() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::vector<int> efficiency_cores;
  for (const auto &core : cores_) {
    if (core.type == CoreType::EFFICIENCY) {
      efficiency_cores.push_back(core.id);
    }
  }

  return efficiency_cores;
}

std::vector<int> ProcessorCoreManager::getCoresShareL3Cache(int core_id) const {
  std::lock_guard<std::mutex> lock(mutex_);

  // Find the socket for this core
  int socket = -1;
  for (const auto &core : cores_) {
    if (core.id == core_id) {
      socket = core.socket;
      break;
    }
  }

  if (socket == -1) {
    return {}; // Core not found
  }

  // Get all cores in the same socket (sharing L3 cache)
  auto it = socket_to_cores_.find(socket);
  if (it != socket_to_cores_.end()) {
    return it->second;
  }

  return {};
}

size_t ProcessorCoreManager::getPhysicalCoreCount() const {
  std::lock_guard<std::mutex> lock(mutex_);

  std::set<int> physical_cores;
  for (const auto &core : cores_) {
    physical_cores.insert(core.physical_id);
  }

  return physical_cores.size();
}

int ProcessorCoreManager::reserveCore(const std::string &purpose,
                                      bool prefer_performance) {
  std::lock_guard<std::mutex> lock(mutex_);

  // First, try to get a core of the preferred type
  CoreType preferred_type =
      prefer_performance ? CoreType::PERFORMANCE : CoreType::EFFICIENCY;

  for (const auto &core : cores_) {
    if (core.type == preferred_type &&
        allocations_.find(core.id) == allocations_.end()) {
      // Found an unallocated core of the preferred type
      allocations_[core.id] = purpose;
      return core.id;
    }
  }

  // If no preferred cores available, try any core
  for (const auto &core : cores_) {
    if (allocations_.find(core.id) == allocations_.end()) {
      // Found an unallocated core
      allocations_[core.id] = purpose;
      return core.id;
    }
  }

  // No cores available
  return -1;
}

std::vector<int> ProcessorCoreManager::reserveCores(size_t count,
                                                    const std::string &purpose,
                                                    bool prefer_performance,
                                                    bool consecutive) {
  std::lock_guard<std::mutex> lock(mutex_);

  if (count == 0) {
    return {};
  }

  std::vector<int> result;

  if (consecutive) {
    // Try to find consecutive cores
    // Get all unallocated cores
    std::vector<int> unallocated;
    for (const auto &core : cores_) {
      if (allocations_.find(core.id) == allocations_.end()) {
        unallocated.push_back(core.id);
      }
    }

    // Sort by core ID
    std::sort(unallocated.begin(), unallocated.end());

    // Find a consecutive range of the required size
    for (size_t i = 0; i <= unallocated.size() - count; ++i) {
      bool is_consecutive = true;
      for (size_t j = 1; j < count; ++j) {
        if (unallocated[i + j] != unallocated[i + j - 1] + 1) {
          is_consecutive = false;
          break;
        }
      }

      if (is_consecutive) {
        // Found a consecutive range
        for (size_t j = 0; j < count; ++j) {
          int core_id = unallocated[i + j];
          allocations_[core_id] = purpose;
          result.push_back(core_id);
        }
        return result;
      }
    }
  }

  // If not consecutive or couldn't find consecutive range,
  // just get individual cores

  // First, try to get cores of the preferred type
  CoreType preferred_type =
      prefer_performance ? CoreType::PERFORMANCE : CoreType::EFFICIENCY;

  for (const auto &core : cores_) {
    if (core.type == preferred_type &&
        allocations_.find(core.id) == allocations_.end()) {
      // Found an unallocated core of the preferred type
      allocations_[core.id] = purpose;
      result.push_back(core.id);

      if (result.size() == count) {
        return result;
      }
    }
  }

  // If not enough preferred cores, get any cores
  for (const auto &core : cores_) {
    if (allocations_.find(core.id) == allocations_.end() &&
        std::find(result.begin(), result.end(), core.id) == result.end()) {
      // Found an unallocated core not already in result
      allocations_[core.id] = purpose;
      result.push_back(core.id);

      if (result.size() == count) {
        return result;
      }
    }
  }

  // Couldn't get enough cores, return what we have
  return result;
}

bool ProcessorCoreManager::releaseCore(int core_id) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = allocations_.find(core_id);
  if (it != allocations_.end()) {
    allocations_.erase(it);
    return true;
  }

  return false;
}

size_t ProcessorCoreManager::releaseAllCores(const std::string &purpose) {
  std::lock_guard<std::mutex> lock(mutex_);

  size_t count = 0;
  auto it = allocations_.begin();
  while (it != allocations_.end()) {
    if (it->second == purpose) {
      it = allocations_.erase(it);
      count++;
    } else {
      ++it;
    }
  }

  return count;
}

std::unordered_map<int, std::string>
ProcessorCoreManager::getAllocations() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return allocations_;
}

bool ProcessorCoreManager::pinThreadImpl(std::thread::native_handle_type thread,
                                         int core_id) {
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(core_id, &cpuset);

  int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  return result == 0;
#elif defined(_WIN32)
  DWORD_PTR mask = 1ULL << core_id;
  return SetThreadAffinityMask(thread, mask) != 0;
#else
  // Unsupported platform
  return false;
#endif
}

bool ProcessorCoreManager::pinThreadToMultipleCoresImpl(
    std::thread::native_handle_type thread, const std::vector<int> &core_ids) {
#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (int core_id : core_ids) {
    CPU_SET(core_id, &cpuset);
  }

  int result = pthread_setaffinity_np(thread, sizeof(cpu_set_t), &cpuset);
  return result == 0;
#elif defined(_WIN32)
  DWORD_PTR mask = 0;

  for (int core_id : core_ids) {
    mask |= (1ULL << core_id);
  }

  return SetThreadAffinityMask(thread, mask) != 0;
#else
  // Unsupported platform
  return false;
#endif
}

bool ProcessorCoreManager::detectCoresImpl() {
#ifdef __linux__
  // On Linux, read from /proc/cpuinfo
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (!cpuinfo) {
    std::cerr << "Failed to open /proc/cpuinfo" << std::endl;
    return false;
  }

  CoreInfo current_core;
  bool has_current_core = false;

  std::string line;
  while (std::getline(cpuinfo, line)) {
    if (line.empty()) {
      // Empty line means end of a processor entry
      if (has_current_core) {
        cores_.push_back(current_core);
        has_current_core = false;
      }
      current_core = CoreInfo();
      continue;
    }

    // Parse key-value pair
    size_t colon_pos = line.find(':');
    if (colon_pos != std::string::npos) {
      std::string key = trim(line.substr(0, colon_pos));
      std::string value = trim(line.substr(colon_pos + 1));

      if (key == "processor") {
        current_core.id = std::stoi(value);
        has_current_core = true;
      } else if (key == "physical id") {
        current_core.socket = std::stoi(value);
      } else if (key == "core id") {
        current_core.physical_id = std::stoi(value);
      } else if (key == "model name") {
        current_core.model_name = value;
      } else if (key == "cpu MHz") {
        current_core.max_frequency_mhz = std::stod(value);
      } else if (key == "siblings") {
        // This is the number of threads per core
      } else if (key == "cpu cores") {
        // This is the number of physical cores per socket
      }
    }
  }

  // Add the last core if any
  if (has_current_core) {
    cores_.push_back(current_core);
  }

  // If we couldn't find any cores, fall back to sysconf
  if (cores_.empty()) {
    long num_cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (num_cores > 0) {
      for (long i = 0; i < num_cores; i++) {
        CoreInfo core;
        core.id = static_cast<int>(i);
        core.socket = 0;
        core.physical_id = static_cast<int>(i);
        cores_.push_back(core);
      }
    } else {
      std::cerr << "Failed to determine number of cores" << std::endl;
      return false;
    }
  }

  // Try to find cache information
  for (auto &core : cores_) {
    std::stringstream path;
    path << "/sys/devices/system/cpu/cpu" << core.id << "/cache/";

    // Check if the directory exists
    DIR *dir = opendir(path.str().c_str());
    if (dir) {
      closedir(dir);

      // Find cache indices
      std::vector<int> indices;
      for (int i = 0; i < 4; i++) {
        std::stringstream index_path;
        index_path << path.str() << "index" << i;
        DIR *index_dir = opendir(index_path.str().c_str());
        if (index_dir) {
          indices.push_back(i);
          closedir(index_dir);
        }
      }

      // For each cache index, read information
      for (int index : indices) {
        std::stringstream level_path;
        level_path << path.str() << "index" << index << "/level";
        std::ifstream level_file(level_path.str());
        if (!level_file) {
          continue;
        }

        int level;
        level_file >> level;
        level_file.close();

        std::stringstream type_path;
        type_path << path.str() << "index" << index << "/type";
        std::ifstream type_file(type_path.str());
        if (!type_file) {
          continue;
        }

        std::string type;
        type_file >> type;
        type_file.close();

        std::stringstream size_path;
        size_path << path.str() << "index" << index << "/size";
        std::ifstream size_file(size_path.str());
        if (!size_file) {
          continue;
        }

        std::string size_str;
        size_file >> size_str;
        size_file.close();

        // Parse cache size (e.g., "32K" -> 32)
        int size_kb = 0;
        try {
          size_t pos = size_str.find_first_not_of("0123456789");
          int size = std::stoi(size_str.substr(0, pos));
          char unit = (pos != std::string::npos) ? size_str[pos] : 'K';

          if (unit == 'K' || unit == 'k') {
            size_kb = size;
          } else if (unit == 'M' || unit == 'm') {
            size_kb = size * 1024;
          }
        } catch (...) {
          continue;
        }

        // Assign to the appropriate cache field
        if (level == 1) {
          if (type == "Data") {
            core.l1d_cache_size_kb = size_kb;
          } else if (type == "Instruction") {
            core.l1i_cache_size_kb = size_kb;
          }
        } else if (level == 2) {
          core.l2_cache_size_kb = size_kb;
        } else if (level == 3) {
          core.l3_cache_size_kb = size_kb;
        }
      }
    }
  }

  // Find hyperthreading siblings
  for (auto &core : cores_) {
    std::stringstream topology_path;
    topology_path << "/sys/devices/system/cpu/cpu" << core.id
                  << "/topology/thread_siblings_list";
    std::ifstream siblings_file(topology_path.str());
    if (siblings_file) {
      std::string siblings_str;
      siblings_file >> siblings_str;
      siblings_file.close();

      // Parse the siblings list (e.g., "0,4" or "0-1")
      std::vector<int> siblings;
      size_t pos = 0;
      while (pos < siblings_str.length()) {
        size_t next_pos = siblings_str.find_first_of(",", pos);
        std::string token = siblings_str.substr(pos, next_pos - pos);

        // Check if it's a range
        size_t dash_pos = token.find('-');
        if (dash_pos != std::string::npos) {
          int start = std::stoi(token.substr(0, dash_pos));
          int end = std::stoi(token.substr(dash_pos + 1));
          for (int i = start; i <= end; i++) {
            siblings.push_back(i);
          }
        } else {
          siblings.push_back(std::stoi(token));
        }

        if (next_pos == std::string::npos) {
          break;
        }
        pos = next_pos + 1;
      }

      // Find the sibling that isn't this core
      for (int sibling : siblings) {
        if (sibling != core.id) {
          core.sibling = sibling;
          break;
        }
      }
    }
  }

  return !cores_.empty();
#elif defined(_WIN32)
  // On Windows, use GetLogicalProcessorInformation
  typedef BOOL(WINAPI * LPFN_GLPI)(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION,
                                   PDWORD);

  LPFN_GLPI glpi = (LPFN_GLPI)GetProcAddress(GetModuleHandle(TEXT("kernel32")),
                                             "GetLogicalProcessorInformation");

  if (glpi == NULL) {
    std::cerr << "GetLogicalProcessorInformation is not supported" << std::endl;
    return false;
  }

  DWORD length = 0;
  glpi(NULL, &length);

  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER) {
    std::cerr << "Error " << GetLastError()
              << " from GetLogicalProcessorInformation" << std::endl;
    return false;
  }

  std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(
      length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));

  if (!glpi(buffer.data(), &length)) {
    std::cerr << "Error " << GetLastError()
              << " from GetLogicalProcessorInformation" << std::endl;
    return false;
  }

  struct PhysicalCore {
    int socket;
    int id;
    DWORD_PTR mask;
    int l1d_cache_size_kb;
    int l1i_cache_size_kb;
    int l2_cache_size_kb;
    int l3_cache_size_kb;
  };

  std::vector<PhysicalCore> physical_cores;
  int next_core_id = 0;

  // Process the buffer to extract information
  for (DWORD i = 0; i < buffer.size(); i++) {
    const auto &info = buffer[i];

    switch (info.Relationship) {
    case RelationProcessorCore: {
      // Each processor core can have multiple logical processors
      DWORD_PTR mask = info.ProcessorMask;
      PhysicalCore core;
      core.socket = info.ProcessorCore.Flags & 1; // Socket ID
      core.id = next_core_id++;
      core.mask = mask;
      core.l1d_cache_size_kb = 0;
      core.l1i_cache_size_kb = 0;
      core.l2_cache_size_kb = 0;
      core.l3_cache_size_kb = 0;
      physical_cores.push_back(core);
    } break;

    case RelationCache: {
      // Cache information
      const auto &cache = info.Cache;
      DWORD_PTR mask = info.ProcessorMask;

      // Find all cores that share this cache
      for (auto &core : physical_cores) {
        if ((core.mask & mask) != 0) {
          // This core has access to this cache
          if (cache.Level == 1) {
            if (cache.Type == CacheData) {
              core.l1d_cache_size_kb = cache.Size / 1024;
            } else if (cache.Type == CacheInstruction) {
              core.l1i_cache_size_kb = cache.Size / 1024;
            }
          } else if (cache.Level == 2) {
            core.l2_cache_size_kb = cache.Size / 1024;
          } else if (cache.Level == 3) {
            core.l3_cache_size_kb = cache.Size / 1024;
          }
        }
      }
    } break;
    }
  }

  // Create logical cores
  cores_.clear();
  for (const auto &physical_core : physical_cores) {
    DWORD_PTR mask = physical_core.mask;

    // Count logical cores from mask
    int core_id = 0;
    while (mask != 0) {
      if (mask & 1) {
        CoreInfo core;
        core.id = core_id;
        core.socket = physical_core.socket;
        core.physical_id = physical_core.id;
        core.l1d_cache_size_kb = physical_core.l1d_cache_size_kb;
        core.l1i_cache_size_kb = physical_core.l1i_cache_size_kb;
        core.l2_cache_size_kb = physical_core.l2_cache_size_kb;
        core.l3_cache_size_kb = physical_core.l3_cache_size_kb;

        // Get processor info
        SYSTEM_INFO sysinfo;
        GetSystemInfo(&sysinfo);
        core.model_name = "Unknown";

        // Try to get frequency
        HKEY hKey;
        if (RegOpenKeyEx(HKEY_LOCAL_MACHINE,
                         "HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0",
                         0, KEY_READ, &hKey) == ERROR_SUCCESS) {
          DWORD mhz;
          DWORD dataSize = sizeof(mhz);
          if (RegQueryValueEx(hKey, "~MHz", NULL, NULL, (LPBYTE)&mhz,
                              &dataSize) == ERROR_SUCCESS) {
            core.max_frequency_mhz = mhz;
          }

          char processorName[64];
          dataSize = sizeof(processorName);
          if (RegQueryValueEx(hKey, "ProcessorNameString", NULL, NULL,
                              (LPBYTE)processorName,
                              &dataSize) == ERROR_SUCCESS) {
            core.model_name = processorName;
          }

          RegCloseKey(hKey);
        }

        cores_.push_back(core);
      }

      mask >>= 1;
      core_id++;
    }
  }

  // Find hyperthreading siblings
  for (size_t i = 0; i < cores_.size(); i++) {
    for (size_t j = i + 1; j < cores_.size(); j++) {
      if (cores_[i].physical_id == cores_[j].physical_id) {
        cores_[i].sibling = cores_[j].id;
        cores_[j].sibling = cores_[i].id;
      }
    }
  }

  return !cores_.empty();
#else
  // Generic fallback - assume one socket with all cores
  int num_cores = std::thread::hardware_concurrency();

  for (int i = 0; i < num_cores; i++) {
    CoreInfo core;
    core.id = i;
    core.socket = 0;
    core.physical_id = i;
    cores_.push_back(core);
  }

  return !cores_.empty();
#endif
}

void ProcessorCoreManager::organizeBySocket() {
  socket_to_cores_.clear();

  for (const auto &core : cores_) {
    socket_to_cores_[core.socket].push_back(core.id);
  }
}

void ProcessorCoreManager::detectCoreTypes() {
  // This is a heuristic approach to detect core types based on frequency
  // In reality, this should be done using platform-specific APIs

  // First, check if we have a heterogeneous system (different max frequencies)
  bool has_different_frequencies = false;
  double min_freq = std::numeric_limits<double>::max();
  double max_freq = 0.0;

  for (const auto &core : cores_) {
    if (core.max_frequency_mhz > 0) {
      min_freq = std::min(min_freq, core.max_frequency_mhz);
      max_freq = std::max(max_freq, core.max_frequency_mhz);
    }
  }

  has_different_frequencies =
      (max_freq - min_freq) > 100.0; // More than 100MHz difference

  if (has_different_frequencies) {
    // Use a threshold to differentiate between P-cores and E-cores
    double threshold = (min_freq + max_freq) / 2.0;

    for (auto &core : cores_) {
      if (core.max_frequency_mhz >= threshold) {
        core.type = CoreType::PERFORMANCE;
      } else {
        core.type = CoreType::EFFICIENCY;
      }
    }
  } else {
    // Homogeneous system, treat all cores as performance cores
    for (auto &core : cores_) {
      core.type = CoreType::PERFORMANCE;
    }
  }

#ifdef __linux__
  // On Linux, try to detect hybrid architecture with more accuracy
  // This is a very simplistic approach and may not work on all systems

  // Check for Intel hybrid architecture
  std::ifstream cpuinfo("/proc/cpuinfo");
  if (cpuinfo) {
    std::string line;
    std::regex p_core_pattern(".*Core.*i[579]-\\d+.*");
    std::regex e_core_pattern(".*Atom.*");
    int current_processor = -1;

    while (std::getline(cpuinfo, line)) {
      if (line.find("processor") == 0) {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
          current_processor = std::stoi(trim(line.substr(colon_pos + 1)));
        }
      } else if (line.find("model name") == 0 && current_processor >= 0) {
        size_t colon_pos = line.find(':');
        if (colon_pos != std::string::npos) {
          std::string model = trim(line.substr(colon_pos + 1));

          if (std::regex_match(model, p_core_pattern)) {
            // This is a P-core
            for (auto &core : cores_) {
              if (core.id == current_processor) {
                core.type = CoreType::PERFORMANCE;
                break;
              }
            }
          } else if (std::regex_match(model, e_core_pattern)) {
            // This is an E-core
            for (auto &core : cores_) {
              if (core.id == current_processor) {
                core.type = CoreType::EFFICIENCY;
                break;
              }
            }
          }
        }
      }
    }
  }
#endif
}

void ProcessorCoreManager::printCoreInfo() const {
  std::cout << "Processor Core Information:" << std::endl;
  std::cout << "Total cores: " << cores_.size() << std::endl;

  for (const auto &core : cores_) {
    std::cout << "Core " << core.id << ":" << std::endl;
    std::cout << "  Socket: " << core.socket << std::endl;
    std::cout << "  Physical ID: " << core.physical_id << std::endl;

    if (core.sibling != -1) {
      std::cout << "  Hyperthreading sibling: " << core.sibling << std::endl;
    }

    std::cout << "  Type: "
              << (core.type == CoreType::PERFORMANCE  ? "Performance"
                  : core.type == CoreType::EFFICIENCY ? "Efficiency"
                                                      : "Unknown")
              << std::endl;

    std::cout << "  Model: " << core.model_name << std::endl;
    std::cout << "  Max frequency: " << core.max_frequency_mhz << " MHz"
              << std::endl;

    std::cout << "  Caches:" << std::endl;
    if (core.l1d_cache_size_kb > 0) {
      std::cout << "    L1 Data: " << core.l1d_cache_size_kb << " KB"
                << std::endl;
    }
    if (core.l1i_cache_size_kb > 0) {
      std::cout << "    L1 Instruction: " << core.l1i_cache_size_kb << " KB"
                << std::endl;
    }
    if (core.l2_cache_size_kb > 0) {
      std::cout << "    L2: " << core.l2_cache_size_kb << " KB" << std::endl;
    }
    if (core.l3_cache_size_kb > 0) {
      std::cout << "    L3: " << core.l3_cache_size_kb << " KB" << std::endl;
    }

    std::cout << std::endl;
  }
}

ScopedCoreReservation::ScopedCoreReservation(const std::string &purpose,
                                             bool prefer_performance)
    : purpose_(purpose) {

  core_id_ = ProcessorCoreManager::getInstance().reserveCore(
      purpose, prefer_performance);
}

ScopedCoreReservation::~ScopedCoreReservation() {
  if (core_id_ >= 0) {
    ProcessorCoreManager::getInstance().releaseCore(core_id_);
  }
}

bool ScopedCoreReservation::pinCurrentThread() {
  if (core_id_ >= 0) {
    return ProcessorCoreManager::getInstance().pinCurrentThread(core_id_);
  }
  return false;
}

bool ScopedCoreReservation::pinThread(std::thread &thread) {
  if (core_id_ >= 0) {
    return ProcessorCoreManager::getInstance().pinThread(thread, core_id_);
  }
  return false;
}

ScopedMultiCoreReservation::ScopedMultiCoreReservation(
    size_t count, const std::string &purpose, bool prefer_performance,
    bool consecutive)
    : purpose_(purpose) {

  core_ids_ = ProcessorCoreManager::getInstance().reserveCores(
      count, purpose, prefer_performance, consecutive);
}

ScopedMultiCoreReservation::~ScopedMultiCoreReservation() {
  // Release all reserved cores
  for (int core_id : core_ids_) {
    ProcessorCoreManager::getInstance().releaseCore(core_id);
  }
}

bool ScopedMultiCoreReservation::pinCurrentThread(size_t index) {
  if (index < core_ids_.size()) {
    return ProcessorCoreManager::getInstance().pinCurrentThread(
        core_ids_[index]);
  }
  return false;
}

bool ScopedMultiCoreReservation::pinThread(std::thread &thread, size_t index) {
  if (index < core_ids_.size()) {
    return ProcessorCoreManager::getInstance().pinThread(thread,
                                                         core_ids_[index]);
  }
  return false;
}

bool ScopedMultiCoreReservation::pinThreadToAllCores(std::thread &thread) {
  if (!core_ids_.empty()) {
    return ProcessorCoreManager::getInstance().pinThreadToMultipleCores(
        thread, core_ids_);
  }
  return false;
}

} // namespace system
} // namespace engine
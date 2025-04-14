#include "shmem.h"
#include <sys/time.h>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <random>
#include <sstream>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <mutex>
#include <unistd.h>  // For close() function

namespace engine {
namespace determine {

// SharedMemoryChannel implementation
SharedMemoryChannel::SharedMemoryChannel(const std::string& keyFile, bool create)
    : isCreator_(create), keyFile_(keyFile), sharedBlock_(nullptr) {
    
    // Generate IPC key from file path
    key_ = ftok(keyFile.c_str(), 'R');
    if (key_ == -1) {
        // If the key file doesn't exist, create it
        if (errno == ENOENT && create) {
            int fd = open(keyFile.c_str(), O_CREAT | O_WRONLY, 0666);
            if (fd != -1) {
                close(fd);
                key_ = ftok(keyFile.c_str(), 'R');
            }
        }
        
        if (key_ == -1) {
            throw std::runtime_error("Failed to generate IPC key: " + 
                                   std::string(strerror(errno)));
        }
    }
    
    // Create or get shared memory segment
    if (create) {
        shmId_ = shmget(key_, sizeof(SharedBlock), IPC_CREAT | IPC_EXCL | 0666);
        if (shmId_ == -1 && errno == EEXIST) {
            // Segment exists, try to get it
            shmId_ = shmget(key_, sizeof(SharedBlock), 0666);
        }
    } else {
        shmId_ = shmget(key_, sizeof(SharedBlock), 0666);
    }
    
    if (shmId_ == -1) {
        throw std::runtime_error("Failed to create shared memory segment: " + 
                               std::string(strerror(errno)));
    }
    
    initialize();
}

SharedMemoryChannel::~SharedMemoryChannel() {
    if (sharedBlock_ != nullptr && sharedBlock_ != (void*)-1) {
        // Detach from shared memory
        shmdt(sharedBlock_);
    }
    
    // If we created the segment, mark it for deletion when all processes detach
    if (isCreator_ && shmId_ != -1) {
        shmctl(shmId_, IPC_RMID, nullptr);
    }
}

void SharedMemoryChannel::initialize() {
    // Attach the shared memory segment
    sharedBlock_ = static_cast<SharedBlock*>(shmat(shmId_, nullptr, 0));
    
    if (sharedBlock_ == (void*)-1) {
        throw std::runtime_error("Failed to attach shared memory segment: " + 
                               std::string(strerror(errno)));
    }
    
    // Initialize if creating the segment
    if (isCreator_) {
        // Use placement new to initialize the shared block
        new (sharedBlock_) SharedBlock();
    }
}

uint64_t SharedMemoryChannel::getLastHeartbeatNanos() const {
    if (!isValid()) {
        return 0;
    }
    return sharedBlock_->lastHeartbeatNanos.load(std::memory_order_acquire);
}

void SharedMemoryChannel::updateHeartbeat() {
    if (!isValid()) {
        return;
    }
    
    // Get current time in nanoseconds
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        return;
    }
    
    uint64_t nanoTimestamp = static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + 
                           static_cast<uint64_t>(ts.tv_nsec);
    
    sharedBlock_->lastHeartbeatNanos.store(nanoTimestamp, std::memory_order_release);
}

std::string SharedMemoryChannel::generateUniqueKey(const std::string& keyPrefix) {
    static std::mutex keyMutex;
    std::lock_guard<std::mutex> lock(keyMutex);
    
    // Create a unique key based on current time and a random number
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    auto nano = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count() % 1000000000;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(10000, 99999);
    
    std::stringstream ss;
    ss << "/tmp/" << keyPrefix << "_" 
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(9) << nano
       << "_" << dis(gen);
    
    return ss.str();
}

// SharedMemoryManager implementation
SharedMemoryManager& SharedMemoryManager::getInstance() {
    static SharedMemoryManager instance;
    return instance;
}

std::shared_ptr<SharedMemoryChannel> SharedMemoryManager::getChannel(
    const std::string& keyFile, bool create) {
    
    // Check if we already have this channel
    auto it = channels_.find(keyFile);
    if (it != channels_.end() && it->second) {
        return it->second;
    }
    
    // Create a new channel
    try {
        auto channel = std::make_shared<SharedMemoryChannel>(keyFile, create);
        channels_[keyFile] = channel;
        return channel;
    } catch (const std::exception& e) {
        std::cerr << "Error creating shared memory channel: " << e.what() << std::endl;
        return nullptr;
    }
}

void SharedMemoryManager::removeChannel(const std::string& keyFile) {
    channels_.erase(keyFile);
}

void SharedMemoryManager::updateAllHeartbeats() {
    for (auto& pair : channels_) {
        if (pair.second) {
            pair.second->updateHeartbeat();
        }
    }
}

} 
}
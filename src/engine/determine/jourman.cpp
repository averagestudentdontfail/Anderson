#include "jourman.h"
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>
#include <ctime>
#include <sstream>
#include <iomanip>
#include <sys/time.h>
#include <unistd.h>
#include <fcntl.h>

namespace engine {
namespace determine {

// EventJournal implementation
EventJournal::EventJournal(const std::string& filename, const std::string& mode)
    : filename_(filename) {
    
    file_ = fopen(filename.c_str(), mode.c_str());
    if (!file_) {
        std::cerr << "Failed to open event journal: " 
                  << filename << ": " << strerror(errno) << std::endl;
        return;
    }
    
    // Use unbuffered I/O for more deterministic behavior
    setvbuf(file_, nullptr, _IONBF, 0);
    
    // Ensure file is synced to disk
    int fd = fileno(file_);
    if (fd != -1) {
        // Set file flags for direct I/O if possible
        int flags = fcntl(fd, F_GETFL);
        if (flags != -1) {
            // O_DIRECT may not be available on all systems
            #ifdef O_DIRECT
            fcntl(fd, F_SETFL, flags | O_DIRECT);
            #endif
        }
        
        fsync(fd);
    }
    
    // If opening for read or append, determine the current sequence number
    if (mode != "wb" && mode != "w") {
        // Seek to the end
        fseek(file_, 0, SEEK_END);
        long size = ftell(file_);
        
        if (size > 0) {
            // Read the last event to get its sequence number
            fseek(file_, 0, SEEK_SET);
            
            Event event;
            uint64_t lastSequence = 0;
            
            // Find the last valid event
            while (readEvent(event)) {
                lastSequence = event.sequenceNumber;
            }
            
            // Set the sequence counter to continue from the last event
            sequence_.store(lastSequence + 1, std::memory_order_release);
            
            // Reset file position to the beginning for reading
            fseek(file_, 0, SEEK_SET);
        }
    }
}

EventJournal::~EventJournal() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
}

uint64_t EventJournal::recordEvent(EventType type, const void* data, size_t size, 
                                  uint32_t sourceId, uint32_t flags) {
    if (!file_) {
        return 0;
    }
    
    // Lock to ensure thread safety during write
    std::lock_guard<std::mutex> lock(journalMutex_);
    
    Event event;
    event.type = type;
    event.timestamp = getCurrentNanos();
    event.sequenceNumber = sequence_.fetch_add(1, std::memory_order_acq_rel);
    event.sourceId = sourceId;
    event.flags = flags;
    
    // Copy data to the appropriate union field based on type
    if (type == REQUEST_EVENT && size == sizeof(PricingRequest)) {
        memcpy(&event.request, data, size);
    } else if (type == RESULT_EVENT && size == sizeof(PricingResult)) {
        memcpy(&event.result, data, size);
    } else if (type == MARKET_DATA_EVENT && size == sizeof(MarketUpdate)) {
        memcpy(&event.marketData, data, size);
    } else if (type == SYSTEM_EVENT && size <= sizeof(event.system)) {
        memcpy(&event.system, data, size);
    } else {
        // For custom events, copy raw data into the generic field
        size_t copySize = std::min(size, sizeof(event.generic.data));
        memcpy(&event.generic.data, data, copySize);
    }
    
    // Write the event to the journal
    writeEvent(event);
    
    return event.sequenceNumber;
}

uint64_t EventJournal::recordRequest(const PricingRequest& request, uint32_t sourceId) {
    return recordEvent(REQUEST_EVENT, &request, sizeof(request), sourceId);
}

uint64_t EventJournal::recordResult(const PricingResult& result, uint32_t sourceId) {
    return recordEvent(RESULT_EVENT, &result, sizeof(result), sourceId);
}

uint64_t EventJournal::recordMarketUpdate(const MarketUpdate& update, uint32_t sourceId) {
    return recordEvent(MARKET_DATA_EVENT, &update, sizeof(update), sourceId);
}

uint64_t EventJournal::recordSystemEvent(uint32_t eventCode, uint32_t param1, 
                                        uint32_t param2, uint64_t param3,
                                        uint32_t sourceId) {
    struct {
        uint32_t eventCode;
        uint32_t param1;
        uint32_t param2;
        uint64_t param3;
    } sysEvent = {eventCode, param1, param2, param3};
    
    return recordEvent(SYSTEM_EVENT, &sysEvent, sizeof(sysEvent), sourceId);
}

bool EventJournal::readNextEvent(Event& event) {
    if (!file_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(journalMutex_);
    return readEvent(event);
}

bool EventJournal::seekToSequence(uint64_t sequence) {
    if (!file_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(journalMutex_);
    
    // Rewind to the beginning of the file
    rewind(file_);
    
    Event event;
    while (readEvent(event)) {
        if (event.sequenceNumber >= sequence) {
            // Found the sequence, seek back to read this event next time
            long pos = ftell(file_);
            fseek(file_, pos - sizeof(Event), SEEK_SET);
            return true;
        }
    }
    
    // Sequence not found, reset to beginning
    rewind(file_);
    return false;
}

bool EventJournal::seekToTimestamp(uint64_t timestamp) {
    if (!file_) {
        return false;
    }
    
    std::lock_guard<std::mutex> lock(journalMutex_);
    
    // Rewind to the beginning of the file
    rewind(file_);
    
    Event event;
    Event prevEvent;
    bool foundPrev = false;
    
    while (readEvent(event)) {
        if (event.timestamp >= timestamp) {
            // Found an event after the timestamp
            
            // If we have a previous event, decide which is closer
            if (foundPrev) {
                // Calculate distances
                uint64_t prevDist = timestamp - prevEvent.timestamp;
                uint64_t currDist = event.timestamp - timestamp;
                
                // If the previous event is closer, seek to it
                if (prevDist < currDist) {
                    long pos = ftell(file_);
                    fseek(file_, pos - 2 * sizeof(Event), SEEK_SET);
                    return true;
                }
            }
            
            // Seek back to read this event next time
            long pos = ftell(file_);
            fseek(file_, pos - sizeof(Event), SEEK_SET);
            return true;
        }
        
        // Update previous event
        prevEvent = event;
        foundPrev = true;
    }
    
    // Timestamp not found, reset to beginning
    rewind(file_);
    return false;
}

void EventJournal::flush(bool forceSync) {
    if (!file_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(journalMutex_);
    
    // Flush file buffers
    fflush(file_);
    
    if (forceSync) {
        // Sync to disk for durability
        int fd = fileno(file_);
        if (fd != -1) {
            // fdatasync is faster than fsync as it doesn't update metadata
            fdatasync(fd);
        }
    }
}

uint64_t EventJournal::getCurrentNanos() {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        // Fallback if clock_gettime fails
        return time(nullptr) * 1000000000ULL;
    }
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + 
           static_cast<uint64_t>(ts.tv_nsec);
}

void EventJournal::reset() {
    if (!file_) {
        return;
    }
    
    std::lock_guard<std::mutex> lock(journalMutex_);
    rewind(file_);
}

bool EventJournal::writeEvent(const Event& event) {
    if (!file_) {
        return false;
    }
    
    // Write the event to the journal file
    size_t written = fwrite(&event, sizeof(Event), 1, file_);
    if (written != 1) {
        std::cerr << "Warning: Failed to write event to journal: " 
                  << strerror(errno) << std::endl;
        return false;
    }
    
    // Flush to ensure data is written
    fflush(file_);
    
    return true;
}

bool EventJournal::readEvent(Event& event) {
    if (!file_) {
        return false;
    }
    
    // Read the event from the journal file
    size_t read = fread(&event, sizeof(Event), 1, file_);
    return read == 1;
}

// JournalManager implementation
JournalManager& JournalManager::getInstance() {
    static JournalManager instance;
    return instance;
}

std::shared_ptr<EventJournal> JournalManager::getJournal(const std::string& name, 
                                                       const std::string& filename,
                                                       const std::string& mode) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    // Check if we already have this journal
    auto it = journals_.find(name);
    if (it != journals_.end() && it->second) {
        return it->second;
    }
    
    // Determine the filename to use
    std::string journalFile = filename.empty() ? generateJournalFilename(name) : filename;
    
    // Create a new journal
    try {
        auto journal = std::make_shared<EventJournal>(journalFile, mode);
        if (journal->isValid()) {
            journals_[name] = journal;
            return journal;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error creating journal: " << e.what() << std::endl;
    }
    
    return nullptr;
}

void JournalManager::removeJournal(const std::string& name) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    journals_.erase(name);
}

void JournalManager::flushAll(bool forceSync) {
    std::lock_guard<std::mutex> lock(managerMutex_);
    
    for (auto& pair : journals_) {
        if (pair.second) {
            pair.second->flush(forceSync);
        }
    }
}

std::string JournalManager::generateJournalFilename(const std::string& name) {
    // Get current time
    auto now = std::chrono::system_clock::now();
    auto now_time_t = std::chrono::system_clock::to_time_t(now);
    
    std::stringstream ss;
    ss << "/tmp/journal_" << name << "_"
       << std::put_time(std::localtime(&now_time_t), "%Y%m%d_%H%M%S")
       << ".bin";
    
    return ss.str();
}

// JournalReplayEngine implementation
JournalReplayEngine::JournalReplayEngine(std::shared_ptr<EventJournal> journal)
    : journal_(journal) {
}

void JournalReplayEngine::setEventHandler(EventType type, 
                                         std::function<void(const Event&)> handler) {
    handlers_[type] = handler;
}

void JournalReplayEngine::startReplay(double speedFactor) {
    if (replaying_.exchange(true)) {
        // Already replaying
        return;
    }
    
    speedFactor_ = speedFactor;
    
    // Reset journal position
    journal_->reset();
    
    // Start replay in a separate thread
    std::thread replayThread(&JournalReplayEngine::replayThreadFunc, this);
    replayThread.detach();
}

void JournalReplayEngine::stopReplay() {
    replaying_.store(false);
}

void JournalReplayEngine::replayThreadFunc() {
    if (!journal_) {
        replaying_.store(false);
        return;
    }
    
    Event event;
    uint64_t lastTimestamp = 0;
    
    // Using replayStartTime for properly handling real-time replay
    auto replayStartTime = std::chrono::high_resolution_clock::now();
    uint64_t journalStartTimestamp = 0;
    bool isFirstEvent = true;
    
    while (replaying_.load() && journal_->readNextEvent(event)) {
        // Capture the first event timestamp as the journal start time
        if (isFirstEvent) {
            journalStartTimestamp = event.timestamp;
            isFirstEvent = false;
        }
        
        // Calculate delay between events
        if (lastTimestamp > 0 && speedFactor_ > 0.0) {
            uint64_t delay = event.timestamp - lastTimestamp;
            
            // Adjust by speed factor
            delay = static_cast<uint64_t>(delay / speedFactor_);
            
            // Sleep for the scaled delay
            std::this_thread::sleep_for(std::chrono::nanoseconds(delay));
        }
        
        // Update last timestamp
        lastTimestamp = event.timestamp;
        
        // Dispatch to appropriate handler
        auto it = handlers_.find(event.type);
        if (it != handlers_.end() && it->second) {
            it->second(event);
        }
    }
    
    replaying_.store(false);
}

} 
} 
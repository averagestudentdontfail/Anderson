#ifndef ENGINE_DETERMINE_JOURMAN_H
#define ENGINE_DETERMINE_JOURMAN_H

#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include <mutex>
#include <unordered_map>
#include "shmem.h" // For event type definitions

namespace engine {
namespace determine {

// Event types for journaling
enum EventType {
    REQUEST_EVENT, 
    RESULT_EVENT, 
    MARKET_DATA_EVENT,
    SYSTEM_EVENT,
    CUSTOM_EVENT,
    HEARTBEAT_EVENT
};

// Generic event structure
struct Event {
    EventType type;
    uint64_t timestamp;
    uint64_t sequenceNumber;
    uint32_t sourceId;  // Identifies the event source (e.g., process ID, thread ID)
    uint32_t flags;     // Additional flags for event processing
    
    // Union to store different event data types
    union {
        PricingRequest request;
        PricingResult result;
        MarketUpdate marketData;
        
        // For system events
        struct {
            uint32_t eventCode;
            uint32_t param1;
            uint32_t param2;
            uint64_t param3;
        } system;
        
        // Generic event data
        struct {
            uint64_t data[8]; // 64 bytes of raw data
        } generic;
    };
};

/**
 * @brief Journal for recording and replaying events in a deterministic system
 */
class EventJournal {
public:
    /**
     * @brief Create a new event journal
     * @param filename Path to the journal file
     * @param mode Journal mode (read, write, or append)
     */
    EventJournal(const std::string& filename, const std::string& mode = "ab+");
    
    /**
     * @brief Close the journal
     */
    ~EventJournal();
    
    /**
     * @brief Record an event to the journal
     * @param type Event type
     * @param data Pointer to event data
     * @param size Size of event data
     * @param sourceId Source ID for the event
     * @param flags Additional flags
     * @return Sequence number of the recorded event
     */
    uint64_t recordEvent(EventType type, const void* data, size_t size, 
                        uint32_t sourceId = 0, uint32_t flags = 0);
    
    /**
     * @brief Record a pricing request event
     * @param request The pricing request to record
     * @param sourceId Source ID for the event
     * @return Sequence number of the recorded event
     */
    uint64_t recordRequest(const PricingRequest& request, uint32_t sourceId = 0);
    
    /**
     * @brief Record a pricing result event
     * @param result The pricing result to record
     * @param sourceId Source ID for the event
     * @return Sequence number of the recorded event
     */
    uint64_t recordResult(const PricingResult& result, uint32_t sourceId = 0);
    
    /**
     * @brief Record a market data update event
     * @param update The market data update to record
     * @param sourceId Source ID for the event
     * @return Sequence number of the recorded event
     */
    uint64_t recordMarketUpdate(const MarketUpdate& update, uint32_t sourceId = 0);
    
    /**
     * @brief Record a system event
     * @param eventCode The system event code
     * @param param1 First parameter
     * @param param2 Second parameter
     * @param param3 Third parameter
     * @param sourceId Source ID for the event
     * @return Sequence number of the recorded event
     */
    uint64_t recordSystemEvent(uint32_t eventCode, uint32_t param1 = 0, 
                             uint32_t param2 = 0, uint64_t param3 = 0,
                             uint32_t sourceId = 0);
    
    /**
     * @brief Read the next event from the journal
     * @param event Reference to an Event struct to be populated
     * @return True if an event was read, false if end of journal
     */
    bool readNextEvent(Event& event);
    
    /**
     * @brief Seek to a specific sequence number in the journal
     * @param sequence The sequence number to seek to
     * @return True if successful, false otherwise
     */
    bool seekToSequence(uint64_t sequence);
    
    /**
     * @brief Seek to a specific timestamp in the journal
     * @param timestamp The timestamp to seek to
     * @return True if successful, false otherwise
     */
    bool seekToTimestamp(uint64_t timestamp);
    
    /**
     * @brief Flush the journal to disk
     * @param forceSync Force a sync to disk
     */
    void flush(bool forceSync = false);
    
    /**
     * @brief Get the current monotonic time in nanoseconds
     * @return Current time in nanoseconds
     */
    static uint64_t getCurrentNanos();
    
    /**
     * @brief Check if the journal is valid
     * @return True if the journal is valid
     */
    bool isValid() const { return file_ != nullptr; }
    
    /**
     * @brief Get the current sequence number
     * @return Current sequence number
     */
    uint64_t getCurrentSequence() const { 
        return sequence_.load(std::memory_order_acquire); 
    }
    
    /**
     * @brief Reset the journal position to the beginning
     */
    void reset();

private:
    FILE* file_;
    std::atomic<uint64_t> sequence_{0};
    std::string filename_;
    std::mutex journalMutex_;
    
    /**
     * @brief Write an event to the journal file
     * @param event The event to write
     * @return True if successful, false otherwise
     */
    bool writeEvent(const Event& event);
    
    /**
     * @brief Read an event from the journal file
     * @param event Reference to an Event struct to be populated
     * @return True if successful, false otherwise
     */
    bool readEvent(Event& event);
};

/**
 * @brief Manager for multiple event journals
 * 
 * Handles creation, access, and coordination of multiple event journals.
 */
class JournalManager {
public:
    /**
     * @brief Get the singleton instance
     * @return Reference to the journal manager
     */
    static JournalManager& getInstance();
    
    /**
     * @brief Get or create a journal
     * @param name Name of the journal
     * @param filename Path to the journal file (if not provided, a default path is used)
     * @param mode Journal mode (read, write, or append)
     * @return Shared pointer to the journal
     */
    std::shared_ptr<EventJournal> getJournal(const std::string& name, 
                                           const std::string& filename = "",
                                           const std::string& mode = "ab+");
    
    /**
     * @brief Remove a journal from the manager
     * @param name Name of the journal to remove
     */
    void removeJournal(const std::string& name);
    
    /**
     * @brief Flush all journals
     * @param forceSync Force a sync to disk
     */
    void flushAll(bool forceSync = false);
    
    /**
     * @brief Generate a default journal filename
     * @param name Name of the journal
     * @return Default journal filename
     */
    std::string generateJournalFilename(const std::string& name);

private:
    JournalManager() = default;
    ~JournalManager() = default;
    
    // Disable copy and move
    JournalManager(const JournalManager&) = delete;
    JournalManager& operator=(const JournalManager&) = delete;
    JournalManager(JournalManager&&) = delete;
    JournalManager& operator=(JournalManager&&) = delete;
    
    std::unordered_map<std::string, std::shared_ptr<EventJournal>> journals_;
    std::mutex managerMutex_;
};

/**
 * @brief Replay engine for event journals
 * 
 * Provides functionality to replay events from a journal for deterministic testing.
 */
class JournalReplayEngine {
public:
    /**
     * @brief Create a replay engine for the specified journal
     * @param journal The journal to replay
     */
    explicit JournalReplayEngine(std::shared_ptr<EventJournal> journal);
    
    /**
     * @brief Set the event handler for a specific event type
     * @param type Event type to handle
     * @param handler Function to call for each event of this type
     */
    void setEventHandler(EventType type, 
                         std::function<void(const Event&)> handler);
    
    /**
     * @brief Start replay from the beginning
     * @param speedFactor Speed factor (1.0 = real-time, 0.0 = as fast as possible)
     */
    void startReplay(double speedFactor = 1.0);
    
    /**
     * @brief Stop the current replay
     */
    void stopReplay();
    
    /**
     * @brief Check if replay is currently active
     * @return True if replay is active
     */
    bool isReplaying() const { return replaying_; }

private:
    std::shared_ptr<EventJournal> journal_;
    std::unordered_map<EventType, std::function<void(const Event&)>> handlers_;
    std::atomic<bool> replaying_{false};
    double speedFactor_{1.0};
    
    /**
     * @brief Replay thread function
     */
    void replayThreadFunc();
};

} // namespace determine
} // namespace engine

#endif 
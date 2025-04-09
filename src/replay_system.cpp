// replay_system.cpp
// Event replay system for validating deterministic execution

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <cstring>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <numeric>
#include <cerrno>
#include <cmath>
#include <sys/stat.h>
#include <fcntl.h>
#include <signal.h>
#include "deterministic_pricing_system.h"

// Global flag for graceful shutdown
std::atomic<bool> g_running{true};

// Signal handler
void signalHandler(int signal) {
    std::cout << "Received signal " << signal << ", initiating shutdown..." << std::endl;
    g_running = false;
}

// Class for replaying recorded events
class ReplayEngine {
private:
    std::ifstream journal_;
    DeterministicPricer pricer_;
    std::vector<Event> events_;
    bool verbose_;
    
    // Statistics for performance tracking
    struct Stats {
        size_t totalRequests = 0;
        size_t matchedResults = 0;
        size_t mismatchedResults = 0;
        size_t skippedResults = 0;
        
        // Latency statistics
        double totalLatency = 0;
        size_t latencyCount = 0;
        uint64_t minLatency = UINT64_MAX;
        uint64_t maxLatency = 0;
        
        // Histogram buckets in microseconds: 0-100, 100-200, 200-500, 500-1000, 1000+
        std::vector<uint64_t> latencyBuckets = std::vector<uint64_t>(5, 0);
        
        // Performance metrics
        std::chrono::high_resolution_clock::time_point startTime;
        std::chrono::high_resolution_clock::time_point endTime;
    };
    
    Stats stats_;
    
public:
    explicit ReplayEngine(const std::string& journalFile, bool verbose = false)
        : verbose_(verbose) {
        // Register signal handlers for graceful shutdown
        signal(SIGINT, signalHandler);
        signal(SIGTERM, signalHandler);
        
        // First check if the file exists
        struct stat fileStat;
        if (stat(journalFile.c_str(), &fileStat) != 0) {
            throw std::runtime_error("Journal file does not exist: " + journalFile + " - " + 
                                  std::string(strerror(errno)));
        }
        
        // Open the journal file
        journal_.open(journalFile, std::ios::binary);
        if (!journal_) {
            throw std::runtime_error("Failed to open journal for replay: " + journalFile + " - " + 
                                  std::string(strerror(errno)));
        }
        
        // Get file size for progress reporting
        journal_.seekg(0, std::ios::end);
        size_t fileSize = journal_.tellg();
        journal_.seekg(0, std::ios::beg);
        
        std::cout << "Reading events from journal (" << (fileSize / (1024 * 1024)) << " MB)..." << std::endl;
        
        // Read all events
        Event event;
        size_t eventsRead = 0;
        size_t bytesRead = 0;
        
        // Header size: type + timestamp + sequence number
        const size_t headerSize = sizeof(EventType) + sizeof(uint64_t) * 2;
        
        // Track start time for performance reporting
        stats_.startTime = std::chrono::high_resolution_clock::now();
        
        // Keep reading until we reach the end of the file or user interrupts
        while (g_running) {
            // Read event header
            if (!journal_.read(reinterpret_cast<char*>(&event), headerSize)) {
                if (journal_.eof()) {
                    break;  // End of file reached
                }
                throw std::runtime_error("Error reading journal file: " + 
                                      std::string(strerror(errno)));
            }
            
            bytesRead += headerSize;
            
            // Sanity check the event type
            if (event.type < REQUEST_EVENT || event.type > MARKET_DATA_EVENT) {
                throw std::runtime_error("Invalid event type in journal: " + 
                                      std::to_string(static_cast<int>(event.type)));
            }
            
            // Read event data based on type
            size_t dataSize = 0;
            switch (event.type) {
                case REQUEST_EVENT:
                    dataSize = sizeof(PricingRequest);
                    break;
                case RESULT_EVENT:
                    dataSize = sizeof(PricingResult);
                    break;
                case MARKET_DATA_EVENT:
                    dataSize = sizeof(MarketUpdate);
                    break;
                default:
                    throw std::runtime_error("Unknown event type in journal");
            }
            
            if (!journal_.read(reinterpret_cast<char*>(&event) + headerSize, dataSize)) {
                throw std::runtime_error("Corrupted journal file - incomplete event data");
            }
            
            bytesRead += dataSize;
            
            // Add event to our collection
            events_.push_back(event);
            eventsRead++;
            
            // Show progress every 10,000 events or 100MB
            if (eventsRead % 10000 == 0 || bytesRead % (100 * 1024 * 1024) == 0) {
                auto now = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
                    now - stats_.startTime).count();
                
                double progress = static_cast<double>(bytesRead) / fileSize * 100.0;
                double rate = eventsRead / (elapsed > 0 ? elapsed : 1);
                
                std::cout << "Read " << eventsRead << " events (" 
                          << std::fixed << std::setprecision(1) << progress << "%), "
                          << "Events/sec: " << static_cast<int>(rate) << "\r" << std::flush;
            }
        }
        
        std::cout << std::endl;
        
        // Check if we were interrupted
        if (!g_running) {
            throw std::runtime_error("Journal loading interrupted by user");
        }
        
        // Sort by sequence number to ensure correct order
        std::cout << "Sorting events by sequence number..." << std::endl;
        std::sort(events_.begin(), events_.end(), 
                 [](const Event& a, const Event& b) {
                     return a.sequenceNumber < b.sequenceNumber;
                 });
        
        // Count event types
        size_t requestCount = 0, resultCount = 0, marketDataCount = 0;
        for (const auto& e : events_) {
            switch (e.type) {
                case REQUEST_EVENT: requestCount++; break;
                case RESULT_EVENT: resultCount++; break;
                case MARKET_DATA_EVENT: marketDataCount++; break;
            }
        }
        
        stats_.totalRequests = requestCount;
        
        // Record end time and calculate load performance
        stats_.endTime = std::chrono::high_resolution_clock::now();
        auto loadTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            stats_.endTime - stats_.startTime).count();
        
        // Print summary of loaded events
        std::cout << "Finished reading " << events_.size() << " events in " 
                  << loadTime << " ms (" 
                  << static_cast<int>(events_.size() * 1000.0 / loadTime) << " events/sec)." 
                  << std::endl;
        
        std::cout << "Event breakdown:" << std::endl;
        std::cout << "  Requests: " << requestCount << std::endl;
        std::cout << "  Results: " << resultCount << std::endl;
        std::cout << "  Market Data: " << marketDataCount << std::endl;
        
        // Check for potential issues
        if (resultCount < requestCount * 0.9) {
            std::cout << "Warning: Only " << (resultCount * 100.0 / requestCount) 
                      << "% of requests have matching results." << std::endl;
        }
    }
    
    // Run replay and verify results
    void replay() {
        std::cout << "Starting replay..." << std::endl;
        
        // Reset and start timing the replay
        stats_.startTime = std::chrono::high_resolution_clock::now();
        
        // Map to store requests by ID for later matching
        std::unordered_map<uint64_t, PricingRequest> requestMap;
        
        // Map to store recorded results by request ID
        std::unordered_map<uint64_t, PricingResult> recordedResults;
        
        // Process events in sequence order
        std::cout << "Processing events and building request/result maps..." << std::endl;
        
        for (const auto& event : events_) {
            if (!g_running) {
                throw std::runtime_error("Replay interrupted by user");
            }
            
            if (event.type == REQUEST_EVENT) {
                // Store request for later matching
                requestMap[event.request.requestId] = event.request;
            } 
            else if (event.type == RESULT_EVENT) {
                // Store the recorded result
                recordedResults[event.result.requestId] = event.result;
            }
        }
        
        // Replay each request and compare with recorded result
        std::cout << "Replaying " << requestMap.size() << " pricing requests..." << std::endl;
        
        size_t progress = 0;
        const size_t progressInterval = std::max<size_t>(1, requestMap.size() / 20);
        
        // Prepare the pricer with a warm-up run
        PricingRequest warmupRequest;
        warmupRequest.S = 100.0;
        warmupRequest.K = 100.0;
        warmupRequest.r = 0.05;
        warmupRequest.q = 0.01;
        warmupRequest.vol = 0.2;
        warmupRequest.T = 1.0;
        warmupRequest.requestId = 0;       
        warmupRequest.instrumentId = 0; 
        
        for (int i = 0; i < 10; i++) {
            PricingResult* warmupResult = pricer_.price(warmupRequest);
            if (warmupResult) {
                pricer_.release(warmupResult);
            }
        }
        
        // Open a CSV file for detailed result logging if verbose mode is enabled
        std::ofstream resultCsv;
        if (verbose_) {
            resultCsv.open("replay_results.csv");
            if (resultCsv.is_open()) {
                resultCsv << "RequestID,S,K,r,q,vol,T,RecordedPrice,ReplayedPrice,Difference,Status" << std::endl;
            } else {
                std::cerr << "Warning: Could not open replay_results.csv for writing" << std::endl;
            }
        }
        
        for (const auto& [requestId, request] : requestMap) {
            if (!g_running) {
                throw std::runtime_error("Replay interrupted by user");
            }
            
            // Check if we have a recorded result for this request
            auto recordedIt = recordedResults.find(requestId);
            if (recordedIt == recordedResults.end()) {
                stats_.skippedResults++;
                if (verbose_) {
                    std::cout << "No recorded result for request ID " << requestId << std::endl;
                }
                continue;
            }
            
            // Replay the pricing calculation
            PricingResult* replayedResult = pricer_.price(request);
            if (!replayedResult) {
                std::cerr << "Failed to replay request ID " << requestId << std::endl;
                stats_.skippedResults++;
                continue;
            }
            
            // Compare with recorded result
            const PricingResult& recordedResult = recordedIt->second;
            
            // Check if prices match within tolerance
            constexpr double tolerance = 1e-10;
            bool priceMatches = std::abs(replayedResult->price - recordedResult.price) <= tolerance;
            
            if (priceMatches) {
                stats_.matchedResults++;
            } else {
                stats_.mismatchedResults++;
                if (verbose_ || stats_.mismatchedResults < 10) {
                    std::cout << "Replay mismatch for request ID " << requestId << ":" << std::endl;
                    std::cout << "  Recorded: " << recordedResult.price << std::endl;
                    std::cout << "  Replayed: " << replayedResult->price << std::endl;
                    std::cout << "  Difference: " << (replayedResult->price - recordedResult.price) << std::endl;
                    std::cout << "  Parameters: S=" << request.S << ", K=" << request.K 
                              << ", r=" << request.r << ", q=" << request.q 
                              << ", vol=" << request.vol << ", T=" << request.T << std::endl;
                }
            }
            
            // Log to CSV if enabled
            if (verbose_ && resultCsv.is_open()) {
                resultCsv << requestId << ","
                          << request.S << ","
                          << request.K << ","
                          << request.r << ","
                          << request.q << ","
                          << request.vol << ","
                          << request.T << ","
                          << recordedResult.price << ","
                          << replayedResult->price << ","
                          << (replayedResult->price - recordedResult.price) << ","
                          << (priceMatches ? "Match" : "Mismatch") << std::endl;
            }
            
            // Release the result
            pricer_.release(replayedResult);
            
            // Show progress
            progress++;
            if (progress % progressInterval == 0 || progress == requestMap.size()) {
                std::cout << "Progress: " << progress << "/" << requestMap.size() 
                          << " (" << (progress * 100 / requestMap.size()) << "%)" << std::endl;
            }
        }
        
        // Close the CSV file if open
        if (resultCsv.is_open()) {
            resultCsv.close();
            if (verbose_) {
                std::cout << "Detailed results written to replay_results.csv" << std::endl;
            }
        }
        
        // Record end time and calculate replay performance
        stats_.endTime = std::chrono::high_resolution_clock::now();
        auto replayTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            stats_.endTime - stats_.startTime).count();
        
        // Print summary
        std::cout << "\nReplay Summary:" << std::endl;
        std::cout << "  Total Requests: " << requestMap.size() << std::endl;
        std::cout << "  Matched Results: " << stats_.matchedResults 
                  << " (" << (stats_.matchedResults * 100.0 / requestMap.size()) << "%)" << std::endl;
        std::cout << "  Mismatched Results: " << stats_.mismatchedResults 
                  << " (" << (stats_.mismatchedResults * 100.0 / requestMap.size()) << "%)" << std::endl;
        std::cout << "  Skipped Results: " << stats_.skippedResults 
                  << " (" << (stats_.skippedResults * 100.0 / requestMap.size()) << "%)" << std::endl;
        std::cout << "  Replay Time: " << replayTime << " ms (" 
                  << static_cast<int>(requestMap.size() * 1000.0 / replayTime) << " requests/sec)" 
                  << std::endl;
        
        if (stats_.matchedResults == requestMap.size() - stats_.skippedResults) {
            std::cout << "✅ REPLAY SUCCESSFUL: All processed results matched exactly!" << std::endl;
        } else {
            std::cout << "❌ REPLAY FAILED: " << stats_.mismatchedResults
                      << " results did not match." << std::endl;
        }
    }
    
    // Run latency analysis on the recorded events
    void analyzeLatency() {
        std::cout << "Analyzing latency from recorded events..." << std::endl;
        
        // Map request timestamps by ID
        std::unordered_map<uint64_t, uint64_t> requestTimestamps;
        
        // Reset stats
        stats_.totalLatency = 0;
        stats_.latencyCount = 0;
        stats_.minLatency = UINT64_MAX;
        stats_.maxLatency = 0;
        std::fill(stats_.latencyBuckets.begin(), stats_.latencyBuckets.end(), 0);
        
        stats_.startTime = std::chrono::high_resolution_clock::now();
        
        // First pass: record request timestamps
        std::cout << "Processing request timestamps..." << std::endl;
        for (const auto& event : events_) {
            if (!g_running) {
                throw std::runtime_error("Analysis interrupted by user");
            }
            
            if (event.type == REQUEST_EVENT) {
                requestTimestamps[event.request.requestId] = event.timestamp;
            }
        }
        
        // Prepare for P50, P95, P99 calculations
        std::vector<uint64_t> allLatencies;
        allLatencies.reserve(events_.size() / 2); // Rough estimate
        
        // Open a CSV file for detailed latency logging if verbose mode is enabled
        std::ofstream latencyCsv;
        if (verbose_) {
            latencyCsv.open("latency_results.csv");
            if (latencyCsv.is_open()) {
                latencyCsv << "RequestID,RequestTime,ResultTime,LatencyNanos,LatencyMicros" << std::endl;
            } else {
                std::cerr << "Warning: Could not open latency_results.csv for writing" << std::endl;
            }
        }
        
        // Second pass: calculate latency for each result
        std::cout << "Calculating latencies..." << std::endl;
        for (const auto& event : events_) {
            if (!g_running) {
                throw std::runtime_error("Analysis interrupted by user");
            }
            
            if (event.type == RESULT_EVENT) {
                // Find the corresponding request
                auto it = requestTimestamps.find(event.result.requestId);
                if (it != requestTimestamps.end()) {
                    uint64_t requestTime = it->second;
                    uint64_t latencyNanos = event.timestamp - requestTime;
                    uint64_t latencyMicros = latencyNanos / 1000;
                    
                    // Update statistics
                    stats_.totalLatency += latencyNanos;
                    stats_.latencyCount++;
                    stats_.minLatency = std::min(stats_.minLatency, latencyNanos);
                    stats_.maxLatency = std::max(stats_.maxLatency, latencyNanos);
                    
                    // Add to vector for percentile calculations
                    allLatencies.push_back(latencyNanos);
                    
                    // Update histogram
                    if (latencyMicros < 100) {
                        stats_.latencyBuckets[0]++;
                    } else if (latencyMicros < 200) {
                        stats_.latencyBuckets[1]++;
                    } else if (latencyMicros < 500) {
                        stats_.latencyBuckets[2]++;
                    } else if (latencyMicros < 1000) {
                        stats_.latencyBuckets[3]++;
                    } else {
                        stats_.latencyBuckets[4]++;
                    }
                    
                    // Log to CSV if enabled
                    if (verbose_ && latencyCsv.is_open()) {
                        latencyCsv << event.result.requestId << ","
                                  << requestTime << ","
                                  << event.timestamp << ","
                                  << latencyNanos << ","
                                  << latencyMicros << std::endl;
                    }
                }
            }
        }
        
        // Close the CSV file if open
        if (latencyCsv.is_open()) {
            latencyCsv.close();
            if (verbose_) {
                std::cout << "Detailed latency data written to latency_results.csv" << std::endl;
            }
        }
        
        // Sort latencies for percentile calculations
        std::sort(allLatencies.begin(), allLatencies.end());
        
        // Record end time and calculate analysis performance
        stats_.endTime = std::chrono::high_resolution_clock::now();
        auto analysisTime = std::chrono::duration_cast<std::chrono::milliseconds>(
            stats_.endTime - stats_.startTime).count();
        
        // Calculate percentiles
        uint64_t p50 = 0, p95 = 0, p99 = 0, p999 = 0;
        if (!allLatencies.empty()) {
            p50 = allLatencies[allLatencies.size() * 0.5];
            p95 = allLatencies[allLatencies.size() * 0.95];
            p99 = allLatencies[allLatencies.size() * 0.99];
            p999 = allLatencies[allLatencies.size() * 0.999];
        }
        
        // Print latency statistics
        if (stats_.latencyCount > 0) {
            double avgLatencyNanos = stats_.totalLatency / stats_.latencyCount;
            double avgLatencyMicros = avgLatencyNanos / 1000.0;
            
            std::cout << "\nLatency Statistics:" << std::endl;
            std::cout << "  Total Measurements: " << stats_.latencyCount << std::endl;
            std::cout << "  Average Latency: " << std::fixed << std::setprecision(2) 
                      << avgLatencyMicros << " μs" << std::endl;
            std::cout << "  Minimum Latency: " << (stats_.minLatency / 1000.0) << " μs" << std::endl;
            std::cout << "  Maximum Latency: " << (stats_.maxLatency / 1000.0) << " μs" << std::endl;
            std::cout << "  P50 Latency: " << (p50 / 1000.0) << " μs" << std::endl;
            std::cout << "  P95 Latency: " << (p95 / 1000.0) << " μs" << std::endl;
            std::cout << "  P99 Latency: " << (p99 / 1000.0) << " μs" << std::endl;
            std::cout << "  P99.9 Latency: " << (p999 / 1000.0) << " μs" << std::endl;
            std::cout << "  Analysis Time: " << analysisTime << " ms" << std::endl;
            
            std::cout << "\nLatency Distribution:" << std::endl;
            std::cout << "  0-100 μs: " << stats_.latencyBuckets[0] 
                      << " (" << (stats_.latencyBuckets[0] * 100.0 / stats_.latencyCount) << "%)" << std::endl;
            std::cout << "  100-200 μs: " << stats_.latencyBuckets[1] 
                      << " (" << (stats_.latencyBuckets[1] * 100.0 / stats_.latencyCount) << "%)" << std::endl;
            std::cout << "  200-500 μs: " << stats_.latencyBuckets[2] 
                      << " (" << (stats_.latencyBuckets[2] * 100.0 / stats_.latencyCount) << "%)" << std::endl;
            std::cout << "  500-1000 μs: " << stats_.latencyBuckets[3] 
                      << " (" << (stats_.latencyBuckets[3] * 100.0 / stats_.latencyCount) << "%)" << std::endl;
            std::cout << "  1000+ μs: " << stats_.latencyBuckets[4] 
                      << " (" << (stats_.latencyBuckets[4] * 100.0 / stats_.latencyCount) << "%)" << std::endl;
        } else {
            std::cout << "No latency data available." << std::endl;
        }
    }
};

// Command-line arguments handler with more robust error handling
struct Args {
    std::string journalFile;
    bool verbose = false;
    bool analyzeLatency = false;
    
    static void printUsage(const char* programName) {
        std::cout << "Usage: " << programName << " [options] <journal_file>" << std::endl;
        std::cout << "Options:" << std::endl;
        std::cout << "  -v, --verbose     Enable verbose output and detailed CSV logging" << std::endl;
        std::cout << "  -l, --latency     Analyze request-to-response latency" << std::endl;
        std::cout << "  -h, --help        Show this help message" << std::endl;
        std::cout << std::endl;
        std::cout << "Examples:" << std::endl;
        std::cout << "  " << programName << " pricing_events.bin              # Replay pricing events" << std::endl;
        std::cout << "  " << programName << " -l pricing_events.bin           # Analyze latency" << std::endl;
        std::cout << "  " << programName << " -v -l pricing_events.bin        # Verbose latency analysis" << std::endl;
    }
    
    static Args parse(int argc, char** argv) {
        Args args;
        
        if (argc < 2) {
            printUsage(argv[0]);
            exit(1);
        }
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            
            if (arg == "-h" || arg == "--help") {
                printUsage(argv[0]);
                exit(0);
            } else if (arg == "-v" || arg == "--verbose") {
                args.verbose = true;
            } else if (arg == "-l" || arg == "--latency") {
                args.analyzeLatency = true;
            } else if (arg[0] == '-') {
                std::cerr << "Unknown option: " << arg << std::endl;
                printUsage(argv[0]);
                exit(1);
            } else {
                if (!args.journalFile.empty()) {
                    std::cerr << "Error: Multiple journal files specified" << std::endl;
                    printUsage(argv[0]);
                    exit(1);
                }
                args.journalFile = arg;
            }
        }
        
        if (args.journalFile.empty()) {
            std::cerr << "Error: Journal file not specified" << std::endl;
            printUsage(argv[0]);
            exit(1);
        }
        
        return args;
    }
};

// Main function
int main(int argc, char** argv) {
    // Register signal handlers
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // Disable stdio buffering for more responsive output
    setvbuf(stdout, nullptr, _IONBF, 0);
    setvbuf(stderr, nullptr, _IONBF, 0);
    
    // Parse command-line arguments
    Args args = Args::parse(argc, argv);
    
    std::cout << "==================================================" << std::endl;
    std::cout << "Deterministic Derivatives Pricing System Replay" << std::endl;
    std::cout << "==================================================" << std::endl;
    
    try {
        // Create and initialize the replay engine
        ReplayEngine engine(args.journalFile, args.verbose);
        
        // Run the appropriate analysis
        if (args.analyzeLatency) {
            // Analyze latency
            engine.analyzeLatency();
        } else {
            // Run the replay and verify results
            engine.replay();
        }
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
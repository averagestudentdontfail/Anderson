#include "../../engine/determine/shmem.h"
#include "../../engine/determine/priceman.h"
#include "../../common/profile/perfmon.h"
#include "../../common/profile/timemon.h"
#include <iostream>
#include <iomanip>
#include <string>
#include <thread>
#include <chrono>
#include <atomic>
#include <signal.h>
#include <cstring>

// Flag for graceful shutdown
std::atomic<bool> running(true);

// Signal handler
void signal_handler(int signal) {
    if (signal == SIGINT || signal == SIGTERM) {
        running = false;
    }
}

// Simple console utilities
namespace console {
    // Clear the screen
    void clear() {
        #ifdef _WIN32
        system("cls");
        #else
        system("clear");
        #endif
    }
    
    // Move cursor to position
    void move_to(int row, int col) {
        std::cout << "\033[" << row << ";" << col << "H";
    }
    
    // Set text color
    enum class Color {
        RED = 31,
        GREEN = 32,
        YELLOW = 33,
        BLUE = 34,
        MAGENTA = 35,
        CYAN = 36,
        WHITE = 37,
        DEFAULT = 39
    };
    
    void set_color(Color color) {
        std::cout << "\033[" << static_cast<int>(color) << "m";
    }
    
    void reset_color() {
        std::cout << "\033[0m";
    }
    
    // Draw a progress bar
    void progress_bar(int row, int col, int width, double percentage, Color color = Color::BLUE) {
        move_to(row, col);
        set_color(color);
        
        int filled = static_cast<int>(width * percentage);
        
        std::cout << "[";
        for (int i = 0; i < width; ++i) {
            if (i < filled) {
                std::cout << "=";
            } else if (i == filled) {
                std::cout << ">";
            } else {
                std::cout << " ";
            }
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << (percentage * 100.0) << "%";
        reset_color();
    }
}

// Monitoring application
int main(int argc, char* argv[]) {
    // Set up signal handlers
    signal(SIGINT, signal_handler);
    signal(SIGTERM, signal_handler);
    
    std::cout << "ALO Pricing System Monitor\n";
    std::cout << "==========================\n";
    std::cout << "Connecting to shared memory...\n";
    
    // Connect to shared memory
    SharedMemoryChannel channel;
    bool connected = false;
    
    // Try to connect to shared memory
    for (int attempt = 0; attempt < 5; ++attempt) {
        try {
            channel.connect("alo_pricing_system");
            connected = true;
            break;
        } catch (const std::exception& e) {
            std::cout << "Connection attempt " << (attempt + 1) << " failed: " << e.what() << "\n";
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    }
    
    if (!connected) {
        std::cerr << "Failed to connect to shared memory. Is the pricing system running?\n";
        return 1;
    }
    
    std::cout << "Connected to shared memory successfully.\n";
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    // Create counters for tracking statistics
    int total_requests = 0;
    int completed_requests = 0;
    int pending_requests = 0;
    int64_t last_heartbeat = 0;
    double avg_latency_us = 0.0;
    double max_latency_us = 0.0;
    std::string system_status = "Unknown";
    
    // Timer for updates
    auto last_update_time = std::chrono::high_resolution_clock::now();
    
    // Main monitoring loop
    while (running) {
        // Clear the screen
        console::clear();
        
        // Calculate time since last update
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed_sec = std::chrono::duration<double>(now - last_update_time).count();
        last_update_time = now;
        
        // Read shared memory data
        SharedBlock* block = channel.getSharedBlock();
        if (block) {
            // Update statistics
            int64_t current_heartbeat = block->heartbeat;
            if (current_heartbeat != last_heartbeat) {
                // System is alive
                system_status = "Running";
                last_heartbeat = current_heartbeat;
            } else {
                // No heartbeat change - possible issue
                auto heartbeat_age = std::chrono::duration_cast<std::chrono::seconds>(
                    std::chrono::high_resolution_clock::now() - 
                    std::chrono::system_clock::time_point(std::chrono::milliseconds(block->heartbeat))
                ).count();
                
                if (heartbeat_age > 5) {
                    system_status = "Not Responding";
                }
            }
            
            // Read queue information
            pending_requests = block->requestQueueSize;
            
            // Read performance metrics if available
            avg_latency_us = block->avgProcessingTimeUs;
            max_latency_us = block->maxProcessingTimeUs;
            
            // Read counters
            total_requests = block->totalRequests;
            completed_requests = block->completedRequests;
        }
        
        // Display system status
        console::move_to(1, 1);
        std::cout << "ALO Pricing System Monitor";
        
        console::move_to(3, 1);
        std::cout << "System Status: ";
        if (system_status == "Running") {
            console::set_color(console::Color::GREEN);
        } else {
            console::set_color(console::Color::RED);
        }
        std::cout << system_status;
        console::reset_color();
        
        // Display request statistics
        console::move_to(5, 1);
        std::cout << "Request Statistics:";
        
        console::move_to(6, 3);
        std::cout << "Total Requests: " << total_requests;
        
        console::move_to(7, 3);
        std::cout << "Completed Requests: " << completed_requests;
        
        console::move_to(8, 3);
        std::cout << "Pending Requests: " << pending_requests;
        
        // Display latency information
        console::move_to(10, 1);
        std::cout << "Performance Metrics:";
        
        console::move_to(11, 3);
        std::cout << "Average Processing Time: " << std::fixed << std::setprecision(2) 
                  << (avg_latency_us / 1000.0) << " ms";
        
        console::move_to(12, 3);
        std::cout << "Maximum Processing Time: " << std::fixed << std::setprecision(2) 
                  << (max_latency_us / 1000.0) << " ms";
        
        // Display queue utilization
        console::move_to(14, 1);
        std::cout << "Queue Utilization:";
        
        double queue_utilization = pending_requests / 100.0; // Assuming max queue size is 100
        queue_utilization = std::min(1.0, std::max(0.0, queue_utilization));
        
        console::move_to(15, 3);
        console::progress_bar(15, 3, 50, queue_utilization, 
                             queue_utilization > 0.8 ? console::Color::RED : 
                             queue_utilization > 0.5 ? console::Color::YELLOW : 
                             console::Color::GREEN);
        
        // Display throughput (if we have enough information)
        if (elapsed_sec > 0) {
            static int last_completed = 0;
            int completed_delta = completed_requests - last_completed;
            double throughput = completed_delta / elapsed_sec;
            last_completed = completed_requests;
            
            console::move_to(17, 1);
            std::cout << "Throughput: " << std::fixed << std::setprecision(1) 
                      << throughput << " requests/sec";
        }
        
        // Display cache hit rate (if available)
        if (block && block->cacheHitRatio >= 0) {
            console::move_to(18, 1);
            std::cout << "Cache Hit Rate: " << std::fixed << std::setprecision(1) 
                      << (block->cacheHitRatio * 100.0) << "%";
        }
        
        // Display instructions
        console::move_to(20, 1);
        console::set_color(console::Color::YELLOW);
        std::cout << "Press Ctrl+C to exit";
        console::reset_color();
        
        // Update at regular intervals
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    // Cleanup and exit
    console::clear();
    std::cout << "Disconnecting from shared memory...\n";
    channel.disconnect();
    std::cout << "Monitor terminated.\n";
    
    return 0;
}
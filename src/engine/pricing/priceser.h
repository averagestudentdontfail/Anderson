#ifndef PRICE_SERVICE_H
#define PRICE_SERVICE_H

#include "../../common/command.h"
#include "../../common/memory/objpool.h"
#include "../../common/concurrency/threadpool.h"
#include "../../alo/aloengine.h"
#include "../execution/pricecom.h"
#include "pricebat.h"
#include <memory>
#include <future>
#include <vector>
#include <map>

/**
 * @brief Service for option pricing operations
 * 
 * This service provides an interface for pricing options using the ALO engine.
 * It manages command execution, threading, and memory pooling.
 */
class PriceService {
public:
    /**
     * @brief Constructor
     * 
     * @param engine ALO pricing engine
     * @param command_queue Command queue for operations
     * @param thread_pool Thread pool for parallel processing
     * @param result_pool Memory pool for pricing results
     */
    PriceService(
        std::shared_ptr<ALOEngine> engine,
        std::shared_ptr<CommandQueue> command_queue,
        std::shared_ptr<ThreadPool> thread_pool,
        std::shared_ptr<PricingResultPool> result_pool)
        : engine_(engine), 
          command_queue_(command_queue), 
          thread_pool_(thread_pool), 
          result_pool_(result_pool) {}
    
    /**
     * @brief Price a single option
     * 
     * @param request Pricing request
     * @return Future containing the pricing result
     */
    std::future<std::shared_ptr<PricingResult>> priceOption(const PricingRequest& request) {
        auto promise = std::make_shared<std::promise<std::shared_ptr<PricingResult>>>();
        auto future = promise->get_future();
        
        // Create a pricing command
        auto command = std::make_shared<PriceCommand>(engine_, request, result_pool_);
        
        // Submit the command to the thread pool
        thread_pool_->enqueue([command, promise]() {
            if (command->execute()) {
                promise->set_value(command->getResult());
            } else {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("Failed to price option: " + command->getErrorMessage())));
            }
        });
        
        return future;
    }
    
    /**
     * @brief Price multiple options with the same parameters except strikes
     * 
     * @param spot Spot price
     * @param strikes Vector of strike prices
     * @param r Risk-free rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity
     * @param is_put True for puts, false for calls
     * @param is_american True for American options, false for European
     * @return Future containing vector of pricing results
     */
    std::future<std::vector<std::shared_ptr<PricingResult>>> priceOptions(
        double spot,
        const std::vector<double>& strikes,
        double r, double q, double vol, double T,
        bool is_put = true, bool is_american = true) {
        
        auto promise = std::make_shared<std::promise<std::vector<std::shared_ptr<PricingResult>>>>();
        auto future = promise->get_future();
        
        // Create a batch pricing command
        auto command = std::make_shared<BatchPriceCommand>(
            engine_, spot, strikes, r, q, vol, T, is_put, is_american, result_pool_);
        
        // Submit the command to the thread pool
        thread_pool_->enqueue([command, promise]() {
            if (command->execute()) {
                promise->set_value(command->getResults());
            } else {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("Failed to price options: " + command->getErrorMessage())));
            }
        });
        
        return future;
    }
    
    /**
     * @brief Price multiple options with different parameters
     * 
     * @param requests Vector of pricing requests
     * @return Future containing vector of pricing results
     */
    std::future<std::vector<std::shared_ptr<PricingResult>>> priceOptions(
        const std::vector<PricingRequest>& requests) {
        
        auto promise = std::make_shared<std::promise<std::vector<std::shared_ptr<PricingResult>>>>();
        auto future = promise->get_future();
        
        // Create a composite command with individual pricing commands
        auto composite = std::make_shared<CompositeCommand>("BatchPricing");
        
        // Vector to collect individual results
        auto results = std::make_shared<std::vector<std::shared_ptr<PricingResult>>>(
            requests.size());
        
        // Create individual commands for each request
        for (size_t i = 0; i < requests.size(); ++i) {
            auto command = std::make_shared<PriceCommand>(engine_, requests[i], result_pool_);
            
            // Capture the index to ensure correct order
            auto wrapped_command = std::make_shared<Command>();
            *wrapped_command = [command, results, i]() -> bool {
                bool success = command->execute();
                if (success) {
                    (*results)[i] = command->getResult();
                }
                return success;
            };
            
            composite->addCommand(wrapped_command);
        }
        
        // Submit the composite command to the thread pool
        thread_pool_->enqueue([composite, promise, results]() {
            if (composite->execute()) {
                promise->set_value(*results);
            } else {
                promise->set_exception(std::make_exception_ptr(
                    std::runtime_error("Failed to price options batch")));
            }
        });
        
        return future;
    }
    
    /**
     * @brief Queue a pricing command for later execution
     * 
     * @param request Pricing request
     * @return True if the command was queued successfully, false otherwise
     */
    bool queuePricingCommand(const PricingRequest& request) {
        return command_queue_->addCommand(
            std::make_shared<PriceCommand>(engine_, request, result_pool_));
    }
    
    /**
     * @brief Execute all queued commands
     * 
     * @return Number of commands executed successfully
     */
    size_t executeQueuedCommands() {
        return command_queue_->executeAll();
    }
    
    /**
     * @brief Get statistics about the pricing service
     * 
     * @return Map of statistic names to values
     */
    std::map<std::string, double> getStatistics() const {
        auto [allocated, deallocated, active, blocks, utilization] = result_pool_->getStats();
        
        return {
            {"result_pool.allocated", static_cast<double>(allocated)},
            {"result_pool.deallocated", static_cast<double>(deallocated)},
            {"result_pool.active", static_cast<double>(active)},
            {"result_pool.blocks", static_cast<double>(blocks)},
            {"result_pool.utilization", utilization},
            {"command_queue.size", static_cast<double>(command_queue_->size())},
            {"thread_pool.size", static_cast<double>(thread_pool_->size())},
            {"thread_pool.active", static_cast<double>(thread_pool_->activeCount())}
        };
    }
    
private:
    std::shared_ptr<ALOEngine> engine_;
    std::shared_ptr<CommandQueue> command_queue_;
    std::shared_ptr<ThreadPool> thread_pool_;
    std::shared_ptr<PricingResultPool> result_pool_;
};

#endif 
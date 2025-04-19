#ifndef PRICE_BATCH_H
#define PRICE_BATCH_H

#include "../../common/command.h"
#include "../../common/memory/objpool.h"
#include "../../alo/aloengine.h"
#include <memory>
#include <vector>
#include <future>
#include <algorithm>
#include <unordered_map>
#include <tuple>

/**
 * @class PriceBatch
 * @brief Handler for efficient batch processing of option prices
 * 
 * This class optimizes the pricing of multiple options by grouping
 * similar requests and processing them in batches.
 */
class PriceBatch {
public:
    /**
     * @brief Constructor
     * 
     * @param engine ALO pricing engine
     * @param result_pool Memory pool for pricing results
     */
    PriceBatch(
        std::shared_ptr<ALOEngine> engine,
        std::shared_ptr<PricingResultPool> result_pool)
        : engine_(engine), result_pool_(result_pool) {}
    
    /**
     * @brief Process multiple pricing requests efficiently
     * 
     * Groups similar requests and processes them in batches
     * 
     * @param requests Vector of pricing requests
     * @return Vector of pricing results
     */
    std::vector<std::shared_ptr<PricingResult>> processBatch(
        const std::vector<PricingRequest>& requests) {
        
        // Return empty results for empty requests
        if (requests.empty()) {
            return {};
        }
        
        // Prepare result vector
        std::vector<std::shared_ptr<PricingResult>> results(requests.size());
        
        // Group requests by common parameters
        groupRequests(requests, results);
        
        return results;
    }
    
private:
    /**
     * @brief Group similar requests for batch processing
     * 
     * @param requests Vector of pricing requests
     * @param results Vector to store the results
     */
    void groupRequests(
        const std::vector<PricingRequest>& requests,
        std::vector<std::shared_ptr<PricingResult>>& results) {
        
        // Group by all parameters except strike
        std::unordered_map<
            std::string,  // Key: S_r_q_vol_T_is_put_is_american
            std::vector<std::pair<size_t, double>>  // Value: vector of (request_index, strike)
        > groups;
        
        // Process each request
        for (size_t i = 0; i < requests.size(); ++i) {
            const auto& req = requests[i];
            
            // Create a key for grouping
            std::string key = createGroupKey(req);
            
            // Add to the appropriate group
            groups[key].emplace_back(i, req.K);
        }
        
        // Process each group
        for (auto& [key, indices_strikes] : groups) {
            processGroup(key, indices_strikes, requests, results);
        }
    }
    
    /**
     * @brief Create a key for grouping similar requests
     * 
     * @param req Pricing request
     * @return Key string
     */
    std::string createGroupKey(const PricingRequest& req) {
        std::ostringstream oss;
        oss << req.S << "_" << req.r << "_" << req.q << "_" 
            << req.vol << "_" << req.T << "_" 
            << (req.is_put ? "P" : "C") << "_"
            << (req.is_american ? "A" : "E");
        return oss.str();
    }
    
    /**
     * @brief Process a group of similar requests
     * 
     * @param key Group key
     * @param indices_strikes Vector of (request_index, strike) pairs
     * @param requests Original requests
     * @param results Results vector to populate
     */
    void processGroup(
        const std::string& key,
        const std::vector<std::pair<size_t, double>>& indices_strikes,
        const std::vector<PricingRequest>& requests,
        std::vector<std::shared_ptr<PricingResult>>& results) {
        
        // Get parameters from the first request in the group
        const size_t first_index = indices_strikes[0].first;
        const auto& first_req = requests[first_index];
        
        // Extract all strikes for batch processing
        std::vector<double> strikes;
        strikes.reserve(indices_strikes.size());
        
        for (const auto& [_, strike] : indices_strikes) {
            strikes.push_back(strike);
        }
        
        // Process the batch based on option type
        std::vector<double> prices;
        
        if (first_req.is_american) {
            if (first_req.is_put) {
                prices = engine_->batchCalculatePut(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T);
            } else {
                prices = engine_->batchCalculateCall(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T);
            }
        } else {
            // European options
            prices.resize(strikes.size());
            for (size_t i = 0; i < strikes.size(); ++i) {
                if (first_req.is_put) {
                    prices[i] = ALOEngine::blackScholesPut(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                } else {
                    prices[i] = ALOEngine::blackScholesCall(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                }
            }
        }
        
        // Calculate European prices for early exercise premium
        std::vector<double> european_prices;
        if (first_req.is_american) {
            european_prices.resize(strikes.size());
            for (size_t i = 0; i < strikes.size(); ++i) {
                if (first_req.is_put) {
                    european_prices[i] = ALOEngine::blackScholesPut(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                } else {
                    european_prices[i] = ALOEngine::blackScholesCall(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                }
            }
        }
        
        // Create result objects and assign prices
        for (size_t i = 0; i < indices_strikes.size(); ++i) {
            const size_t req_index = indices_strikes[i].first;
            
            // Create a result object from the pool
            auto result = result_pool_->get();
            
            // Set basic result properties
            result->price = prices[i];
            result->is_american = first_req.is_american;
            
            // Calculate early exercise premium if applicable
            if (first_req.is_american) {
                result->early_exercise_premium = prices[i] - european_prices[i];
            } else {
                result->early_exercise_premium = 0.0;
            }
            
            // Store in the results vector
            results[req_index] = result;
        }
        
        // Calculate Greeks for each option in the batch
        calculateGreeksForBatch(indices_strikes, requests, results);
    }
    
    /**
     * @brief Calculate Greeks for a batch of similar options
     * 
     * @param indices_strikes Vector of (request_index, strike) pairs
     * @param requests Original requests
     * @param results Results vector to populate with Greeks
     */
    void calculateGreeksForBatch(
        const std::vector<std::pair<size_t, double>>& indices_strikes,
        const std::vector<PricingRequest>& requests,
        std::vector<std::shared_ptr<PricingResult>>& results) {
        
        // Get parameters from the first request in the group
        const size_t first_index = indices_strikes[0].first;
        const auto& first_req = requests[first_index];
        
        // Step sizes for finite difference approximations
        const double h_S = std::max(0.01, first_req.S * 0.001);  // 0.1% of spot
        const double h_vol = std::max(0.001, first_req.vol * 0.01);  // 1% of vol
        const double h_r = 0.0001;  // 1 bp
        const double h_T = std::max(0.0001, first_req.T * 0.01);  // 1% of time
        
        // Extract all strikes
        std::vector<double> strikes;
        strikes.reserve(indices_strikes.size());
        
        for (const auto& [_, strike] : indices_strikes) {
            strikes.push_back(strike);
        }
        
        // Calculate prices for the up and down shifts
        std::vector<double> prices_up_S, prices_down_S;
        std::vector<double> prices_up_vol, prices_down_vol;
        std::vector<double> prices_down_T;
        std::vector<double> prices_up_r, prices_down_r;
        
        // Up/down spot prices
        if (first_req.is_american) {
            if (first_req.is_put) {
                prices_up_S = engine_->batchCalculatePut(
                    first_req.S + h_S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T);
                prices_down_S = engine_->batchCalculatePut(
                    first_req.S - h_S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T);
            } else {
                prices_up_S = engine_->batchCalculateCall(
                    first_req.S + h_S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T);
                prices_down_S = engine_->batchCalculateCall(
                    first_req.S - h_S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T);
            }
        } else {
            prices_up_S.resize(strikes.size());
            prices_down_S.resize(strikes.size());
            
            for (size_t i = 0; i < strikes.size(); ++i) {
                if (first_req.is_put) {
                    prices_up_S[i] = ALOEngine::blackScholesPut(
                        first_req.S + h_S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                    prices_down_S[i] = ALOEngine::blackScholesPut(
                        first_req.S - h_S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                } else {
                    prices_up_S[i] = ALOEngine::blackScholesCall(
                        first_req.S + h_S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                    prices_down_S[i] = ALOEngine::blackScholesCall(
                        first_req.S - h_S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T);
                }
            }
        }
        
        // Up/down volatility
        if (first_req.is_american) {
            if (first_req.is_put) {
                prices_up_vol = engine_->batchCalculatePut(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol + h_vol, first_req.T);
                prices_down_vol = engine_->batchCalculatePut(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol - h_vol, first_req.T);
            } else {
                prices_up_vol = engine_->batchCalculateCall(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol + h_vol, first_req.T);
                prices_down_vol = engine_->batchCalculateCall(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol - h_vol, first_req.T);
            }
        } else {
            prices_up_vol.resize(strikes.size());
            prices_down_vol.resize(strikes.size());
            
            for (size_t i = 0; i < strikes.size(); ++i) {
                if (first_req.is_put) {
                    prices_up_vol[i] = ALOEngine::blackScholesPut(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol + h_vol, first_req.T);
                    prices_down_vol[i] = ALOEngine::blackScholesPut(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol - h_vol, first_req.T);
                } else {
                    prices_up_vol[i] = ALOEngine::blackScholesCall(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol + h_vol, first_req.T);
                    prices_down_vol[i] = ALOEngine::blackScholesCall(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol - h_vol, first_req.T);
                }
            }
        }
        
        // Down time
        if (first_req.is_american) {
            if (first_req.is_put) {
                prices_down_T = engine_->batchCalculatePut(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T - h_T);
            } else {
                prices_down_T = engine_->batchCalculateCall(
                    first_req.S, strikes, first_req.r, first_req.q, first_req.vol, first_req.T - h_T);
            }
        } else {
            prices_down_T.resize(strikes.size());
            
            for (size_t i = 0; i < strikes.size(); ++i) {
                if (first_req.is_put) {
                    prices_down_T[i] = ALOEngine::blackScholesPut(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T - h_T);
                } else {
                    prices_down_T[i] = ALOEngine::blackScholesCall(
                        first_req.S, strikes[i], first_req.r, first_req.q, first_req.vol, first_req.T - h_T);
                }
            }
        }
        
        // Up/down rate
        if (first_req.is_american) {
            if (first_req.is_put) {
                prices_up_r = engine_->batchCalculatePut(
                    first_req.S, strikes, first_req.r + h_r, first_req.q, first_req.vol, first_req.T);
                prices_down_r = engine_->batchCalculatePut(
                    first_req.S, strikes, first_req.r - h_r, first_req.q, first_req.vol, first_req.T);
            } else {
                prices_up_r = engine_->batchCalculateCall(
                    first_req.S, strikes, first_req.r + h_r, first_req.q, first_req.vol, first_req.T);
                prices_down_r = engine_->batchCalculateCall(
                    first_req.S, strikes, first_req.r - h_r, first_req.q, first_req.vol, first_req.T);
            }
        } else {
            prices_up_r.resize(strikes.size());
            prices_down_r.resize(strikes.size());
            
            for (size_t i = 0; i < strikes.size(); ++i) {
                if (first_req.is_put) {
                    prices_up_r[i] = ALOEngine::blackScholesPut(
                        first_req.S, strikes[i], first_req.r + h_r, first_req.q, first_req.vol, first_req.T);
                    prices_down_r[i] = ALOEngine::blackScholesPut(
                        first_req.S, strikes[i], first_req.r - h_r, first_req.q, first_req.vol, first_req.T);
                } else {
                    prices_up_r[i] = ALOEngine::blackScholesCall(
                        first_req.S, strikes[i], first_req.r + h_r, first_req.q, first_req.vol, first_req.T);
                    prices_down_r[i] = ALOEngine::blackScholesCall(
                        first_req.S, strikes[i], first_req.r - h_r, first_req.q, first_req.vol, first_req.T);
                }
            }
        }
        
        // Calculate Greeks for each option
        for (size_t i = 0; i < indices_strikes.size(); ++i) {
            const size_t req_index = indices_strikes[i].first;
            auto& result = results[req_index];
            
            // Delta: dV/dS
            result->delta = (prices_up_S[i] - prices_down_S[i]) / (2 * h_S);
            
            // Gamma: d²V/dS²
            result->gamma = (prices_up_S[i] - 2 * result->price + prices_down_S[i]) / (h_S * h_S);
            
            // Vega: dV/dσ
            result->vega = (prices_up_vol[i] - prices_down_vol[i]) / (2 * h_vol) * 0.01;  // Convert to 1% move
            
            // Theta: -dV/dT
            result->theta = -(result->price - prices_down_T[i]) / h_T / 365.0;  // Daily theta
            
            // Rho: dV/dr
            result->rho = (prices_up_r[i] - prices_down_r[i]) / (2 * h_r) * 0.01;  // Convert to 1% move
        }
    }
    
private:
    std::shared_ptr<ALOEngine> engine_;
    std::shared_ptr<PricingResultPool> result_pool_;
};

#endif 
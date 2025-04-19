#ifndef PRICE_COMMAND_H
#define PRICE_COMMAND_H

#include "../../common/command.h"
#include "../../common/memory/objpool.h"
#include "../../alo/aloengine.h"
#include <memory>
#include <string>
#include <map>
#include <any>

/**
 * @brief Command for pricing a single option
 */
class PriceCommand : public Command {
public:
    /**
     * @brief Constructor
     * 
     * @param engine ALO pricing engine
     * @param request Pricing request
     * @param result_pool Pool for pricing results
     */
    PriceCommand(
        std::shared_ptr<ALOEngine> engine,
        const PricingRequest& request,
        std::shared_ptr<PricingResultPool> result_pool)
        : engine_(engine), request_(request), result_pool_(result_pool), result_() {}
    
    /**
     * @brief Execute the pricing command
     * 
     * @return True if pricing was successful, false otherwise
     */
    bool execute() override {
        return executeWithTiming([this]() {
            try {
                // Get a new result from the pool
                result_ = result_pool_->get();
                
                // Price the option
                double price;
                if (request_.is_american) {
                    if (request_.is_put) {
                        price = engine_->calculateOption(
                            request_.S, request_.K, request_.r, request_.q, 
                            request_.vol, request_.T, PUT);
                        
                        // Calculate early exercise premium
                        double european_price = ALOEngine::blackScholesPut(
                            request_.S, request_.K, request_.r, request_.q, 
                            request_.vol, request_.T);
                        
                        result_->early_exercise_premium = price - european_price;
                    } else {
                        price = engine_->calculateOption(
                            request_.S, request_.K, request_.r, request_.q, 
                            request_.vol, request_.T, CALL);
                        
                        // Calculate early exercise premium
                        double european_price = ALOEngine::blackScholesCall(
                            request_.S, request_.K, request_.r, request_.q, 
                            request_.vol, request_.T);
                        
                        result_->early_exercise_premium = price - european_price;
                    }
                } else {
                    // European option
                    if (request_.is_put) {
                        price = ALOEngine::blackScholesPut(
                            request_.S, request_.K, request_.r, request_.q, 
                            request_.vol, request_.T);
                    } else {
                        price = ALOEngine::blackScholesCall(
                            request_.S, request_.K, request_.r, request_.q, 
                            request_.vol, request_.T);
                    }
                    result_->early_exercise_premium = 0.0;
                }
                
                // Store the price
                result_->price = price;
                result_->is_american = request_.is_american;
                
                // Calculate Greeks (simple finite difference approximation)
                calculateGreeks();
                
                return true;
            } catch (const std::exception& e) {
                // Handle any exceptions
                error_message_ = e.what();
                return false;
            }
        });
    }
    
    /**
     * @brief Undo the pricing command (not supported)
     * 
     * @return Always false
     */
    bool undo() override {
        // No meaningful way to undo a pricing calculation
        return false;
    }
    
    /**
     * @brief Check if the command can be undone
     * 
     * @return Always false
     */
    bool canUndo() const override {
        return false;
    }
    
    /**
     * @brief Get the name of the command
     * 
     * @return "PriceOption"
     */
    std::string name() const override {
        return "PriceOption";
    }
    
    /**
     * @brief Get the type of the command
     * 
     * @return "Pricing"
     */
    std::string type() const override {
        return "Pricing";
    }
    
    /**
     * @brief Get the pricing result
     * 
     * @return Shared pointer to the pricing result
     */
    std::shared_ptr<PricingResult> getResult() const {
        return result_;
    }
    
    /**
     * @brief Get the error message if execution failed
     * 
     * @return Error message
     */
    std::string getErrorMessage() const {
        return error_message_;
    }
    
    /**
     * @brief Get the parameters of the command
     * 
     * @return Map of parameter names to values
     */
    std::map<std::string, std::any> parameters() const override {
        std::map<std::string, std::any> params;
        params["S"] = request_.S;
        params["K"] = request_.K;
        params["r"] = request_.r;
        params["q"] = request_.q;
        params["vol"] = request_.vol;
        params["T"] = request_.T;
        params["is_put"] = request_.is_put;
        params["is_american"] = request_.is_american;
        return params;
    }
    
    /**
     * @brief Check if the command execution is deterministic
     * 
     * @return Always true
     */
    bool isDeterministic() const override {
        // Pricing calculations are deterministic
        return true;
    }
    
private:
    /**
     * @brief Calculate the Greeks using finite difference
     */
    void calculateGreeks() {
        const double h_S = std::max(0.01, request_.S * 0.001);  // 0.1% of spot
        const double h_vol = std::max(0.001, request_.vol * 0.01);  // 1% of vol
        const double h_r = 0.0001;  // 1 bp
        const double h_T = std::max(0.0001, request_.T * 0.01);  // 1% of time
        
        // Delta: dV/dS
        PricingRequest up_S = request_;
        up_S.S += h_S;
        
        PricingRequest down_S = request_;
        down_S.S -= h_S;
        
        double price_up_S, price_down_S;
        
        if (request_.is_american) {
            price_up_S = engine_->calculateOption(
                up_S.S, up_S.K, up_S.r, up_S.q, up_S.vol, up_S.T, 
                request_.is_put ? PUT : CALL);
            
            price_down_S = engine_->calculateOption(
                down_S.S, down_S.K, down_S.r, down_S.q, down_S.vol, down_S.T, 
                request_.is_put ? PUT : CALL);
        } else {
            if (request_.is_put) {
                price_up_S = ALOEngine::blackScholesPut(
                    up_S.S, up_S.K, up_S.r, up_S.q, up_S.vol, up_S.T);
                price_down_S = ALOEngine::blackScholesPut(
                    down_S.S, down_S.K, down_S.r, down_S.q, down_S.vol, down_S.T);
            } else {
                price_up_S = ALOEngine::blackScholesCall(
                    up_S.S, up_S.K, up_S.r, up_S.q, up_S.vol, up_S.T);
                price_down_S = ALOEngine::blackScholesCall(
                    down_S.S, down_S.K, down_S.r, down_S.q, down_S.vol, down_S.T);
            }
        }
        
        result_->delta = (price_up_S - price_down_S) / (2 * h_S);
        
        // Gamma: d²V/dS²
        result_->gamma = (price_up_S - 2 * result_->price + price_down_S) / (h_S * h_S);
        
        // Vega: dV/dσ
        PricingRequest up_vol = request_;
        up_vol.vol += h_vol;
        
        PricingRequest down_vol = request_;
        down_vol.vol -= h_vol;
        
        double price_up_vol, price_down_vol;
        
        if (request_.is_american) {
            price_up_vol = engine_->calculateOption(
                up_vol.S, up_vol.K, up_vol.r, up_vol.q, up_vol.vol, up_vol.T, 
                request_.is_put ? PUT : CALL);
            
            price_down_vol = engine_->calculateOption(
                down_vol.S, down_vol.K, down_vol.r, down_vol.q, down_vol.vol, down_vol.T, 
                request_.is_put ? PUT : CALL);
        } else {
            if (request_.is_put) {
                price_up_vol = ALOEngine::blackScholesPut(
                    up_vol.S, up_vol.K, up_vol.r, up_vol.q, up_vol.vol, up_vol.T);
                price_down_vol = ALOEngine::blackScholesPut(
                    down_vol.S, down_vol.K, down_vol.r, down_vol.q, down_vol.vol, down_vol.T);
            } else {
                price_up_vol = ALOEngine::blackScholesCall(
                    up_vol.S, up_vol.K, up_vol.r, up_vol.q, up_vol.vol, up_vol.T);
                price_down_vol = ALOEngine::blackScholesCall(
                    down_vol.S, down_vol.K, down_vol.r, down_vol.q, down_vol.vol, down_vol.T);
            }
        }
        
        result_->vega = (price_up_vol - price_down_vol) / (2 * h_vol) * 0.01;  // Convert to 1% move
        
        // Theta: -dV/dT
        PricingRequest down_T = request_;
        down_T.T -= h_T;
        
        double price_down_T;
        
        if (request_.is_american) {
            price_down_T = engine_->calculateOption(
                down_T.S, down_T.K, down_T.r, down_T.q, down_T.vol, down_T.T, 
                request_.is_put ? PUT : CALL);
        } else {
            if (request_.is_put) {
                price_down_T = ALOEngine::blackScholesPut(
                    down_T.S, down_T.K, down_T.r, down_T.q, down_T.vol, down_T.T);
            } else {
                price_down_T = ALOEngine::blackScholesCall(
                    down_T.S, down_T.K, down_T.r, down_T.q, down_T.vol, down_T.T);
            }
        }
        
        // Theta is the negative of the derivative
        result_->theta = -(result_->price - price_down_T) / h_T / 365.0;  // Daily theta
        
        // Rho: dV/dr
        PricingRequest up_r = request_;
        up_r.r += h_r;
        
        PricingRequest down_r = request_;
        down_r.r -= h_r;
        
        double price_up_r, price_down_r;
        
        if (request_.is_american) {
            price_up_r = engine_->calculateOption(
                up_r.S, up_r.K, up_r.r, up_r.q, up_r.vol, up_r.T, 
                request_.is_put ? PUT : CALL);
            
            price_down_r = engine_->calculateOption(
                down_r.S, down_r.K, down_r.r, down_r.q, down_r.vol, down_r.T, 
                request_.is_put ? PUT : CALL);
        } else {
            if (request_.is_put) {
                price_up_r = ALOEngine::blackScholesPut(
                    up_r.S, up_r.K, up_r.r, up_r.q, up_r.vol, up_r.T);
                price_down_r = ALOEngine::blackScholesPut(
                    down_r.S, down_r.K, down_r.r, down_r.q, down_r.vol, down_r.T);
            } else {
                price_up_r = ALOEngine::blackScholesCall(
                    up_r.S, up_r.K, up_r.r, up_r.q, up_r.vol, up_r.T);
                price_down_r = ALOEngine::blackScholesCall(
                    down_r.S, down_r.K, down_r.r, down_r.q, down_r.vol, down_r.T);
            }
        }
        
        result_->rho = (price_up_r - price_down_r) / (2 * h_r) * 0.01;  // Convert to 1% move
    }
    
private:
    std::shared_ptr<ALOEngine> engine_;
    PricingRequest request_;
    std::shared_ptr<PricingResultPool> result_pool_;
    std::shared_ptr<PricingResult> result_;
    std::string error_message_;
};

/**
 * @brief Command for batch pricing of options
 */
class BatchPriceCommand : public Command {
public:
    /**
     * @brief Constructor
     * 
     * @param engine ALO pricing engine
     * @param spot Spot price
     * @param strikes Vector of strike prices
     * @param r Risk-free rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity
     * @param is_put True for puts, false for calls
     * @param is_american True for American options, false for European
     * @param result_pool Pool for pricing results
     */
    BatchPriceCommand(
        std::shared_ptr<ALOEngine> engine,
        double spot,
        const std::vector<double>& strikes,
        double r, double q, double vol, double T,
        bool is_put, bool is_american,
        std::shared_ptr<PricingResultPool> result_pool)
        : engine_(engine), spot_(spot), strikes_(strikes), 
          r_(r), q_(q), vol_(vol), T_(T),
          is_put_(is_put), is_american_(is_american),
          result_pool_(result_pool) {}
    
    /**
     * @brief Execute the batch pricing command
     * 
     * @return True if pricing was successful, false otherwise
     */
    bool execute() override {
        return executeWithTiming([this]() {
            try {
                // Reserve space for results
                results_.clear();
                results_.reserve(strikes_.size());
                
                // Create pricing results from the pool
                for (size_t i = 0; i < strikes_.size(); ++i) {
                    results_.push_back(result_pool_->get());
                }
                
                // Price the options in batch
                std::vector<double> prices;
                
                if (is_american_) {
                    if (is_put_) {
                        prices = engine_->batchCalculatePut(spot_, strikes_, r_, q_, vol_, T_);
                    } else {
                        prices = engine_->batchCalculateCall(spot_, strikes_, r_, q_, vol_, T_);
                    }
                } else {
                    // European options
                    prices.resize(strikes_.size());
                    for (size_t i = 0; i < strikes_.size(); ++i) {
                        if (is_put_) {
                            prices[i] = ALOEngine::blackScholesPut(spot_, strikes_[i], r_, q_, vol_, T_);
                        } else {
                            prices[i] = ALOEngine::blackScholesCall(spot_, strikes_[i], r_, q_, vol_, T_);
                        }
                    }
                }
                
                // Store the prices
                for (size_t i = 0; i < strikes_.size(); ++i) {
                    results_[i]->price = prices[i];
                    results_[i]->is_american = is_american_;
                    
                    // Calculate early exercise premium for American options
                    if (is_american_) {
                        double european_price;
                        if (is_put_) {
                            european_price = ALOEngine::blackScholesPut(spot_, strikes_[i], r_, q_, vol_, T_);
                        } else {
                            european_price = ALOEngine::blackScholesCall(spot_, strikes_[i], r_, q_, vol_, T_);
                        }
                        results_[i]->early_exercise_premium = prices[i] - european_price;
                    } else {
                        results_[i]->early_exercise_premium = 0.0;
                    }
                }
                
                // Calculate Greeks for each option (can be optimized further)
                calculateGreeks();
                
                return true;
            } catch (const std::exception& e) {
                // Handle any exceptions
                error_message_ = e.what();
                return false;
            }
        });
    }
    
    /**
     * @brief Undo the batch pricing command (not supported)
     * 
     * @return Always false
     */
    bool undo() override {
        // No meaningful way to undo a pricing calculation
        return false;
    }
    
    /**
     * @brief Check if the command can be undone
     * 
     * @return Always false
     */
    bool canUndo() const override {
        return false;
    }
    
    /**
     * @brief Get the name of the command
     * 
     * @return "BatchPriceOptions"
     */
    std::string name() const override {
        return "BatchPriceOptions";
    }
    
    /**
     * @brief Get the type of the command
     * 
     * @return "Pricing"
     */
    std::string type() const override {
        return "Pricing";
    }
    
    /**
     * @brief Get the batch pricing results
     * 
     * @return Vector of shared pointers to pricing results
     */
    std::vector<std::shared_ptr<PricingResult>> getResults() const {
        return results_;
    }
    
    /**
     * @brief Get the error message if execution failed
     * 
     * @return Error message
     */
    std::string getErrorMessage() const {
        return error_message_;
    }
    
    /**
     * @brief Get the parameters of the command
     * 
     * @return Map of parameter names to values
     */
    std::map<std::string, std::any> parameters() const override {
        std::map<std::string, std::any> params;
        params["spot"] = spot_;
        params["strikes_count"] = strikes_.size();
        params["r"] = r_;
        params["q"] = q_;
        params["vol"] = vol_;
        params["T"] = T_;
        params["is_put"] = is_put_;
        params["is_american"] = is_american_;
        return params;
    }
    
    /**
     * @brief Check if the command execution is deterministic
     * 
     * @return Always true
     */
    bool isDeterministic() const override {
        // Pricing calculations are deterministic
        return true;
    }
    
private:
    /**
     * @brief Calculate the Greeks for all options in the batch
     */
    void calculateGreeks() {
        // Simple implementation for now - can be optimized with SIMD and batch calculations
        for (size_t i = 0; i < strikes_.size(); ++i) {
            calculateGreeksForOption(i);
        }
    }
    
    /**
     * @brief Calculate the Greeks for a single option in the batch
     * 
     * @param index Index of the option
     */
    void calculateGreeksForOption(size_t index) {
        const double strike = strikes_[index];
        const double price = results_[index]->price;
        
        const double h_S = std::max(0.01, spot_ * 0.001);  // 0.1% of spot
        const double h_vol = std::max(0.001, vol_ * 0.01);  // 1% of vol
        const double h_r = 0.0001;  // 1 bp
        const double h_T = std::max(0.0001, T_ * 0.01);  // 1% of time
        
        // Delta: dV/dS
        double price_up_S, price_down_S;
        
        if (is_american_) {
            if (is_put_) {
                price_up_S = engine_->calculatePut(spot_ + h_S, strike, r_, q_, vol_, T_);
                price_down_S = engine_->calculatePut(spot_ - h_S, strike, r_, q_, vol_, T_);
            } else {
                price_up_S = engine_->calculateOption(spot_ + h_S, strike, r_, q_, vol_, T_, CALL);
                price_down_S = engine_->calculateOption(spot_ - h_S, strike, r_, q_, vol_, T_, CALL);
            }
        } else {
            if (is_put_) {
                price_up_S = ALOEngine::blackScholesPut(spot_ + h_S, strike, r_, q_, vol_, T_);
                price_down_S = ALOEngine::blackScholesPut(spot_ - h_S, strike, r_, q_, vol_, T_);
            } else {
                price_up_S = ALOEngine::blackScholesCall(spot_ + h_S, strike, r_, q_, vol_, T_);
                price_down_S = ALOEngine::blackScholesCall(spot_ - h_S, strike, r_, q_, vol_, T_);
            }
        }
        
        results_[index]->delta = (price_up_S - price_down_S) / (2 * h_S);
        
        // Gamma: d²V/dS²
        results_[index]->gamma = (price_up_S - 2 * price + price_down_S) / (h_S * h_S);
        
        // Vega: dV/dσ
        double price_up_vol, price_down_vol;
        
        if (is_american_) {
            if (is_put_) {
                price_up_vol = engine_->calculatePut(spot_, strike, r_, q_, vol_ + h_vol, T_);
                price_down_vol = engine_->calculatePut(spot_, strike, r_, q_, vol_ - h_vol, T_);
            } else {
                price_up_vol = engine_->calculateOption(spot_, strike, r_, q_, vol_ + h_vol, T_, CALL);
                price_down_vol = engine_->calculateOption(spot_, strike, r_, q_, vol_ - h_vol, T_, CALL);
            }
        } else {
            if (is_put_) {
                price_up_vol = ALOEngine::blackScholesPut(spot_, strike, r_, q_, vol_ + h_vol, T_);
                price_down_vol = ALOEngine::blackScholesPut(spot_, strike, r_, q_, vol_ - h_vol, T_);
            } else {
                price_up_vol = ALOEngine::blackScholesCall(spot_, strike, r_, q_, vol_ + h_vol, T_);
                price_down_vol = ALOEngine::blackScholesCall(spot_, strike, r_, q_, vol_ - h_vol, T_);
            }
        }
        
        results_[index]->vega = (price_up_vol - price_down_vol) / (2 * h_vol) * 0.01;  // Convert to 1% move
        
        // Theta: -dV/dT
        double price_down_T;
        
        if (is_american_) {
            if (is_put_) {
                price_down_T = engine_->calculatePut(spot_, strike, r_, q_, vol_, T_ - h_T);
            } else {
                price_down_T = engine_->calculateOption(spot_, strike, r_, q_, vol_, T_ - h_T, CALL);
            }
        } else {
            if (is_put_) {
                price_down_T = ALOEngine::blackScholesPut(spot_, strike, r_, q_, vol_, T_ - h_T);
            } else {
                price_down_T = ALOEngine::blackScholesCall(spot_, strike, r_, q_, vol_, T_ - h_T);
            }
        }
        
        // Theta is the negative of the derivative
        results_[index]->theta = -(price - price_down_T) / h_T / 365.0;  // Daily theta
        
        // Rho: dV/dr
        double price_up_r, price_down_r;
        
        if (is_american_) {
            if (is_put_) {
                price_up_r = engine_->calculatePut(spot_, strike, r_ + h_r, q_, vol_, T_);
                price_down_r = engine_->calculatePut(spot_, strike, r_ - h_r, q_, vol_, T_);
            } else {
                price_up_r = engine_->calculateOption(spot_, strike, r_ + h_r, q_, vol_, T_, CALL);
                price_down_r = engine_->calculateOption(spot_, strike, r_ - h_r, q_, vol_, T_, CALL);
            }
        } else {
            if (is_put_) {
                price_up_r = ALOEngine::blackScholesPut(spot_, strike, r_ + h_r, q_, vol_, T_);
                price_down_r = ALOEngine::blackScholesPut(spot_, strike, r_ - h_r, q_, vol_, T_);
            } else {
                price_up_r = ALOEngine::blackScholesCall(spot_, strike, r_ + h_r, q_, vol_, T_);
                price_down_r = ALOEngine::blackScholesCall(spot_, strike, r_ - h_r, q_, vol_, T_);
            }
        }
        
        results_[index]->rho = (price_up_r - price_down_r) / (2 * h_r) * 0.01;  // Convert to 1% move
    }
    
private:
    std::shared_ptr<ALOEngine> engine_;
    double spot_;
    std::vector<double> strikes_;
    double r_, q_, vol_, T_;
    bool is_put_, is_american_;
    std::shared_ptr<PricingResultPool> result_pool_;
    std::vector<std::shared_ptr<PricingResult>> results_;
    std::string error_message_;
};

#endif
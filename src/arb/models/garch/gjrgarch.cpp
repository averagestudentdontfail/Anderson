#include "gjr_garch.h"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <iostream>
#include <random>
#include <chrono>
#include <gsl/gsl_multimin.h>
#include <gsl/gsl_vector.h>

namespace vol_arb {

// Static wrapper for GSL optimization
struct GJRGARCHLikelihoodWrapper {
    const std::vector<double>* returns;
    LikelihoodType likelihoodType;
    double studentDOF;
    
    static double evaluate(const gsl_vector* parameters, void* params) {
        auto wrapper = static_cast<GJRGARCHLikelihoodWrapper*>(params);
        GJRGARCHParameters modelParams;
        
        // Extract parameters from GSL vector
        modelParams.omega = gsl_vector_get(parameters, 0);
        modelParams.alpha = gsl_vector_get(parameters, 1);
        modelParams.gamma = gsl_vector_get(parameters, 2);
        modelParams.beta = gsl_vector_get(parameters, 3);
        
        // Check if parameters are valid
        if (!modelParams.isValid()) {
            return 1e10; // High penalty for invalid parameters
        }
        
        // Create temporary model with these parameters
        GJRGARCHModel tempModel(modelParams);
        
        // Calculate negative log-likelihood (for minimization)
        double likelihood = -tempModel.getLogLikelihood(*wrapper->returns, 
                                                     wrapper->likelihoodType, 
                                                     wrapper->studentDOF);
        return likelihood;
    }
};

// Constructor
GJRGARCHModel::GJRGARCHModel(const GJRGARCHParameters& params,
                         double unconditionalVariance,
                         size_t maxHistoryLength)
    : params_(params),
      unconditionalVariance_(unconditionalVariance),
      maxHistoryLength_(maxHistoryLength) {
    
    if (!params.isValid()) {
        throw std::invalid_argument("Invalid GJR-GARCH parameters");
    }
    
    // Initialize history with unconditional variance
    variances_.push_back(unconditionalVariance_);
    
    // Initialize random number generator
    initializeRNG();
}

// Update the model with a new return observation
void GJRGARCHModel::update(double returnValue) {
    // Add the new return to history
    returns_.push_back(returnValue);
    
    // Calculate the new conditional variance
    double prevVariance = variances_.back();
    double newVariance = calculateNextVariance(prevVariance, returnValue);
    
    // Add the new variance to history
    variances_.push_back(newVariance);
    
    // Trim history if it exceeds the maximum length
    if (returns_.size() > maxHistoryLength_) {
        returns_.pop_front();
        variances_.pop_front();
    }
}

// Update the model with multiple return observations
void GJRGARCHModel::update(const std::vector<double>& returns) {
    for (double returnValue : returns) {
        update(returnValue);
    }
}

// Calculate the next conditional variance given return and previous variance
double GJRGARCHModel::calculateNextVariance(double prevVariance, double return_t) const {
    double result = params_.omega;
    result += params_.alpha * return_t * return_t;
    
    // Add leverage effect term for negative returns
    if (return_t < 0) {
        result += params_.gamma * return_t * return_t;
    }
    
    result += params_.beta * prevVariance;
    
    return result;
}

// Forecast variance h steps ahead
double GJRGARCHModel::forecastVariance(int horizon) const {
    if (horizon <= 0) {
        throw std::invalid_argument("Forecast horizon must be positive");
    }
    
    return forecastVarianceInternal(horizon);
}

// Internal forecasting method
double GJRGARCHModel::forecastVarianceInternal(int horizon, const double* shockSequence) const {
    if (variances_.empty()) {
        return unconditionalVariance_;
    }
    
    double forecastedVar = variances_.back();
    
    // For h-step ahead forecast
    for (int i = 0; i < horizon; ++i) {
        // If shock sequence is provided, use it; otherwise, use expected value
        double expectedSquaredShock = 1.0; // E[z²] = 1 for standard normal
        
        if (shockSequence != nullptr) {
            double shock = shockSequence[i];
            expectedSquaredShock = shock * shock;
        }
        
        // Calculate next variance using GARCH recursion
        // For forecasting we use E[(gamma*I_{t+1} + alpha)*e_{t+1}²] = (gamma/2 + alpha) for standard normal shocks
        // Since P(z < 0) = 0.5 for standard normal
        forecastedVar = params_.omega + (params_.alpha + 0.5 * params_.gamma) * expectedSquaredShock * forecastedVar + params_.beta * forecastedVar;
    }
    
    return forecastedVar;
}

// Forecast volatility h steps ahead
double GJRGARCHModel::forecastVolatility(int horizon) const {
    return std::sqrt(forecastVariance(horizon));
}

// Get the current conditional variance
double GJRGARCHModel::getCurrentVariance() const {
    if (variances_.empty()) {
        return unconditionalVariance_;
    }
    return variances_.back();
}

// Get the current conditional volatility
double GJRGARCHModel::getCurrentVolatility() const {
    return std::sqrt(getCurrentVariance());
}

// Estimate parameters from historical returns
void GJRGARCHModel::calibrate(const std::vector<double>& returns, 
                            LikelihoodType likelihoodType,
                            double studentDOF) {
    if (returns.size() < 30) {
        throw std::invalid_argument("Insufficient data for calibration (need at least 30 points)");
    }
    
    // Calculate initial unconditional variance estimate
    double initialVariance = 0.0;
    for (double ret : returns) {
        initialVariance += ret * ret;
    }
    initialVariance /= returns.size();
    
    // Set up GSL minimizer
    const gsl_multimin_fminimizer_type* T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer* s = nullptr;
    gsl_vector* ss = nullptr;
    gsl_vector* x = nullptr;
    gsl_multimin_function minex_func;
    
    int status;
    size_t iter = 0;
    const size_t maxIter = 1000;
    
    try {
        // Initial parameter values
        x = gsl_vector_alloc(4);
        gsl_vector_set(x, 0, 0.00001);  // omega
        gsl_vector_set(x, 1, 0.05);     // alpha
        gsl_vector_set(x, 2, 0.1);      // gamma
        gsl_vector_set(x, 3, 0.85);     // beta
        
        // Set initial step sizes
        ss = gsl_vector_alloc(4);
        gsl_vector_set(ss, 0, 0.00001);
        gsl_vector_set(ss, 1, 0.01);
        gsl_vector_set(ss, 2, 0.01);
        gsl_vector_set(ss, 3, 0.01);
        
        // Initialize wrapper for likelihood function
        GJRGARCHLikelihoodWrapper wrapper;
        wrapper.returns = &returns;
        wrapper.likelihoodType = likelihoodType;
        wrapper.studentDOF = studentDOF;
        
        // Initialize minimizer function
        minex_func.n = 4;
        minex_func.f = GJRGARCHLikelihoodWrapper::evaluate;
        minex_func.params = &wrapper;
        
        // Initialize minimizer
        s = gsl_multimin_fminimizer_alloc(T, 4);
        gsl_multimin_fminimizer_set(s, &minex_func, x, ss);
        
        // Minimization loop
        const double epsabs = 1e-4;
        do {
            iter++;
            status = gsl_multimin_fminimizer_iterate(s);
            
            if (status != GSL_SUCCESS) {
                break;
            }
            
            double size = gsl_multimin_fminimizer_size(s);
            status = gsl_multimin_test_size(size, epsabs);
            
        } while (status == GSL_CONTINUE && iter < maxIter);
        
        // Extract optimized parameters
        GJRGARCHParameters optimizedParams;
        optimizedParams.omega = gsl_vector_get(s->x, 0);
        optimizedParams.alpha = gsl_vector_get(s->x, 1);
        optimizedParams.gamma = gsl_vector_get(s->x, 2);
        optimizedParams.beta = gsl_vector_get(s->x, 3);
        
        // Check if parameters are valid
        if (!optimizedParams.isValid()) {
            std::cerr << "Warning: Optimized parameters are invalid, using default values" << std::endl;
            // Keep the current parameters
        } else {
            // Update model parameters
            params_ = optimizedParams;
        }
        
        // Recalculate unconditional variance
        unconditionalVariance_ = getUnconditionalVariance();
        
        // Reset model state and incorporate all returns
        reset(unconditionalVariance_);
        update(returns);
        
    } catch (const std::exception& e) {
        std::cerr << "Error in GJR-GARCH calibration: " << e.what() << std::endl;
        // Free GSL resources before rethrowing
        if (x) gsl_vector_free(x);
        if (ss) gsl_vector_free(ss);
        if (s) gsl_multimin_fminimizer_free(s);
        throw;
    }
    
    // Free GSL resources
    gsl_vector_free(x);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free(s);
}

// Set model parameters directly
void GJRGARCHModel::setParameters(const GJRGARCHParameters& params) {
    if (!params.isValid()) {
        throw std::invalid_argument("Invalid GJR-GARCH parameters");
    }
    
    params_ = params;
    
    // Update unconditional variance
    unconditionalVariance_ = getUnconditionalVariance();
}

// Get estimated long-run (unconditional) variance
double GJRGARCHModel::getUnconditionalVariance() const {
    if (!params_.isStationary()) {
        // Return the current variance if model is non-stationary
        return getCurrentVariance();
    }
    
    // For GJR-GARCH, the unconditional variance is:
    // omega / (1 - alpha - beta - gamma/2)
    return params_.omega / (1.0 - params_.alpha - params_.beta - 0.5 * params_.gamma);
}

// Reset the model to initial state
void GJRGARCHModel::reset(double unconditionalVariance) {
    returns_.clear();
    variances_.clear();
    unconditionalVariance_ = unconditionalVariance;
    variances_.push_back(unconditionalVariance_);
}

// Get the log-likelihood of the model given the data
double GJRGARCHModel::getLogLikelihood(const std::vector<double>& returns, 
                                     LikelihoodType likelihoodType,
                                     double studentDOF) const {
    if (returns.empty()) {
        return 0.0;
    }
    
    const double LOG_2PI = std::log(2.0 * M_PI);
    double logLikelihood = 0.0;
    
    // Initialize with unconditional variance
    double variance = unconditionalVariance_;
    
    for (double ret : returns) {
        if (likelihoodType == GAUSSIAN) {
            // Log-likelihood for normal distribution
            logLikelihood += -0.5 * (LOG_2PI + std::log(variance) + ret * ret / variance);
        } else {
            // Log-likelihood for Student's t distribution
            double nu = studentDOF;
            logLikelihood += std::lgamma((nu + 1.0) / 2.0) - std::lgamma(nu / 2.0) - 0.5 * std::log(M_PI * (nu - 2.0) * variance);
            logLikelihood += -0.5 * (nu + 1.0) * std::log(1.0 + (ret * ret) / (variance * (nu - 2.0)));
        }
        
        // Update variance for next observation
        variance = calculateNextVariance(variance, ret);
    }
    
    return logLikelihood;
}

// Generate simulated paths for Monte Carlo analysis
std::vector<std::vector<double>> GJRGARCHModel::simulatePaths(int numPaths, 
                                                           int horizonDays,
                                                           const std::function<double()>& randomGenerator) const {
    if (numPaths <= 0 || horizonDays <= 0) {
        throw std::invalid_argument("Number of paths and horizon must be positive");
    }
    
    std::vector<std::vector<double>> paths(numPaths, std::vector<double>(horizonDays));
    
    // Get starting variance
    double startVariance = getCurrentVariance();
    
    // Define random number generator if not provided
    std::function<double()> rng;
    if (randomGenerator) {
        rng = randomGenerator;
    } else {
        rng = [this]() {
            return normalDist_(rng_);
        };
    }
    
    // Generate paths
    for (int path = 0; path < numPaths; ++path) {
        double variance = startVariance;
        
        for (int day = 0; day < horizonDays; ++day) {
            // Generate random shock
            double z = rng();
            
            // Calculate return based on volatility
            double ret = std::sqrt(variance) * z;
            paths[path][day] = ret;
            
            // Update variance for next day
            variance = calculateNextVariance(variance, ret);
        }
    }
    
    return paths;
}

// Initialize random number generator
void GJRGARCHModel::initializeRNG() {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    rng_ = std::mt19937(seed);
    normalDist_ = std::normal_distribution<double>(0.0, 1.0);
}

// Utility functions implementation
namespace gjr_garch_utils {

// Convert daily volatility to annualized
double dailyToAnnualizedVol(double dailyVol, int tradingDaysPerYear) {
    return dailyVol * std::sqrt(static_cast<double>(tradingDaysPerYear));
}

// Convert annualized volatility to daily
double annualizedToDailyVol(double annualizedVol, int tradingDaysPerYear) {
    return annualizedVol / std::sqrt(static_cast<double>(tradingDaysPerYear));
}

// Calculate realized volatility from returns
double calculateRealizedVolatility(const std::vector<double>& returns, bool annualized) {
    if (returns.empty()) {
        return 0.0;
    }
    
    double sumSquared = 0.0;
    for (double ret : returns) {
        sumSquared += ret * ret;
    }
    
    double variance = sumSquared / returns.size();
    double stdDev = std::sqrt(variance);
    
    if (annualized) {
        return dailyToAnnualizedVol(stdDev);
    } else {
        return stdDev;
    }
}

// Calculate log returns from price series
std::vector<double> calculateLogReturns(const std::vector<double>& prices) {
    if (prices.size() < 2) {
        return {};
    }
    
    std::vector<double> returns(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i-1] <= 0.0 || prices[i] <= 0.0) {
            throw std::invalid_argument("Prices must be positive for log returns");
        }
        returns[i-1] = std::log(prices[i] / prices[i-1]);
    }
    
    return returns;
}

// Calculate percentage returns from price series
std::vector<double> calculatePercentReturns(const std::vector<double>& prices) {
    if (prices.size() < 2) {
        return {};
    }
    
    std::vector<double> returns(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        if (prices[i-1] <= 0.0) {
            throw std::invalid_argument("Previous price must be positive for percent returns");
        }
        returns[i-1] = (prices[i] - prices[i-1]) / prices[i-1];
    }
    
    return returns;
}

} // namespace gjr_garch_utils

} // namespace vol_arb
// gjr_garch.h
// GJR-GARCH model for volatility forecasting with leverage effects

#ifndef GJR_GARCH_H
#define GJR_GARCH_H

#include <vector>
#include <deque>
#include <memory>
#include <cmath>
#include <string>
#include <random>
#include <functional>
#include <unordered_map>

namespace vol_arb {

// Structure to hold GJR-GARCH model parameters
struct GJRGARCHParameters {
    double omega;    // Constant term
    double alpha;    // ARCH parameter (impact of past squared returns)
    double gamma;    // Leverage effect parameter (additional impact of negative returns)
    double beta;     // GARCH parameter (persistence of volatility)
    
    // Default constructor with typical values
    GJRGARCHParameters(double omega = 0.000001, 
                      double alpha = 0.05, 
                      double gamma = 0.1, 
                      double beta = 0.85)
        : omega(omega), alpha(alpha), gamma(gamma), beta(beta) {}
    
    // Check if parameters satisfy stationarity condition
    bool isStationary() const {
        return (alpha + beta + 0.5 * gamma < 1.0);
    }
    
    // Check if parameters are valid (positive and in reasonable range)
    bool isValid() const {
        return (omega > 0.0 && alpha >= 0.0 && gamma >= 0.0 && beta >= 0.0 && 
                alpha + beta < 1.0 && isStationary());
    }
};

// Enum for likelihood function calculation method
enum LikelihoodType {
    GAUSSIAN,
    STUDENT_T
};

// Class for GJR-GARCH model
class GJRGARCHModel {
public:
    // Constructor with parameters
    explicit GJRGARCHModel(const GJRGARCHParameters& params = GJRGARCHParameters(),
                         double unconditionalVariance = 0.04, // Default ~ 20% annualized vol
                         size_t maxHistoryLength = 252);  // Default to ~1 year of daily data
    
    // Update the model with a new return observation
    void update(double returnValue);
    
    // Update the model with multiple return observations
    void update(const std::vector<double>& returns);
    
    // Forecast volatility h steps ahead
    double forecastVolatility(int horizon = 1) const;
    
    // Forecast variance h steps ahead
    double forecastVariance(int horizon = 1) const;
    
    // Get the current conditional variance (most recent estimate)
    double getCurrentVariance() const;
    
    // Get the current conditional volatility (sqrt of variance)
    double getCurrentVolatility() const;
    
    // Estimate parameters from historical returns
    void calibrate(const std::vector<double>& returns, 
                  LikelihoodType likelihoodType = GAUSSIAN,
                  double studentDOF = 5.0);
    
    // Set model parameters directly
    void setParameters(const GJRGARCHParameters& params);
    
    // Get current model parameters
    const GJRGARCHParameters& getParameters() const { return params_; }
    
    // Get estimated long-run (unconditional) variance
    double getUnconditionalVariance() const;
    
    // Reset the model to initial state
    void reset(double unconditionalVariance = 0.04);
    
    // Get the log-likelihood of the model given the data
    double getLogLikelihood(const std::vector<double>& returns, 
                          LikelihoodType likelihoodType = GAUSSIAN,
                          double studentDOF = 5.0) const;
    
    // Generate simulated paths for Monte Carlo analysis
    std::vector<std::vector<double>> simulatePaths(int numPaths, 
                                                int horizonDays, 
                                                const std::function<double()>& randomGenerator = nullptr) const;

private:
    GJRGARCHParameters params_;         // Model parameters
    std::deque<double> returns_;        // Historical returns
    std::deque<double> variances_;      // Historical conditional variances
    double unconditionalVariance_;      // Long-run (unconditional) variance
    size_t maxHistoryLength_;           // Maximum length of return/variance history to keep
    
    // Internal forecasting methods
    double forecastVarianceInternal(int horizon, const double* shockSequence = nullptr) const;
    
    // Random number generator for simulation
    mutable std::mt19937 rng_;
    mutable std::normal_distribution<double> normalDist_;
    
    // Calculate the next conditional variance given return and previous variance
    double calculateNextVariance(double prevVariance, double return_t) const;
    
    // Initialize random number generator
    void initializeRNG();
};

// Utility functions for the GJR-GARCH model
namespace gjr_garch_utils {
    // Convert daily volatility to annualized (assuming 252 trading days)
    double dailyToAnnualizedVol(double dailyVol, int tradingDaysPerYear = 252);
    
    // Convert annualized volatility to daily
    double annualizedToDailyVol(double annualizedVol, int tradingDaysPerYear = 252);
    
    // Calculate realized volatility from returns
    double calculateRealizedVolatility(const std::vector<double>& returns, bool annualized = true);
    
    // Calculate log returns from price series
    std::vector<double> calculateLogReturns(const std::vector<double>& prices);
    
    // Calculate percentage returns from price series
    std::vector<double> calculatePercentReturns(const std::vector<double>& prices);
}

} // namespace vol_arb

#endif // GJR_GARCH_H
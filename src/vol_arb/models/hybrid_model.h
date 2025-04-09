// hybrid_model.h
// Combines GJR-GARCH and HMM models for regime-aware volatility forecasting

#ifndef HYBRID_MODEL_H
#define HYBRID_MODEL_H

#include "gjr_garch.h"
#include "hmm.h"
#include <memory>
#include <vector>
#include <string>
#include <chrono>
#include <unordered_map>

namespace vol_arb {

// Structure to hold market data for model updates
struct MarketData {
    std::string symbol;
    double price;
    double volume;
    double open;
    double high;
    double low;
    double close;
    uint64_t timestampNanos;
    
    // Default constructor
    MarketData() : price(0), volume(0), open(0), high(0), low(0), close(0), timestampNanos(0) {}
    
    // Constructor with essential fields
    MarketData(const std::string& symbol, double price, uint64_t timestamp)
        : symbol(symbol), price(price), volume(0), open(0), high(0), low(0), close(0), 
          timestampNanos(timestamp) {}
    
    // Constructor with all fields
    MarketData(const std::string& symbol, double price, double volume, 
              double open, double high, double low, double close,
              uint64_t timestamp)
        : symbol(symbol), price(price), volume(volume), open(open), high(high), low(low),
          close(close), timestampNanos(timestamp) {}
};

// Structure for rolling statistics with time-based expiry
template<typename T>
class RollingStatistics {
public:
    // Constructor with window size in seconds
    explicit RollingStatistics(double windowSizeSeconds = 86400.0) // Default: 1 day
        : windowSizeNanos(static_cast<uint64_t>(windowSizeSeconds * 1e9)) {}
    
    // Add a new data point
    void add(const T& data, uint64_t timestampNanos) {
        dataPoints.emplace_back(data, timestampNanos);
        
        // Remove expired data points
        uint64_t cutoffTime = timestampNanos - windowSizeNanos;
        while (!dataPoints.empty() && dataPoints.front().second < cutoffTime) {
            dataPoints.erase(dataPoints.begin());
        }
    }
    
    // Get all data points in the window
    std::vector<T> getData() const {
        std::vector<T> result;
        result.reserve(dataPoints.size());
        for (const auto& point : dataPoints) {
            result.push_back(point.first);
        }
        return result;
    }
    
    // Get timestamps
    std::vector<uint64_t> getTimestamps() const {
        std::vector<uint64_t> result;
        result.reserve(dataPoints.size());
        for (const auto& point : dataPoints) {
            result.push_back(point.second);
        }
        return result;
    }
    
    // Get the most recent data point
    T getLast() const {
        if (dataPoints.empty()) {
            throw std::runtime_error("No data points in rolling statistics");
        }
        return dataPoints.back().first;
    }
    
    // Get the most recent timestamp
    uint64_t getLastTimestamp() const {
        if (dataPoints.empty()) {
            return 0;
        }
        return dataPoints.back().second;
    }
    
    // Get number of data points
    size_t size() const {
        return dataPoints.size();
    }
    
    // Check if empty
    bool empty() const {
        return dataPoints.empty();
    }
    
    // Clear all data
    void clear() {
        dataPoints.clear();
    }
    
private:
    std::vector<std::pair<T, uint64_t>> dataPoints;
    uint64_t windowSizeNanos;
};

// Time window structure for feature extraction
struct TimeWindow {
    uint64_t startNanos;
    uint64_t endNanos;
    
    TimeWindow() : startNanos(0), endNanos(0) {}
    
    TimeWindow(uint64_t start, uint64_t end) : startNanos(start), endNanos(end) {}
    
    // Create window for the last N seconds
    static TimeWindow lastSeconds(uint64_t seconds, uint64_t nowNanos = 0) {
        if (nowNanos == 0) {
            nowNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::system_clock::now().time_since_epoch()).count();
        }
        return TimeWindow(nowNanos - seconds * 1'000'000'000ULL, nowNanos);
    }
    
    // Duration in seconds
    double durationSeconds() const {
        return static_cast<double>(endNanos - startNanos) / 1'000'000'000.0;
    }
};

// Structure for intraday metrics that feed into the HMM
struct IntradayMetrics {
    double realizedVolatility;     // Annualized volatility based on recent returns
    double volumeRatio;            // Ratio of recent volume to average volume
    double returnSkewness;         // Skewness of recent returns
    double returnKurtosis;         // Excess kurtosis of recent returns
    double priceRange;             // High-Low range as percentage of average price
    double bidAskSpread;           // Average bid-ask spread
    double volatilityOfVolatility; // Volatility of volatility - how stable is vol
    double priceAcceleration;      // Second derivative of price - acceleration
    double volumeSurge;            // Recent volume surge relative to average
    uint64_t timestampNanos;       // Timestamp when metrics were calculated
    
    IntradayMetrics() 
        : realizedVolatility(0), volumeRatio(0), returnSkewness(0), 
          returnKurtosis(0), priceRange(0), bidAskSpread(0),
          volatilityOfVolatility(0), priceAcceleration(0), volumeSurge(0),
          timestampNanos(0) {}
    
    // Convert to feature vector for HMM
    Vector toFeatureVector() const {
        Vector features(6);
        features << realizedVolatility, 
                   volumeRatio, 
                   returnSkewness, 
                   returnKurtosis,
                   volatilityOfVolatility,
                   volumeSurge;
        return features;
    }
};

// Main hybrid model class combining GJR-GARCH with HMM
class HMM_GJRGARCHModel {
public:
    // Constructor
    HMM_GJRGARCHModel(int regimeCount = 3,
                    const std::vector<MarketRegimeParams>& initialRegimeParams = {},
                    double calibrationFreqMins = 30.0);
    
    // Update model with new market data
    void updateWithMarketData(const MarketData& data);
    
    // Recalibrate models based on recent data
    void recalibrateModels(bool forceRecalibration = false);
    
    // Forecast volatility for a specific symbol
    double forecastVolatility(const std::string& symbol, int daysAhead = 1);
    
    // Get regime probabilities
    std::vector<double> getRegimeProbabilities() const;
    
    // Get probability of regime transition
    double getProbabilityOfRegimeTransition() const;
    
    // Get the current dominant regime
    MarketRegime getCurrentDominantRegime() const;
    
    // Get confidence level in volatility forecast (0-1)
    double getVolatilityForecastConfidence() const;
    
    // Extract volatility features from market data
    IntradayMetrics extractVolatilityFeatures(const std::string& symbol, 
                                            const TimeWindow& window);
                                            
    // Set the HMM transition matrix for enhanced regime detection
    void setHMMTransitionMatrix(const Matrix& transitionMatrix);
    
    // Force regime state for testing purposes
    void forceRegimeState(const Vector& probabilities);
    
    // Helper method to calculate volatility-of-volatility
    double calculateVolOfVol(const std::vector<double>& returns);
    
    // Helper method to calculate price acceleration
    double calculatePriceAcceleration(const std::vector<double>& prices);
    
private:
    // Core models
    HiddenMarkovModel hmm_;
    std::unordered_map<std::string, GJRGARCHModel> garchModels_;
    
    // Model parameters for each regime
    std::vector<GJRGARCHParameters> regimeGarchParams_;
    
    // Data storage
    std::unordered_map<std::string, RollingStatistics<double>> priceSeries_;
    std::unordered_map<std::string, RollingStatistics<double>> returnSeries_;
    std::unordered_map<std::string, RollingStatistics<double>> volumeSeries_;
    RollingStatistics<IntradayMetrics> volatilityMetrics_;
    
    // Calibration variables
    std::chrono::steady_clock::time_point lastCalibrationTime_;
    double calibrationFrequencyMins_;
    
    // Helper methods
    void updateReturnSeries(const std::string& symbol, double price, uint64_t timestamp);
    GJRGARCHParameters getRegimeAdaptedParameters(const std::string& symbol) const;
    void initializeDefaultRegimeParameters();
};

} // namespace vol_arb

#endif // HYBRID_MODEL_H
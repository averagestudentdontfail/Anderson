#include "hybrid_model.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>

namespace vol_arb {

HMM_GJRGARCHModel::HMM_GJRGARCHModel(int regimeCount,
                                  const std::vector<MarketRegimeParams>& initialRegimeParams,
                                  double calibrationFreqMins)
    : hmm_(regimeCount),
      volatilityMetrics_(86400.0), // 1 day of metrics
      calibrationFrequencyMins_(calibrationFreqMins) {
    
    // Initialize HMM if parameters provided
    if (!initialRegimeParams.empty()) {
        if (static_cast<int>(initialRegimeParams.size()) != regimeCount) {
            throw std::invalid_argument("Number of regime parameters does not match regime count");
        }
        
        // Set emission distributions
        std::vector<EmissionDistribution> distributions;
        std::vector<std::string> regimeNames;
        
        for (const auto& params : initialRegimeParams) {
            distributions.push_back(params.distribution);
            regimeNames.push_back(params.name);
        }
        
        hmm_.setEmissionDistributions(distributions);
        hmm_.setRegimeNames(regimeNames);
    } else {
        // Initialize with default parameters
        hmm_.initializeWithDefaultRegimes(6); // Updated to 6-dimensional features
    }
    
    // Initialize regime-specific GARCH parameters
    initializeDefaultRegimeParameters();
    
    // Initialize calibration time
    lastCalibrationTime_ = std::chrono::steady_clock::now();
}

void HMM_GJRGARCHModel::updateWithMarketData(const MarketData& data) {
    // Add to price series
    priceSeries_[data.symbol].add(data.price, data.timestampNanos);
    
    // Add to volume series if available
    if (data.volume > 0) {
        volumeSeries_[data.symbol].add(data.volume, data.timestampNanos);
    }
    
    // Update return series
    updateReturnSeries(data.symbol, data.price, data.timestampNanos);
    
    // Check if it's time to recalibrate
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> timeSinceCalibration = now - lastCalibrationTime_;
    double minutesSinceCalibration = timeSinceCalibration.count() / 60.0;
    
    if (minutesSinceCalibration >= calibrationFrequencyMins_) {
        recalibrateModels(false);
    }
}

void HMM_GJRGARCHModel::recalibrateModels(bool forceRecalibration) {
    // Skip if no data is available
    if (priceSeries_.empty()) {
        return;
    }
    
    // Compute intraday metrics for the last few hours
    TimeWindow window = TimeWindow::lastSeconds(4 * 3600); // Last 4 hours
    
    for (const auto& [symbol, _] : priceSeries_) {
        IntradayMetrics metrics = extractVolatilityFeatures(symbol, window);
        volatilityMetrics_.add(metrics, metrics.timestampNanos);
        
        // Update HMM with latest feature vector
        Vector featureVector = metrics.toFeatureVector();
        hmm_.update(featureVector);
        
        // Update GARCH model with latest return
        if (!returnSeries_[symbol].empty() && returnSeries_[symbol].size() >= 2) {
            auto returns = returnSeries_[symbol].getData();
            double latestReturn = returns.back();
            
            // Create or update GARCH model for this symbol
            if (garchModels_.find(symbol) == garchModels_.end()) {
                // Create new model with adapted parameters based on current regime
                GJRGARCHParameters params = getRegimeAdaptedParameters(symbol);
                garchModels_[symbol] = GJRGARCHModel(params);
            }
            
            // Update the model with new return
            garchModels_[symbol].update(latestReturn);
        }
    }
    
    // Full recalibration with longer history if forced or needed
    if (forceRecalibration) {
        // Collect feature vectors for HMM training
        std::vector<Vector> featureVectors;
        for (const auto& metrics : volatilityMetrics_.getData()) {
            featureVectors.push_back(metrics.toFeatureVector());
        }
        
        // Only train if we have enough data
        if (featureVectors.size() >= 30) {
            try {
                hmm_.train(featureVectors, 100, 1e-4);
            } catch (const std::exception& e) {
                std::cerr << "Error training HMM: " << e.what() << std::endl;
            }
        }
        
        // Recalibrate GARCH models for each symbol
        for (auto& [symbol, garchModel] : garchModels_) {
            if (returnSeries_[symbol].size() >= 100) {  // Need sufficient history
                std::vector<double> returns = returnSeries_[symbol].getData();
                try {
                    garchModel.calibrate(returns);
                } catch (const std::exception& e) {
                    std::cerr << "Error calibrating GARCH model for " << symbol 
                              << ": " << e.what() << std::endl;
                }
            }
        }
    }
    
    // Update calibration time
    lastCalibrationTime_ = std::chrono::steady_clock::now();
}

double HMM_GJRGARCHModel::forecastVolatility(const std::string& symbol, int daysAhead) {
    // Check if we have data for this symbol
    if (priceSeries_.find(symbol) == priceSeries_.end() ||
        returnSeries_.find(symbol) == returnSeries_.end() || 
        returnSeries_[symbol].empty()) {
        
        // If no data is available, return a default volatility
        // based on current regime probabilities
        std::vector<double> regimeProbs = getRegimeProbabilities();
        double defaultVol = 0.0;
        
        // Weight by regime probabilities
        if (regimeProbs.size() >= 3) {
            defaultVol += regimeProbs[0] * 0.15; // Low vol regime ~ 15%
            defaultVol += regimeProbs[1] * 0.25; // Medium vol regime ~ 25%
            defaultVol += regimeProbs[2] * 0.40; // High vol regime ~ 40%
        } else {
            defaultVol = 0.20; // Default 20% if no regime info
        }
        
        return defaultVol;
    }
   
    // Ensure GARCH model exists
    if (garchModels_.find(symbol) == garchModels_.end()) {
        // Create new model with regime-adapted parameters
        GJRGARCHParameters params = getRegimeAdaptedParameters(symbol);
        garchModels_[symbol] = GJRGARCHModel(params);
       
        // Initialize with available returns
        if (returnSeries_[symbol].size() >= 2) {
            std::vector<double> returns = returnSeries_[symbol].getData();
            garchModels_[symbol].update(returns);
        }
    }
   
    // Get current regime probabilities
    std::vector<double> regimeProbs = getRegimeProbabilities();
   
    // Forecast volatility for each regime
    double weightedVolatility = 0.0;
   
    // If we have a trained GARCH model, use it for base forecast
    double baseVolatility = garchModels_[symbol].forecastVolatility(daysAhead);
   
    // Apply regime-specific adjustments
    int numRegimes = regimeProbs.size();
    for (int i = 0; i < numRegimes; ++i) {
        double regimeProb = regimeProbs[i];
       
        // Adjust forecast based on regime
        double regimeAdjustment = 1.0;  // Default: no adjustment
       
        if (i == 0) {  // Low volatility regime
            regimeAdjustment = 0.85;
        } else if (i == numRegimes - 1) {  // High volatility regime
            regimeAdjustment = 1.3;
        } else if (i == 1 || i == numRegimes - 2) {  // Transition regimes
            regimeAdjustment = 1.1;
        }
       
        weightedVolatility += regimeProb * baseVolatility * regimeAdjustment;
    }
   
    return weightedVolatility;
}

std::vector<double> HMM_GJRGARCHModel::getRegimeProbabilities() const {
    Vector probs = hmm_.getStateProbabilities();
    std::vector<double> result(probs.size());
    for (int i = 0; i < probs.size(); ++i) {
        result[i] = probs(i);
    }
    return result;
}

double HMM_GJRGARCHModel::getProbabilityOfRegimeTransition() const {
    // Use a 5-day horizon for transition probability
    return hmm_.predictRegimeChangeProbability(5);
}

MarketRegime HMM_GJRGARCHModel::getCurrentDominantRegime() const {
    return hmm_.getCurrentRegime();
}

double HMM_GJRGARCHModel::getVolatilityForecastConfidence() const {
    // Confidence is higher when one regime has high probability
    Vector probs = hmm_.getStateProbabilities();
    
    // Calculate entropy of probability distribution (normalized by max entropy)
    double entropy = 0.0;
    double maxEntropy = std::log(probs.size());
    
    for (int i = 0; i < probs.size(); ++i) {
        if (probs(i) > 0) {
            entropy -= probs(i) * std::log(probs(i));
        }
    }
    
    // Convert to confidence (1 - normalized entropy)
    double normalizedEntropy = entropy / maxEntropy;
    return 1.0 - normalizedEntropy;
}

IntradayMetrics HMM_GJRGARCHModel::extractVolatilityFeatures(const std::string& symbol, 
                                                         const TimeWindow& window) {
    IntradayMetrics metrics;
    
    // Check if we have data for this symbol
    if (priceSeries_.find(symbol) == priceSeries_.end()) {
        return metrics;
    }
    
    auto priceData = priceSeries_[symbol].getData();
    auto priceTimestamps = priceSeries_[symbol].getTimestamps();
    
    if (priceData.empty()) {
        return metrics;
    }
    
    // Filter data within the time window
    std::vector<double> windowPrices;
    std::vector<double> windowReturns;
    std::vector<double> volumeData;
    
    for (size_t i = 0; i < priceData.size(); ++i) {
        if (priceTimestamps[i] >= window.startNanos && priceTimestamps[i] <= window.endNanos) {
            windowPrices.push_back(priceData[i]);
            
            if (i > 0) {
                double ret = std::log(priceData[i] / priceData[i-1]);
                windowReturns.push_back(ret);
            }
        }
    }
    
    // Also collect volume data if available
    if (volumeSeries_.find(symbol) != volumeSeries_.end()) {
        auto volData = volumeSeries_[symbol].getData();
        auto volTimestamps = volumeSeries_[symbol].getTimestamps();
        
        for (size_t i = 0; i < volData.size(); ++i) {
            if (volTimestamps[i] >= window.startNanos && volTimestamps[i] <= window.endNanos) {
                volumeData.push_back(volData[i]);
            }
        }
    }
    
    // Need at least a few points for meaningful calculation
    if (windowPrices.size() < 3 || windowReturns.size() < 2) {
        return metrics;
    }
    
    // Calculate realized volatility
    double sum = 0.0, sumSq = 0.0;
    for (double ret : windowReturns) {
        sum += ret;
        sumSq += ret * ret;
    }
    double mean = sum / windowReturns.size();
    double variance = (sumSq / windowReturns.size()) - (mean * mean);
    double volatility = std::sqrt(std::max(0.0, variance));
    
    // Annualize volatility (assuming returns are in appropriate frequency)
    double periodsPerYear = 252.0 * 6.5 * 3600.0 / window.durationSeconds();
    
    // Make volatility feature more responsive
    metrics.realizedVolatility = volatility * std::sqrt(periodsPerYear) * 1.5; // Amplify the signal
    
    // Add more regime-sensitive features
    if (window.durationSeconds() > 0) {
        // Calculate volatility of volatility (important regime indicator)
        // This measures how much the volatility itself is changing
        double volOfVol = calculateVolOfVol(windowReturns);
        metrics.volatilityOfVolatility = volOfVol;
        
        // Add price acceleration (second derivative)
        metrics.priceAcceleration = calculatePriceAcceleration(windowPrices);
        
        // Add volume surge metric
        if (!volumeData.empty()) {
            double recentAvgVolume = std::accumulate(volumeData.begin(), volumeData.end(), 0.0) / volumeData.size();
            double historicalAvgVolume = 0.0;
            int count = 0;
            
            // Get historical average from longer period
            for (const auto& vol : volumeSeries_[symbol].getData()) {
                historicalAvgVolume += vol;
                count++;
            }
            
            double avgVolume = count > 0 ? historicalAvgVolume / count : 0.0;
            metrics.volumeSurge = recentAvgVolume / (avgVolume + 1e-10) - 1.0;
        }
    }
    
    // Calculate volume ratio if volume data available
    if (!volumeData.empty()) {
        // Get average volume over a longer period
        double totalVolume = 0.0;
        int volumeCount = 0;
        
        for (const auto& vol : volumeSeries_[symbol].getData()) {
            totalVolume += vol;
            volumeCount++;
        }
        
        double avgVolume = volumeCount > 0 ? totalVolume / volumeCount : 0.0;
        
        // Calculate recent volume
        double windowVolume = std::accumulate(volumeData.begin(), volumeData.end(), 0.0);
        double recentAvgVolume = volumeData.size() > 0 ? windowVolume / volumeData.size() : 0.0;
        
        metrics.volumeRatio = avgVolume > 0 ? recentAvgVolume / avgVolume : 1.0;
    }
    
    // Calculate return skewness and kurtosis
    double sumCubed = 0.0;
    double sumQuad = 0.0;
    
    for (double ret : windowReturns) {
        double centered = ret - mean;
        double centered2 = centered * centered;
        sumCubed += centered * centered2;
        sumQuad += centered2 * centered2;
    }
    
    if (volatility > 0) {
        metrics.returnSkewness = (sumCubed / windowReturns.size()) / std::pow(variance, 1.5);
        metrics.returnKurtosis = (sumQuad / windowReturns.size()) / (variance * variance) - 3.0;
    }
    
    // Calculate price range
    if (windowPrices.size() >= 2) {
        double minPrice = *std::min_element(windowPrices.begin(), windowPrices.end());
        double maxPrice = *std::max_element(windowPrices.begin(), windowPrices.end());
        double avgPrice = std::accumulate(windowPrices.begin(), windowPrices.end(), 0.0) / windowPrices.size();
        
        metrics.priceRange = avgPrice > 0 ? (maxPrice - minPrice) / avgPrice : 0.0;
    }
    
    // Set timestamp to the most recent data point
    metrics.timestampNanos = window.endNanos;
    
    return metrics;
}

double HMM_GJRGARCHModel::calculateVolOfVol(const std::vector<double>& returns) {
    if (returns.size() < 10) return 0.0;  // Need enough data points
    
    // Calculate rolling volatility windows
    std::vector<double> rollingVols;
    size_t windowSize = std::min<size_t>(5, returns.size() / 2);
    
    for (size_t i = 0; i <= returns.size() - windowSize; ++i) {
        // Calculate volatility for this window
        double sumSq = 0.0;
        for (size_t j = i; j < i + windowSize; ++j) {
            sumSq += returns[j] * returns[j];
        }
        double variance = sumSq / windowSize;
        rollingVols.push_back(std::sqrt(variance));
    }
    
    // Now calculate standard deviation of these rolling volatilities
    if (rollingVols.size() < 2) return 0.0;
    
    double mean = std::accumulate(rollingVols.begin(), rollingVols.end(), 0.0) / rollingVols.size();
    double sumSqDev = 0.0;
    
    for (double vol : rollingVols) {
        double dev = vol - mean;
        sumSqDev += dev * dev;
    }
    
    return std::sqrt(sumSqDev / (rollingVols.size() - 1));
}

double HMM_GJRGARCHModel::calculatePriceAcceleration(const std::vector<double>& prices) {
    if (prices.size() < 3) return 0.0;  // Need at least 3 points for acceleration
    
    // Calculate first differences (velocities)
    std::vector<double> velocities;
    for (size_t i = 1; i < prices.size(); ++i) {
        velocities.push_back(prices[i] - prices[i-1]);
    }
    
    // Calculate second differences (accelerations)
    std::vector<double> accelerations;
    for (size_t i = 1; i < velocities.size(); ++i) {
        accelerations.push_back(velocities[i] - velocities[i-1]);
    }
    
    // Return average acceleration
    return std::accumulate(accelerations.begin(), accelerations.end(), 0.0) / accelerations.size();
}

void HMM_GJRGARCHModel::updateReturnSeries(const std::string& symbol, double price, uint64_t timestamp) {
    // Check if we already have prices for this symbol
    if (priceSeries_[symbol].size() < 1) {
        return; // Need at least one previous price to calculate return
    }
    
    // Get the previous price
    double prevPrice = priceSeries_[symbol].getData().back();
    uint64_t prevTimestamp = priceSeries_[symbol].getTimestamps().back();
    
    // Skip if this is the same price point (timestamp hasn't moved forward)
    if (timestamp <= prevTimestamp) {
        return;
    }
    
    // Calculate log return
    double logReturn = std::log(price / prevPrice);
    
    // Add to return series
    returnSeries_[symbol].add(logReturn, timestamp);
}

GJRGARCHParameters HMM_GJRGARCHModel::getRegimeAdaptedParameters(const std::string& symbol) const {
    // Get current regime probabilities
    std::vector<double> regimeProbs = getRegimeProbabilities();
    
    // Get the most likely regime
    int mostLikelyRegime = 0;
    double maxProb = regimeProbs[0];
    
    for (size_t i = 1; i < regimeProbs.size(); ++i) {
        if (regimeProbs[i] > maxProb) {
            maxProb = regimeProbs[i];
            mostLikelyRegime = i;
        }
    }
    
    // Use parameters for the most likely regime
    return regimeGarchParams_[mostLikelyRegime];
}

void HMM_GJRGARCHModel::initializeDefaultRegimeParameters() {
    int regimeCount = hmm_.getNumStates();
    regimeGarchParams_.resize(regimeCount);
    
    // Low volatility regime - higher persistence, lower volatility of volatility
    regimeGarchParams_[0] = GJRGARCHParameters(0.000001, 0.03, 0.05, 0.92);
    
    if (regimeCount >= 3) {
        // Medium volatility regime - balanced parameters
        regimeGarchParams_[1] = GJRGARCHParameters(0.000005, 0.07, 0.10, 0.85);
        
        // High volatility regime - higher volatility of volatility, lower persistence
        regimeGarchParams_[regimeCount-1] = GJRGARCHParameters(0.000010, 0.10, 0.15, 0.75);
        
        // For transition regimes (if any)
        for (int i = 2; i < regimeCount - 1; ++i) {
            regimeGarchParams_[i] = GJRGARCHParameters(0.000008, 0.08, 0.12, 0.80);
        }
    } else if (regimeCount == 2) {
        // Just low and high regimes
        regimeGarchParams_[1] = GJRGARCHParameters(0.000010, 0.10, 0.15, 0.75);
    }
}

void HMM_GJRGARCHModel::setHMMTransitionMatrix(const Matrix& transitionMatrix) {
    // Forward the call to the underlying HMM
    hmm_.setTransitionMatrix(transitionMatrix);
}

void HMM_GJRGARCHModel::forceRegimeState(const Vector& probabilities) {
    // Forward the call to the underlying HMM
    hmm_.forceRegimeProbabilities(probabilities);
}

} // namespace vol_arb
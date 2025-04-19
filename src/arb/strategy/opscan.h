// opportunity_scanner.h
// Scanner for volatility arbitrage opportunities

#ifndef OPPORTUNITY_SCANNER_H
#define OPPORTUNITY_SCANNER_H

#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include "vol_arb_strategy.h"

namespace vol_arb {

// Forward declarations
class HMM_GJRGARCHModel;
class MarketDataSystem;

// Structure to represent a market opportunity
struct ArbitrageOpportunity {
    OptionData option;
    ArbitrageSignal signal;
    double score;  // Overall score (higher = better opportunity)
    double expectedValue;
    double confidence;
    double liquidity;  // Measure of option liquidity
    
    // Constructor
    ArbitrageOpportunity()
        : score(0), expectedValue(0), confidence(0), liquidity(0) {}
    
    // Constructor with signal
    ArbitrageOpportunity(const ArbitrageSignal& signalIn)
        : option(signalIn.option), signal(signalIn),
          score(0), expectedValue(signalIn.expectedValue),
          confidence(signalIn.confidence), liquidity(0) {}
};

// Option chain structure
struct OptionChain {
    std::string underlyingSymbol;
    double underlyingPrice;
    std::vector<OptionData> calls;
    std::vector<OptionData> puts;
    std::vector<double> expirations;  // Available expiration days
    std::vector<double> strikes;      // Available strike prices
    
    // Get all options as a flat list
    std::vector<OptionData> getAllOptions() const {
        std::vector<OptionData> allOptions;
        allOptions.reserve(calls.size() + puts.size());
        allOptions.insert(allOptions.end(), calls.begin(), calls.end());
        allOptions.insert(allOptions.end(), puts.begin(), puts.end());
        return allOptions;
    }
    
    // Find option by type, strike, and expiry
    OptionData* findOption(char type, double strike, double expiry) {
        auto& options = (type == 'C') ? calls : puts;
        
        for (auto& option : options) {
            if (std::abs(option.strike - strike) < 0.001 &&
                std::abs(option.expiryDays - expiry) < 0.1) {
                return &option;
            }
        }
        
        return nullptr;
    }
};

// Simplified market data system for testing
class MarketDataSystem {
public:
    // Get current underlying price
    double getUnderlyingPrice(const std::string& symbol) const;
    
    // Get option chain for a symbol
    OptionChain getOptionChain(const std::string& symbol) const;
    
    // Get risk-free rate
    double getRiskFreeRate() const;
    
    // Get dividend yield for a symbol
    double getDividendYield(const std::string& symbol) const;
    
private:
    // In a real system, this would connect to market data providers
    // For testing, we'll just generate synthetic data
    OptionChain generateSyntheticOptionChain(const std::string& symbol) const;
};

// Main class for scanning and finding arbitrage opportunities
class HMM_GJRGARCHOpportunityScanner {
public:
    // Constructor
    HMM_GJRGARCHOpportunityScanner(
        std::shared_ptr<VolatilityArbitrageStrategy> strategy,
        std::shared_ptr<MarketDataSystem> dataSystem,
        std::shared_ptr<HMM_GJRGARCHModel> volModel);
    
    // Scan an option chain for arbitrage opportunities
    std::vector<ArbitrageOpportunity> scanOptionChain(const std::string& symbol);
    
    // Rank opportunities by expected value
    std::vector<ArbitrageOpportunity> rankByExpectedValue();
    
    // Filter opportunities by liquidity and execution feasibility
    std::vector<ArbitrageOpportunity> filterByLiquidityAndExecution(double minSpreadRatio = 0.1);
    
    // Scan for special opportunities during regime transitions
    std::vector<ArbitrageOpportunity> scanForRegimeTransitionPlays();
    
    // Identify volatility term structure arbitrage
    std::vector<ArbitrageOpportunity> identifyVolatilityTermStructureArbitrage();
    
    // Calculate option liquidity score (0-1)
    double calculateLiquidityScore(const OptionData& option) const;
    
    // Set minimum thresholds
    void setMinimumThresholds(double minVol, double minVega, int maxOpportunities);
    
private:
    std::shared_ptr<VolatilityArbitrageStrategy> strategy_;
    std::shared_ptr<MarketDataSystem> dataSystem_;
    std::shared_ptr<HMM_GJRGARCHModel> volModel_;
    
    // Currently discovered opportunities
    std::vector<ArbitrageOpportunity> opportunities_;
    
    // Scanner parameters
    double minImpliedVolDifference_;
    double minVegaValue_;
    int maxSimultaneousOpportunities_;
    
    // Calculate overall opportunity score
    double calculateOpportunityScore(const ArbitrageSignal& signal, double liquidity);
    
    // Check if we already have enough positions in the underlying
    bool hasEnoughPositionsInUnderlying(const std::string& symbol, int maxPerUnderlying = 3) const;
};

} // namespace vol_arb

#endif // OPPORTUNITY_SCANNER_H
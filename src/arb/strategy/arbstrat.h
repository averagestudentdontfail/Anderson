// vol_arb_strategy.h
// Core implementation of volatility arbitrage strategy

#ifndef VOL_ARB_STRATEGY_H
#define VOL_ARB_STRATEGY_H

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "../models/hybrid_model.h"
#include "../../alo/alo_engine.h"

namespace vol_arb {

// Forward declarations
class EnhancedIVSolver;
class PositionManager;

// Option data structure
struct OptionData {
    std::string symbol;        // Underlying symbol
    std::string optionSymbol;  // Option symbol/identifier
    char type;                 // 'C' for call, 'P' for put
    double strike;             // Strike price
    double expiryDays;         // Days to expiration
    double bid;                // Bid price
    double ask;                // Ask price
    double lastPrice;          // Last traded price
    double volume;             // Trading volume
    double openInterest;       // Open interest
    double underlyingPrice;    // Current price of underlying
    double dividendYield;      // Dividend yield (annual)
    double riskFreeRate;       // Risk-free rate (annual)
    double impliedVol;         // Implied volatility
    bool isAmerican;           // True for American options, false for European
    
    // Constructor with essential fields
    OptionData(const std::string& symbol, const std::string& optSym, char type,
              double strike, double expiry, double underlyingPrice, 
              double bid, double ask)
        : symbol(symbol), optionSymbol(optSym), type(type), 
          strike(strike), expiryDays(expiry), bid(bid), ask(ask),
          lastPrice(0), volume(0), openInterest(0), 
          underlyingPrice(underlyingPrice), dividendYield(0), 
          riskFreeRate(0), impliedVol(0), isAmerican(true) {}
    
    // Default constructor
    OptionData() : type('P'), strike(0), expiryDays(0), bid(0), ask(0),
                   lastPrice(0), volume(0), openInterest(0), underlyingPrice(0),
                   dividendYield(0), riskFreeRate(0), impliedVol(0), isAmerican(true) {}
};

// Structure for arbitrage signals
struct ArbitrageSignal {
    enum Direction {
        BUY,
        SELL,
        NEUTRAL
    };
    
    // Signal information
    OptionData option;         // Option data
    Direction direction;       // Trade direction
    double forecastVol;        // Forecasted volatility
    double impliedVol;         // Current implied volatility 
    double volDifference;      // Difference between implied and forecast volatility
    double expectedValue;      // Expected value of the trade
    double confidence;         // Confidence level (0-1)
    double vegaExposure;       // Expected vega exposure
    double priceDifference;    // Difference between model and market price
    
    // Default constructor
    ArbitrageSignal() : direction(NEUTRAL), forecastVol(0), impliedVol(0),
                        volDifference(0), expectedValue(0), confidence(0),
                        vegaExposure(0), priceDifference(0) {}
};

// Position size recommendation
struct PositionSize {
    int contracts;             // Number of contracts
    double notionalValue;      // Notional value of position
    double maxRisk;            // Maximum risk exposure
    double vegaRisk;           // Vega risk exposure
    
    PositionSize() : contracts(0), notionalValue(0), maxRisk(0), vegaRisk(0) {}
    
    PositionSize(int contracts, double notional, double risk, double vega)
        : contracts(contracts), notionalValue(notional), maxRisk(risk), vegaRisk(vega) {}
};

// Strategy parameters that can be adjusted by regime
struct StrategyParams {
    double minVolSpread;       // Minimum volatility difference to trade
    double maxPositionSize;    // Maximum position size as % of capital
    double confidenceThreshold; // Minimum confidence level to trade
    double exitThreshold;      // Volatility convergence threshold for exit
    double maxDaysToHold;      // Maximum days to hold a position
    
    StrategyParams() 
        : minVolSpread(0.03), maxPositionSize(0.05), 
          confidenceThreshold(0.6), exitThreshold(0.01), maxDaysToHold(21) {}
    
    StrategyParams(double spread, double posSize, double confidence, 
                 double exit, double maxDays)
        : minVolSpread(spread), maxPositionSize(posSize),
          confidenceThreshold(confidence), exitThreshold(exit), 
          maxDaysToHold(maxDays) {}
};

// Option position structure
struct OptionPosition {
    std::string id;            // Position ID
    OptionData option;         // Option data
    int quantity;              // Position quantity (+ for long, - for short)
    double entryPrice;         // Entry price per contract
    double entryImpliedVol;    // Implied volatility at entry
    double entryForecastVol;   // Forecasted volatility at entry
    double currentPrice;       // Current price per contract
    double currentImpliedVol;  // Current implied volatility
    double currentForecastVol; // Current forecasted volatility
    double currentPnL;         // Current unrealized P&L
    uint64_t entryTimestamp;   // Entry timestamp
    uint64_t lastUpdateTimestamp; // Last update timestamp
    
    // Greeks at entry and current
    double entryDelta;
    double entryGamma;
    double entryVega;
    double entryTheta;
    double currentDelta;
    double currentGamma;
    double currentVega;
    double currentTheta;
    
    // Hedging information
    int hedgeQuantity;         // Quantity of underlying hedges
    double hedgeCost;          // Average cost of hedges
    
    // Constructor
    OptionPosition() 
        : quantity(0), entryPrice(0), entryImpliedVol(0), entryForecastVol(0),
          currentPrice(0), currentImpliedVol(0), currentForecastVol(0),
          currentPnL(0), entryTimestamp(0), lastUpdateTimestamp(0),
          entryDelta(0), entryGamma(0), entryVega(0), entryTheta(0),
          currentDelta(0), currentGamma(0), currentVega(0), currentTheta(0),
          hedgeQuantity(0), hedgeCost(0) {}
};

// Current portfolio of positions
struct Portfolio {
    std::vector<OptionPosition> positions;
    double totalVegaExposure;
    double totalDeltaExposure;
    double totalGammaExposure;
    double totalThetaExposure;
    double allocatedCapital;
    double totalNotional;
    
    // Get position by ID
    OptionPosition* getPosition(const std::string& id) {
        for (auto& pos : positions) {
            if (pos.id == id) {
                return &pos;
            }
        }
        return nullptr;
    }
    
    // Calculate total exposure by underlying
    std::unordered_map<std::string, double> getVegaByUnderlying() const {
        std::unordered_map<std::string, double> result;
        for (const auto& pos : positions) {
            result[pos.option.symbol] += pos.currentVega * pos.quantity;
        }
        return result;
    }
    
    // Calculate allocation percentage
    double getAllocationPercentage(double totalCapital) {
        return totalCapital > 0 ? allocatedCapital / totalCapital : 0.0;
    }
};

// Trade execution plan
class ExecutionPlan {
public:
    // Constructor with option and direction
    ExecutionPlan(const OptionData& option, ArbitrageSignal::Direction direction,
                 double targetPrice = 0.0);
    
    // Execute the plan with specified position size
    OptionPosition execute(const PositionSize& size);
    
    // Get expected execution price
    double getExpectedPrice() const;
    
    // Get expected execution cost
    double getExpectedCost(const PositionSize& size) const;
    
    // Check if the plan is valid
    bool isValid() const;
    
private:
    OptionData option_;
    ArbitrageSignal::Direction direction_;
    double targetPrice_;
    bool valid_;
};

// Enhanced implied volatility solver
class EnhancedIVSolver {
public:
    // Constructor
    EnhancedIVSolver(std::shared_ptr<ALOEngine> engine);
    
    // Solve for implied volatility
    double solveForImpliedVol(const OptionData& option);
    
    // Handle special cases for deep ITM/OTM options
    double handleDeepITMOptions(const OptionData& option);
    double handleDeepOTMOptions(const OptionData& option);
    
    // Apply Brent's method for faster convergence
    double applyBrentsMethod(const std::function<double(double)>& priceDiffFunc,
                           double volLower, double volUpper);
    
    // Fit local volatility smile for more robust IV
    std::vector<double> fitLocalVolatilitySmile(const std::vector<OptionData>& optionChain);
    
private:
    std::shared_ptr<ALOEngine> engine_;
    const double MIN_VOL = 0.001;
    const double MAX_VOL = 2.0;
    const double PRECISION = 1e-7;
    const int MAX_ITERATIONS = 100;
};

// Main volatility arbitrage strategy class
class VolatilityArbitrageStrategy {
public:
    // Constructor
    VolatilityArbitrageStrategy(std::shared_ptr<ALOEngine> engine,
                              std::shared_ptr<HMM_GJRGARCHModel> volModel,
                              std::shared_ptr<EnhancedIVSolver> ivSolver = nullptr);
    
    // Evaluate an option for arbitrage opportunities
    ArbitrageSignal evaluateOption(const OptionData& option);
    
    // Calculate the difference between implied and forecasted volatility
    double calculateImpliedVolSpread(const OptionData& option);
    
    // Calculate optimal position size
    PositionSize calculateOptimalPositionSize(const ArbitrageSignal& signal,
                                          const Portfolio& currentPositions);
    
    // Adjust strategy for current market regime
    void adjustStrategyForCurrentRegime();
    
    // Get volatility convergence speed estimate based on regime
    double getVolatilityConvergenceSpeed(MarketRegime regime);
    
    // Create execution plan for a signal
    ExecutionPlan createExecutionPlan(const ArbitrageSignal& signal);
    
    // Monitor active positions and generate updates
    void monitorActivePositions(std::vector<OptionPosition>& positions);
    
    // Set strategy parameters
    void setStrategyParameters(const StrategyParams& params);
    
    // Set risk limits
    void setRiskLimits(double vegaLimit, double deltaLimit, double positionSizeLimit,
                     double confidenceThreshold);
    
    // Set total capital
    void setTotalCapital(double capital);
    
    // Get current strategy parameters
    const StrategyParams& getStrategyParameters() const;
    
private:
    std::shared_ptr<ALOEngine> engine_;
    std::shared_ptr<HMM_GJRGARCHModel> volModel_;
    std::shared_ptr<EnhancedIVSolver> ivSolver_;
    
    // Strategy parameters
    StrategyParams params_;
    
    // Risk limits
    double vegaExposureLimit_;
    double deltaExposureLimit_;
    double positionSizeLimit_;
    double totalCapital_;
    
    // Regime-specific parameter sets
    std::unordered_map<MarketRegime, StrategyParams> regimeParams_;
    
    // Helper methods
    double calculateModelPrice(const OptionData& option, double volatility);
    double calculateModelImpliedVol(const OptionData& option);
    double calculateExpectedValue(const OptionData& option, double forecastVol, double impliedVol);
    double calculateConfidence(const OptionData& option, double forecastVol, double volSpread);
};

} // namespace vol_arb

#endif // VOL_ARB_STRATEGY_H
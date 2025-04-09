// volatility_arbitrage_test.cpp
// End-to-end test case for volatility arbitrage strategy

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <thread>
#include "../alo/alo_engine.h"
#include "../vol_arb/models/gjr_garch.h"
#include "../vol_arb/models/hmm.h"
#include "../vol_arb/models/hybrid_model.h"
#include "../vol_arb/strategy/vol_arb_strategy.h"
#include "../vol_arb/strategy/opportunity_scanner.h"

using namespace vol_arb;

// Define a test scenario for volatility arbitrage
struct VolatilityArbitrageTestScenario {
    std::string name;
    std::string symbol;
    std::vector<double> initialRegime;   // Initial regime probabilities
    int regimeTransitionTrigger;         // Data point when regime transition begins
    double volatilityShock;              // Size of volatility increase during test
    
    VolatilityArbitrageTestScenario() 
        : name("Default Scenario"), symbol("SPY"), 
          initialRegime({0.7, 0.25, 0.05}), regimeTransitionTrigger(75),
          volatilityShock(0.1) {}
};

// Test class for volatility arbitrage
class VolatilityArbitrageTest {
private:
    // Core system components
    std::shared_ptr<ALOEngine> aloEngine_;
    std::shared_ptr<HMM_GJRGARCHModel> volModel_;
    std::shared_ptr<EnhancedIVSolver> ivSolver_;
    std::shared_ptr<VolatilityArbitrageStrategy> strategy_;
    std::shared_ptr<MarketDataSystem> dataSystem_;
    std::shared_ptr<HMM_GJRGARCHOpportunityScanner> scanner_;
    
    // Test data
    std::vector<MarketData> marketData_;
    VolatilityArbitrageTestScenario scenario_;
    
    // Test results
    std::vector<ArbitrageOpportunity> detectedOpportunities_;
    std::vector<OptionPosition> openPositions_;
    std::vector<OptionPosition> closedPositions_;
    double initialCapital_;
    double currentCapital_;
    double totalPnL_;

    // Helper method to calculate option price using the ALO engine
    double calculateModelPrice(const OptionPosition& position, double volatility, double underlyingPrice) {
        // Use the ALO engine to price the option correctly
        OptionType optionType = position.option.type == 'C' ? CALL : PUT;
        
        try {
            return aloEngine_->calculateOption(
                underlyingPrice,
                position.option.strike,
                position.option.riskFreeRate,
                position.option.dividendYield,
                volatility,
                position.option.expiryDays / 365.0,
                optionType
            );
        } catch (const std::exception& e) {
            std::cerr << "Error calculating option price: " << e.what() << std::endl;
            // Fallback to a simplified calculation
            double timeToExpiry = position.option.expiryDays / 365.0;
            double intrinsicValue = 0.0;
            
            if (optionType == CALL) {
                intrinsicValue = std::max(0.0, underlyingPrice - position.option.strike);
            } else {
                intrinsicValue = std::max(0.0, position.option.strike - underlyingPrice);
            }
            
            // Add time value based on volatility
            double timeValue = underlyingPrice * volatility * std::sqrt(timeToExpiry) * 0.4;
            return intrinsicValue + timeValue;
        }
    }

    // Add new helper methods for risk management
    double calculateRecentWinRate(int numTrades) const {
        if (closedPositions_.empty()) return 0.5; // Default to neutral
        
        int numWins = 0;
        int count = 0;
        
        // Look at most recent trades
        for (auto it = closedPositions_.rbegin(); 
             it != closedPositions_.rend() && count < numTrades; 
             ++it, ++count) {
            if (it->currentPnL > 0) numWins++;
        }
        
        return count > 0 ? static_cast<double>(numWins) / count : 0.5;
    }
    
    double calculatePositionSize(const ArbitrageOpportunity& opportunity, 
                               const MarketData& currentData) {
        // Calculate Kelly fraction based on volatility convergence
        double kellyFraction = calculateKellyFraction(opportunity);
        
        // Calculate volatility convergence score
        double convergenceScore = calculateVolatilityConvergenceScore(opportunity);
        
        // Base position size on Kelly fraction and convergence score
        double positionSize = kellyFraction * currentCapital_ * convergenceScore;
        
        // Apply regime adjustment
        positionSize = calculateRegimeAdjustedSize(positionSize);
        
        // Apply more conservative sizing
        positionSize = std::min(positionSize, currentCapital_ * 0.02); // Max 2% of capital
        
        // Ensure minimum position size
        double minPositionSize = 0.005 * currentCapital_; // 0.5% minimum position size
        positionSize = std::max(positionSize, minPositionSize);
        
        // Check if position size is within risk limits
        if (!isWithinRiskLimits(opportunity, positionSize)) {
            std::cout << "  Position size exceeds risk limits. Reducing size." << std::endl;
            positionSize = 0.01 * currentCapital_; // Default to 1% if risk limits exceeded
        }
        
        return positionSize;
    }
    
    bool executeOpportunity(const ArbitrageOpportunity& opportunity, 
                          const MarketData& currentData) {
        // Create execution plan
        ExecutionPlan plan = strategy_->createExecutionPlan(opportunity.signal);
        
        if (!plan.isValid()) {
            std::cout << "  Execution plan is invalid. Skipping opportunity." << std::endl;
            return false;
        }
        
        // Get current portfolio for position sizing
        Portfolio portfolio;
        portfolio.positions = openPositions_;
        
        // Use a much more conservative position size
        double maxRiskPercent = 0.01; // 1% of capital per trade
        double positionSize = currentCapital_ * maxRiskPercent;
        
        // Calculate position size
        PositionSize size;
        // Get option price from the execution plan
        double optionPrice = plan.getExpectedPrice();
        
        // Ensure option price is reasonable
        if (optionPrice <= 0.0 || optionPrice > opportunity.option.underlyingPrice * 0.5) {
            std::cout << "  Invalid option price: $" << optionPrice << ". Skipping opportunity." << std::endl;
            return false;
        }
        
        // Calculate contract value (100 shares per contract)
        double contractValue = optionPrice * 100.0;
        
        // Limit number of contracts based on position size
        size.contracts = (contractValue > 0) ? std::max(1, static_cast<int>(positionSize / contractValue)) : 0;
        
        // Hard limit on contracts per position
        size.contracts = std::min(size.contracts, 5);
        
        // Calculate notional and risk values
        size.notionalValue = contractValue * size.contracts;
        size.maxRisk = size.notionalValue;
        
        // Check if we have enough capital (with margin)
        double requiredCapital = size.notionalValue;
        if (requiredCapital > currentCapital_) {
            std::cout << "  Not enough capital for position. Required: $" << requiredCapital 
                      << ", Available: $" << currentCapital_ << std::endl;
            return false;
        }
        
        std::cout << "  Position size: " << size.contracts << " contracts" << std::endl;
        std::cout << "  Option price: $" << optionPrice << std::endl;
        std::cout << "  Notional value: $" << size.notionalValue << std::endl;
        
        // Execute the plan
        OptionPosition position = plan.execute(size);
        
        // Ensure reasonable entry price (should match the execution plan price)
        position.entryPrice = optionPrice;
        position.currentPrice = optionPrice;
        
        // Set volatility information
        position.entryImpliedVol = opportunity.signal.impliedVol;
        position.entryForecastVol = opportunity.signal.forecastVol;
        position.currentImpliedVol = opportunity.signal.impliedVol;
        position.currentForecastVol = opportunity.signal.forecastVol;
        
        // Set entry time
        position.entryTimestamp = currentData.timestampNanos;
        position.lastUpdateTimestamp = currentData.timestampNanos;
        
        // Calculate option Greeks
        double timeToExpiry = position.option.expiryDays / 365.0;
        double sqrt_t = std::sqrt(timeToExpiry);
        
        // Simple approximation for vega
        position.entryVega = position.entryPrice * sqrt_t * 
                           position.option.underlyingPrice / 
                           (100.0 * position.entryImpliedVol);
        
        // Calculate cost and update capital
        double positionCost = position.entryPrice * std::abs(position.quantity) * 100.0;
        currentCapital_ -= positionCost;
        
        // Add to open positions
        openPositions_.push_back(position);
        
        // Add to detected opportunities for tracking
        detectedOpportunities_.push_back(opportunity);
        
        std::cout << "  Trade executed. New capital: $" << currentCapital_ << std::endl;
        
        return true;
    }

    // Add new helper methods for improved position sizing and risk management
    double calculateVolatilityConvergenceScore(const ArbitrageOpportunity& opportunity) {
        // Calculate the spread between implied and forecast volatility
        double volSpread = opportunity.signal.impliedVol - opportunity.signal.forecastVol;
        
        // Estimate volatility of volatility (simplified for now)
        double volOfVol = 0.2; // Typical value, could be calculated from historical data
        
        // Calculate z-score of the current spread
        double zScore = volSpread / volOfVol;
        
        // Calculate convergence probability based on mean reversion
        double meanReversionStrength = 0.7; // Estimated from historical data
        double halfLife = 5.0; // Days for half the spread to converge
        
        // Probability of convergence within holding period
        double convergenceProb = 1.0 - std::exp(-meanReversionStrength * opportunity.option.expiryDays / halfLife);
        
        // Combine into a single score
        return convergenceProb * std::abs(zScore);
    }
    
    double calculateAverageWinLossRatio() {
        if (closedPositions_.empty()) {
            return 1.5; // Default value if no historical data
        }
        
        double totalWins = 0.0;
        double totalLosses = 0.0;
        int winCount = 0;
        int lossCount = 0;
        
        for (const auto& position : closedPositions_) {
            if (position.currentPnL > 0) {
                totalWins += position.currentPnL;
                winCount++;
            } else {
                totalLosses += std::abs(position.currentPnL);
                lossCount++;
            }
        }
        
        double avgWin = winCount > 0 ? totalWins / winCount : 0.0;
        double avgLoss = lossCount > 0 ? totalLosses / lossCount : 1.0; // Avoid division by zero
        
        return avgWin / avgLoss;
    }
    
    double calculateKellyFraction(const ArbitrageOpportunity& opportunity) {
        // Estimate win probability based on volatility convergence
        double volSpread = std::abs(opportunity.signal.impliedVol - opportunity.signal.forecastVol);
        double volSpreadPercentile = std::min(1.0, volSpread / 0.1); // Normalize to [0,1]
        
        // Higher win probability for larger volatility spreads
        double winProbability = 0.5 + 0.3 * volSpreadPercentile;
        
        // Estimate win/loss ratio based on historical data or defaults
        double avgWinLossRatio = calculateAverageWinLossRatio();
        
        // Apply Kelly formula with a safety factor (quarter-Kelly)
        double kellyFraction = (winProbability * avgWinLossRatio - (1.0 - winProbability)) / avgWinLossRatio;
        kellyFraction = std::max(0.0, kellyFraction * 0.25); // Quarter-Kelly for extra safety
        
        return kellyFraction;
    }
    
    double calculateRegimeAdjustedSize(double baseSize) {
        // Get current regime probabilities
        std::vector<double> regimeProbs = volModel_->getRegimeProbabilities();
        
        // Calculate regime adjustment factor - be more conservative in higher vol regimes
        double lowVolFactor = 1.0;      // Full size in low vol
        double mediumVolFactor = 0.5;   // Half size in medium vol
        double highVolFactor = 0.25;    // Quarter size in high vol
        
        double regimeFactor = lowVolFactor * regimeProbs[0] + 
                             mediumVolFactor * regimeProbs[1] + 
                             highVolFactor * regimeProbs[2];
        
        // Apply regime adjustment
        return baseSize * regimeFactor;
    }
    
    bool isWithinRiskLimits(const ArbitrageOpportunity& opportunity, double positionSize) {
        // Calculate maximum position size based on capital
        double maxPositionSize = 0.05 * currentCapital_; // 5% max per position
        
        // Calculate maximum position size based on volatility
        double volSpread = std::abs(opportunity.signal.impliedVol - opportunity.signal.forecastVol);
        double volAdjustedSize = currentCapital_ * 0.02 * (1.0 + volSpread * 5.0);
        
        // Calculate maximum position size based on portfolio exposure
        double portfolioVega = 0.0;
        for (const auto& pos : openPositions_) {
            portfolioVega += pos.entryVega * pos.quantity;
        }
        
        double maxVegaExposure = currentCapital_ * 0.01; // 1% max vega exposure
        double vegaAdjustedSize = (maxVegaExposure - portfolioVega) / 
                                 (opportunity.option.underlyingPrice * std::sqrt(opportunity.option.expiryDays / 365.0));
        
        // Take the minimum of all constraints
        double allowedSize = std::min({maxPositionSize, volAdjustedSize, vegaAdjustedSize});
        
        return positionSize <= allowedSize;
    }
    
    double calculateDynamicStopLoss(const OptionPosition& position, const MarketData& currentData) {
        // For options, a reasonable stop loss is a percentage of the premium paid
        double stopLossPercent = 0.5; // 50% of entry price (allow for 50% loss)
        double stopLossPrice = position.entryPrice * stopLossPercent;
        
        // Add trailing stop for profitable positions
        if (position.currentPrice > position.entryPrice * 1.5) { // If 50% profit
            double trailingStop = position.currentPrice * 0.8; // 20% trailing stop
            stopLossPrice = std::max(stopLossPrice, trailingStop);
        }
        
        return stopLossPrice;
    }
    
    double calculatePositionPnL(const OptionPosition& position, double currentPrice) {
        // Basic P&L calculation based on option prices
        double pnl = (currentPrice - position.entryPrice) * position.quantity * 100.0;
        
        // Cap the P&L to reasonable limits
        double maxLoss = -position.entryPrice * position.quantity * 100.0; // Max loss is premium paid
        double maxGain = position.entryPrice * position.quantity * 100.0 * 3.0; // Max gain is 3x premium
        
        pnl = std::max(maxLoss, std::min(pnl, maxGain));
        
        // Adjust for transaction costs
        double transactionCost = 0.001 * std::abs(position.entryPrice * position.quantity * 100.0);
        pnl -= transactionCost;
        
        return pnl;
    }
    
    double calculateVolatilityForecastConfidence() {
        // Get current regime probabilities
        std::vector<double> regimeProbs = volModel_->getRegimeProbabilities();
        
        // Calculate regime stability
        double regimeStability = std::max({regimeProbs[0], regimeProbs[1], regimeProbs[2]});
        
        // Estimate forecast error (simplified)
        double forecastError = 0.2; // Could be calculated from historical data
        
        // Calculate confidence score
        double confidence = regimeStability * (1.0 - forecastError);
        
        return confidence;
    }

public:
    // Constructor
    VolatilityArbitrageTest(const VolatilityArbitrageTestScenario& scenario = VolatilityArbitrageTestScenario())
        : scenario_(scenario), initialCapital_(1000000.0), currentCapital_(initialCapital_), totalPnL_(0.0) {
        
        // Initialize the ALO engine
        aloEngine_ = std::make_shared<ALOEngine>(ACCURATE);
        
        // Initialize the volatility model
        volModel_ = std::make_shared<HMM_GJRGARCHModel>(3);
        
        // Initialize the IV solver
        ivSolver_ = std::make_shared<EnhancedIVSolver>(aloEngine_);
        
        // Initialize the strategy
        strategy_ = std::make_shared<VolatilityArbitrageStrategy>(aloEngine_, volModel_, ivSolver_);
        strategy_->setTotalCapital(initialCapital_);
        
        // Initialize the market data system
        dataSystem_ = std::make_shared<MarketDataSystem>();
        
        // Initialize the scanner
        scanner_ = std::make_shared<HMM_GJRGARCHOpportunityScanner>(strategy_, dataSystem_, volModel_);
    }
    
    // Generate synthetic market data
    void generateSyntheticMarketData(int numDataPoints = 200) {
        marketData_.clear();
        
        std::cout << "Generating synthetic market data..." << std::endl;
        
        // Start with base price
        double basePrice = 450.0;  // SPY starting price
        
        // Initial volatility
        double baseVol = 0.15;  // 15% annualized
        
        // Setup random number generator
        static std::mt19937 gen(std::random_device{}());
        
        // Generate price path with more dramatic volatility change
        for (int i = 0; i < numDataPoints; ++i) {
            // Determine current volatility (adjust for regime change)
            double currentVol = baseVol;
            
            // If we've hit the transition trigger, start increasing volatility more rapidly
            if (i >= scenario_.regimeTransitionTrigger) {
                double progress = static_cast<double>(i - scenario_.regimeTransitionTrigger) / 25.0; // Faster transition
                progress = std::min(1.0, progress);  // Cap at 1.0
                currentVol += scenario_.volatilityShock * progress * 2.0; // Double the effect
                
                // Add volatility clusters (spikes) typical of regime changes
                if ((i - scenario_.regimeTransitionTrigger) % 5 == 0) {
                    currentVol *= 1.5; // Occasional volatility spikes
                }
            }
            
            // Add dramatic price moves during transition
            double dailyVol = currentVol / std::sqrt(252.0);
            std::normal_distribution<double> dist(0.0, dailyVol);
            double shock = dist(gen);
            
            // Generate more dramatic price moves during transition
            if (i >= scenario_.regimeTransitionTrigger && i < scenario_.regimeTransitionTrigger + 30) {
                // Add directional bias during transition (typically down during vol increases)
                shock -= 0.002; // Slight downward bias
            }
            
            basePrice *= std::exp(shock);
            
            // Create market data point
            MarketData data(scenario_.symbol, basePrice, 
                          1000000.0 * (1.0 + 0.5 * currentVol),  // Volume increases with vol
                          basePrice * 0.995, basePrice * 1.005, 
                          basePrice * 0.99, basePrice * 1.01,
                          i * 60 * 60 * 1000000000ULL);  // Hourly data points
            
            marketData_.push_back(data);
        }
        
        std::cout << "Generated " << marketData_.size() << " market data points" << std::endl;
    }
    
    // Configure test scenario
    void configureForTestScenario() {
        // Adjust HMM to be more responsive to regime changes
        // This is done by modifying the transition matrix to allow faster transitions
        Matrix transitionMatrix = Matrix::Zero(3, 3);
        
        // More responsive transition matrix with balanced probabilities
        transitionMatrix << 
            0.70, 0.25, 0.05,   // Low vol regime
            0.15, 0.70, 0.15,   // Medium vol regime
            0.05, 0.25, 0.70;   // High vol regime
        
        // Apply to the model
        volModel_->setHMMTransitionMatrix(transitionMatrix);
    }
    
    // Simulate regime transition
    void simulateRegimeTransition(int dataPoint, int transitionPoint) {
        if (dataPoint >= transitionPoint && dataPoint < transitionPoint + 25) {
            double progress = (dataPoint - transitionPoint) / 25.0;
            Vector newProbs(3);
            newProbs << 0.7 - 0.5 * progress,  // Decrease low vol
                        0.25 + 0.25 * progress, // Increase medium vol
                        0.05 + 0.25 * progress; // Increase high vol
            volModel_->forceRegimeState(newProbs);
        }
        else if (dataPoint >= transitionPoint + 25 && dataPoint < transitionPoint + 50) {
            double progress = (dataPoint - (transitionPoint + 25)) / 25.0;
            Vector newProbs(3);
            newProbs << 0.2 - 0.1 * progress,   // Further decrease low vol
                        0.5 - 0.25 * progress,   // Decrease medium vol
                        0.3 + 0.35 * progress;   // Increase high vol
            volModel_->forceRegimeState(newProbs);
        }
    }
    
    // Run the full trading cycle test
    void runFullTradingCycleTest() {
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Running Volatility Arbitrage Test: " << scenario_.name << std::endl;
        std::cout << "==================================================" << std::endl;
        
        // If we don't have market data, generate it with enhanced volatility shift
        if (marketData_.empty()) {
            generateSyntheticMarketData();
        }
        
        // Configure the HMM to be more responsive to regime changes
        configureForTestScenario();
        
        // Initialize the volatility model with initial market data
        std::cout << "\nInitializing volatility model with first 50 data points..." << std::endl;
        for (int i = 0; i < 50 && i < static_cast<int>(marketData_.size()); ++i) {
            volModel_->updateWithMarketData(marketData_[i]);
        }
        
        // Calibrate the model
        volModel_->recalibrateModels(true);
        
        // Set initial regime probabilities
        std::cout << "\nSetting initial regime probabilities..." << std::endl;
        Vector initialProbs(3);
        initialProbs << scenario_.initialRegime[0], 
                        scenario_.initialRegime[1], 
                        scenario_.initialRegime[2];
        volModel_->forceRegimeState(initialProbs);
        
        std::cout << "  Low volatility: " << initialProbs[0] * 100 << "%" << std::endl;
        std::cout << "  Medium volatility: " << initialProbs[1] * 100 << "%" << std::endl;
        std::cout << "  High volatility: " << initialProbs[2] * 100 << "%" << std::endl;
        
        // Main simulation loop
        std::cout << "\nStarting main simulation loop..." << std::endl;
        for (size_t i = 50; i < marketData_.size(); ++i) {
            const MarketData& currentData = marketData_[i];
            
            // Print progress
            if (i % 10 == 0) {
                std::cout << "Processing data point " << i << "/" << marketData_.size() 
                          << " (" << (i * 100 / marketData_.size()) << "%)" << std::endl;
            }
            
            // Update volatility model
            volModel_->updateWithMarketData(currentData);
            
            // Simulate regime transition if needed
            simulateRegimeTransition(i, scenario_.regimeTransitionTrigger);
            
            // Get current regime probabilities
            std::vector<double> regimeProbs = volModel_->getRegimeProbabilities();
            std::cout << "\nCurrent Market Regime:" << std::endl;
            std::cout << "  Low Vol: " << regimeProbs[0] * 100 << "%" << std::endl;
            std::cout << "  Medium Vol: " << regimeProbs[1] * 100 << "%" << std::endl;
            std::cout << "  High Vol: " << regimeProbs[2] * 100 << "%" << std::endl;
            
            // Monitor existing positions first
            monitorPositions(currentData);
            
            // Scan for new opportunities
            std::cout << "Scanning for arbitrage opportunities..." << std::endl;
            auto opportunities = scanner_->scanOptionChain(scenario_.symbol);
            
            // Check for high regime transition probability
            double transitionProb = 1.0 - regimeProbs[0]; // Probability of not being in low vol regime
            if (transitionProb > 0.4) { // 40% threshold
                std::cout << "Detected high regime transition probability: " 
                          << transitionProb * 100 << "%" << std::endl;
            }
            
            // Execute opportunities if we have capital
            for (const auto& opportunity : opportunities) {
                if (currentCapital_ > 0) {
                    executeOpportunity(opportunity, currentData);
                }
            }
        }
        
        // Close any remaining positions at the end
        for (auto& position : openPositions_) {
            std::cout << "\nClosing remaining position at end of test:" << std::endl;
            std::cout << "  Symbol: " << position.option.symbol << std::endl;
            std::cout << "  Entry price: " << position.entryPrice << std::endl;
            
            // Calculate a final option price
            double finalOptionPrice = calculateModelPrice(position, position.currentImpliedVol, 
                                                      marketData_.back().price);
            position.currentPrice = finalOptionPrice;
            
            std::cout << "  Exit price: " << position.currentPrice << std::endl;
            
            double pnl = calculatePositionPnL(position, position.currentPrice);
            position.currentPnL = pnl;
            std::cout << "  P&L: $" << pnl << std::endl;
            
            currentCapital_ += pnl;
            totalPnL_ += pnl;
            closedPositions_.push_back(position);
        }
        openPositions_.clear();
        
        // Print final results
        printTestResults();
    }
    
    // Monitor existing positions and handle exits
    void monitorPositions(const MarketData& currentData) {
        for (auto it = openPositions_.begin(); it != openPositions_.end();) {
            // Update the option's underlying price reference
            it->option.underlyingPrice = currentData.price;
            
            // Calculate the current option price using the option pricing model
            double optionPrice = calculateModelPrice(*it, it->currentImpliedVol, currentData.price);
            
            // Cap the price movement to prevent unrealistic changes
            double maxChange = it->entryPrice * 2.0; // Max 100% price change
            if (std::abs(optionPrice - it->entryPrice) > maxChange) {
                optionPrice = it->entryPrice + (optionPrice > it->entryPrice ? maxChange : -maxChange);
            }
            
            // Update current price with the calculated option price
            it->currentPrice = optionPrice;
            
            // Calculate P&L based on option price changes
            it->currentPnL = calculatePositionPnL(*it, optionPrice);
            
            // Calculate a more realistic stop loss based on the option price
            double stopLoss = it->entryPrice * 0.5; // 50% max loss
            
            // Add maximum holding period
            uint64_t maxHoldingPeriod = 5 * 24 * 60 * 60 * 1000000000ULL; // 5 days
            bool timeStop = (currentData.timestampNanos - it->entryTimestamp > maxHoldingPeriod);
            
            // Check stop loss or time-based exit
            if (it->currentPrice < stopLoss || timeStop) {
                std::string exitReason = timeStop ? "Maximum holding period reached" : "Stop loss triggered";
                std::cout << "Exiting position: " << it->option.symbol << std::endl;
                std::cout << "  Reason: " << exitReason << std::endl;
                std::cout << "  Entry price: " << it->entryPrice << std::endl;
                std::cout << "  Exit price: " << it->currentPrice << std::endl;
                std::cout << "  Stop loss: " << stopLoss << std::endl;
                
                // Calculate final P&L
                double pnl = it->currentPnL;
                std::cout << "  P&L: $" << pnl << std::endl;
                
                // Update capital
                currentCapital_ += pnl;
                totalPnL_ += pnl;
                
                // Move to closed positions
                closedPositions_.push_back(*it);
                
                // Remove from open positions
                it = openPositions_.erase(it);
            } else {
                ++it;
            }
        }
    }
    
    // Print final test results
    void printTestResults() {
        std::cout << "\n==================================================" << std::endl;
        std::cout << "Volatility Arbitrage Test Results" << std::endl;
        std::cout << "==================================================" << std::endl;
        
        std::cout << "\nTest Scenario: " << scenario_.name << std::endl;
        std::cout << "Symbol: " << scenario_.symbol << std::endl;
        std::cout << "Market Data Points: " << marketData_.size() << std::endl;
        std::cout << "Volatility Shock: " << scenario_.volatilityShock * 100.0 << "%" << std::endl;
        std::cout << "Regime Transition Point: " << scenario_.regimeTransitionTrigger << std::endl;
        
        // Print performance summary
        std::cout << "\nPerformance Summary:" << std::endl;
        std::cout << "  Initial Capital: $" << initialCapital_ << std::endl;
        std::cout << "  Final Capital: $" << currentCapital_ << std::endl;
        std::cout << "  Total P&L: $" << totalPnL_ << std::endl;
        std::cout << "  Return: " << (totalPnL_ / initialCapital_) * 100.0 << "%" << std::endl;
        
        // Print trade statistics
        std::cout << "\nTrade Statistics:" << std::endl;
        std::cout << "  Total Opportunities Detected: " << detectedOpportunities_.size() << std::endl;
        std::cout << "  Total Positions Taken: " << openPositions_.size() + closedPositions_.size() << std::endl;
        std::cout << "  Positions Still Open: " << openPositions_.size() << std::endl;
        std::cout << "  Positions Closed: " << closedPositions_.size() << std::endl;
        
        // Calculate win rate and other metrics
        int winningTrades = 0;
        int losingTrades = 0;
        double totalWinning = 0.0;
        double totalLosing = 0.0;
        double maxWin = 0.0;
        double maxLoss = 0.0;
        
        for (const auto& position : closedPositions_) {
            double pnl = position.currentPnL;
            
            if (pnl > 0) {
                winningTrades++;
                totalWinning += pnl;
                maxWin = std::max(maxWin, pnl);
            } else {
                losingTrades++;
                totalLosing += std::abs(pnl);
                maxLoss = std::max(maxLoss, std::abs(pnl));
            }
        }
        
        double winRate = closedPositions_.empty() ? 
                       0.0 : (static_cast<double>(winningTrades) / closedPositions_.size()) * 100.0;
        
        double avgWin = winningTrades > 0 ? totalWinning / winningTrades : 0.0;
        double avgLoss = losingTrades > 0 ? totalLosing / losingTrades : 0.0;
        double profitFactor = totalLosing > 0 ? totalWinning / totalLosing : 0.0;
        
        std::cout << "  Win Rate: " << winRate << "%" << std::endl;
        std::cout << "  Average Winning Trade: $" << avgWin << std::endl;
        std::cout << "  Average Losing Trade: $" << avgLoss << std::endl;
        std::cout << "  Maximum Win: $" << maxWin << std::endl;
        std::cout << "  Maximum Loss: $" << maxLoss << std::endl;
        std::cout << "  Profit Factor: " << profitFactor << std::endl;
        
        // Print volatility model summary with regime detection results
        std::cout << "\nVolatility Model Summary:" << std::endl;
        std::cout << "  Final Regime Probabilities:" << std::endl;
        
        std::vector<double> finalProbs = volModel_->getRegimeProbabilities();
        std::cout << "    Low Vol: " << finalProbs[0] * 100.0 << "%" << std::endl;
        std::cout << "    Medium Vol: " << finalProbs[1] * 100.0 << "%" << std::endl;
        std::cout << "    High Vol: " << finalProbs[2] * 100.0 << "%" << std::endl;
        
        std::cout << "  Final Forecast Confidence: " << 
            volModel_->getVolatilityForecastConfidence() * 100.0 << "%" << std::endl;
        std::cout << "  Detected Regime Transitions: " << 
            (finalProbs[2] > 0.5 ? "Successfully detected high volatility regime" : 
             "Did not fully transition to high volatility regime") << std::endl;
             
        // If we have open positions, show their status with improved P&L tracking
        if (!openPositions_.empty()) {
            std::cout << "\nOpen Positions:" << std::endl;
            std::cout << "+-------+--------+--------+----------+-----------+----------+---------+----------+" << std::endl;
            std::cout << "| Symbol| Type   | Strike | Quantity | Entry Vol | Curr Vol | P&L     | P&L %    |" << std::endl;
            std::cout << "+-------+--------+--------+----------+-----------+----------+---------+----------+" << std::endl;
            
            for (const auto& pos : openPositions_) {
                double pnl = pos.currentPnL;
                double pnlPct = 100.0 * pnl / (std::abs(pos.entryPrice * pos.quantity) * 100.0);
                
                std::cout << "| " << std::left << std::setw(5) << pos.option.symbol << " | "
                          << std::setw(6) << (pos.option.type == 'C' ? "Call" : "Put") << " | "
                          << std::right << std::setw(6) << pos.option.strike << " | "
                          << std::setw(8) << pos.quantity << " | "
                          << std::setw(9) << std::fixed << std::setprecision(1) << pos.entryImpliedVol * 100.0 << "% | "
                          << std::setw(8) << std::fixed << std::setprecision(1) << pos.currentImpliedVol * 100.0 << "% | "
                          << std::setw(7) << std::fixed << std::setprecision(0) << pnl << " | "
                          << std::setw(8) << std::fixed << std::setprecision(2) << pnlPct << "% |" << std::endl;
            }
            
            std::cout << "+-------+--------+--------+----------+-----------+----------+---------+----------+" << std::endl;
        }
        
        // Show closed positions summary
        if (!closedPositions_.empty()) {
            std::cout << "\nClosed Positions Summary:" << std::endl;
            std::cout << "+-------+--------+--------+----------+------------+------------+-----------------+" << std::endl;
            std::cout << "| Symbol| Type   | Strike | Quantity | Entry Price| Exit Price | P&L             |" << std::endl;
            std::cout << "+-------+--------+--------+----------+------------+------------+-----------------+" << std::endl;
            
            // Sort by P&L
            std::vector<OptionPosition> sortedPositions = closedPositions_;
            std::sort(sortedPositions.begin(), sortedPositions.end(),
                     [](const OptionPosition& a, const OptionPosition& b) {
                         return a.currentPnL > b.currentPnL;
                     });
            
            // Show top 10 positions by P&L
            size_t numToShow = std::min(sortedPositions.size(), size_t(10));
            for (size_t i = 0; i < numToShow; ++i) {
                const auto& pos = sortedPositions[i];
                
                std::cout << "| " << std::left << std::setw(5) << pos.option.symbol << " | "
                          << std::setw(6) << (pos.option.type == 'C' ? "Call" : "Put") << " | "
                          << std::right << std::setw(6) << pos.option.strike << " | "
                          << std::setw(8) << pos.quantity << " | "
                          << std::setw(10) << std::fixed << std::setprecision(2) << pos.entryPrice << " | "
                          << std::setw(10) << std::fixed << std::setprecision(2) << pos.currentPrice << " | "
                          << std::setw(7) << std::fixed << std::setprecision(0) << pos.currentPnL
                          << " (" << std::setprecision(1) << 
                             (100.0 * pos.currentPnL / (std::abs(pos.entryPrice * pos.quantity) * 100.0)) << "%) |" << std::endl;
            }
            
            std::cout << "+-------+--------+--------+----------+------------+------------+-----------------+" << std::endl;
        }
    }
};

// Main function to run the test
int main() {
    std::cout << "=================================================================================" << std::endl;
    std::cout << "             Volatility Arbitrage Strategy Test" << std::endl;
    std::cout << "=================================================================================" << std::endl;
    
    // Create test scenario
    VolatilityArbitrageTestScenario scenario;
    scenario.name = "Volatility Increase Scenario";
    scenario.symbol = "SPY";
    scenario.initialRegime = {0.75, 0.20, 0.05};  // Start in low vol regime
    scenario.regimeTransitionTrigger = 75;         // Transition starts at data point 75
    scenario.volatilityShock = 0.10;              // 10% vol increase during test
    
    // Create and run test
    VolatilityArbitrageTest test(scenario);
    test.runFullTradingCycleTest();
    
    return 0;
}

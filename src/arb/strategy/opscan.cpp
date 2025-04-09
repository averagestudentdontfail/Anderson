#include "opportunity_scanner.h"
#include "../models/hybrid_model.h"
#include <algorithm>
#include <cmath>
#include <random>
#include <unordered_set>
#include <iostream>

namespace vol_arb {

//------------------------------------------------------------------------------
// MarketDataSystem Implementation
//------------------------------------------------------------------------------

double MarketDataSystem::getUnderlyingPrice(const std::string& symbol) const {
    // For testing, generate deterministic price based on symbol
    double basePrice = 0.0;
    
    if (symbol == "SPY") basePrice = 450.0;
    else if (symbol == "QQQ") basePrice = 380.0;
    else if (symbol == "AAPL") basePrice = 175.0;
    else if (symbol == "MSFT") basePrice = 320.0;
    else if (symbol == "AMZN") basePrice = 140.0;
    else if (symbol == "GOOGL") basePrice = 130.0;
    else if (symbol == "META") basePrice = 330.0;
    else if (symbol == "TSLA") basePrice = 240.0;
    else if (symbol == "NVDA") basePrice = 450.0;
    else basePrice = 100.0;
    
    // Add small random variation for testing
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> dist(0.0, basePrice * 0.002);
    return basePrice + dist(rng);
}

double MarketDataSystem::getRiskFreeRate() const {
    // Current approximate 3-month Treasury yield
    return 0.05;
}

double MarketDataSystem::getDividendYield(const std::string& symbol) const {
    // Default dividend yields for testing
    if (symbol == "SPY") return 0.015;
    else if (symbol == "QQQ") return 0.006;
    else if (symbol == "AAPL") return 0.005;
    else if (symbol == "MSFT") return 0.008;
    else if (symbol == "AMZN") return 0.0;
    else if (symbol == "GOOGL") return 0.0;
    else if (symbol == "META") return 0.005;
    else if (symbol == "TSLA") return 0.0;
    else if (symbol == "NVDA") return 0.001;
    else return 0.01;
}

OptionChain MarketDataSystem::getOptionChain(const std::string& symbol) const {
    return generateSyntheticOptionChain(symbol);
}

OptionChain MarketDataSystem::generateSyntheticOptionChain(const std::string& symbol) const {
    OptionChain chain;
    chain.underlyingSymbol = symbol;
    chain.underlyingPrice = getUnderlyingPrice(symbol);
    
    // Generate standard expiration dates (in days)
    std::vector<double> expirations = {7, 14, 30, 45, 60, 90, 180};
    chain.expirations = expirations;
    
    // Generate strikes around current price
    double basePrice = chain.underlyingPrice;
    double minStrike = basePrice * 0.8;
    double maxStrike = basePrice * 1.2;
    double strikeIncrement = basePrice * 0.025; // About 2.5% increments
    
    std::vector<double> strikes;
    for (double strike = minStrike; strike <= maxStrike; strike += strikeIncrement) {
        strikes.push_back(strike);
    }
    chain.strikes = strikes;
    
    // Get risk-free rate and dividend yield
    double riskFreeRate = getRiskFreeRate();
    double dividendYield = getDividendYield(symbol);
    
    // Create options for each combination of expiration and strike
    static std::mt19937 rng(std::random_device{}());
    std::normal_distribution<double> volNoise(0.0, 0.02); // Random noise for vol
    
    for (double expiry : expirations) {
        for (double strike : strikes) {
            // Base implied volatility parameters
            double baseVol = 0.20; // 20% base volatility
            
            // Apply skew based on moneyness
            double moneyness = strike / basePrice;
            double skew = (moneyness < 1.0) ? 
                          (1.0 - moneyness) * 0.2 :  // Add vol for OTM puts
                          (moneyness - 1.0) * 0.05;  // Slight increase for OTM calls
            
            // Apply term structure - higher vol for longer dated options
            double termAdjustment = std::sqrt(expiry / 30.0) * 0.03;
            
            // Calculate option-specific vol with noise
            double callVol = baseVol + skew + termAdjustment + volNoise(rng);
            double putVol = baseVol + skew + termAdjustment + volNoise(rng);
            
            // Create option symbols
            std::string callSymbol = symbol + "_C_" + std::to_string(static_cast<int>(strike)) + 
                                    "_" + std::to_string(static_cast<int>(expiry));
            
            std::string putSymbol = symbol + "_P_" + std::to_string(static_cast<int>(strike)) + 
                                   "_" + std::to_string(static_cast<int>(expiry));
            
            // Calculate theoretical prices using Black-Scholes (simplified)
            double timeToExpiry = expiry / 365.0;
            double sqrtTime = std::sqrt(timeToExpiry);
            
            // Simplified BS calculation
            double d1call = (std::log(basePrice / strike) + 
                           (riskFreeRate - dividendYield + 0.5 * callVol * callVol) * timeToExpiry) / 
                           (callVol * sqrtTime);
            double d2call = d1call - callVol * sqrtTime;
            
            double d1put = (std::log(basePrice / strike) + 
                          (riskFreeRate - dividendYield + 0.5 * putVol * putVol) * timeToExpiry) / 
                          (putVol * sqrtTime);
            double d2put = d1put - putVol * sqrtTime;
            
            // Normal CDF approximation
            auto normalCDF = [](double x) -> double {
                return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
            };
            
            double callPrice = basePrice * std::exp(-dividendYield * timeToExpiry) * normalCDF(d1call) - 
                              strike * std::exp(-riskFreeRate * timeToExpiry) * normalCDF(d2call);
            
            double putPrice = strike * std::exp(-riskFreeRate * timeToExpiry) * normalCDF(-d2put) - 
                             basePrice * std::exp(-dividendYield * timeToExpiry) * normalCDF(-d1put);
            
            // Add bid-ask spread (wider for less liquid options)
            double liquidityFactor = std::exp(-(std::abs(moneyness - 1.0) * 3.0 + expiry / 180.0));
            double minSpread = 0.05;
            double maxSpread = 0.20;
            double spreadRatio = minSpread + (1.0 - liquidityFactor) * (maxSpread - minSpread);
            
            double callSpread = std::max(0.01, callPrice * spreadRatio);
            double putSpread = std::max(0.01, putPrice * spreadRatio);
            
            double callBid = std::max(0.01, callPrice - callSpread / 2.0);
            double callAsk = callPrice + callSpread / 2.0;
            
            double putBid = std::max(0.01, putPrice - putSpread / 2.0);
            double putAsk = putPrice + putSpread / 2.0;
            
            // Create option data objects
            OptionData callOption(symbol, callSymbol, 'C', strike, expiry, basePrice, callBid, callAsk);
            callOption.lastPrice = callPrice;
            callOption.impliedVol = callVol;
            callOption.riskFreeRate = riskFreeRate;
            callOption.dividendYield = dividendYield;
            callOption.volume = 100.0 * liquidityFactor * (1000.0 + rng() % 10000);
            callOption.openInterest = 10.0 * liquidityFactor * (10000.0 + rng() % 100000);
            
            OptionData putOption(symbol, putSymbol, 'P', strike, expiry, basePrice, putBid, putAsk);
            putOption.lastPrice = putPrice;
            putOption.impliedVol = putVol;
            putOption.riskFreeRate = riskFreeRate;
            putOption.dividendYield = dividendYield;
            putOption.volume = 100.0 * liquidityFactor * (1000.0 + rng() % 10000);
            putOption.openInterest = 10.0 * liquidityFactor * (10000.0 + rng() % 100000);
            
            // Add to chain
            chain.calls.push_back(callOption);
            chain.puts.push_back(putOption);
        }
    }
    
    return chain;
}

//------------------------------------------------------------------------------
// HMM_GJRGARCHOpportunityScanner Implementation
//------------------------------------------------------------------------------

HMM_GJRGARCHOpportunityScanner::HMM_GJRGARCHOpportunityScanner(
    std::shared_ptr<VolatilityArbitrageStrategy> strategy,
    std::shared_ptr<MarketDataSystem> dataSystem,
    std::shared_ptr<HMM_GJRGARCHModel> volModel)
    : strategy_(strategy), dataSystem_(dataSystem), volModel_(volModel),
      minImpliedVolDifference_(0.03), minVegaValue_(0.1), maxSimultaneousOpportunities_(5) {
    
    if (!strategy_) {
        throw std::invalid_argument("Strategy pointer cannot be null");
    }
    
    if (!volModel_) {
        throw std::invalid_argument("Volatility model pointer cannot be null");
    }
    
    // Create a data system if not provided
    if (!dataSystem_) {
        dataSystem_ = std::make_shared<MarketDataSystem>();
    }
}

std::vector<ArbitrageOpportunity> HMM_GJRGARCHOpportunityScanner::scanOptionChain(
    const std::string& symbol) {
    
    // Clear previous opportunities
    opportunities_.clear();
    
    // Get option chain
    OptionChain chain = dataSystem_->getOptionChain(symbol);
    
    // Get all options
    std::vector<OptionData> options = chain.getAllOptions();
    
    // Evaluate each option for arbitrage signals
    for (const auto& option : options) {
        // Skip options with zero or negative bid/ask
        if (option.bid <= 0.0 || option.ask <= 0.0) {
            continue;
        }
        
        // Evaluate option for arbitrage
        ArbitrageSignal signal = strategy_->evaluateOption(option);
        
        // Check if the signal meets minimum criteria
        if (signal.direction != ArbitrageSignal::NEUTRAL &&
            std::abs(signal.volDifference) >= minImpliedVolDifference_ &&
            std::abs(signal.vegaExposure) >= minVegaValue_) {
            
            // Create an opportunity
            ArbitrageOpportunity opportunity(signal);
            
            // Calculate liquidity score
            opportunity.liquidity = calculateLiquidityScore(option);
            
            // Calculate overall score
            opportunity.score = calculateOpportunityScore(signal, opportunity.liquidity);
            
            // Add to opportunities list
            opportunities_.push_back(opportunity);
        }
    }
    
    // Sort opportunities by score (descending)
    std::sort(opportunities_.begin(), opportunities_.end(),
             [](const ArbitrageOpportunity& a, const ArbitrageOpportunity& b) {
                 return a.score > b.score;
             });
    
    // Limit to maximum number of opportunities
    if (opportunities_.size() > static_cast<size_t>(maxSimultaneousOpportunities_)) {
        opportunities_.resize(maxSimultaneousOpportunities_);
    }
    
    return opportunities_;
}

std::vector<ArbitrageOpportunity> HMM_GJRGARCHOpportunityScanner::rankByExpectedValue() {
    // Sort by expected value (descending)
    std::vector<ArbitrageOpportunity> result = opportunities_;
    
    std::sort(result.begin(), result.end(),
             [](const ArbitrageOpportunity& a, const ArbitrageOpportunity& b) {
                 return a.expectedValue > b.expectedValue;
             });
    
    return result;
}

std::vector<ArbitrageOpportunity> HMM_GJRGARCHOpportunityScanner::filterByLiquidityAndExecution(
    double minSpreadRatio) {
    
    std::vector<ArbitrageOpportunity> filtered;
    
    for (const auto& opportunity : opportunities_) {
        const OptionData& option = opportunity.option;
        
        // Calculate spread ratio (spread / mid price)
        double spread = option.ask - option.bid;
        double midPrice = (option.bid + option.ask) / 2.0;
        double spreadRatio = midPrice > 0.0 ? spread / midPrice : 1.0;
        
        // Check if the spread is tight enough
        if (spreadRatio <= minSpreadRatio) {
            // Check minimum volume and open interest requirements
            if (option.volume >= 10.0 && option.openInterest >= 100.0) {
                filtered.push_back(opportunity);
            }
        }
    }
    
    // Sort by combined score of expected value and liquidity
    std::sort(filtered.begin(), filtered.end(),
             [](const ArbitrageOpportunity& a, const ArbitrageOpportunity& b) {
                 return (a.expectedValue * a.liquidity) > (b.expectedValue * b.liquidity);
             });
    
    return filtered;
}

std::vector<ArbitrageOpportunity> HMM_GJRGARCHOpportunityScanner::scanForRegimeTransitionPlays() {
    std::vector<ArbitrageOpportunity> transitionPlays;
    
    // Get probability of regime transition
    double transitionProb = volModel_->getProbabilityOfRegimeTransition();
    
    // Only look for transition plays if probability is significant
    if (transitionProb < 0.25) {
        return transitionPlays;
    }
    
    // Get current regime
    MarketRegime currentRegime = volModel_->getCurrentDominantRegime();
    
    // Determine target symbols based on regime
    std::vector<std::string> symbols;
    
    if (currentRegime == LOW_VOLATILITY || currentRegime == MEDIUM_VOLATILITY) {
        // Transitioning to higher volatility - look at index options
        symbols = {"SPY", "QQQ"};
    } else {
        // Transitioning to lower volatility - look at individual stocks
        symbols = {"AAPL", "MSFT", "GOOGL", "META", "AMZN"};
    }
    
    // For each symbol, look for specific transition plays
    for (const std::string& symbol : symbols) {
        OptionChain chain = dataSystem_->getOptionChain(symbol);
        
        // Filter for options with:
        // - Longer dated (45+ days)
        // - OTM by 5-15% (for calls if going to higher vol, puts if going to lower vol)
        
        for (const auto& option : chain.getAllOptions()) {
            // Skip shorter-dated options
            if (option.expiryDays < 45.0) {
                continue;
            }
            
            // Calculate moneyness
            double moneyness = option.strike / option.underlyingPrice;
            
            bool isCandidate = false;
            
            if (currentRegime == LOW_VOLATILITY || currentRegime == MEDIUM_VOLATILITY) {
                // Going to higher vol - look for OTM calls
                if (option.type == 'C' && moneyness >= 1.05 && moneyness <= 1.15) {
                    isCandidate = true;
                }
            } else {
                // Going to lower vol - look for OTM puts
                if (option.type == 'P' && moneyness <= 0.95 && moneyness >= 0.85) {
                    isCandidate = true;
                }
            }
            
            if (isCandidate) {
                // Evaluate option
                ArbitrageSignal signal = strategy_->evaluateOption(option);
                
                // For transition plays, we may override the signal direction
                if (currentRegime == LOW_VOLATILITY || currentRegime == MEDIUM_VOLATILITY) {
                    // Going to higher vol - buy options
                    signal.direction = ArbitrageSignal::BUY;
                } else {
                    // Going to lower vol - sell options
                    signal.direction = ArbitrageSignal::SELL;
                }
                
                // Adjust confidence based on transition probability
                signal.confidence = std::max(signal.confidence, transitionProb);
                
                // Create opportunity
                ArbitrageOpportunity opportunity(signal);
                opportunity.liquidity = calculateLiquidityScore(option);
                
                // Calculate special score that factors in transition probability
                opportunity.score = calculateOpportunityScore(signal, opportunity.liquidity) * 
                                   (1.0 + transitionProb);
                
                transitionPlays.push_back(opportunity);
            }
        }
    }
    
    // Sort by score
    std::sort(transitionPlays.begin(), transitionPlays.end(),
             [](const ArbitrageOpportunity& a, const ArbitrageOpportunity& b) {
                 return a.score > b.score;
             });
    
    // Limit number of plays
    int maxTransitionPlays = 3;
    if (transitionPlays.size() > static_cast<size_t>(maxTransitionPlays)) {
        transitionPlays.resize(maxTransitionPlays);
    }
    
    return transitionPlays;
}

std::vector<ArbitrageOpportunity> HMM_GJRGARCHOpportunityScanner::identifyVolatilityTermStructureArbitrage() {
    std::vector<ArbitrageOpportunity> termStructurePlays;
    
    // List of symbols to check
    std::vector<std::string> symbols = {"SPY", "QQQ", "AAPL", "MSFT", "GOOGL"};
    
    for (const std::string& symbol : symbols) {
        OptionChain chain = dataSystem_->getOptionChain(symbol);
        
        // Group options by strike
        std::unordered_map<double, std::vector<OptionData>> strikeMap;
        
        for (const auto& option : chain.getAllOptions()) {
            strikeMap[option.strike].push_back(option);
        }
        
        // For each strike, look for vol term structure anomalies
        for (auto& [strike, options] : strikeMap) {
            // Skip if we don't have enough options at this strike
            if (options.size() < 3) {
                continue;
            }
            
            // Sort by expiration
            std::sort(options.begin(), options.end(),
                     [](const OptionData& a, const OptionData& b) {
                         return a.expiryDays < b.expiryDays;
                     });
            
            // Look for term structure inconsistencies
            for (size_t i = 1; i < options.size() - 1; ++i) {
                const OptionData& prev = options[i-1];
                const OptionData& curr = options[i];
                const OptionData& next = options[i+1];
                
                // Skip if not all the same type
                if (prev.type != curr.type || curr.type != next.type) {
                    continue;
                }
                
                // Calculate implied vol differences
                double volDiff1 = curr.impliedVol - prev.impliedVol;
                double volDiff2 = next.impliedVol - curr.impliedVol;
                
                // Time differences in days
                double timeDiff1 = curr.expiryDays - prev.expiryDays;
                double timeDiff2 = next.expiryDays - curr.expiryDays;
                
                // Calculate vol change per day
                double volPerDay1 = timeDiff1 > 0 ? volDiff1 / timeDiff1 : 0;
                double volPerDay2 = timeDiff2 > 0 ? volDiff2 / timeDiff2 : 0;
                
                // Look for significant difference in term structure slope
                if (std::abs(volPerDay2 - volPerDay1) > 0.001) {
                    // Potentially anomalous term structure
                    
                    // Create a "butterfly" trade: short the middle expiration, 
                    // long the shorter and longer expirations
                    ArbitrageSignal signal;
                    signal.option = curr;
                    signal.direction = ArbitrageSignal::SELL;
                    signal.confidence = 0.7;  // Assign reasonable confidence
                    signal.expectedValue = std::abs(volPerDay2 - volPerDay1) * 100;  // Scale for scoring
                    
                    ArbitrageOpportunity opportunity(signal);
                    opportunity.liquidity = calculateLiquidityScore(curr);
                    opportunity.score = calculateOpportunityScore(signal, opportunity.liquidity);
                    
                    termStructurePlays.push_back(opportunity);
                }
            }
        }
    }
    
    // Sort by score
    std::sort(termStructurePlays.begin(), termStructurePlays.end(),
             [](const ArbitrageOpportunity& a, const ArbitrageOpportunity& b) {
                 return a.score > b.score;
             });
    
    // Limit number of plays
    int maxPlays = 3;
    if (termStructurePlays.size() > static_cast<size_t>(maxPlays)) {
        termStructurePlays.resize(maxPlays);
    }
    
    return termStructurePlays;
}

double HMM_GJRGARCHOpportunityScanner::calculateLiquidityScore(const OptionData& option) const {
    // Factors affecting liquidity:
    // 1. Bid-ask spread relative to price
    // 2. Trading volume
    // 3. Open interest
    // 4. Option moneyness (ATM tends to be more liquid)
    
    // Calculate spread ratio
    double spreadRatio = 1.0;
    double midPrice = (option.bid + option.ask) / 2.0;
    if (midPrice > 0.0) {
        spreadRatio = (option.ask - option.bid) / midPrice;
    }
    
    // Spread score (lower spread = higher score)
    double spreadScore = std::exp(-10.0 * spreadRatio);
    
    // Volume score
    double volumeScore = std::min(1.0, option.volume / 1000.0);
    
    // Open interest score
    double oiScore = std::min(1.0, option.openInterest / 10000.0);
    
    // Moneyness score (1.0 at ATM, decreasing as we move away)
    double moneyness = option.strike / option.underlyingPrice;
    if (option.type == 'P') {
        moneyness = 1.0 / moneyness;  // For puts, invert moneyness
    }
    double moneynessScore = std::exp(-4.0 * std::pow(moneyness - 1.0, 2));
    
    // Combine scores with weights
    double liquidityScore = 0.4 * spreadScore + 
                           0.2 * volumeScore + 
                           0.2 * oiScore + 
                           0.2 * moneynessScore;
    
    return std::min(1.0, std::max(0.0, liquidityScore));
}

void HMM_GJRGARCHOpportunityScanner::setMinimumThresholds(double minVol, double minVega, int maxOpportunities) {
    minImpliedVolDifference_ = minVol;
    minVegaValue_ = minVega;
    maxSimultaneousOpportunities_ = maxOpportunities;
}

double HMM_GJRGARCHOpportunityScanner::calculateOpportunityScore(const ArbitrageSignal& signal, double liquidity) {
    // Weights for different factors
    const double EXPECTED_VALUE_WEIGHT = 0.35;
    const double CONFIDENCE_WEIGHT = 0.25;
    const double LIQUIDITY_WEIGHT = 0.20;
    const double VOL_DIFF_WEIGHT = 0.20;
    
    // Normalize expected value (arbitrary scaling factor)
    double normalizedEV = std::min(1.0, signal.expectedValue / 2.0);
    
    // Normalize vol difference
    double normalizedVolDiff = std::min(1.0, std::abs(signal.volDifference) / 0.1);
    
    // Combine factors
    double score = EXPECTED_VALUE_WEIGHT * normalizedEV +
                  CONFIDENCE_WEIGHT * signal.confidence +
                  LIQUIDITY_WEIGHT * liquidity +
                  VOL_DIFF_WEIGHT * normalizedVolDiff;
    
    return score;
}

bool HMM_GJRGARCHOpportunityScanner::hasEnoughPositionsInUnderlying(
    const std::string& symbol, int maxPerUnderlying) const {
    
    // Count how many opportunities we already have for this underlying
    int count = 0;
    
    for (const auto& opp : opportunities_) {
        if (opp.option.symbol == symbol) {
            count++;
        }
    }
    
    return count >= maxPerUnderlying;
}

} // namespace vol_arb
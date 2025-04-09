#include "vol_arb_strategy.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <random>
#include <limits>

namespace vol_arb {

//------------------------------------------------------------------------------
// ExecutionPlan Implementation
//------------------------------------------------------------------------------

ExecutionPlan::ExecutionPlan(const OptionData& option, ArbitrageSignal::Direction direction,
                         double targetPrice)
    : option_(option), direction_(direction), targetPrice_(targetPrice), valid_(true) {
    
    // Validate the plan
    if (direction == ArbitrageSignal::BUY && (option.ask <= 0.0 || std::isnan(option.ask))) {
        valid_ = false;
    } else if (direction == ArbitrageSignal::SELL && (option.bid <= 0.0 || std::isnan(option.bid))) {
        valid_ = false;
    }
    
    // Set target price if not provided
    if (targetPrice_ <= 0.0) {
        if (direction == ArbitrageSignal::BUY) {
            targetPrice_ = option.ask;
        } else if (direction == ArbitrageSignal::SELL) {
            targetPrice_ = option.bid;
        } else {
            targetPrice_ = (option.bid + option.ask) / 2.0;
        }
    }
}

OptionPosition ExecutionPlan::execute(const PositionSize& size) {
    // Create new position
    OptionPosition position;
    position.option = option_;
    position.quantity = direction_ == ArbitrageSignal::BUY ? size.contracts : -size.contracts;
    position.entryPrice = getExpectedPrice();
    position.entryTimestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count();
    position.lastUpdateTimestamp = position.entryTimestamp;
    
    // Generate unique ID
    std::stringstream ss;
    ss << option_.symbol << "_" << (option_.type == 'C' ? "CALL" : "PUT") << "_"
       << std::fixed << std::setprecision(2) << option_.strike << "_"
       << std::setprecision(0) << option_.expiryDays << "d_"
       << position.entryTimestamp;
    position.id = ss.str();
    
    // Set current values same as entry values initially
    position.currentPrice = position.entryPrice;
    position.currentImpliedVol = position.entryImpliedVol;
    position.currentForecastVol = position.entryForecastVol;
    position.currentDelta = position.entryDelta;
    position.currentGamma = position.entryGamma;
    position.currentVega = position.entryVega;
    position.currentTheta = position.entryTheta;
    
    return position;
}

double ExecutionPlan::getExpectedPrice() const {
    if (!valid_) {
        return 0.0;
    }
    
    // Return target price if explicitly set
    if (targetPrice_ > 0.0) {
        return targetPrice_;
    }
    
    // Otherwise use bid/ask as appropriate
    if (direction_ == ArbitrageSignal::BUY) {
        return option_.ask;
    } else if (direction_ == ArbitrageSignal::SELL) {
        return option_.bid;
    }
    
    // For neutral, use mid price
    return (option_.bid + option_.ask) / 2.0;
}

double ExecutionPlan::getExpectedCost(const PositionSize& size) const {
    return getExpectedPrice() * size.contracts * (option_.type == 'C' ? 100.0 : 100.0);
}

bool ExecutionPlan::isValid() const {
    return valid_;
}

//------------------------------------------------------------------------------
// EnhancedIVSolver Implementation
//------------------------------------------------------------------------------

EnhancedIVSolver::EnhancedIVSolver(std::shared_ptr<ALOEngine> engine)
    : engine_(engine) {
    
    if (!engine_) {
        throw std::invalid_argument("ALOEngine pointer cannot be null");
    }
}

double EnhancedIVSolver::solveForImpliedVol(const OptionData& option) {
    // Handle special cases
    if (option.bid <= 0.0 && option.ask <= 0.0 && option.lastPrice <= 0.0) {
        // No valid price reference
        return 0.0;
    }
    
    // Use mid price by default, or last traded price if bid/ask not available
    double targetPrice;
    if (option.bid > 0.0 && option.ask > 0.0) {
        targetPrice = (option.bid + option.ask) / 2.0;
    } else if (option.lastPrice > 0.0) {
        targetPrice = option.lastPrice;
    } else if (option.bid > 0.0) {
        targetPrice = option.bid;
    } else {
        targetPrice = option.ask;
    }
    
    // Check for deep ITM/OTM cases
    if (targetPrice <= 0.01) {
        return handleDeepOTMOptions(option);
    }
    
    double intrinsicValue = 0.0;
    if (option.type == 'C') {
        intrinsicValue = std::max(0.0, option.underlyingPrice - option.strike);
    } else {
        intrinsicValue = std::max(0.0, option.strike - option.underlyingPrice);
    }
    
    if (targetPrice <= intrinsicValue * 1.01) {
        return handleDeepITMOptions(option);
    }
    
    // Define price difference function for numerical solving
    auto priceDiffFunction = [this, &option, targetPrice](double vol) -> double {
        OptionType optionType = option.type == 'C' ? CALL : PUT;
        double modelPrice = 0.0;
        
        try {
            modelPrice = engine_->calculateOption(
                option.underlyingPrice,
                option.strike,
                option.riskFreeRate,
                option.dividendYield,
                vol,
                option.expiryDays / 365.0,
                optionType
            );
        } catch (const std::exception& e) {
            // Handle calculation errors
            std::cerr << "Error in price calculation: " << e.what() << std::endl;
            return std::numeric_limits<double>::max();
        }
        
        return modelPrice - targetPrice;
    };
    
    // Apply Brent's method for numerical solving
    try {
        return applyBrentsMethod(priceDiffFunction, MIN_VOL, MAX_VOL);
    } catch (const std::exception& e) {
        std::cerr << "Error in implied volatility solver: " << e.what() << std::endl;
        
        // Fallback to bracketing search
        double lo = MIN_VOL;
        double hi = MAX_VOL;
        double mid;
        
        for (int i = 0; i < MAX_ITERATIONS; ++i) {
            mid = (lo + hi) / 2.0;
            double diff = priceDiffFunction(mid);
            
            if (std::abs(diff) < PRECISION) {
                return mid;
            }
            
            if (diff > 0) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        
        return mid;
    }
}

double EnhancedIVSolver::handleDeepITMOptions(const OptionData& option) {
    // For deep ITM options, we use a combination of historical vol and skew adjustment
    
    // Start with a low volatility estimate
    double baseVol = 0.15;  // Default 15% volatility as starting point
    
    // Apply moneyness adjustment (reduce vol for deeper ITM options)
    double moneyness = option.underlyingPrice / option.strike;
    if (option.type == 'P') {
        moneyness = 1.0 / moneyness;  // Adjust for puts
    }
    
    // Apply skew adjustment (deeper ITM typically has lower implied vol)
    double skewAdjustment = std::max(0.6, 1.0 - 0.2 * (moneyness - 1.0));
    
    return baseVol * skewAdjustment;
}

double EnhancedIVSolver::handleDeepOTMOptions(const OptionData& option) {
    // For deep OTM options, we use a combination of historical vol and skew adjustment
    
    // Start with a moderate volatility estimate
    double baseVol = 0.20;  // Default 20% volatility as starting point
    
    // Apply moneyness adjustment (increase vol for deeper OTM options)
    double moneyness = option.underlyingPrice / option.strike;
    if (option.type == 'P') {
        moneyness = 1.0 / moneyness;  // Adjust for puts
    }
    
    // Apply skew adjustment (deeper OTM typically has higher implied vol)
    double skewAdjustment = std::min(1.5, 1.0 + 0.25 * (1.0 - moneyness));
    
    return baseVol * skewAdjustment;
}

double EnhancedIVSolver::applyBrentsMethod(const std::function<double(double)>& priceDiffFunc,
                                       double volLower, double volUpper) {
    // Implementation of Brent's method for finding roots
    const double EPSILON = 1e-10;
    
    double a = volLower;
    double b = volUpper;
    double c = volUpper;
    double d = 0.0;
    
    double fa = priceDiffFunc(a);
    double fb = priceDiffFunc(b);
    
    if (fa * fb > 0) {
        throw std::runtime_error("Root not bracketed");
    }
    
    double fc = fb;
    
    for (int i = 0; i < MAX_ITERATIONS; ++i) {
        // Check if we need to swap bounds
        if (fb * fc > 0) {
            c = a;
            fc = fa;
            d = b - a;
        }
        
        if (std::abs(fc) < std::abs(fb)) {
            a = b;
            b = c;
            c = a;
            fa = fb;
            fb = fc;
            fc = fa;
        }
        
        // Convergence check
        double tol = 2.0 * EPSILON * std::abs(b) + PRECISION;
        double m = 0.5 * (c - b);
        
        if (std::abs(m) <= tol || std::abs(fb) < PRECISION) {
            return b;
        }
        
        // Decide which method to use
        if (std::abs(fa) > std::abs(fb) && std::abs(fb) > std::abs(fc)) {
            // Use inverse quadratic interpolation
            double s = fb / fa;
            double t = fb / fc;
            double u = fa / fc;
            
            double p = s * (t * (u - t) * (c - b) - (1.0 - t) * (b - a));
            double q = (t - 1.0) * (u - 1.0) * (s - 1.0);
            
            d = p / q;
        } else {
            // Use secant method
            d = (b - a) * fb / (fb - fa);
        }
        
        // Check if we need to use bisection
        if (std::abs(d) > 0.5 * std::abs(b - a) ||
            (b + d) < volLower || (b + d) > volUpper) {
            d = m;  // bisection
        }
        
        // Update bounds
        a = b;
        fa = fb;
        
        b = b + d;
        fb = priceDiffFunc(b);
    }
    
    return b;  // Return best estimate after MAX_ITERATIONS
}

std::vector<double> EnhancedIVSolver::fitLocalVolatilitySmile(const std::vector<OptionData>& optionChain) {
    if (optionChain.empty()) {
        return {};
    }
    
    // Filter for valid options of the same type and expiry
    char optionType = optionChain[0].type;
    double expiryDays = optionChain[0].expiryDays;
    
    std::vector<std::pair<double, double>> strikeVolPairs;
    
    for (const auto& option : optionChain) {
        if (option.type == optionType && 
            std::abs(option.expiryDays - expiryDays) < 0.1) {
            
            double iv = solveForImpliedVol(option);
            if (iv > MIN_VOL && iv < MAX_VOL) {
                // Store strike-vol pairs
                strikeVolPairs.emplace_back(option.strike, iv);
            }
        }
    }
    
    // Need at least a few points for meaningful interpolation
    if (strikeVolPairs.size() < 3) {
        return {};
    }
    
    // Sort by strike
    std::sort(strikeVolPairs.begin(), strikeVolPairs.end(),
             [](const auto& a, const auto& b) { return a.first < b.first; });
    
    // Extract strikes and vols
    std::vector<double> strikes, vols;
    for (const auto& pair : strikeVolPairs) {
        strikes.push_back(pair.first);
        vols.push_back(pair.second);
    }
    
    // Perform simple cubic spline interpolation (simplified implementation)
    // In a real system, you'd use a more sophisticated fitting algorithm
    
    // For now, just return the sorted volatilities
    std::vector<double> result;
    for (double vol : vols) {
        result.push_back(vol);
    }
    
    return result;
}

//------------------------------------------------------------------------------
// VolatilityArbitrageStrategy Implementation
//------------------------------------------------------------------------------

VolatilityArbitrageStrategy::VolatilityArbitrageStrategy(
    std::shared_ptr<ALOEngine> engine,
    std::shared_ptr<HMM_GJRGARCHModel> volModel,
    std::shared_ptr<EnhancedIVSolver> ivSolver)
    : engine_(engine), volModel_(volModel), vegaExposureLimit_(1000000.0),
      deltaExposureLimit_(1000000.0), positionSizeLimit_(0.05), totalCapital_(1000000.0) {
    
    if (!engine_) {
        throw std::invalid_argument("ALOEngine pointer cannot be null");
    }
    
    if (!volModel_) {
        throw std::invalid_argument("HMM_GJRGARCHModel pointer cannot be null");
    }
    
    // Create IV solver if not provided
    if (!ivSolver) {
        ivSolver_ = std::make_shared<EnhancedIVSolver>(engine_);
    } else {
        ivSolver_ = ivSolver;
    }
    
    // Initialize with default strategy parameters
    params_ = StrategyParams();
    
    // Setup regime-specific parameters
    // Low volatility regime - smaller spreads, higher position sizes
    regimeParams_[LOW_VOLATILITY] = StrategyParams(0.02, 0.08, 0.55, 0.008, 28);
    
    // Medium volatility regime - balanced approach
    regimeParams_[MEDIUM_VOLATILITY] = StrategyParams(0.03, 0.05, 0.60, 0.01, 21);
    
    // High volatility regime - larger spreads, smaller positions, shorter holding periods
    regimeParams_[HIGH_VOLATILITY] = StrategyParams(0.05, 0.03, 0.70, 0.015, 14);
    
    // Transition regime - cautious approach
    regimeParams_[TRANSITION_REGIME] = StrategyParams(0.04, 0.04, 0.65, 0.012, 17);
    
    // Default to medium volatility parameters initially
    params_ = regimeParams_[MEDIUM_VOLATILITY];
}

ArbitrageSignal VolatilityArbitrageStrategy::evaluateOption(const OptionData& option) {
    ArbitrageSignal signal;
    signal.option = option;
    
    // Calculate implied volatility if not provided
    double impliedVol = option.impliedVol;
    if (impliedVol <= 0.0) {
        impliedVol = calculateModelImpliedVol(option);
    }
    
    // Get forecast volatility from model
    double forecastVol = volModel_->forecastVolatility(option.symbol, 
                                                    static_cast<int>(option.expiryDays / 7.0));
    
    // Calculate spread between implied and forecast volatility
    double volSpread = impliedVol - forecastVol;
    
    // Calculate expected value of trade
    double expectedValue = calculateExpectedValue(option, forecastVol, impliedVol);
    
    // Calculate confidence level
    double confidence = calculateConfidence(option, forecastVol, volSpread);
    
    // Determine trade direction
    ArbitrageSignal::Direction direction = ArbitrageSignal::NEUTRAL;
    
    if (volSpread > params_.minVolSpread && confidence >= params_.confidenceThreshold) {
        // Implied vol > forecast vol -> sell option
        direction = ArbitrageSignal::SELL;
    } else if (volSpread < -params_.minVolSpread && confidence >= params_.confidenceThreshold) {
        // Implied vol < forecast vol -> buy option
        direction = ArbitrageSignal::BUY;
    }
    
    // Calculate price difference
    double modelPrice = calculateModelPrice(option, forecastVol);
    double marketPrice = (option.bid + option.ask) / 2.0;
    double priceDiff = marketPrice - modelPrice;
    
    // Calculate vega exposure
    // Approximate vega = price * sqrt(time) * S / (100 * vol)
    double spot = option.underlyingPrice;
    double timeInYears = option.expiryDays / 365.0;
    double vegaApprox = marketPrice * std::sqrt(timeInYears) * spot / (100.0 * impliedVol);
    
    // Populate the signal
    signal.direction = direction;
    signal.forecastVol = forecastVol;
    signal.impliedVol = impliedVol;
    signal.volDifference = volSpread;
    signal.expectedValue = expectedValue;
    signal.confidence = confidence;
    signal.vegaExposure = vegaApprox;
    signal.priceDifference = priceDiff;
    
    return signal;
}

double VolatilityArbitrageStrategy::calculateImpliedVolSpread(const OptionData& option) {
    // Calculate implied volatility
    double impliedVol = option.impliedVol;
    if (impliedVol <= 0.0) {
        impliedVol = calculateModelImpliedVol(option);
    }
    
    // Get forecast volatility
    double forecastVol = volModel_->forecastVolatility(option.symbol, 
                                                    static_cast<int>(option.expiryDays / 7.0));
    
    return impliedVol - forecastVol;
}

PositionSize VolatilityArbitrageStrategy::calculateOptimalPositionSize(
    const ArbitrageSignal& signal, const Portfolio& currentPositions) {
    
    // Default response - no position
    PositionSize size;
    
    // Return empty size if signal is neutral
    if (signal.direction == ArbitrageSignal::NEUTRAL) {
        return size;
    }
    
    // Check for minimum criteria
    if (std::abs(signal.volDifference) < params_.minVolSpread ||
        signal.confidence < params_.confidenceThreshold) {
        return size;
    }
    
    // Get symbol and check existing exposure
    const std::string& symbol = signal.option.symbol;
    
    // Calculate current vega exposure to this underlying
    double currentVegaExposure = 0.0;
    auto vegaByUnderlying = currentPositions.getVegaByUnderlying();
    if (vegaByUnderlying.find(symbol) != vegaByUnderlying.end()) {
        currentVegaExposure = vegaByUnderlying[symbol];
    }
    
    // Calculate maximum vega exposure for this symbol
    double maxSymbolVegaExposure = vegaExposureLimit_ * 0.25;  // Max 25% to any one symbol
    
    // Calculate remaining vega capacity
    double remainingVegaCapacity = maxSymbolVegaExposure - std::abs(currentVegaExposure);
    if (remainingVegaCapacity <= 0) {
        return size;  // No capacity remaining
    }
    
    // Calculate position size based on vega exposure
    double vegaPerContract = signal.vegaExposure;
    
    // Adjust for confidence
    double confidenceAdjustment = 0.5 + 0.5 * signal.confidence;  // 0.5 to 1.0 scaling
    
    // Adjust for regime and volatility spread
    double spreadAdjustment = std::min(1.0, std::abs(signal.volDifference) / 0.05);
    
    // Calculate target vega exposure
    double targetVegaExposure = remainingVegaCapacity * confidenceAdjustment * spreadAdjustment;
    
    // Calculate contracts based on vega
    int maxContractsByVega = vegaPerContract > 0 ? 
        static_cast<int>(targetVegaExposure / vegaPerContract) : 0;
    
    // Calculate contracts based on capital allocation limit
    double optionPrice = signal.direction == ArbitrageSignal::BUY ? 
        signal.option.ask : signal.option.bid;
    double contractSize = optionPrice * 100.0;  // Assuming standard 100 multiplier
    
    double availableCapital = totalCapital_ * params_.maxPositionSize;
    int maxContractsByCapital = contractSize > 0 ?
        static_cast<int>(availableCapital / contractSize) : 0;
    
    // Take the minimum of the two constraints
    int contracts = std::min(maxContractsByVega, maxContractsByCapital);
    
    // Ensure at least 1 contract if we're trading
    if (contracts > 0) {
        size.contracts = contracts;
        size.notionalValue = contractSize * contracts;
        size.maxRisk = size.notionalValue;  // Simplified - full premium at risk
        size.vegaRisk = vegaPerContract * contracts;
    }
    
    return size;
}

void VolatilityArbitrageStrategy::adjustStrategyForCurrentRegime() {
    // Get current market regime
    MarketRegime currentRegime = volModel_->getCurrentDominantRegime();
    
    // Check if we have parameters for this regime
    if (regimeParams_.find(currentRegime) != regimeParams_.end()) {
        params_ = regimeParams_[currentRegime];
    } else {
        // Default to medium volatility regime if not found
        params_ = regimeParams_[MEDIUM_VOLATILITY];
    }
}

double VolatilityArbitrageStrategy::getVolatilityConvergenceSpeed(MarketRegime regime) {
    // Estimate of fraction of vol difference that converges per day
    
    switch (regime) {
        case LOW_VOLATILITY:
            return 0.03;  // Slower convergence in low vol
            
        case MEDIUM_VOLATILITY:
            return 0.05;  // Moderate convergence
            
        case HIGH_VOLATILITY:
            return 0.08;  // Faster convergence in high vol
            
        case TRANSITION_REGIME:
            return 0.07;  // Fairly fast convergence in transition periods
            
        default:
            return 0.05;  // Default to moderate speed
    }
}

ExecutionPlan VolatilityArbitrageStrategy::createExecutionPlan(const ArbitrageSignal& signal) {
    double targetPrice = 0.0;
    
    // Calculate target price with a small buffer for better execution
    if (signal.direction == ArbitrageSignal::BUY) {
        // When buying, aim slightly above mid but below ask
        targetPrice = signal.option.bid + (signal.option.ask - signal.option.bid) * 0.6;
    } else if (signal.direction == ArbitrageSignal::SELL) {
        // When selling, aim slightly below mid but above bid
        targetPrice = signal.option.bid + (signal.option.ask - signal.option.bid) * 0.4;
    }
    
    return ExecutionPlan(signal.option, signal.direction, targetPrice);
}

void VolatilityArbitrageStrategy::monitorActivePositions(std::vector<OptionPosition>& positions) {
    // Update each position with current market data
    for (auto& position : positions) {
        // Copy the option data to get current values
        OptionData currentOption = position.option;
        
        // Update option data with current values (would come from market data in real system)
        // For this implementation, we'll simulate some time decay and vol convergence
        
        // Get days since entry
        uint64_t currentTime = std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        double daysSinceEntry = static_cast<double>(currentTime - position.entryTimestamp) / 
                               (1e9 * 3600 * 24);  // Convert ns to days
        
        // Update days to expiry
        currentOption.expiryDays = position.option.expiryDays - daysSinceEntry;
        if (currentOption.expiryDays <= 0) {
            // Option has expired, handle appropriately
            continue;
        }
        
        // Get current forecast volatility
        double currentForecastVol = volModel_->forecastVolatility(
            currentOption.symbol, static_cast<int>(currentOption.expiryDays / 7.0));
        
        // Estimate volatility convergence
        MarketRegime currentRegime = volModel_->getCurrentDominantRegime();
        double convergenceSpeed = getVolatilityConvergenceSpeed(currentRegime);
        
        // Calculate expected current implied vol based on convergence
        double expectedConvergence = (position.entryForecastVol - position.entryImpliedVol) *
                                    (1.0 - std::pow(1.0 - convergenceSpeed, daysSinceEntry));
        double currentImpliedVol = position.entryImpliedVol + expectedConvergence;
        
        // Update option with new implied vol
        currentOption.impliedVol = currentImpliedVol;
        
        // Calculate current price
        double currentPrice = calculateModelPrice(currentOption, currentImpliedVol);
        
        // Update position
        position.currentPrice = currentPrice;
        position.currentImpliedVol = currentImpliedVol;
        position.currentForecastVol = currentForecastVol;
        position.lastUpdateTimestamp = currentTime;
        
        // Update Greeks (simplified calculation)
        // In a real implementation, these would be calculated from the model
        double timeToExpiry = currentOption.expiryDays / 365.0;
        double sqrt_t = std::sqrt(timeToExpiry);
        double spot = currentOption.underlyingPrice;
        
        // Very simplified Greeks calculations (not accurate, just for example)
        position.currentVega = currentPrice * sqrt_t * spot / (100.0 * currentImpliedVol);
        position.currentTheta = -currentPrice * spot * currentImpliedVol / (200.0 * sqrt_t * 365.0);
        
        // Delta calculation depends on option type and moneyness
        double moneyness = spot / currentOption.strike;
        if (currentOption.type == 'P') {
            moneyness = 1.0 / moneyness;
            position.currentDelta = -0.5 + 0.5 * (1.0 - moneyness) / (currentImpliedVol * sqrt_t);
        } else {
            position.currentDelta = 0.5 + 0.5 * (moneyness - 1.0) / (currentImpliedVol * sqrt_t);
        }
        
        // Bound delta between -1 and 1
        position.currentDelta = std::max(-1.0, std::min(1.0, position.currentDelta));
        
        // Gamma is second derivative of delta with respect to spot
        position.currentGamma = position.currentVega / (spot * spot * currentImpliedVol * sqrt_t);
    }
}

void VolatilityArbitrageStrategy::setStrategyParameters(const StrategyParams& params) {
    params_ = params;
}

void VolatilityArbitrageStrategy::setRiskLimits(
    double vegaLimit, double deltaLimit, double positionSizeLimit, double confidenceThreshold) {
    
    vegaExposureLimit_ = vegaLimit;
    deltaExposureLimit_ = deltaLimit;
    positionSizeLimit_ = positionSizeLimit;
    params_.confidenceThreshold = confidenceThreshold;
}

void VolatilityArbitrageStrategy::setTotalCapital(double capital) {
    totalCapital_ = capital;
}

const StrategyParams& VolatilityArbitrageStrategy::getStrategyParameters() const {
    return params_;
}

double VolatilityArbitrageStrategy::calculateModelPrice(const OptionData& option, double volatility) {
    try {
        OptionType optionType = option.type == 'C' ? CALL : PUT;
        
        return engine_->calculateOption(
            option.underlyingPrice,
            option.strike,
            option.riskFreeRate,
            option.dividendYield,
            volatility,
            option.expiryDays / 365.0,
            optionType
        );
    } catch (const std::exception& e) {
        std::cerr << "Error calculating model price: " << e.what() << std::endl;
        return 0.0;
    }
}

double VolatilityArbitrageStrategy::calculateModelImpliedVol(const OptionData& option) {
    return ivSolver_->solveForImpliedVol(option);
}

double VolatilityArbitrageStrategy::calculateExpectedValue(
    const OptionData& option, double forecastVol, double impliedVol) {
    
    // Calculate model prices using both volatilities
    double modelPriceWithForecast = calculateModelPrice(option, forecastVol);
    double modelPriceWithImplied = calculateModelPrice(option, impliedVol);
    
    // Expected value is the price difference (depends on trade direction)
    if (impliedVol > forecastVol) {
        // Overpriced option - expected value comes from selling
        return modelPriceWithImplied - modelPriceWithForecast;
    } else {
        // Underpriced option - expected value comes from buying
        return modelPriceWithForecast - modelPriceWithImplied;
    }
}

double VolatilityArbitrageStrategy::calculateConfidence(
    const OptionData& option, double forecastVol, double volSpread) {
    
    // Base confidence from the volatility model
    double baseConfidence = volModel_->getVolatilityForecastConfidence();
    
    // Adjust based on spread size (higher spread = higher confidence)
    double spreadFactor = std::min(1.0, std::abs(volSpread) / 0.05);
    
    // Adjust based on time to expiry (longer expiry = more time for convergence)
    double timeFactor = std::min(1.0, option.expiryDays / 45.0);
    
    // Adjust based on liquidity (tighter bid-ask = higher confidence)
    double bidAskSpread = option.ask - option.bid;
    double midPrice = (option.bid + option.ask) / 2.0;
    double spreadPct = midPrice > 0.0 ? bidAskSpread / midPrice : 1.0;
    double liquidityFactor = std::max(0.5, 1.0 - spreadPct);
    
    // Combine factors with different weights
    double weightedConfidence = baseConfidence * 0.4 +  // 40% weight to model confidence
                               spreadFactor * 0.3 +     // 30% weight to spread size
                               timeFactor * 0.2 +       // 20% weight to time factor
                               liquidityFactor * 0.1;   // 10% weight to liquidity
    
    return weightedConfidence;
}

} // namespace vol_arb
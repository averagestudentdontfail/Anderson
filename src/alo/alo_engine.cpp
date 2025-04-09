#include "alo_engine.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>

// ALOEngine constructor
ALOEngine::ALOEngine(ALOScheme scheme, FixedPointEquation eq) 
    : equation_(eq) {
    // Set up the scheme based on parameter
    setScheme(scheme);
}

// Schema setter
void ALOEngine::setScheme(ALOScheme scheme) {
    switch (scheme) {
        case FAST:
            scheme_ = createFastScheme();
            break;
        case ACCURATE:
            scheme_ = createAccurateScheme();
            break;
        case HIGH_PRECISION:
            scheme_ = createHighPrecisionScheme();
            break;
        default:
            throw std::invalid_argument("Unknown ALO scheme");
    }
}

// Fixed point equation setter
void ALOEngine::setFixedPointEquation(FixedPointEquation eq) {
    equation_ = eq;
}

// Black-Scholes formula for European put
double ALOEngine::blackScholesPut(double S, double K, double r, double q, double vol, double T) {
    if (vol <= 0.0 || T <= 0.0) {
        return std::max(0.0, K - S);
    }
    
    double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
    double d2 = d1 - vol * std::sqrt(T);
    
    double nd1 = 0.5 * (1.0 + std::erf(-d1 / std::sqrt(2.0)));
    double nd2 = 0.5 * (1.0 + std::erf(-d2 / std::sqrt(2.0)));
    
    return K * std::exp(-r * T) * nd2 - S * std::exp(-q * T) * nd1;
}

// Black-Scholes formula for European call
double ALOEngine::blackScholesCall(double S, double K, double r, double q, double vol, double T) {
    if (vol <= 0.0 || T <= 0.0) {
        return std::max(0.0, S - K);
    }
    
    double d1 = (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
    double d2 = d1 - vol * std::sqrt(T);
    
    double nd1 = 0.5 * (1.0 + std::erf(d1 / std::sqrt(2.0)));
    double nd2 = 0.5 * (1.0 + std::erf(d2 / std::sqrt(2.0)));
    
    return S * std::exp(-q * T) * nd1 - K * std::exp(-r * T) * nd2;
}

// Main calculation function that routes to appropriate implementation
double ALOEngine::calculateOption(double S, double K, double r, double q, double vol, double T, OptionType type) const {
    if (type == PUT) {
        return calculatePutImpl(S, K, r, q, vol, T);
    } else { // CALL
        return calculateCallImpl(S, K, r, q, vol, T);
    }
}

// Legacy method for backward compatibility
double ALOEngine::calculatePut(double S, double K, double r, double q, double vol, double T) const {
    return calculatePutImpl(S, K, r, q, vol, T);
}

// Calculate early exercise premium
double ALOEngine::calculateEarlyExercisePremium(double S, double K, double r, double q, double vol, double T, 
                                               OptionType type) const {
    double american = calculateOption(S, K, r, q, vol, T, type);
    double european = (type == PUT) ? 
        blackScholesPut(S, K, r, q, vol, T) : 
        blackScholesCall(S, K, r, q, vol, T);
    return american - european;
}

// Maximum early exercise boundary value for puts
double ALOEngine::xMaxPut(double K, double r, double q) const {
    // Table 2 from the paper for puts
    if (r > 0.0 && q > 0.0)
        return K * std::min(1.0, r / q);
    else if (r > 0.0 && q <= 0.0)
        return K;
    else if (r == 0.0 && q < 0.0)
        return K;
    else if (r == 0.0 && q >= 0.0)
        return 0.0; // European case
    else if (r < 0.0 && q >= 0.0)
        return 0.0; // European case
    else if (r < 0.0 && q < r)
        return K; // double boundary case
    else if (r < 0.0 && r <= q && q < 0.0)
        return 0.0; // European case
    else
        throw std::runtime_error("Internal error in xMaxPut calculation");
}

// Maximum early exercise boundary value for calls
double ALOEngine::xMaxCall(double K, double r, double q) const {
    // For call options, the early exercise boundary is different
    if (q > 0.0 && r >= 0.0)
        return K * std::max(1.0, r / q);
    else if (q <= 0.0 && r >= 0.0)
        return HUGE_VAL; // Effectively infinite, early exercise never optimal
    else if (q >= r && r < 0.0)
        return HUGE_VAL; // European case
    else if (q < r && r < 0.0)
        return K; // Double boundary case
    else
        throw std::runtime_error("Internal error in xMaxCall calculation");
}

// Implementation for American put options
double ALOEngine::calculatePutImpl(double S, double K, double r, double q, double vol, double T) const {
    // Check for special cases
    if (K <= 0.0 || S <= 0.0 || vol <= 0.0 || T <= 0.0) {
        return std::max(0.0, K - S);
    }
    
    // Cache key for memoization
    std::ostringstream key_stream;
    key_stream << std::fixed << std::setprecision(10)
              << S << "_" << K << "_" << r << "_" << q << "_" << vol << "_" << T << "_PUT";
    std::string cache_key = key_stream.str();
    
    // Check cache
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
        return it->second;
    }
    
    // Handle special cases from the Leif Andersen, Mark Lake paper
    if (r < 0.0 && q < r) {
        throw std::runtime_error("Double-boundary case q<r<0 for a put option is not implemented");
    }
    
    // Check if we're in the European case
    if (r <= 0.0 && r <= q) {
        double european_value = blackScholesPut(S, K, r, q, vol, T);
        cache_[cache_key] = european_value;
        return european_value;
    }
    
    // Calculate early exercise boundary
    auto boundary = calculatePutExerciseBoundary(S, K, r, q, vol, T);
    
    // Calculate early exercise premium
    double earlyExercisePremium = calculatePutExercisePremium(S, K, r, q, vol, T, boundary);
    
    // Calculate European put value
    double europeanValue = blackScholesPut(S, K, r, q, vol, T);
    
    // American put value = European put value + early exercise premium
    double result = std::max(europeanValue, 0.0) + std::max(0.0, earlyExercisePremium);
    
    // Cache result
    cache_[cache_key] = result;
    
    return result;
}

// Implementation for American call options
double ALOEngine::calculateCallImpl(double S, double K, double r, double q, double vol, double T) const {
    // Check for special cases
    if (K <= 0.0 || S <= 0.0 || vol <= 0.0 || T <= 0.0) {
        return std::max(0.0, S - K);
    }
    
    // Cache key for memoization
    std::ostringstream key_stream;
    key_stream << std::fixed << std::setprecision(10)
              << S << "_" << K << "_" << r << "_" << q << "_" << vol << "_" << T << "_CALL";
    std::string cache_key = key_stream.str();
    
    // Check cache
    auto it = cache_.find(cache_key);
    if (it != cache_.end()) {
        return it->second;
    }
    
    // For calls with no dividends, early exercise is never optimal
    if (q <= 0.0) {
        double european_value = blackScholesCall(S, K, r, q, vol, T);
        cache_[cache_key] = european_value;
        return european_value;
    }
    
    // Check if we're in the European case
    if (r < 0.0 && q >= r) {
        double european_value = blackScholesCall(S, K, r, q, vol, T);
        cache_[cache_key] = european_value;
        return european_value;
    }
    
    // Calculate early exercise boundary
    auto boundary = calculateCallExerciseBoundary(S, K, r, q, vol, T);
    
    // Calculate early exercise premium
    double earlyExercisePremium = calculateCallExercisePremium(S, K, r, q, vol, T, boundary);
    
    // Calculate European call value
    double europeanValue = blackScholesCall(S, K, r, q, vol, T);
    
    // American call value = European call value + early exercise premium
    double result = std::max(europeanValue, 0.0) + std::max(0.0, earlyExercisePremium);
    
    // Cache result
    cache_[cache_key] = result;
    
    return result;
}

// Calculate early exercise boundary for puts using Chebyshev interpolation
std::shared_ptr<numerics::ChebyshevInterpolation> ALOEngine::calculatePutExerciseBoundary(
    double S, double K, double r, double q, double vol, double T) const {
    
    const size_t n = scheme_->getNumChebyshevNodes();
    const size_t m = scheme_->getNumFixedPointIterations();
    const double xmax = xMaxPut(K, r, q);
    
    // Initialize interpolation nodes
    std::vector<double> nodes(n);
    std::vector<double> y(n, 0.0); // Boundary function values at nodes
    
    // Chebyshev nodes of the second kind in [-1, 1]
    for (size_t i = 0; i < n; ++i) {
        nodes[i] = std::cos(M_PI * i / (n - 1)); // x_i in [-1, 1]
    }
    
    // Create initial interpolation
    auto interp = std::make_shared<numerics::ChebyshevInterpolation>(
        nodes, y, numerics::SECOND_KIND, -1.0, 1.0);
    
    // Function to map tau to z in [-1, 1]
    auto tauToZ = [T](double tau) -> double {
        return 2.0 * std::sqrt(tau / T) - 1.0;
    };
    
    // Function to get boundary value at tau
    auto B = [xmax, interp, tauToZ](double tau) -> double {
        if (tau <= 0.0) return xmax;
        
        const double z = tauToZ(tau);
        return xmax * std::exp(-std::sqrt(std::max(0.0, (*interp)(z, true))));
    };
    
    // Function to map boundary value to interpolation y-value
    auto h = [xmax](double fv) -> double {
        return std::pow(std::log(fv / xmax), 2);
    };
    
    // Create fixed point evaluator 
    auto evaluator = createFixedPointEvaluator(K, r, q, vol, B);
    
    // Perform fixed point iterations
    // First is a Jacobi-Newton step
    for (size_t k = 0; k < 1; ++k) {
        for (size_t i = 1; i < n; ++i) { // Skip first node (corresponds to tau=0)
            const double z = nodes[i];
            const double tau = T * std::pow(0.5 * (1.0 + z), 2);
            const double b = B(tau);
            
            const auto [N, D, fv] = evaluator->evaluate(tau, b);
            
            if (tau < 1e-10) {
                y[i] = h(fv);
            } else {
                const auto [Nd, Dd] = evaluator->derivatives(tau, b);
                
                // Newton step
                const double fd = K * std::exp(-(r - q) * tau) * (Nd / D - Dd * N / (D * D));
                const double b_new = b - (fv - b) / (fd - 1.0);
                
                y[i] = h(b_new);
            }
        }
        
        // Update interpolation
        interp->updateValues(y);
    }
    
    // Remaining iterations are standard fixed point
    for (size_t k = 1; k < m; ++k) {
        for (size_t i = 1; i < n; ++i) { // Skip first node (corresponds to tau=0)
            const double z = nodes[i];
            const double tau = T * std::pow(0.5 * (1.0 + z), 2);
            const double b = B(tau);
            
            const auto [N, D, fv] = evaluator->evaluate(tau, b);
            
            y[i] = h(fv);
        }
        
        // Update interpolation
        interp->updateValues(y);
    }
    
    return interp;
}

// Calculate early exercise boundary for calls using Chebyshev interpolation
std::shared_ptr<numerics::ChebyshevInterpolation> ALOEngine::calculateCallExerciseBoundary(
    double S, double K, double r, double q, double vol, double T) const {
    
    const size_t n = scheme_->getNumChebyshevNodes();
    const size_t m = scheme_->getNumFixedPointIterations();
    const double xmax = xMaxCall(K, r, q);
    
    // For infinite boundary (no early exercise), return an empty boundary
    if (std::isinf(xmax) || xmax > 1e12) {
        std::vector<double> nodes(n);
        std::vector<double> y(n, 0.0);
        
        for (size_t i = 0; i < n; ++i) {
            nodes[i] = std::cos(M_PI * i / (n - 1));
        }
        
        return std::make_shared<numerics::ChebyshevInterpolation>(
            nodes, y, numerics::SECOND_KIND, -1.0, 1.0);
    }
    
    // Initialize interpolation nodes
    std::vector<double> nodes(n);
    std::vector<double> y(n, 0.0); // Boundary function values at nodes
    
    // Chebyshev nodes of the second kind in [-1, 1]
    for (size_t i = 0; i < n; ++i) {
        nodes[i] = std::cos(M_PI * i / (n - 1)); // x_i in [-1, 1]
    }
    
    // Create initial interpolation
    auto interp = std::make_shared<numerics::ChebyshevInterpolation>(
        nodes, y, numerics::SECOND_KIND, -1.0, 1.0);
    
    // Function to map tau to z in [-1, 1]
    auto tauToZ = [T](double tau) -> double {
        return 2.0 * std::sqrt(tau / T) - 1.0;
    };
    
    // Function to get boundary value at tau
    auto B = [xmax, interp, tauToZ](double tau) -> double {
        if (tau <= 0.0) return xmax;
        
        const double z = tauToZ(tau);
        return xmax * std::exp(-std::sqrt(std::max(0.0, (*interp)(z, true))));
    };
    
    // Function to map boundary value to interpolation y-value
    auto h = [xmax](double fv) -> double {
        return std::pow(std::log(fv / xmax), 2);
    };
    
    // For calls, we swap r and q in the fixed point equation (put-call symmetry)
    auto evaluator = createFixedPointEvaluator(K, q, r, vol, B);
    
    // Perform fixed point iterations
    // First is a Jacobi-Newton step
    for (size_t k = 0; k < 1; ++k) {
        for (size_t i = 1; i < n; ++i) { // Skip first node (corresponds to tau=0)
            const double z = nodes[i];
            const double tau = T * std::pow(0.5 * (1.0 + z), 2);
            const double b = B(tau);
            
            const auto [N, D, fv] = evaluator->evaluate(tau, b);
            
            if (tau < 1e-10) {
                y[i] = h(fv);
            } else {
                const auto [Nd, Dd] = evaluator->derivatives(tau, b);
                
                // Newton step
                const double fd = K * std::exp(-(q - r) * tau) * (Nd / D - Dd * N / (D * D));
                const double b_new = b - (fv - b) / (fd - 1.0);
                
                y[i] = h(b_new);
            }
        }
        
        // Update interpolation
        interp->updateValues(y);
    }
    
    // Remaining iterations are standard fixed point
    for (size_t k = 1; k < m; ++k) {
        for (size_t i = 1; i < n; ++i) { // Skip first node (corresponds to tau=0)
            const double z = nodes[i];
            const double tau = T * std::pow(0.5 * (1.0 + z), 2);
            const double b = B(tau);
            
            const auto [N, D, fv] = evaluator->evaluate(tau, b);
            
            y[i] = h(fv);
        }
        
        // Update interpolation
        interp->updateValues(y);
    }
    
    return interp;
}

// Calculate early exercise premium for puts
double ALOEngine::calculatePutExercisePremium(
    double S, double K, double r, double q, double vol, double T,
    const std::shared_ptr<numerics::ChebyshevInterpolation>& boundary) const {
    
    const double xmax = xMaxPut(K, r, q);
    
    // Function to map tau to z in [-1, 1]
    auto tauToZ = [T](double tau) -> double {
        return 2.0 * std::sqrt(tau / T) - 1.0;
    };
    
    // Function to get boundary value at tau
    auto B = [xmax, boundary, tauToZ](double tau) -> double {
        if (tau <= 0.0) return xmax;
        
        const double z = tauToZ(tau);
        return xmax * std::exp(-std::sqrt(std::max(0.0, (*boundary)(z, true))));
    };
    
    // Integrand for put early exercise premium
    auto integrand = [S, K, r, q, vol, B](double z) -> double {
        const double t = z * z; // tau = z^2
        const double b_t = B(t);
        
        if (b_t <= 0.0) return 0.0;
        
        const double dr = std::exp(-r * t);
        const double dq = std::exp(-q * t);
        const double v = vol * std::sqrt(t);
        
        if (v < 1e-10) {
            if (std::abs(S * dq - b_t * dr) < 1e-10)
                return z * (r * K * dr - q * S * dq);
            else if (b_t * dr > S * dq)
                return 2.0 * z * (r * K * dr - q * S * dq);
            else
                return 0.0;
        }
        
        const double dp = (std::log(S * dq / (b_t * dr)) / v) + 0.5 * v;
        
        return 2.0 * z * (r * K * dr * (0.5 * (1.0 + std::erf((-dp + v) / std::sqrt(2.0)))) - 
                          q * S * dq * (0.5 * (1.0 + std::erf(-dp / std::sqrt(2.0)))));
    };
    
    // Integrate to get early exercise premium
    return scheme_->getPricingIntegrator()->integrate(integrand, 0.0, std::sqrt(T));
}

// Calculate early exercise premium for calls
double ALOEngine::calculateCallExercisePremium(
    double S, double K, double r, double q, double vol, double T,
    const std::shared_ptr<numerics::ChebyshevInterpolation>& boundary) const {
    
    // If zero dividend, calls should never be exercised early
    if (q <= 0.0) {
        return 0.0;
    }
    
    const double xmax = xMaxCall(K, r, q);
    
    // For infinite boundary (no early exercise), return zero premium
    if (std::isinf(xmax) || xmax > 1e12) {
        return 0.0;
    }
    
    // Function to map tau to z in [-1, 1]
    auto tauToZ = [T](double tau) -> double {
        return 2.0 * std::sqrt(tau / T) - 1.0;
    };
    
    // Function to get boundary value at tau
    auto B = [xmax, boundary, tauToZ](double tau) -> double {
        if (tau <= 0.0) return xmax;
        
        const double z = tauToZ(tau);
        return xmax * std::exp(-std::sqrt(std::max(0.0, (*boundary)(z, true))));
    };
    
    // Integrand for call early exercise premium
    auto integrand = [S, K, r, q, vol, B](double z) -> double {
        const double t = z * z; // tau = z^2
        const double b_t = B(t);
        
        if (b_t <= 0.0 || std::isinf(b_t)) return 0.0;
        
        const double dr = std::exp(-r * t);
        const double dq = std::exp(-q * t);
        const double v = vol * std::sqrt(t);
        
        if (v < 1e-10) {
            if (std::abs(b_t * dq - K * dr) < 1e-10)
                return z * (q * b_t * dq - r * K * dr);
            else if (b_t * dq > K * dr)
                return 2.0 * z * (q * b_t * dq - r * K * dr);
            else
                return 0.0;
        }
        
        const double dp = (std::log(b_t * dq / (K * dr)) / v) + 0.5 * v;
        
        return 2.0 * z * (q * b_t * dq * (0.5 * (1.0 + std::erf(dp / std::sqrt(2.0)))) - 
                          r * K * dr * (0.5 * (1.0 + std::erf((dp - v) / std::sqrt(2.0)))));
    };
    
    // Integrate to get early exercise premium
    return scheme_->getPricingIntegrator()->integrate(integrand, 0.0, std::sqrt(T));
}
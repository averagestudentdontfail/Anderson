#include "aloengine.h"
#include "mod/american.h"
#include "mod/european.h"
#include "num/integrator.h"
#include "num/chebyshev.h"
#include "opt/cache.h"
#include "opt/simd.h"
#include "opt/vector.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <future>
#include <thread>
#include <array>
#include <vector>

namespace engine {
namespace alo {
 
ALOIterationScheme::ALOIterationScheme(size_t n, size_t m, 
                                      std::shared_ptr<num::Integrator> fpIntegrator,
                                      std::shared_ptr<num::Integrator> pricingIntegrator)
     : n_(n), m_(m), 
       fpIntegrator_(fpIntegrator),
       pricingIntegrator_(pricingIntegrator) {
     
     if (n_ < 2) {
         throw std::invalid_argument("ALOIterationScheme: Number of Chebyshev nodes must be at least 2");
     }
     
     if (m_ < 1) {
         throw std::invalid_argument("ALOIterationScheme: Number of fixed point iterations must be at least 1");
     }
     
     if (!fpIntegrator_) {
         throw std::invalid_argument("ALOIterationScheme: Fixed point integrator cannot be null");
     }
     
     if (!pricingIntegrator_) {
         throw std::invalid_argument("ALOIterationScheme: Pricing integrator cannot be null");
     }
}
 
std::string ALOIterationScheme::getDescription() const {
     return "ChebyshevNodes: " + std::to_string(n_) + 
            ", FixedPointIterations: " + std::to_string(m_) +
            ", FPIntegrator: " + fpIntegrator_->name() +
            ", PricingIntegrator: " + pricingIntegrator_->name();
}
 
ALOEngine::ALOEngine(ALOScheme scheme, FixedPointEquation eq) 
     : equation_(eq) {
     // Set up the scheme based on parameter
     setScheme(scheme);
}
 
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
 
void ALOEngine::setFixedPointEquation(FixedPointEquation eq) {
    equation_ = eq;
}
 
std::string ALOEngine::getSchemeDescription() const {
    return scheme_->getDescription();
}
 
std::string ALOEngine::getEquationName() const {
    switch (equation_) {
        case FP_A: return "Equation A";
        case FP_B: return "Equation B";
        case AUTO: return "Auto";
        default: return "Unknown";
    }
}
 
void ALOEngine::clearCache() const {
    // Clear both the legacy cache and the new thread-local cache
    opt::getThreadLocalCache().clear();
    
    // Also clear the legacy string cache for backward compatibility
    legacy_cache_.clear();
}
 
size_t ALOEngine::getCacheSize() const {
    // Return combined size of both caches
    return opt::getThreadLocalCache().size() + legacy_cache_.size();
}
 
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
 
std::shared_ptr<ALOIterationScheme> ALOEngine::createFastScheme() {
    // Legendre-Legendre (7,2,7)-27 scheme
    try {
        auto fpIntegrator = num::createIntegrator("GaussLegendre", 7);
        auto pricingIntegrator = num::createIntegrator("GaussLegendre", 27);
        
        return std::make_shared<ALOIterationScheme>(7, 2, fpIntegrator, pricingIntegrator);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error creating fast scheme: ") + e.what());
    }
}
 
std::shared_ptr<ALOIterationScheme> ALOEngine::createAccurateScheme() {
    // Legendre-TanhSinh (25,5,13)-1e-8 scheme
    try {
        auto fpIntegrator = num::createIntegrator("GaussLegendre", 25);
        auto pricingIntegrator = num::createIntegrator("TanhSinh", 0, 1e-8);
        
        return std::make_shared<ALOIterationScheme>(13, 5, fpIntegrator, pricingIntegrator);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error creating accurate scheme: ") + e.what());
    }
}
 
std::shared_ptr<ALOIterationScheme> ALOEngine::createHighPrecisionScheme() {
    // TanhSinh-TanhSinh (10,30)-1e-10 scheme
    try {
        auto fpIntegrator = num::createIntegrator("TanhSinh", 0, 1e-10);
        auto pricingIntegrator = num::createIntegrator("TanhSinh", 0, 1e-10);
        
        return std::make_shared<ALOIterationScheme>(30, 10, fpIntegrator, pricingIntegrator);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error creating high precision scheme: ") + e.what());
    }
}
 
double ALOEngine::calculateOption(double S, double K, double r, double q, double vol, double T, OptionType type) const {
    if (type == PUT) {
        return calculatePutImpl(S, K, r, q, vol, T);
    } else { // CALL
        return calculateCallImpl(S, K, r, q, vol, T);
    }
}
 
double ALOEngine::calculatePut(double S, double K, double r, double q, double vol, double T) const {
    return calculatePutImpl(S, K, r, q, vol, T);
}
 
double ALOEngine::calculateEarlyExercisePremium(double S, double K, double r, double q, double vol, double T, 
                                             OptionType type) const {
    double american = calculateOption(S, K, r, q, vol, T, type);
    double european = (type == PUT) ? 
        blackScholesPut(S, K, r, q, vol, T) : 
        blackScholesCall(S, K, r, q, vol, T);
    return american - european;
}

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
 
double ALOEngine::xMaxCall(double K, double r, double q) const {
    // For call options, the early exercise boundary is different
    if (q > 0.0 && r >= 0.0)
        return K * std::max(1.0, r / q);
    else if (q <= 0.0 && r >= 0.0)
        return std::numeric_limits<double>::infinity(); // Effectively infinite, early exercise never optimal
    else if (q >= r && r < 0.0)
        return std::numeric_limits<double>::infinity(); // European case
    else if (q < r && r < 0.0)
        return K; // Double boundary case
    else
        throw std::runtime_error("Internal error in xMaxCall calculation");
}
 
double ALOEngine::calculatePutImpl(double S, double K, double r, double q, double vol, double T) const {
    // Check for special cases
    if (K <= 0.0 || S <= 0.0 || vol <= 0.0 || T <= 0.0) {
        return std::max(0.0, K - S);
    }
    
    // Create efficient binary cache key
    opt::OptionParams params{S, K, r, q, vol, T, 0}; // 0 = PUT
    
    // Use thread-local cache with optimized key
    return opt::getCachedPrice(params, [&]() {
        // Handle special cases from the paper
        if (r < 0.0 && q < r) {
            throw std::runtime_error("Double-boundary case q<r<0 for a put option is not implemented");
        }
        
        // Check if we're in the European case
        if (r <= 0.0 && r <= q) {
            return blackScholesPut(S, K, r, q, vol, T);
        }
        
        // Create American put model
        mod::AmericanPut american_put(scheme_->getPricingIntegrator());
        
        // Calculate early exercise boundary
        auto boundary = american_put.calculateExerciseBoundary(
            S, K, r, q, vol, T,
            scheme_->getNumChebyshevNodes(),
            scheme_->getNumFixedPointIterations(),
            scheme_->getFixedPointIntegrator()
        );
        
        // Calculate early exercise premium
        double earlyExercisePremium = american_put.calculateEarlyExercisePremium(
            S, K, r, q, vol, T, boundary);
        
        // Calculate European put value
        mod::EuropeanPut european_put;
        double europeanValue = european_put.calculatePrice(S, K, r, q, vol, T);
        
        // American put value = European put value + early exercise premium
        return std::max(europeanValue, 0.0) + std::max(0.0, earlyExercisePremium);
    });
}
 
double ALOEngine::calculateCallImpl(double S, double K, double r, double q, double vol, double T) const {
    // Check for special cases
    if (K <= 0.0 || S <= 0.0 || vol <= 0.0 || T <= 0.0) {
        return std::max(0.0, S - K);
    }
    
    // Create efficient binary cache key
    opt::OptionParams params{S, K, r, q, vol, T, 1}; // 1 = CALL
    
    // Use thread-local cache with optimized key
    return opt::getCachedPrice(params, [&]() {
        // For calls with no dividends, early exercise is never optimal
        if (q <= 0.0) {
            mod::EuropeanCall european_call;
            return european_call.calculatePrice(S, K, r, q, vol, T);
        }
        
        // Check if we're in the European case
        if (r < 0.0 && q >= r) {
            mod::EuropeanCall european_call;
            return european_call.calculatePrice(S, K, r, q, vol, T);
        }
        
        // Create American call model
        mod::AmericanCall american_call(scheme_->getPricingIntegrator());
        
        // Calculate early exercise boundary
        auto boundary = american_call.calculateExerciseBoundary(
            S, K, r, q, vol, T,
            scheme_->getNumChebyshevNodes(),
            scheme_->getNumFixedPointIterations(),
            scheme_->getFixedPointIntegrator()
        );
        
        // Calculate early exercise premium
        double earlyExercisePremium = american_call.calculateEarlyExercisePremium(
            S, K, r, q, vol, T, boundary);
        
        // Calculate European call value
        mod::EuropeanCall european_call;
        double europeanValue = european_call.calculatePrice(S, K, r, q, vol, T);
        
        // American call value = European call value + early exercise premium
        return std::max(europeanValue, 0.0) + std::max(0.0, earlyExercisePremium);
    });
}

// Helper method for processing a chunk of American put options
void ALOEngine::processAmericanPutChunk(const double* S, const double* K, const double* r,
                                     const double* q, const double* vol, const double* T,
                                     double* results, size_t n) const {
    // Calculate European prices first (vectorized)
    std::vector<double> european_prices(n);
    for (size_t i = 0; i < n; ++i) {
        european_prices[i] = blackScholesPut(S[i], K[i], r[i], q[i], vol[i], T[i]);
    }
    
    // Calculate early exercise premiums
    for (size_t i = 0; i < n; ++i) {
        // Use cached early exercise premium calculation
        opt::OptionParams params{S[i], K[i], r[i], q[i], vol[i], T[i], 0}; // 0 = PUT
        
        results[i] = opt::getCachedPrice(params, [&]() {
            // Skip full calculation if it's a European-only case
            if (r[i] <= 0.0 && r[i] <= q[i]) {
                return european_prices[i];
            }
            
            // Create American put model
            mod::AmericanPut american_put(scheme_->getPricingIntegrator());
            
            // Calculate early exercise boundary
            auto boundary = american_put.calculateExerciseBoundary(
                S[i], K[i], r[i], q[i], vol[i], T[i],
                scheme_->getNumChebyshevNodes(),
                scheme_->getNumFixedPointIterations(),
                scheme_->getFixedPointIntegrator()
            );
            
            // Calculate early exercise premium
            double earlyExercisePremium = american_put.calculateEarlyExercisePremium(
                S[i], K[i], r[i], q[i], vol[i], T[i], boundary);
            
            // American put value = European put value + early exercise premium
            return std::max(european_prices[i], 0.0) + std::max(0.0, earlyExercisePremium);
        });
    }
}
 
std::vector<double> ALOEngine::batchCalculatePut(double S, const std::vector<double>& strikes,
                                              double r, double q, double vol, double T) const {
    const size_t n = strikes.size();
    if (n == 0) return {};
    
    std::vector<double> results(n);
    
    // Check for European-only condition once (r <= 0.0 && r <= q)
    bool use_european = (r <= 0.0 && r <= q);
    
    // Strategy 1: For very small batches, use scalar computation
    if (n <= 4) {
        if (use_european) {
            for (size_t i = 0; i < n; ++i) {
                results[i] = blackScholesPut(S, strikes[i], r, q, vol, T);
            }
        } else {
            for (size_t i = 0; i < n; ++i) {
                results[i] = calculatePutImpl(S, strikes[i], r, q, vol, T);
            }
        }
        return results;
    }
    
    // Strategy 2: For small-to-medium batches, use SIMD
    if (n <= 64) {
        // European options are easier to vectorize efficiently
        if (use_european) {
            constexpr size_t VECTOR_SIZE = 4;
            size_t i = 0;
            
            // Process in groups of 4 using SIMD
            for (; i + VECTOR_SIZE - 1 < n; i += VECTOR_SIZE) {
                std::array<double, VECTOR_SIZE> K_batch = {
                    strikes[i], strikes[i+1], strikes[i+2], strikes[i+3]
                };
                std::array<double, VECTOR_SIZE> S_batch = {S, S, S, S};
                std::array<double, VECTOR_SIZE> r_batch = {r, r, r, r};
                std::array<double, VECTOR_SIZE> q_batch = {q, q, q, q};
                std::array<double, VECTOR_SIZE> vol_batch = {vol, vol, vol, vol};
                std::array<double, VECTOR_SIZE> T_batch = {T, T, T, T};
                std::array<double, VECTOR_SIZE> result_batch;
                
                // Use optimized vectorized Black-Scholes
                opt::VectorMath::bsPut(
                    S_batch.data(), K_batch.data(), r_batch.data(), q_batch.data(),
                    vol_batch.data(), T_batch.data(), result_batch.data(), VECTOR_SIZE
                );
                
                // Copy results back
                for (size_t j = 0; j < VECTOR_SIZE; ++j) {
                    results[i + j] = result_batch[j];
                }
            }
            
            // Handle remaining options
            for (; i < n; ++i) {
                results[i] = blackScholesPut(S, strikes[i], r, q, vol, T);
            }
        } 
        else {
            // For American options, use the calculatePut4 method for groups of 4
            constexpr size_t VECTOR_SIZE = 4;
            size_t i = 0;
            
            // Process in groups of 4
            for (; i + VECTOR_SIZE - 1 < n; i += VECTOR_SIZE) {
                std::array<double, VECTOR_SIZE> strike_batch = {
                    strikes[i], strikes[i+1], strikes[i+2], strikes[i+3]
                };
                
                // Use optimized batch American put calculation
                auto result_batch = calculatePut4(S, strike_batch, r, q, vol, T);
                
                // Copy results back
                for (size_t j = 0; j < VECTOR_SIZE; ++j) {
                    results[i + j] = result_batch[j];
                }
            }
            
            // Handle remaining options individually
            for (; i < n; ++i) {
                results[i] = calculatePutImpl(S, strikes[i], r, q, vol, T);
            }
        }
        
        return results;
    }
    
    // Strategy 3: For larger batches (> 64), use parallelized implementation
    // Process in larger chunks to amortize thread scheduling costs
    if (n > 64) {
        return parallelBatchCalculatePut(S, strikes, r, q, vol, T);
    }
    
    return results; // Should never reach here, but added for safety
}
 
std::vector<double> ALOEngine::batchCalculatePut(double S, 
                                              const std::vector<std::tuple<double, double, double, double, double>>& options) const {
    // Return empty vector for empty input
    if (options.empty()) {
        return {};
    }
    
    std::vector<double> results(options.size());
    const size_t n = options.size();
    
    // For small batches, just use scalar computation
    if (n <= 4) {
        for (size_t i = 0; i < n; ++i) {
            const auto& [K, r, q, vol, T] = options[i];
            results[i] = calculatePutImpl(S, K, r, q, vol, T);
        }
        return results;
    }
    
    // For larger batches, set up chunked processing
    constexpr size_t CHUNK_SIZE = 64;
    
    // Prepare parameter arrays for vectorized processing
    std::vector<double> S_vec(CHUNK_SIZE, S);
    std::vector<double> K_vec(CHUNK_SIZE);
    std::vector<double> r_vec(CHUNK_SIZE);
    std::vector<double> q_vec(CHUNK_SIZE);
    std::vector<double> vol_vec(CHUNK_SIZE);
    std::vector<double> T_vec(CHUNK_SIZE);
    std::vector<double> chunk_results(CHUNK_SIZE);
    
    // Process in chunks
    for (size_t start = 0; start < n; start += CHUNK_SIZE) {
        const size_t end = std::min(start + CHUNK_SIZE, n);
        const size_t chunk_size = end - start;
        
        // For small remaining chunks, use scalar processing
        if (chunk_size < 8) {
            for (size_t i = start; i < end; ++i) {
                const auto& [K, r, q, vol, T] = options[i];
                results[i] = calculatePutImpl(S, K, r, q, vol, T);
            }
            continue;
        }
        
        // Extract parameters to contiguous buffers
        for (size_t i = 0; i < chunk_size; ++i) {
            const auto& [K, r, q, vol, T] = options[i + start];
            K_vec[i] = K;
            r_vec[i] = r;
            q_vec[i] = q;
            vol_vec[i] = vol;
            T_vec[i] = T;
        }
        
        // Use vectorized processing for this chunk
        processAmericanPutChunk(S_vec.data(), K_vec.data(), r_vec.data(), q_vec.data(), 
                               vol_vec.data(), T_vec.data(), chunk_results.data(), chunk_size);
        
        // Copy results back
        std::copy(chunk_results.begin(), chunk_results.begin() + chunk_size, 
                 results.begin() + start);
    }
    
    return results;
}
 
std::array<double, 4> ALOEngine::calculatePut4(
    const std::array<double, 4>& spots,
    const std::array<double, 4>& strikes,
    const std::array<double, 4>& rs,
    const std::array<double, 4>& qs,
    const std::array<double, 4>& vols,
    const std::array<double, 4>& Ts) const {
    
    // Initialize result array
    std::array<double, 4> results = {0.0, 0.0, 0.0, 0.0};
    
    // Process each option efficiently
    for (size_t i = 0; i < 4; ++i) {
        // Check for degenerate cases
        if (strikes[i] <= 0.0 || spots[i] <= 0.0 || vols[i] <= 0.0 || Ts[i] <= 0.0) {
            results[i] = std::max(0.0, strikes[i] - spots[i]);
            continue;
        }
        
        // Check if this particular option is European-only
        if (rs[i] <= 0.0 && rs[i] <= qs[i]) {
            results[i] = blackScholesPut(spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i]);
        } else {
            // Cache key
            opt::OptionParams params{spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i], 0};
            
            // American option calculation with caching
            results[i] = opt::getCachedPrice(params, [&, i]() {
                return calculatePutImpl(spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i]);
            });
        }
    }
    
    return results;
}
 
std::array<double, 4> ALOEngine::calculatePut4(
    double S,
    const std::array<double, 4>& strikes,
    double r, double q, double vol, double T) const {
    
    // Initialize arrays with the same values for all except strikes
    std::array<double, 4> spots = {S, S, S, S};
    std::array<double, 4> rs = {r, r, r, r};
    std::array<double, 4> qs = {q, q, q, q};
    std::array<double, 4> vols = {vol, vol, vol, vol};
    std::array<double, 4> Ts = {T, T, T, T};
    
    // Check for European-only condition once
    if (r <= 0.0 && r <= q) {
        // Use vectorized Black-Scholes for European options
        std::array<double, 4> results;
        opt::VectorMath::bsPut(
            spots.data(), strikes.data(), rs.data(), qs.data(), 
            vols.data(), Ts.data(), results.data(), 4
        );
        return results;
    }
    
    // Use the full implementation for American options
    return calculatePut4(spots, strikes, rs, qs, vols, Ts);
}
 
std::vector<double> ALOEngine::batchCalculateCall(double S, const std::vector<double>& strikes,
                                               double r, double q, double vol, double T) const {
    // Return empty vector for empty input
    if (strikes.empty()) {
        return {};
    }
    
    std::vector<double> results(strikes.size());
    const size_t n = strikes.size();
    
    // Check for European-only condition (q <= 0.0 or r < 0.0 && q >= r)
    bool use_european = (q <= 0.0 || (r < 0.0 && q >= r));
    
    if (use_european) {
        // Use direct Black-Scholes for European options
        for (size_t i = 0; i < n; ++i) {
            results[i] = blackScholesCall(S, strikes[i], r, q, vol, T);
        }
    } else {
        // For all options, use scalar computation with caching
        for (size_t i = 0; i < n; ++i) {
            results[i] = calculateCallImpl(S, strikes[i], r, q, vol, T);
        }
    }
    
    return results;
}
 
std::vector<double> ALOEngine::parallelBatchCalculatePut(double S, const std::vector<double>& strikes,
                                                     double r, double q, double vol, double T) const {
    const size_t n = strikes.size();
    if (n == 0) return {};
    
    // For small batches, use the regular calculation to avoid thread overhead
    if (n <= 64) {
        return batchCalculatePut(S, strikes, r, q, vol, T);
    }
    
    // Allocate result space
    std::vector<double> results(n);
    
    // Determine European-only condition once
    bool use_european = (r <= 0.0 && r <= q);
    
    // Get thread pool
    auto& pool = get_worker_pool();
    const unsigned int num_cores = std::thread::hardware_concurrency();
    
    // Calculate chunk size to minimize scheduling overhead
    // Using a hybrid approach: more chunks than cores for better load balancing
    const unsigned int num_chunks = std::min(static_cast<unsigned int>(n), num_cores * 2);
    const size_t chunk_size = (n + num_chunks - 1) / num_chunks; // ceiling division
    
    // Submit tasks to thread pool with deterministic behavior
    std::atomic<size_t> completed_chunks(0);
    const size_t total_chunks = (n + chunk_size - 1) / chunk_size;
    
    for (size_t start = 0; start < n; start += chunk_size) {
        const size_t end = std::min(start + chunk_size, n);
        
        pool.enqueue([this, &results, &completed_chunks, S, &strikes, r, q, vol, T, use_european, start, end]() {
            // Process this chunk with bounded execution time
            if (use_european) {
                // Fast path for European options - use vectorized Black-Scholes
                constexpr size_t VECTOR_SIZE = 4;
                size_t i = start;
                
                // Process in groups of 4 using SIMD
                for (; i + VECTOR_SIZE - 1 < end; i += VECTOR_SIZE) {
                    std::array<double, VECTOR_SIZE> K_batch = {
                        strikes[i], strikes[i+1], strikes[i+2], strikes[i+3]
                    };
                    std::array<double, VECTOR_SIZE> S_batch = {S, S, S, S};
                    std::array<double, VECTOR_SIZE> r_batch = {r, r, r, r};
                    std::array<double, VECTOR_SIZE> q_batch = {q, q, q, q};
                    std::array<double, VECTOR_SIZE> vol_batch = {vol, vol, vol, vol};
                    std::array<double, VECTOR_SIZE> T_batch = {T, T, T, T};
                    std::array<double, VECTOR_SIZE> result_batch;
                    
                    // Use optimized vectorized Black-Scholes
                    opt::VectorMath::bsPut(
                        S_batch.data(), K_batch.data(), r_batch.data(), q_batch.data(),
                        vol_batch.data(), T_batch.data(), result_batch.data(), VECTOR_SIZE
                    );
                    
                    // Copy results back
                    for (size_t j = 0; j < VECTOR_SIZE; ++j) {
                        results[i + j] = result_batch[j];
                    }
                }
                
                // Handle remaining options individually
                for (; i < end; ++i) {
                    results[i] = blackScholesPut(S, strikes[i], r, q, vol, T);
                }
            } else {
                // For American options, again process in batches of 4 for SIMD efficiency
                constexpr size_t VECTOR_SIZE = 4;
                size_t i = start;
                
                // Process in groups of 4
                for (; i + VECTOR_SIZE - 1 < end; i += VECTOR_SIZE) {
                    std::array<double, VECTOR_SIZE> strike_batch = {
                        strikes[i], strikes[i+1], strikes[i+2], strikes[i+3]
                    };
                    
                    // Use optimized batch American put calculation
                    auto result_batch = calculatePut4(S, strike_batch, r, q, vol, T);
                    
                    // Copy results back
                    for (size_t j = 0; j < VECTOR_SIZE; ++j) {
                        results[i + j] = result_batch[j];
                    }
                }
                
                // Handle remaining options individually
                for (; i < end; ++i) {
                    results[i] = calculatePutImpl(S, strikes[i], r, q, vol, T);
                }
            }
            
            // Increment completed chunks counter
            completed_chunks.fetch_add(1);
        });
    }
    
    // Wait for all tasks to complete
    pool.wait_all();
    
    return results;
}
  
std::shared_ptr<ALOEngine::FixedPointEvaluator> ALOEngine::createFixedPointEvaluator(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B) const {
    
    // Choose equation type
    FixedPointEquation eq = equation_;
    if (eq == AUTO) {
        // Automatically select equation based on r and q
        eq = (std::abs(r - q) < 0.001) ? FP_A : FP_B;
    }
    
    // Create appropriate evaluator
    if (eq == FP_A) {
        return std::make_shared<EquationA>(K, r, q, vol, B, scheme_->getFixedPointIntegrator());
    } else {
        return std::make_shared<EquationB>(K, r, q, vol, B, scheme_->getFixedPointIntegrator());
    }
}
 
ALOEngine::FixedPointEvaluator::FixedPointEvaluator(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B,
    std::shared_ptr<num::Integrator> integrator)
    : K_(K), r_(r), q_(q), vol_(vol), vol2_(vol * vol), 
      B_(B), integrator_(integrator) {}
 
std::pair<double, double> ALOEngine::FixedPointEvaluator::d(double t, double z) const {
    if (t <= 0.0 || z <= 0.0) {
        return {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    }
    
    const double v = vol_ * std::sqrt(t);
    const double m = (std::log(z) + (r_ - q_) * t) / v + 0.5 * v;
    
    return {m, m - v};
}
 
double ALOEngine::FixedPointEvaluator::normalCDF(double x) const {
    // Using standard error function implementation
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}
 
double ALOEngine::FixedPointEvaluator::normalPDF(double x) const {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}
  
ALOEngine::EquationA::EquationA(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B,
    std::shared_ptr<num::Integrator> integrator)
    : FixedPointEvaluator(K, r, q, vol, B, integrator) {}
 
std::tuple<double, double, double> ALOEngine::EquationA::evaluate(double tau, double b) const {
    double N, D;
    const double eps = 1e-10;
    
    if (tau < eps * eps) {
        // For very small tau, use limit values
        if (std::abs(b - K_) < eps) {
            N = D = 0.5;
        } else {
            N = D = (b > K_) ? 1.0 : 0.0;
        }
    } else {
        const double stv = std::sqrt(tau) / vol_;
        
        // Integrate to get K1 + K2
        auto K12_integrand = [&](double y) -> double {
            const double m = 0.25 * tau * (1.0 + y) * (1.0 + y);
            const double df = std::exp(q_ * tau - q_ * m);
            
            if (y <= 5.0 * eps - 1.0) {
                if (std::abs(b - B_(tau - m)) < eps) {
                    return df * stv / (std::sqrt(2.0 * M_PI));
                } else {
                    return 0.0;
                }
            } else {
                const auto dp = d(m, b / B_(tau - m)).first;
                return df * (0.5 * tau * (y + 1.0) * normalCDF(dp) + stv * normalPDF(dp));
            }
        };
        
        // Integrate to get K3
        auto K3_integrand = [&](double y) -> double {
            const double m = 0.25 * tau * (1.0 + y) * (1.0 + y);
            const double df = std::exp(r_ * tau - r_ * m);
            
            if (y <= 5.0 * eps - 1.0) {
                if (std::abs(b - B_(tau - m)) < eps) {
                    return df * stv / (std::sqrt(2.0 * M_PI));
                } else {
                    return 0.0;
                }
            } else {
                return df * stv * normalPDF(d(m, b / B_(tau - m)).second);
            }
        };
        
        double K12 = integrator_->integrate(K12_integrand, -1.0, 1.0);
        double K3 = integrator_->integrate(K3_integrand, -1.0, 1.0);
        
        const auto dpm = d(tau, b / K_);
        N = normalPDF(dpm.second) / vol_ / std::sqrt(tau) + r_ * K3;
        D = normalPDF(dpm.first) / vol_ / std::sqrt(tau) + normalCDF(dpm.first) + q_ * K12;
    }
    
    // Calculate function value
    const double alpha = K_ * std::exp(-(r_ - q_) * tau);
    double fv;
    
    if (tau < eps * eps) {
        if (std::abs(b - K_) < eps || b > K_) {
            fv = alpha;
        } else {
            if (std::abs(q_) < eps) {
                fv = alpha * r_ * ((q_ < 0.0) ? -1.0 : 1.0) / eps;
            } else {
                fv = alpha * r_ / q_;
            }
        }
    } else {
        fv = alpha * N / D;
    }
    
    return {N, D, fv};
}
 
std::pair<double, double> ALOEngine::EquationA::derivatives(double tau, double b) const {
    double Nd, Dd;
    const double eps = 1e-10;
    
    if (tau < eps * eps) {
        if (std::abs(b - K_) < eps) {
            const double sqTau = std::sqrt(tau);
            Dd = M_2_SQRTPI * M_SQRT1_2 * (
                -(0.5 * vol2_ + r_ - q_) / (b * vol_ * vol2_ * sqTau) + 1.0 / (b * vol_ * sqTau));
            Nd = M_2_SQRTPI * M_SQRT1_2 * (-0.5 * vol2_ + r_ - q_) / (b * vol_ * vol2_ * sqTau);
        } else {
            Dd = Nd = 0.0;
        }
    } else {
        const auto dpm = d(tau, b / K_);
        
        Dd = -normalPDF(dpm.first) * dpm.first / (b * vol2_ * tau) +
             normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau));
        Nd = -normalPDF(dpm.second) * dpm.second / (b * vol2_ * tau);
    }
    
    return {Nd, Dd};
}

ALOEngine::EquationB::EquationB(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B,
    std::shared_ptr<num::Integrator> integrator)
    : FixedPointEvaluator(K, r, q, vol, B, integrator) {}
 
std::tuple<double, double, double> ALOEngine::EquationB::evaluate(double tau, double b) const {
    double N, D;
    const double eps = 1e-10;
    
    if (tau < eps * eps) {
        // For very small tau, use limit values
        if (std::abs(b - K_) < eps) {
            N = D = 0.5;
        } else if (b < K_) {
            N = D = 0.0;
        } else {
            N = D = 1.0;
        }
    } else {
        // Integrate for N and D
        auto N_integrand = [&](double u) -> double {
            const double df = std::exp(r_ * u);
            if (u >= tau * (1.0 - 5.0 * eps)) {
                if (std::abs(b - B_(u)) < eps) {
                    return 0.5 * df;
                } else {
                    return df * ((b < B_(u)) ? 0.0 : 1.0);
                }
            } else {
                return df * normalCDF(d(tau - u, b / B_(u)).second);
            }
        };
        
        auto D_integrand = [&](double u) -> double {
            const double df = std::exp(q_ * u);
            if (u >= tau * (1.0 - 5.0 * eps)) {
                if (std::abs(b - B_(u)) < eps) {
                    return 0.5 * df;
                } else {
                    return df * ((b < B_(u)) ? 0.0 : 1.0);
                }
            } else {
                return df * normalCDF(d(tau - u, b / B_(u)).first);
            }
        };
        
        double ni = integrator_->integrate(N_integrand, 0.0, tau);
        double di = integrator_->integrate(D_integrand, 0.0, tau);
        
        const auto dpm = d(tau, b / K_);
        
        N = normalCDF(dpm.second) + r_ * ni;
        D = normalCDF(dpm.first) + q_ * di;
    }
    
    // Calculate function value
    const double alpha = K_ * std::exp(-(r_ - q_) * tau);
    double fv;
    
    if (tau < eps * eps) {
        if (std::abs(b - K_) < eps || b > K_) {
            fv = alpha;
        } else {
            if (std::abs(q_) < eps) {
                fv = alpha * r_ * ((q_ < 0.0) ? -1.0 : 1.0) / eps;
            } else {
                fv = alpha * r_ / q_;
            }
        }
    } else {
        fv = alpha * N / D;
    }
    
    return {N, D, fv};
}
 
std::pair<double, double> ALOEngine::EquationB::derivatives(double tau, double b) const {
    const double eps = 1e-10;
    
    if (tau < eps * eps) {
        return {0.0, 0.0};
    }
    
    const auto dpm = d(tau, b / K_);
    
    return {
        normalPDF(dpm.second) / (b * vol_ * std::sqrt(tau)),
        normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau))
    };
}

} // namespace alo
} // namespace engine
#ifndef PRICE_SIMD_H
#define PRICE_SIMD_H

#include "../../common/simd/simdops.h"
#include "../../alo/aloengine.h"
#include <array>
#include <vector>
#include <immintrin.h>

/**
 * @class PriceSimd
 * @brief SIMD-accelerated option pricing calculations
 * 
 * This class provides SIMD-optimized implementations of common
 * option pricing calculations using AVX2 instructions.
 */
class PriceSimd {
public:
    /**
     * @brief Constructor
     * 
     * @param engine ALO pricing engine
     */
    explicit PriceSimd(std::shared_ptr<ALOEngine> engine)
        : engine_(engine) {}
    
    /**
     * @brief Calculate Black-Scholes put prices for 4 options at once using AVX2
     * 
     * @param spots Array of 4 spot prices
     * @param strikes Array of 4 strike prices
     * @param rates Array of 4 risk-free rates
     * @param dividends Array of 4 dividend yields
     * @param vols Array of 4 volatilities
     * @param times Array of 4 times to maturity
     * @return Array of 4 put option prices
     */
    std::array<double, 4> blackScholesPut4(
        const std::array<double, 4>& spots,
        const std::array<double, 4>& strikes,
        const std::array<double, 4>& rates,
        const std::array<double, 4>& dividends,
        const std::array<double, 4>& vols,
        const std::array<double, 4>& times) const {
        
        // Load values into SIMD vectors
        __m256d S = simd::SimdOps::load(spots);
        __m256d K = simd::SimdOps::load(strikes);
        __m256d r = simd::SimdOps::load(rates);
        __m256d q = simd::SimdOps::load(dividends);
        __m256d vol = simd::SimdOps::load(vols);
        __m256d T = simd::SimdOps::load(times);
        
        // Calculate Black-Scholes put prices
        __m256d result = simd::SimdOps::bsPut(S, K, r, q, vol, T);
        
        // Store result in array
        std::array<double, 4> prices;
        simd::SimdOps::store(prices, result);
        
        return prices;
    }
    
    /**
     * @brief Calculate Black-Scholes call prices for 4 options at once using AVX2
     * 
     * @param spots Array of 4 spot prices
     * @param strikes Array of 4 strike prices
     * @param rates Array of 4 risk-free rates
     * @param dividends Array of 4 dividend yields
     * @param vols Array of 4 volatilities
     * @param times Array of 4 times to maturity
     * @return Array of 4 call option prices
     */
    std::array<double, 4> blackScholesCall4(
        const std::array<double, 4>& spots,
        const std::array<double, 4>& strikes,
        const std::array<double, 4>& rates,
        const std::array<double, 4>& dividends,
        const std::array<double, 4>& vols,
        const std::array<double, 4>& times) const {
        
        // Use put-call parity to calculate call prices
        // C = P + S * e^(-q*T) - K * e^(-r*T)
        
        // First calculate put prices
        auto put_prices = blackScholesPut4(spots, strikes, rates, dividends, vols, times);
        
        // Load values into SIMD vectors
        __m256d S = simd::SimdOps::load(spots);
        __m256d K = simd::SimdOps::load(strikes);
        __m256d r = simd::SimdOps::load(rates);
        __m256d q = simd::SimdOps::load(dividends);
        __m256d T = simd::SimdOps::load(times);
        __m256d P = simd::SimdOps::load(put_prices);
        
        // Calculate discount factors
        __m256d neg_q_T = simd::SimdOps::mul(
            simd::SimdOps::mul(q, T),
            simd::SimdOps::set1(-1.0)
        );
        __m256d neg_r_T = simd::SimdOps::mul(
            simd::SimdOps::mul(r, T),
            simd::SimdOps::set1(-1.0)
        );
        
        __m256d discount_q = simd::SimdOps::exp(neg_q_T);
        __m256d discount_r = simd::SimdOps::exp(neg_r_T);
        
        // Calculate terms in put-call parity
        __m256d term1 = simd::SimdOps::mul(S, discount_q);
        __m256d term2 = simd::SimdOps::mul(K, discount_r);
        
        // C = P + S * e^(-q*T) - K * e^(-r*T)
        __m256d result = simd::SimdOps::add(
            P, 
            simd::SimdOps::sub(term1, term2)
        );
        
        // Store result in array
        std::array<double, 4> call_prices;
        simd::SimdOps::store(call_prices, result);
        
        return call_prices;
    }
    
    /**
     * @brief Calculate option Greeks for 4 options at once using AVX2
     * 
     * @param spots Array of 4 spot prices
     * @param strikes Array of 4 strike prices
     * @param rates Array of 4 risk-free rates
     * @param dividends Array of 4 dividend yields
     * @param vols Array of 4 volatilities
     * @param times Array of 4 times to maturity
     * @param is_put True for puts, false for calls
     * @return Array of vectors containing [price, delta, gamma, vega, theta, rho]
     */
    std::array<std::array<double, 6>, 4> calculateGreeks4(
        const std::array<double, 4>& spots,
        const std::array<double, 4>& strikes,
        const std::array<double, 4>& rates,
        const std::array<double, 4>& dividends,
        const std::array<double, 4>& vols,
        const std::array<double, 4>& times,
        bool is_put) const {
        
        // Define step sizes for finite differences
        const double h_S_factor = 0.001;  // 0.1% of spot
        const double h_vol_factor = 0.01;  // 1% of vol
        const double h_r = 0.0001;  // 1 bp
        const double h_T_factor = 0.01;  // 1% of time
        
        // Calculate step sizes for each option
        std::array<double, 4> h_S, h_vol, h_T;
        for (int i = 0; i < 4; ++i) {
            h_S[i] = std::max(0.01, spots[i] * h_S_factor);
            h_vol[i] = std::max(0.001, vols[i] * h_vol_factor);
            h_T[i] = std::max(0.0001, times[i] * h_T_factor);
        }
        
        // Calculate base prices
        std::array<double, 4> prices;
        if (is_put) {
            prices = blackScholesPut4(spots, strikes, rates, dividends, vols, times);
        } else {
            prices = blackScholesCall4(spots, strikes, rates, dividends, vols, times);
        }
        
        // Calculate up/down spot prices
        std::array<double, 4> spots_up, spots_down;
        for (int i = 0; i < 4; ++i) {
            spots_up[i] = spots[i] + h_S[i];
            spots_down[i] = spots[i] - h_S[i];
        }
        
        std::array<double, 4> prices_up_S, prices_down_S;
        if (is_put) {
            prices_up_S = blackScholesPut4(spots_up, strikes, rates, dividends, vols, times);
            prices_down_S = blackScholesPut4(spots_down, strikes, rates, dividends, vols, times);
        } else {
            prices_up_S = blackScholesCall4(spots_up, strikes, rates, dividends, vols, times);
            prices_down_S = blackScholesCall4(spots_down, strikes, rates, dividends, vols, times);
        }
        
        // Calculate up/down volatility prices
        std::array<double, 4> vols_up, vols_down;
        for (int i = 0; i < 4; ++i) {
            vols_up[i] = vols[i] + h_vol[i];
            vols_down[i] = vols[i] - h_vol[i];
        }
        
        std::array<double, 4> prices_up_vol, prices_down_vol;
        if (is_put) {
            prices_up_vol = blackScholesPut4(spots, strikes, rates, dividends, vols_up, times);
            prices_down_vol = blackScholesPut4(spots, strikes, rates, dividends, vols_down, times);
        } else {
            prices_up_vol = blackScholesCall4(spots, strikes, rates, dividends, vols_up, times);
            prices_down_vol = blackScholesCall4(spots, strikes, rates, dividends, vols_down, times);
        }
        
        // Calculate down time prices
        std::array<double, 4> times_down;
        for (int i = 0; i < 4; ++i) {
            times_down[i] = times[i] - h_T[i];
        }
        
        std::array<double, 4> prices_down_T;
        if (is_put) {
            prices_down_T = blackScholesPut4(spots, strikes, rates, dividends, vols, times_down);
        } else {
            prices_down_T = blackScholesCall4(spots, strikes, rates, dividends, vols, times_down);
        }
        
        // Calculate up/down rate prices
        std::array<double, 4> rates_up, rates_down;
        for (int i = 0; i < 4; ++i) {
            rates_up[i] = rates[i] + h_r;
            rates_down[i] = rates[i] - h_r;
        }
        
        std::array<double, 4> prices_up_r, prices_down_r;
        if (is_put) {
            prices_up_r = blackScholesPut4(spots, strikes, rates_up, dividends, vols, times);
            prices_down_r = blackScholesPut4(spots, strikes, rates_down, dividends, vols, times);
        } else {
            prices_up_r = blackScholesCall4(spots, strikes, rates_up, dividends, vols, times);
            prices_down_r = blackScholesCall4(spots, strikes, rates_down, dividends, vols, times);
        }
        
        // Calculate Greeks
        std::array<std::array<double, 6>, 4> results;
        for (int i = 0; i < 4; ++i) {
            // price
            results[i][0] = prices[i];
            
            // delta: dV/dS
            results[i][1] = (prices_up_S[i] - prices_down_S[i]) / (2 * h_S[i]);
            
            // gamma: d²V/dS²
            results[i][2] = (prices_up_S[i] - 2 * prices[i] + prices_down_S[i]) / (h_S[i] * h_S[i]);
            
            // vega: dV/dσ (scaled to 1% move)
            results[i][3] = (prices_up_vol[i] - prices_down_vol[i]) / (2 * h_vol[i]) * 0.01;
            
            // theta: -dV/dT (daily)
            results[i][4] = -(prices[i] - prices_down_T[i]) / h_T[i] / 365.0;
            
            // rho: dV/dr (scaled to 1% move)
            results[i][5] = (prices_up_r[i] - prices_down_r[i]) / (2 * h_r) * 0.01;
        }
        
        return results;
    }
    
    /**
     * @brief Calculate implied volatilities for 4 options at once using AVX2
     * 
     * Uses Newton-Raphson algorithm with SIMD acceleration
     * 
     * @param spots Array of 4 spot prices
     * @param strikes Array of 4 strike prices
     * @param rates Array of 4 risk-free rates
     * @param dividends Array of 4 dividend yields
     * @param market_prices Array of 4 market prices
     * @param times Array of 4 times to maturity
     * @param is_put True for puts, false for calls
     * @param max_iter Maximum number of iterations (default: 100)
     * @param tolerance Convergence tolerance (default: 1e-8)
     * @return Array of 4 implied volatilities
     */
    std::array<double, 4> impliedVolatility4(
        const std::array<double, 4>& spots,
        const std::array<double, 4>& strikes,
        const std::array<double, 4>& rates,
        const std::array<double, 4>& dividends,
        const std::array<double, 4>& market_prices,
        const std::array<double, 4>& times,
        bool is_put,
        int max_iter = 100,
        double tolerance = 1e-8) const {
        
        // Initial guess for volatility (use 25% for all options)
        std::array<double, 4> vols = {0.25, 0.25, 0.25, 0.25};
        
        // Load values into SIMD vectors
        __m256d S = simd::SimdOps::load(spots);
        __m256d K = simd::SimdOps::load(strikes);
        __m256d r = simd::SimdOps::load(rates);
        __m256d q = simd::SimdOps::load(dividends);
        __m256d T = simd::SimdOps::load(times);
        __m256d market = simd::SimdOps::load(market_prices);
        __m256d vol = simd::SimdOps::load(vols);
        
        // Constants
        __m256d zero = simd::SimdOps::set1(0.0);
        __m256d one = simd::SimdOps::set1(1.0);
        __m256d half = simd::SimdOps::set1(0.5);
        __m256d tol = simd::SimdOps::set1(tolerance);
        
        // Mask to track which options have converged
        __m256d converged = simd::SimdOps::set1(0.0);
        
        // Small step for vega calculation
        __m256d h_vol = simd::SimdOps::set1(0.001);
        
        // Newton-Raphson iteration
        for (int iter = 0; iter < max_iter; ++iter) {
            // Calculate option prices with current vol
            __m256d price;
            if (is_put) {
                price = simd::SimdOps::bsPut(S, K, r, q, vol, T);
            } else {
                // For calls, we'd implement bsCall or use put-call parity
                // For simplicity, fall back to scalar calculation for now
                std::array<double, 4> vol_array;
                simd::SimdOps::store(vol_array, vol);
                
                std::array<double, 4> price_array;
                for (int i = 0; i < 4; ++i) {
                    price_array[i] = ALOEngine::blackScholesCall(
                        spots[i], strikes[i], rates[i], dividends[i], vol_array[i], times[i]);
                }
                
                price = simd::SimdOps::load(price_array);
            }
            
            // Calculate price difference
            __m256d diff = simd::SimdOps::sub(price, market);
            
            // Check for convergence
            __m256d abs_diff = _mm256_andnot_pd(_mm256_set1_pd(-0.0), diff);  // Absolute value
            __m256d mask = _mm256_cmp_pd(abs_diff, tol, _CMP_LT_OQ);  // mask where |diff| < tol
            
            // Update converged mask
            converged = _mm256_or_pd(converged, mask);
            
            // If all have converged, we're done
            if (_mm256_movemask_pd(converged) == 0xF) {
                break;
            }
            
            // Calculate vega for Newton step
            // Vega = dPrice/dVol
            __m256d vol_up = simd::SimdOps::add(vol, h_vol);
            
            __m256d price_up;
            if (is_put) {
                price_up = simd::SimdOps::bsPut(S, K, r, q, vol_up, T);
            } else {
                // For calls, fall back to scalar calculation for now
                std::array<double, 4> vol_up_array;
                simd::SimdOps::store(vol_up_array, vol_up);
                
                std::array<double, 4> price_up_array;
                for (int i = 0; i < 4; ++i) {
                    price_up_array[i] = ALOEngine::blackScholesCall(
                        spots[i], strikes[i], rates[i], dividends[i], vol_up_array[i], times[i]);
                }
                
                price_up = simd::SimdOps::load(price_up_array);
            }
            
            __m256d vega = simd::SimdOps::div(
                simd::SimdOps::sub(price_up, price),
                h_vol
            );
            
            // Newton step: vol = vol - diff / vega
            __m256d step = simd::SimdOps::div(diff, vega);
            
            // Only update non-converged values
            __m256d new_vol = simd::SimdOps::sub(vol, step);
            
            // Ensure vol stays positive and reasonable (between 0.001 and 10.0)
            __m256d min_vol = simd::SimdOps::set1(0.001);
            __m256d max_vol = simd::SimdOps::set1(10.0);
            new_vol = simd::SimdOps::max(new_vol, min_vol);
            new_vol = simd::SimdOps::min(new_vol, max_vol);
            
            // Update vol where not converged
            vol = _mm256_blendv_pd(new_vol, vol, converged);
        }
        
        // Store result
        std::array<double, 4> result;
        simd::SimdOps::store(result, vol);
        
        return result;
    }
    
    /**
     * @brief Calculate American option prices for 4 options at once with SIMD where possible
     * 
     * @param spots Array of 4 spot prices
     * @param strikes Array of 4 strike prices
     * @param rates Array of 4 risk-free rates
     * @param dividends Array of 4 dividend yields
     * @param vols Array of 4 volatilities
     * @param times Array of 4 times to maturity
     * @param is_put True for puts, false for calls
     * @return Array of 4 option prices
     */
    std::array<double, 4> americanOption4(
        const std::array<double, 4>& spots,
        const std::array<double, 4>& strikes,
        const std::array<double, 4>& rates,
        const std::array<double, 4>& dividends,
        const std::array<double, 4>& vols,
        const std::array<double, 4>& times,
        bool is_put) const {
        
        return engine_->calculatePut4(
            spots, strikes, rates, dividends, vols, times);
    }
    
    /**
     * @brief Calculate American option prices with the same parameters except strikes
     * 
     * @param spot Spot price
     * @param strikes Array of 4 strike prices
     * @param rate Risk-free rate
     * @param dividend Dividend yield
     * @param vol Volatility
     * @param time Time to maturity
     * @param is_put True for puts, false for calls
     * @return Array of 4 option prices
     */
    std::array<double, 4> americanOption4(
        double spot,
        const std::array<double, 4>& strikes,
        double rate, double dividend, double vol, double time,
        bool is_put) const {
        
        if (is_put) {
            return engine_->calculatePut4(
                spot, strikes, rate, dividend, vol, time);
        } else {
            // For calls, create arrays with the same value for all except strikes
            std::array<double, 4> spots = {spot, spot, spot, spot};
            std::array<double, 4> rates = {rate, rate, rate, rate};
            std::array<double, 4> dividends = {dividend, dividend, dividend, dividend};
            std::array<double, 4> vols = {vol, vol, vol, vol};
            std::array<double, 4> times = {time, time, time, time};
            
            return americanOption4(
                spots, strikes, rates, dividends, vols, times, false);
        }
    }
    
private:
    std::shared_ptr<ALOEngine> engine_;
};

#endif 
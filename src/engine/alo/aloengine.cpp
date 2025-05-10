#include "aloengine.h"
#include "mod/american.h" 
#include "mod/european.h" 
#include "num/integrate.h"
#include "num/chebyshev.h"
#include "opt/cache.h"      
#include "opt/simd.h"       
#include "opt/vector.h"     
#include "num/float.h"      

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <array>
#include <vector>
#include <cstring> 
#include <iostream> 
#include <limits> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif

namespace engine {
namespace alo {

namespace {
    const double DOUBLE_EPS = 1e-12; // Epsilon for double comparisons
    const float  FLOAT_EPS  = 1e-7f; // Epsilon for float comparisons
}
 
ALOIterationScheme::ALOIterationScheme(size_t n_nodes, size_t m_iterations, 
                                      std::shared_ptr<num::IntegrateDouble> fp_integrate,
                                      std::shared_ptr<num::IntegrateDouble> pricing_integrate)
     : n_nodes_(n_nodes), m_iterations_(m_iterations), 
       fp_integrate_(std::move(fp_integrate)),
       pricing_integrate_(std::move(pricing_integrate)) {
     
     if (n_nodes_ < 2) {
         throw std::invalid_argument("ALOIterationScheme: Number of Chebyshev nodes must be at least 2");
     }
     if (m_iterations_ < 1) {
         throw std::invalid_argument("ALOIterationScheme: Number of fixed point iterations must be at least 1");
     }
     if (!fp_integrate_) {
         throw std::invalid_argument("ALOIterationScheme: Fixed point integrator cannot be null");
     }
     if (!pricing_integrate_) {
         throw std::invalid_argument("ALOIterationScheme: Pricing integrator cannot be null");
     }
}
 
std::string ALOIterationScheme::getDescription() const {
     return "ChebyshevNodes: " + std::to_string(n_nodes_) + 
            ", FixedPointIterations: " + std::to_string(m_iterations_) +
            ", FPIntegrate: " + (fp_integrate_ ? fp_integrate_->name() : "null") +
            ", PricingIntegrate: " + (pricing_integrate_ ? pricing_integrate_->name() : "null");
}
 
ALOEngine::ALOEngine(ALOScheme scheme_type, FixedPointEquation eq) 
     : equation_choice_(eq) {
     setScheme(scheme_type);
}
 
void ALOEngine::setScheme(ALOScheme scheme_type) {
    switch (scheme_type) {
        case FAST:
            scheme_ptr_ = createFastScheme();
            break;
        case ACCURATE:
            scheme_ptr_ = createAccurateScheme();
            break;
        case HIGH_PRECISION:
            scheme_ptr_ = createHighPrecisionScheme();
            break;
        default:
            throw std::invalid_argument("Unknown ALO scheme type");
    }
}
 
void ALOEngine::setFixedPointEquation(FixedPointEquation eq) {
    equation_choice_ = eq;
}
 
std::string ALOEngine::getSchemeDescription() const {
    return scheme_ptr_ ? scheme_ptr_->getDescription() : "Scheme not initialized";
}
 
std::string ALOEngine::getEquationName() const {
    FixedPointEquation eq_to_show = equation_choice_;
    if (eq_to_show == AUTO) { 
        // For display purposes, we can show what AUTO would likely resolve to for a typical case,
        // though the actual resolution happens dynamically in createFixedPointEvaluator*.
        // Example: if r approx q, it's A, else B. This is just for info.
        eq_to_show = FP_B; // Default to B if auto and no specific params known
    }
    switch (eq_to_show) {
        case FP_A: return "Equation A";
        case FP_B: return "Equation B";
        default: return "AUTO (resolved at calculation)";
    }
}
 
void ALOEngine::clearCache() const {
    opt::getThreadLocalCacheDouble().clear();
    opt::getThreadLocalCacheSingle().clear();
}
 
size_t ALOEngine::getCacheSize() const {
    return opt::getThreadLocalCacheDouble().size() + 
           opt::getThreadLocalCacheSingle().size();
}
 
double ALOEngine::blackScholesPut(double S, double K, double r, double q, double vol, double T) {
    mod::EuropeanPutDouble ep;
    return ep.calculatePrice(S,K,r,q,vol,T);
}
 
double ALOEngine::blackScholesCall(double S, double K, double r, double q, double vol, double T) {
    mod::EuropeanCallDouble ec;
    return ec.calculatePrice(S,K,r,q,vol,T);
}
 
std::shared_ptr<ALOIterationScheme> ALOEngine::createFastScheme() {
    try {
        auto fpIntegrate = num::createIntegrateDouble("GaussLegendre", 7);
        auto pricingIntegrate = num::createIntegrateDouble("GaussLegendre", 27);
        return std::make_shared<ALOIterationScheme>(7, 2, fpIntegrate, pricingIntegrate);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error creating fast scheme: ") + e.what());
    }
}
 
std::shared_ptr<ALOIterationScheme> ALOEngine::createAccurateScheme() {
    try {
        auto fpIntegrate = num::createIntegrateDouble("GaussLegendre", 25);
        auto pricingIntegrate = num::createIntegrateDouble("TanhSinh", 0, 1e-8); // Tolerance for TanhSinh
        return std::make_shared<ALOIterationScheme>(13, 5, fpIntegrate, pricingIntegrate);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error creating accurate scheme: ") + e.what());
    }
}
 
std::shared_ptr<ALOIterationScheme> ALOEngine::createHighPrecisionScheme() {
    try {
        auto fpIntegrate = num::createIntegrateDouble("TanhSinh", 0, 1e-10);
        auto pricingIntegrate = num::createIntegrateDouble("TanhSinh", 0, 1e-10);
        return std::make_shared<ALOIterationScheme>(30, 10, fpIntegrate, pricingIntegrate);
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Error creating high precision scheme: ") + e.what());
    }
}

// --- Public Pricing Methods ---
double ALOEngine::calculateOption(double S, double K, double r, double q, double vol, double T, OptionType type) const {
    if (!scheme_ptr_) throw std::runtime_error("ALOEngine: Scheme not initialized.");
    if (type == PUT) {
        return calculatePutImpl(S, K, r, q, vol, T);
    } else { 
        return calculateCallImpl(S, K, r, q, vol, T);
    }
}
 
double ALOEngine::calculatePut(double S, double K, double r, double q, double vol, double T) const {
    if (!scheme_ptr_) throw std::runtime_error("ALOEngine: Scheme not initialized.");
    return calculatePutImpl(S, K, r, q, vol, T);
}
 
double ALOEngine::calculateEarlyExercisePremium(double S, double K, double r, double q, double vol, double T, 
                                             OptionType type) const {
    if (!scheme_ptr_) throw std::runtime_error("ALOEngine: Scheme not initialized.");
    double american_price = calculateOption(S, K, r, q, vol, T, type);
    double european_price = (type == PUT) ? 
        blackScholesPut(S, K, r, q, vol, T) : 
        blackScholesCall(S, K, r, q, vol, T);
    return american_price - european_price;
}

// --- Double Precision Core Implementation ---
double ALOEngine::calculatePutImpl(double S, double K, double r, double q, double vol, double T) const {
    if (K <= DOUBLE_EPS || S <= DOUBLE_EPS || vol <= DOUBLE_EPS || T <= DOUBLE_EPS) {
        return std::max(0.0, K - S);
    }
    
    opt::OptionKeyDouble params{S, K, r, q, vol, T, 0}; 
    auto& cache = opt::getThreadLocalCacheDouble();
    auto it = cache.find(params);
    if (it != cache.end()) {
        return it->second;
    }
    
    if (r <= 0.0) { // Generally European for puts if r <= 0
        double result = blackScholesPut(S, K, r, q, vol, T);
        cache[params] = result;
        return std::max(result, K-S); 
    }
    if (r < 0.0 && q < r) { 
        throw std::runtime_error("ALOEngine: double-boundary q<r<0 for put not handled by this Impl.");
    }

    mod::AmericanPutDouble american_put_model(scheme_ptr_->getPricingIntegrate());
    
    auto boundary_interp = american_put_model.calculateExerciseBoundary(
        S, K, r, q, vol, T,
        scheme_ptr_->getNumChebyshevNodes(),
        scheme_ptr_->getNumFixedPointIterations(),
        scheme_ptr_->getFixedPointIntegrate() 
    );
    
    double premium = american_put_model.calculateEarlyExercisePremium(
        S, K, r, q, vol, T, boundary_interp);
    
    double european_val = blackScholesPut(S, K, r, q, vol, T);
    // Per ALO paper, American Value = European Value + Early Exercise Premium
    // And must be >= intrinsic value
    double result = std::max({european_val + premium, K - S, 0.0});


    cache[params] = result;
    return result;
}
 
double ALOEngine::calculateCallImpl(double S, double K, double r, double q, double vol, double T) const {
    if (K <= DOUBLE_EPS || S <= DOUBLE_EPS || vol <= DOUBLE_EPS || T <= DOUBLE_EPS) {
        return std::max(0.0, S - K);
    }

    opt::OptionKeyDouble params{S, K, r, q, vol, T, 1}; 
    auto& cache = opt::getThreadLocalCacheDouble();
    auto it = cache.find(params);
    if (it != cache.end()) {
        return it->second;
    }
    
    if (q <= DOUBLE_EPS) { // No (or negligible) dividend
        double result = blackScholesCall(S, K, r, q, vol, T);
        cache[params] = result;
        return std::max(result, S-K);
    }
    // Condition for European call from ALO "Double Boundary" paper Table 1 (q_eff < r_eff for symmetric put is false)
    // Effective put has r_eff=q, q_eff=r. So if q <= r (for original call params) when r_eff < 0 (i.e. q < 0), it's European.
    // More simply, from QuantLib: if r < 0 and q <= r.
    if (r < 0.0 && q <= r) { 
        double result = blackScholesCall(S, K, r, q, vol, T);
        cache[params] = result;
        return std::max(result, S-K);
    }

    mod::AmericanCallDouble american_call_model(scheme_ptr_->getPricingIntegrate());
    
    auto boundary_interp = american_call_model.calculateExerciseBoundary(
        S, K, r, q, vol, T,
        scheme_ptr_->getNumChebyshevNodes(),
        scheme_ptr_->getNumFixedPointIterations(),
        scheme_ptr_->getFixedPointIntegrate()
    );
    
    double premium = american_call_model.calculateEarlyExercisePremium(
        S, K, r, q, vol, T, boundary_interp);
        
    double european_val = blackScholesCall(S, K, r, q, vol, T);
    double result = std::max({european_val + premium, S - K, 0.0});

    cache[params] = result;
    return result;
}


// --- Single Precision Core Implementation ---
float ALOEngine::calculatePutImplSingle(float S, float K, float r, float q, float vol, float T) const {
    if (K <= FLOAT_EPS || S <= FLOAT_EPS || vol <= FLOAT_EPS || T <= FLOAT_EPS) {
        return std::max(0.0f, K - S);
    }

    opt::OptionKeySingle params{S, K, r, q, vol, T, 0}; // 0 = PUT
    auto& cache = opt::getThreadLocalCacheSingle();
    auto it = cache.find(params);
    if (it != cache.end()) {
        return it->second;
    }

    if (r <= 0.0f) {
        mod::EuropeanPutSingle ep_s;
        float european_val_s = ep_s.calculatePrice(S,K,r,q,vol,T);
        float result = std::max(european_val_s, K-S);
        cache[params] = result;
        return result;
    }
     if (r < 0.0f && q < r) {
        throw std::runtime_error("ALOEngine(Single): double-boundary q<r<0 for put not handled.");
    }

    // Create single-precision numerical components based on the double-precision scheme settings
    std::shared_ptr<num::IntegrateSingle> pricing_integrate_s;
    std::shared_ptr<num::IntegrateSingle> fp_integrate_s;
    size_t pricing_integrator_order = 27; // Default from fast scheme for pricing
    size_t fp_integrator_order = 7;       // Default from fast scheme for FP
    float tolerance = 1e-7f;              // Default for TanhSinh single

    if(scheme_ptr_){ // Check if scheme_ptr_ is initialized
        const auto& double_pricing_integrator = scheme_ptr_->getPricingIntegrate();
        const auto& double_fp_integrator = scheme_ptr_->getFixedPointIntegrate();

        // Attempt to infer parameters (this is a simplification)
        // A more robust way would be to have scheme provide parameters directly
        // or have ALOScheme templated.
        // Example: if(dynamic_cast<GaussLegendreIntegrateDoubleImpl*>(double_fp_integrator.get())) { order = ... }

        // For now, use names and map to typical single-precision choices or defaults
        if (double_pricing_integrator->name().find("GaussLegendre") != std::string::npos) {
            pricing_integrate_s = num::createIntegrateSingle("GaussLegendre", 27); // Default good order for single
        } else { // TanhSinh or Adaptive
            pricing_integrate_s = num::createIntegrateSingle("TanhSinh", 0, 1e-7f);
        }

        if (double_fp_integrator->name().find("GaussLegendre") != std::string::npos) {
            fp_integrate_s = num::createIntegrateSingle("GaussLegendre", 7); // Default good order for single
        } else {
            fp_integrate_s = num::createIntegrateSingle("TanhSinh", 0, 1e-7f);
        }
    } else {
        throw std::runtime_error("ALOEngine(Single): Scheme not initialized.");
    }


    mod::AmericanPutSingle american_put_model_s(pricing_integrate_s);
    
    auto boundary_interp_s = american_put_model_s.calculateExerciseBoundary(
        S, K, r, q, vol, T,
        scheme_ptr_->getNumChebyshevNodes(),
        scheme_ptr_->getNumFixedPointIterations(),
        fp_integrate_s
    );
    
    float premium_s = american_put_model_s.calculateEarlyExercisePremium(
        S, K, r, q, vol, T, boundary_interp_s);
    
    mod::EuropeanPutSingle euro_put_s;
    float european_val_s = euro_put_s.calculatePrice(S, K, r, q, vol, T);
    float result = std::max({european_val_s + premium_s, K - S, 0.0f});

    cache[params] = result;
    return result;
}

float ALOEngine::calculateCallImplSingle(float S, float K, float r, float q, float vol, float T) const {
     if (K <= FLOAT_EPS || S <= FLOAT_EPS || vol <= FLOAT_EPS || T <= FLOAT_EPS) {
        return std::max(0.0f, S - K);
    }

    opt::OptionKeySingle params{S, K, r, q, vol, T, 1}; // 1 = CALL
    auto& cache = opt::getThreadLocalCacheSingle();
    auto it = cache.find(params);
    if (it != cache.end()) {
        return it->second;
    }

    if (q <= FLOAT_EPS) {
        mod::EuropeanCallSingle ec_s;
        float result = ec_s.calculatePrice(S,K,r,q,vol,T);
        cache[params] = result;
        return std::max(result, S-K);
    }
    if (r < 0.0f && q <= r) {
        mod::EuropeanCallSingle ec_s;
        float result = ec_s.calculatePrice(S,K,r,q,vol,T);
        cache[params] = result;
        return std::max(result, S-K);
    }

    std::shared_ptr<num::IntegrateSingle> pricing_integrate_s;
    std::shared_ptr<num::IntegrateSingle> fp_integrate_s;
     if(scheme_ptr_){
        const auto& double_pricing_integrator = scheme_ptr_->getPricingIntegrate();
        const auto& double_fp_integrator = scheme_ptr_->getFixedPointIntegrate();
        if (double_pricing_integrator->name().find("GaussLegendre") != std::string::npos) {
            pricing_integrate_s = num::createIntegrateSingle("GaussLegendre", 27);
        } else { pricing_integrate_s = num::createIntegrateSingle("TanhSinh", 0, 1e-7f); }
        if (double_fp_integrator->name().find("GaussLegendre") != std::string::npos) {
            fp_integrate_s = num::createIntegrateSingle("GaussLegendre", 7);
        } else { fp_integrate_s = num::createIntegrateSingle("TanhSinh", 0, 1e-7f); }
    } else {
        throw std::runtime_error("ALOEngine(Single): Scheme not initialized.");
    }


    mod::AmericanCallSingle american_call_model_s(pricing_integrate_s);
    
    auto boundary_interp_s = american_call_model_s.calculateExerciseBoundary(
        S, K, r, q, vol, T,
        scheme_ptr_->getNumChebyshevNodes(),
        scheme_ptr_->getNumFixedPointIterations(),
        fp_integrate_s
    );
    
    float premium_s = american_call_model_s.calculateEarlyExercisePremium(
        S, K, r, q, vol, T, boundary_interp_s);
        
    mod::EuropeanCallSingle euro_call_s;
    float european_val_s = euro_call_s.calculatePrice(S, K, r, q, vol, T);
    float result = std::max({european_val_s + premium_s, S - K, 0.0f});

    cache[params] = result;
    return result;
}

// --- Public Single Precision API methods ---
float ALOEngine::calculateEuropeanSingle(double S_dbl, double K_dbl, double r_dbl, double q_dbl, 
                               double vol_dbl, double T_dbl, int optionType) const {
    if (optionType == 0) { // PUT
        mod::EuropeanPutSingle ep_s;
        return ep_s.calculatePrice(static_cast<float>(S_dbl), static_cast<float>(K_dbl), 
                                   static_cast<float>(r_dbl), static_cast<float>(q_dbl), 
                                   static_cast<float>(vol_dbl), static_cast<float>(T_dbl));
    } else { // CALL
        mod::EuropeanCallSingle ec_s;
        return ec_s.calculatePrice(static_cast<float>(S_dbl), static_cast<float>(K_dbl), 
                                   static_cast<float>(r_dbl), static_cast<float>(q_dbl), 
                                   static_cast<float>(vol_dbl), static_cast<float>(T_dbl));
    }
}

float ALOEngine::calculateAmericanSingle(double S_dbl, double K_dbl, double r_dbl, double q_dbl, 
                               double vol_dbl, double T_dbl, int optionType) const {
    if (!scheme_ptr_) throw std::runtime_error("ALOEngine: Scheme not initialized for single precision calc.");
    if (optionType == 0) { // PUT
        return calculatePutImplSingle(static_cast<float>(S_dbl), static_cast<float>(K_dbl), 
                                     static_cast<float>(r_dbl), static_cast<float>(q_dbl), 
                                     static_cast<float>(vol_dbl), static_cast<float>(T_dbl));
    } else { // CALL
        return calculateCallImplSingle(static_cast<float>(S_dbl), static_cast<float>(K_dbl), 
                                      static_cast<float>(r_dbl), static_cast<float>(q_dbl), 
                                      static_cast<float>(vol_dbl), static_cast<float>(T_dbl));
    }
}


// --- FixedPointEvaluator Factory (Double) ---
std::shared_ptr<ALOEngine::FixedPointEvaluatorDouble> ALOEngine::createFixedPointEvaluatorDouble(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B_boundary_func) const {
    
    FixedPointEquation eq_to_use = equation_choice_;
    if (eq_to_use == AUTO) {
        // Prefer B unless r and q are very close, and not the double boundary put case
        bool is_double_boundary_put = (r < 0.0 && q < r); // For Puts
        eq_to_use = (std::abs(r - q) < 0.001 && !is_double_boundary_put) ? FP_A : FP_B;
    }
    
    if (eq_to_use == FP_A) {
        return std::make_shared<EquationADouble>(K, r, q, vol, B_boundary_func, scheme_ptr_->getFixedPointIntegrate());
    } else { // FP_B
        return std::make_shared<EquationBDouble>(K, r, q, vol, B_boundary_func, scheme_ptr_->getFixedPointIntegrate());
    }
}

// --- FixedPointEvaluator Factory (Single) ---
std::shared_ptr<ALOEngine::FixedPointEvaluatorSingle> ALOEngine::createFixedPointEvaluatorSingle(
    float K_f, float r_f, float q_f, float vol_f, 
    const std::function<float(float)>& B_boundary_func_f) const {
    
    // Create a single-precision integrator based on the double-precision scheme settings
    std::shared_ptr<num::IntegrateSingle> fp_integrate_s;
    if(scheme_ptr_){
        const auto& double_fp_integrator = scheme_ptr_->getFixedPointIntegrate();
        if (double_fp_integrator->name().find("GaussLegendre") != std::string::npos) {
             // Try to get order from double, or use a default single-precision order
            size_t order_s = 7; // Default
            // Example: if (auto gl_double = dynamic_cast<const num::GaussLegendreIntegrateDoubleImpl*>(double_fp_integrator.get())) { order_s = gl_double->getOrder(); }
            // This requires RTTI and knowing the Impl class name. For now, using a default.
            fp_integrate_s = num::createIntegrateSingle("GaussLegendre", order_s);
        } else { // TanhSinh or Adaptive
            fp_integrate_s = num::createIntegrateSingle("TanhSinh", 0, 1e-7f); // Default tolerance
        }
    } else {
         throw std::runtime_error("ALOEngine(Single): Scheme not initialized for FP evaluator.");
    }


    FixedPointEquation eq_to_use = equation_choice_;
    if (eq_to_use == AUTO) {
        bool is_double_boundary_put_s = (r_f < 0.0f && q_f < r_f);
        eq_to_use = (std::abs(r_f - q_f) < 0.001f && !is_double_boundary_put_s) ? FP_A : FP_B;
    }
    
    if (eq_to_use == FP_A) {
        return std::make_shared<EquationASingle>(K_f, r_f, q_f, vol_f, B_boundary_func_f, fp_integrate_s);
    } else { // FP_B
        return std::make_shared<EquationBSingle>(K_f, r_f, q_f, vol_f, B_boundary_func_f, fp_integrate_s);
    }
}


// --- Batch and SIMD methods ---
std::vector<double> ALOEngine::batchCalculatePut(double S, const std::vector<double>& strikes,
                                              double r, double q, double vol, double T) const {
    std::vector<double> results(strikes.size());
    // For larger batches, this could be parallelized or use a more optimized loop.
    // For now, simple loop calling the (cached) Impl.
    for(size_t i=0; i < strikes.size(); ++i) {
        results[i] = calculatePutImpl(S, strikes[i], r, q, vol, T);
    }
    return results;
}

std::vector<double> ALOEngine::batchCalculatePut(double S, 
                                         const std::vector<std::tuple<double, double, double, double, double>>& options) const {
    std::vector<double> results(options.size());
    for(size_t i=0; i < options.size(); ++i) {
        const auto& [K_opt, r_opt, q_opt, vol_opt, T_opt] = options[i]; // structured binding
        results[i] = calculatePutImpl(S, K_opt, r_opt, q_opt, vol_opt, T_opt);
    }
    return results;
}

std::vector<double> ALOEngine::batchCalculateCall(double S, const std::vector<double>& strikes,
                                          double r, double q, double vol, double T) const {
    std::vector<double> results(strikes.size());
    for(size_t i=0; i < strikes.size(); ++i) {
        results[i] = calculateCallImpl(S, strikes[i], r, q, vol, T);
    }
    return results;
}

std::array<double, 4> ALOEngine::calculatePut4(
    const std::array<double, 4>& spots, const std::array<double, 4>& strikes,
    const std::array<double, 4>& rs, const std::array<double, 4>& qs,
    const std::array<double, 4>& vols, const std::array<double, 4>& Ts) const {
    
    std::array<double, 4> results;
    // This is not using opt::SimdOperationDouble::AmericanPutApprox yet because that's an approx.
    // For full ALO, we loop. If BAW is acceptable here, then use SimdOperationDouble.
    for(size_t i=0; i<4; ++i) {
        results[i] = calculatePutImpl(spots[i], strikes[i], rs[i], qs[i], vols[i], Ts[i]);
    }
    return results;
}

std::array<double, 4> ALOEngine::calculatePut4(
    double S, const std::array<double, 4>& strikes,
    double r, double q, double vol, double T) const {
    std::array<double, 4> spots_arr; spots_arr.fill(S);
    std::array<double, 4> r_arr; r_arr.fill(r);
    std::array<double, 4> q_arr; q_arr.fill(q);
    std::array<double, 4> vol_arr; vol_arr.fill(vol);
    std::array<double, 4> T_arr; T_arr.fill(T);
    return calculatePut4(spots_arr, strikes, r_arr, q_arr, vol_arr, T_arr);
}


std::vector<float> ALOEngine::batchCalculatePutSingle(
    float S, const std::vector<float>& strikes,
    float r, float q, float vol, float T) const {
    std::vector<float> results(strikes.size());
    for(size_t i=0; i < strikes.size(); ++i) {
        results[i] = calculatePutImplSingle(S, strikes[i], r, q, vol, T);
    }
    return results;
}

std::vector<float> ALOEngine::batchCalculateCallSingle(
    float S, const std::vector<float>& strikes,
    float r, float q, float vol, float T) const {
    std::vector<float> results(strikes.size());
    for(size_t i=0; i < strikes.size(); ++i) {
        results[i] = calculateCallImplSingle(S, strikes[i], r, q, vol, T);
    }
    return results;
}

std::vector<float> ALOEngine::batchCalculatePutFloat( 
            float S, const std::vector<float>& strikes,
            float r, float q, float vol, float T) const {
    // Assuming this is also full ALO, same as batchCalculatePutSingle
    return batchCalculatePutSingle(S, strikes, r, q, vol, T);
}


void ALOEngine::runBenchmark(int numOptions) {
    if (numOptions <= 0) numOptions = 10000000; // Ensure positive
    
    std::cout << "Preparing test data for " << numOptions << " options (benchmark uses American Put Single Precision Approx)...\n";
    
    std::vector<float> spots_f(numOptions);
    std::vector<float> strikes_f(numOptions);
    std::vector<float> rates_f(numOptions);
    std::vector<float> divs_f(numOptions);
    std::vector<float> vols_f(numOptions);
    std::vector<float> times_f(numOptions);
    
    std::vector<float> results_scalar_f(numOptions);
    std::vector<float> results_simd_f(numOptions);
    
    // Simple data generation
    for (int i = 0; i < numOptions; i++) {
        spots_f[i]   = 90.0f + static_cast<float>(i % 21); // 90 to 110
        strikes_f[i] = 100.0f;
        rates_f[i]   = 0.01f + static_cast<float>(i % 5) * 0.01f; // 0.01 to 0.05
        divs_f[i]    = 0.00f + static_cast<float>(i % 4) * 0.01f; // 0.00 to 0.03
        vols_f[i]    = 0.10f + static_cast<float>(i % 21) * 0.01f; // 0.10 to 0.30
        times_f[i]   = 0.25f + static_cast<float>(i % 8) * 0.25f; // 0.25 to 2.0
    }
    
    mod::AmericanPutSingle put_approximator(nullptr); // BAW approx doesn't need integrator here
                                                     // If it were full ALO, it would.

    std::cout << "Benchmarking scalar single-precision BAW approximation...\n";
    auto start_scalar = std::chrono::high_resolution_clock::now();
    // This loop is calling a member function which contains the scalar BAW.
    put_approximator.batchApproximatePriceBAW(spots_f, strikes_f, rates_f, divs_f, vols_f, times_f, results_scalar_f);
    auto end_scalar = std::chrono::high_resolution_clock::now();
    double scalar_time = std::chrono::duration<double, std::milli>(end_scalar - start_scalar).count();
    
    std::cout << "Scalar BAW Approx time: " << scalar_time << " ms\n";
    if (scalar_time > 0) std::cout << "Scalar BAW Approx options/sec: " << static_cast<double>(numOptions) / (scalar_time / 1000.0) << "\n";

    std::cout << "Benchmarking SIMD single-precision BAW approximation (via VectorSingle)...\n";
    auto start_simd = std::chrono::high_resolution_clock::now();
    opt::VectorSingle::AmericanPut(spots_f.data(), strikes_f.data(), rates_f.data(), divs_f.data(), 
                                   vols_f.data(), times_f.data(), results_simd_f.data(), numOptions);
    auto end_simd = std::chrono::high_resolution_clock::now();
    double simd_time = std::chrono::duration<double, std::milli>(end_simd - start_simd).count();

    std::cout << "SIMD BAW Approx time: " << simd_time << " ms\n";
    if (simd_time > 0) {
        std::cout << "SIMD BAW Approx options/sec: " << static_cast<double>(numOptions) / (simd_time / 1000.0) << "\n";
        if(scalar_time > 0) std::cout << "Speedup (BAW SIMD vs BAW Scalar): " << scalar_time / simd_time << "x\n";
    }

    // Validate a few results
    double max_diff = 0.0;
    for(int i=0; i < std::min(numOptions, 1000); ++i) { 
        max_diff = std::max(max_diff, static_cast<double>(std::abs(results_scalar_f[i] - results_simd_f[i])));
    }
    std::cout << "Max difference between scalar BAW and SIMD BAW (first 1000): " << max_diff << "\n";
}


} // namespace alo
} // namespace engine
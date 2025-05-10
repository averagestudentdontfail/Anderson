#ifndef ENGINE_ALO_ALOENGINE_H
#define ENGINE_ALO_ALOENGINE_H

#include "aloscheme.h"
#include "num/integrate.h" 
#include "num/chebyshev.h" 
#include "opt/cache.h"     

#include <immintrin.h> 
#include <sleef.h>     
#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map> 
#include <array>
#include <tuple>
#include <chrono> 

namespace engine {
namespace alo {

enum FixedPointEquation {
    FP_A, 
    FP_B,  
    AUTO   
};

enum OptionType {
    PUT, 
    CALL  
};

class ALOIterationScheme { // This scheme primarily drives the double-precision engine
public:
    ALOIterationScheme(size_t n_nodes, size_t m_iterations,
                       std::shared_ptr<num::IntegrateDouble> fp_integrate, 
                       std::shared_ptr<num::IntegrateDouble> pricing_integrate); 
    
    size_t getNumChebyshevNodes() const { return n_nodes_; }
    size_t getNumFixedPointIterations() const { return m_iterations_; }
    std::shared_ptr<num::IntegrateDouble> getFixedPointIntegrate() const { return fp_integrate_; }
    std::shared_ptr<num::IntegrateDouble> getPricingIntegrate() const { return pricing_integrate_; }
    std::string getDescription() const;
    
private:
    size_t n_nodes_; 
    size_t m_iterations_; 
    std::shared_ptr<num::IntegrateDouble> fp_integrate_; 
    std::shared_ptr<num::IntegrateDouble> pricing_integrate_;
};

class ALOEngine {
public:
    explicit ALOEngine(ALOScheme scheme_type = ACCURATE, FixedPointEquation eq = AUTO);
    ~ALOEngine() = default;
    
    // Double precision
    double calculateOption(double S, double K, double r, double q, double vol, double T, OptionType type = PUT) const;
    double calculatePut(double S, double K, double r, double q, double vol, double T) const; 
    double calculateEarlyExercisePremium(double S, double K, double r, double q, double vol, double T, 
                                        OptionType type = PUT) const;
    
    static double blackScholesPut(double S, double K, double r, double q, double vol, double T);
    static double blackScholesCall(double S, double K, double r, double q, double vol, double T);
    
    static std::shared_ptr<ALOIterationScheme> createFastScheme();
    static std::shared_ptr<ALOIterationScheme> createAccurateScheme();
    static std::shared_ptr<ALOIterationScheme> createHighPrecisionScheme();
    
    void setScheme(ALOScheme scheme_type); 
    void setFixedPointEquation(FixedPointEquation eq);
    
    std::string getSchemeDescription() const;
    std::string getEquationName() const;
    
    void clearCache() const; 
    size_t getCacheSize() const; 
    
    // Batch double precision
    std::vector<double> batchCalculatePut(double S, const std::vector<double>& strikes,
                                         double r, double q, double vol, double T) const;
    std::vector<double> batchCalculatePut(double S, 
                                         const std::vector<std::tuple<double, double, double, double, double>>& options) const;
    std::vector<double> batchCalculateCall(double S, const std::vector<double>& strikes,
                                          double r, double q, double vol, double T) const;

    // SIMD double precision (AVX2, 4 doubles)
    std::array<double, 4> calculatePut4(
        const std::array<double, 4>& spots, const std::array<double, 4>& strikes,
        const std::array<double, 4>& rs, const std::array<double, 4>& qs,
        const std::array<double, 4>& vols, const std::array<double, 4>& Ts) const;
    std::array<double, 4> calculatePut4(
        double S, const std::array<double, 4>& strikes,
        double r, double q, double vol, double T) const;

    // --- Single precision calculations ---
    // European (takes double inputs for API consistency, returns float)
    float calculateEuropeanSingle(double S_dbl, double K_dbl, double r_dbl, double q_dbl, 
                               double vol_dbl, double T_dbl, int optionType) const; 
    // American - Full ALO (takes double inputs for API consistency, returns float)
    float calculateAmericanSingle(double S_dbl, double K_dbl, double r_dbl, double q_dbl, 
                               double vol_dbl, double T_dbl, int optionType) const; 
    
    // Batch single precision - Full ALO
    std::vector<float> batchCalculatePutSingle( // Full ALO for puts
        float S, const std::vector<float>& strikes,
        float r, float q, float vol, float T) const;
    std::vector<float> batchCalculateCallSingle( // Full ALO for calls
        float S, const std::vector<float>& strikes,
        float r, float q, float vol, float T) const;
    
    // This might be intended for BAW or other approx, kept for compatibility from original.
    // If it's meant to be full ALO, it's redundant with batchCalculatePutSingle.
    std::vector<float> batchCalculatePutFloat( 
            float S, const std::vector<float>& strikes,
            float r, float q, float vol, float T) const;


    void runBenchmark(int numOptions = 10000000);
                        
private:
    // Double precision internal implementations
    double calculatePutImpl(double S, double K, double r, double q, double vol, double T) const;
    double calculateCallImpl(double S, double K, double r, double q, double vol, double T) const;
    
    // Forward declare inner classes for fixed point evaluation (double precision)
    class FixedPointEvaluatorDouble; 
    class EquationADouble;          
    class EquationBDouble;          

    std::shared_ptr<FixedPointEvaluatorDouble> createFixedPointEvaluatorDouble( 
        double K, double r, double q, double vol, 
        const std::function<double(double)>& B_boundary_func) const; 

    // Single precision internal implementations
    float calculatePutImplSingle(float S, float K, float r, float q, float vol, float T) const;
    float calculateCallImplSingle(float S, float K, float r, float q, float vol, float T) const;

    // Forward declare inner classes for fixed point evaluation (single precision)
    class FixedPointEvaluatorSingle; 
    class EquationASingle;          
    class EquationBSingle;    

    std::shared_ptr<FixedPointEvaluatorSingle> createFixedPointEvaluatorSingle( 
        float K, float r, float q, float vol, 
        const std::function<float(float)>& B_boundary_func) const;
    
    // Member variables
    std::shared_ptr<ALOIterationScheme> scheme_ptr_; 
    FixedPointEquation equation_choice_; 
};

} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_ALOENGINE_H
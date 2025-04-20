#ifndef ENGINE_ALO_ALOENGINE_H
#define ENGINE_ALO_ALOENGINE_H

#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include <array>
#include <tuple>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <deque>

// Forward declarations for dist namespace components
namespace engine {
namespace alo {
namespace dist {
    class TaskDispatcher;
}}}

namespace engine {
namespace alo {

/**
 * @enum ALOScheme
 * @brief Different numerical schemes for the ALO algorithm
 */
enum ALOScheme {
    FAST,           ///< Legendre-Legendre (7,2,7)-27 - Fastest but less accurate
    ACCURATE,       ///< Legendre-TanhSinh (25,5,13)-1e-8 - Good balance of speed and accuracy
    HIGH_PRECISION  ///< TanhSinh-TanhSinh (10,30)-1e-10 - Highest accuracy but slower
};

/**
 * @enum FixedPointEquation
 * @brief Different fixed point equations from the ALO paper
 */
enum FixedPointEquation {
    FP_A,  ///< Equation A from the paper
    FP_B,  ///< Equation B from the paper
    AUTO   ///< Automatically choose based on |r-q|
};

/**
 * @enum OptionType
 * @brief Type of option to price
 */
enum OptionType {
    PUT,  ///< Put option
    CALL  ///< Call option
};

// Forward declarations
namespace num {
    class Integrator;
    class ChebyshevInterpolation;
}

/**
 * @class ALOIterationScheme
 * @brief Class to represent the iteration scheme parameters for the ALO algorithm
 */
class ALOIterationScheme {
public:
    /**
     * @brief Constructor
     * 
     * @param n Number of Chebyshev nodes
     * @param m Number of fixed point iterations
     * @param fpIntegrator Integrator for fixed point equation
     * @param pricingIntegrator Integrator for pricing
     */
    ALOIterationScheme(size_t n, size_t m, 
                       std::shared_ptr<num::Integrator> fpIntegrator,
                       std::shared_ptr<num::Integrator> pricingIntegrator);
    
    // Getters
    size_t getNumChebyshevNodes() const { return n_; }
    size_t getNumFixedPointIterations() const { return m_; }
    std::shared_ptr<num::Integrator> getFixedPointIntegrator() const { return fpIntegrator_; }
    std::shared_ptr<num::Integrator> getPricingIntegrator() const { return pricingIntegrator_; }
    
    /**
     * @brief Get a string description of the scheme
     * @return String description
     */
    std::string getDescription() const;
    
private:
    size_t n_; ///< Number of Chebyshev nodes
    size_t m_; ///< Total number of fixed point iterations
    std::shared_ptr<num::Integrator> fpIntegrator_; ///< Integrator for fixed point equation
    std::shared_ptr<num::Integrator> pricingIntegrator_; ///< Integrator for pricing
};

/**
 * @class ALOEngine
 * @brief Main ALO engine class for pricing American options
 * 
 * This class implements the Andersen-Lake-Offengelden algorithm for
 * pricing American options with high accuracy and performance.
 * The implementation follows deterministic execution principles
 * for reliable operation in trading systems.
 */
class ALOEngine {
public:
    /**
     * @brief Constructor
     * 
     * @param scheme Numerical scheme to use
     * @param eq Fixed point equation to use
     */
    explicit ALOEngine(ALOScheme scheme = ACCURATE, FixedPointEquation eq = AUTO);
    
    /**
     * @brief Destructor
     */
    ~ALOEngine() = default;
    
    /**
     * @brief Main pricing function for American options
     * 
     * @param S Current spot price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @param type Option type (PUT or CALL)
     * @return Option price
     */
    double calculateOption(double S, double K, double r, double q, double vol, double T, OptionType type = PUT) const;
    
    /**
     * @brief Legacy method for American put pricing (for backwards compatibility)
     * 
     * @param S Current spot price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @return Put option price
     */
    double calculatePut(double S, double K, double r, double q, double vol, double T) const;
    
    /**
     * @brief European Black-Scholes put pricing
     * 
     * @param S Current spot price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @return European put option price
     */
    static double blackScholesPut(double S, double K, double r, double q, double vol, double T);
    
    /**
     * @brief European Black-Scholes call pricing
     * 
     * @param S Current spot price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @return European call option price
     */
    static double blackScholesCall(double S, double K, double r, double q, double vol, double T);
    
    /**
     * @brief Create a fast scheme
     * @return ALOIterationScheme pointer
     */
    static std::shared_ptr<ALOIterationScheme> createFastScheme();
    
    /**
     * @brief Create an accurate scheme
     * @return ALOIterationScheme pointer
     */
    static std::shared_ptr<ALOIterationScheme> createAccurateScheme();
    
    /**
     * @brief Create a high precision scheme
     * @return ALOIterationScheme pointer
     */
    static std::shared_ptr<ALOIterationScheme> createHighPrecisionScheme();
    
    /**
     * @brief Set the numerical scheme
     * @param scheme Scheme to use
     */
    void setScheme(ALOScheme scheme);
    
    /**
     * @brief Set the fixed point equation
     * @param eq Equation to use
     */
    void setFixedPointEquation(FixedPointEquation eq);
    
    /**
     * @brief Get the current scheme description
     * @return String description of the current scheme
     */
    std::string getSchemeDescription() const;
    
    /**
     * @brief Get the name of the current fixed point equation
     * @return String name of the current equation
     */
    std::string getEquationName() const;
    
    /**
     * @brief Clear the pricing cache
     */
    void clearCache() const;
    
    /**
     * @brief Get the size of the cache
     * @return Number of cached prices
     */
    size_t getCacheSize() const;
    
    /**
     * @brief Calculate the early exercise premium
     * 
     * @param S Current spot price
     * @param K Strike price
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @param type Option type (PUT or CALL)
     * @return Early exercise premium
     */
    double calculateEarlyExercisePremium(double S, double K, double r, double q, double vol, double T, 
                                        OptionType type = PUT) const;
    
    /**
     * @brief Batch pricing for multiple put options with the same parameters except strikes
     * 
     * @param S Current spot price
     * @param strikes Vector of strike prices
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @return Vector of put option prices
     */
    std::vector<double> batchCalculatePut(double S, const std::vector<double>& strikes,
                                         double r, double q, double vol, double T) const;

    /**
     * @brief Batch pricing for multiple put options with the same spot but varying parameters
     * 
     * @param S Current spot price
     * @param options Vector of option parameters (strike, r, q, vol, T)
     * @return Vector of put option prices
     */
    std::vector<double> batchCalculatePut(double S, 
                                         const std::vector<std::tuple<double, double, double, double, double>>& options) const;

    /**
     * @brief SIMD-accelerated pricing for 4 put options at once with AVX2
     * 
     * @param spots Array of 4 spot prices
     * @param strikes Array of 4 strike prices
     * @param rs Array of 4 risk-free rates
     * @param qs Array of 4 dividend yields
     * @param vols Array of 4 volatilities
     * @param Ts Array of 4 times to maturity
     * @return Array of 4 put option prices
     */
    std::array<double, 4> calculatePut4(
        const std::array<double, 4>& spots,
        const std::array<double, 4>& strikes,
        const std::array<double, 4>& rs,
        const std::array<double, 4>& qs,
        const std::array<double, 4>& vols,
        const std::array<double, 4>& Ts) const;

    /**
     * @brief SIMD-accelerated pricing for 4 put options with same parameters except strikes
     * 
     * @param S Spot price
     * @param strikes Array of 4 strike prices
     * @param r Risk-free rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity
     * @return Array of 4 put option prices
     */
    std::array<double, 4> calculatePut4(
        double S,
        const std::array<double, 4>& strikes,
        double r, double q, double vol, double T) const;

    /**
     * @brief Batch pricing for multiple call options with the same parameters except strikes
     * 
     * @param S Current spot price
     * @param strikes Vector of strike prices
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @return Vector of call option prices
     */
    std::vector<double> batchCalculateCall(double S, const std::vector<double>& strikes,
                                          double r, double q, double vol, double T) const;
    
    /**
     * @brief Parallel batch pricing for large numbers of put options
     * 
     * Uses OpenMP to accelerate batch pricing across multiple threads
     * 
     * @param S Current spot price
     * @param strikes Vector of strike prices
     * @param r Risk-free interest rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity in years
     * @return Vector of put option prices
     */
    std::vector<double> parallelBatchCalculatePut(double S, const std::vector<double>& strikes,
                                                 double r, double q, double vol, double T) const;

    /**
     * @brief Process a chunk of put options with SIMD acceleration
     * 
     * @param S Spot price (constant)
     * @param strikes Array of strike prices
     * @param r Risk-free rate (constant)
     * @param q Dividend yield (constant)
     * @param vol Volatility (constant)
     * @param T Time to maturity (constant)
     * @param results Output array for results
     * @param size Number of options to process
     */
    void processSIMDChunk(double S, const double* strikes, double r, double q, 
                        double vol, double T, double* results, size_t size) const;
    
private:
    /**
     * @brief Implementation of American put option pricing
     */
    double calculatePutImpl(double S, double K, double r, double q, double vol, double T) const;
    
    /**
     * @brief Implementation of American call option pricing
     */
    double calculateCallImpl(double S, double K, double r, double q, double vol, double T) const;
    
    /**
     * @brief Helper method for processing a chunk of American put options
     */
    void processAmericanPutChunk(const double* S, const double* K, const double* r,
                               const double* q, const double* vol, const double* T,
                               double* results, size_t n) const;
    
    /**
     * @brief Calculate the early exercise boundary for puts
     */
    std::shared_ptr<num::ChebyshevInterpolation> calculatePutExerciseBoundary(
        double S, double K, double r, double q, double vol, double T) const;
    
    /**
     * @brief Calculate the early exercise boundary for calls
     */
    std::shared_ptr<num::ChebyshevInterpolation> calculateCallExerciseBoundary(
        double S, double K, double r, double q, double vol, double T) const;
    
    /**
     * @brief Calculate the maximum early exercise boundary value for puts
     */
    double xMaxPut(double K, double r, double q) const;
    
    /**
     * @brief Calculate the maximum early exercise boundary value for calls
     */
    double xMaxCall(double K, double r, double q) const;
    
    /**
     * @brief Calculate the early exercise premium for puts
     */
    double calculatePutExercisePremium(
        double S, double K, double r, double q, double vol, double T,
        const std::shared_ptr<num::ChebyshevInterpolation>& boundary) const;
    
    /**
     * @brief Calculate the early exercise premium for calls
     */
    double calculateCallExercisePremium(
        double S, double K, double r, double q, double vol, double T,
        const std::shared_ptr<num::ChebyshevInterpolation>& boundary) const;
    
    /**
     * @class FixedPointEvaluator
     * @brief Base class for fixed point equation evaluators
     */
    class FixedPointEvaluator {
    public:
        FixedPointEvaluator(double K, double r, double q, double vol, 
                          const std::function<double(double)>& B,
                          std::shared_ptr<num::Integrator> integrator);
        
        virtual ~FixedPointEvaluator() = default;
        
        // Main evaluation functions
        virtual std::tuple<double, double, double> evaluate(double tau, double b) const = 0;
        virtual std::pair<double, double> derivatives(double tau, double b) const = 0;
        
    protected:
        // Helper functions
        std::pair<double, double> d(double t, double z) const;
        
        double K_;
        double r_;
        double q_;
        double vol_;
        double vol2_; // vol^2, precomputed
        std::function<double(double)> B_;
        std::shared_ptr<num::Integrator> integrator_;
        
        // Normal distribution functions
        double normalCDF(double x) const;
        double normalPDF(double x) const;
    };
    
    /**
     * @class EquationA
     * @brief Implementation of Equation A from the ALO paper
     */
    class EquationA : public FixedPointEvaluator {
    public:
        EquationA(double K, double r, double q, double vol, 
                 const std::function<double(double)>& B,
                 std::shared_ptr<num::Integrator> integrator);
        
        std::tuple<double, double, double> evaluate(double tau, double b) const override;
        std::pair<double, double> derivatives(double tau, double b) const override;
    };
    
    /**
     * @class EquationB
     * @brief Implementation of Equation B from the ALO paper
     */
    class EquationB : public FixedPointEvaluator {
    public:
        EquationB(double K, double r, double q, double vol, 
                 const std::function<double(double)>& B,
                 std::shared_ptr<num::Integrator> integrator);
        
        std::tuple<double, double, double> evaluate(double tau, double b) const override;
        std::pair<double, double> derivatives(double tau, double b) const override;
    };
    
    /**
     * @brief Create a fixed point evaluator based on the equation type
     */
    std::shared_ptr<FixedPointEvaluator> createFixedPointEvaluator(
        double K, double r, double q, double vol, 
        const std::function<double(double)>& B) const;
    
    // Member variables
    std::shared_ptr<ALOIterationScheme> scheme_;
    FixedPointEquation equation_;
    
    // Legacy cache for backward compatibility
    mutable std::unordered_map<std::string, double> legacy_cache_;
};

} // namespace alo
} // namespace engine


#include "alodistribute.h"

#endif // ENGINE_ALO_ALOENGINE_H
#ifndef ALO_ENGINE_H
#define ALO_ENGINE_H

#include <memory>
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>
#include "../numerics/integration.h"
#include "../numerics/chebyshev.h"

// Forward declarations
namespace numerics {
    class Integrator;
    class ChebyshevInterpolation;
}

/**
 * @enum ALOScheme
 * @brief Different numerical schemes for the ALO algorithm with varying performance/accuracy tradeoffs
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
                       std::shared_ptr<numerics::Integrator> fpIntegrator,
                       std::shared_ptr<numerics::Integrator> pricingIntegrator);
    
    // Getters
    size_t getNumChebyshevNodes() const { return n_; }
    size_t getNumFixedPointIterations() const { return m_; }
    std::shared_ptr<numerics::Integrator> getFixedPointIntegrator() const { return fpIntegrator_; }
    std::shared_ptr<numerics::Integrator> getPricingIntegrator() const { return pricingIntegrator_; }
    
    /**
     * @brief Get a string description of the scheme
     * @return String description
     */
    std::string getDescription() const {
        return "ChebyshevNodes: " + std::to_string(n_) + 
               ", FixedPointIterations: " + std::to_string(m_) +
               ", FPIntegrator: " + fpIntegrator_->name() +
               ", PricingIntegrator: " + pricingIntegrator_->name();
    }
    
private:
    size_t n_; ///< Number of Chebyshev nodes
    size_t m_; ///< Total number of fixed point iterations
    std::shared_ptr<numerics::Integrator> fpIntegrator_; ///< Integrator for fixed point equation
    std::shared_ptr<numerics::Integrator> pricingIntegrator_; ///< Integrator for pricing
};

/**
 * @class ALOEngine
 * @brief Main ALO engine class for pricing American options
 * 
 * This class implements the Andersen-Lake-Offengenden algorithm for
 * pricing American options with high accuracy and performance.
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
    std::string getSchemeDescription() const {
        return scheme_->getDescription();
    }
    
    /**
     * @brief Get the name of the current fixed point equation
     * @return String name of the current equation
     */
    std::string getEquationName() const {
        switch (equation_) {
            case FP_A: return "Equation A";
            case FP_B: return "Equation B";
            case AUTO: return "Auto";
            default: return "Unknown";
        }
    }
    
    /**
     * @brief Clear the pricing cache
     */
    void clearCache() {
        cache_.clear();
    }
    
    /**
     * @brief Get the size of the cache
     * @return Number of cached prices
     */
    size_t getCacheSize() const {
        return cache_.size();
    }
    
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
     * @brief Calculate the early exercise boundary for puts
     */
    std::shared_ptr<numerics::ChebyshevInterpolation> calculatePutExerciseBoundary(
        double S, double K, double r, double q, double vol, double T) const;
    
    /**
     * @brief Calculate the early exercise boundary for calls
     */
    std::shared_ptr<numerics::ChebyshevInterpolation> calculateCallExerciseBoundary(
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
        const std::shared_ptr<numerics::ChebyshevInterpolation>& boundary) const;
    
    /**
     * @brief Calculate the early exercise premium for calls
     */
    double calculateCallExercisePremium(
        double S, double K, double r, double q, double vol, double T,
        const std::shared_ptr<numerics::ChebyshevInterpolation>& boundary) const;
    
    /**
     * @class FixedPointEvaluator
     * @brief Base class for fixed point equation evaluators
     */
    class FixedPointEvaluator {
    public:
        FixedPointEvaluator(double K, double r, double q, double vol, 
                          const std::function<double(double)>& B,
                          std::shared_ptr<numerics::Integrator> integrator);
        
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
        std::shared_ptr<numerics::Integrator> integrator_;
        
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
                 std::shared_ptr<numerics::Integrator> integrator);
        
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
                 std::shared_ptr<numerics::Integrator> integrator);
        
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
    
    // Cache for performance
    mutable std::unordered_map<std::string, double> cache_;
};

#endif // ALO_ENGINE_H
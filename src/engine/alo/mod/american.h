#ifndef ENGINE_ALO_MOD_AMERICAN_H
#define ENGINE_ALO_MOD_AMERICAN_H

#include "../num/chebyshev.h"
#include "../num/integrate.h"
#include <array>
#include <functional>
#include <memory>
#include <vector>

namespace engine {
namespace alo {
namespace mod {

/**
 * @class AmericanOption
 * @brief Base class for American option pricing models
 *
 * This class provides common functionality for American option pricing
 * using the Anderson-Lake-Offengelden algorithm.
 */
class AmericanOption {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanOption(std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Destructor
   */
  virtual ~AmericanOption() = default;

  /**
   * @brief Calculate the early exercise premium
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param boundary Early exercise boundary
   * @return Early exercise premium
   */
  virtual double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolation> &boundary) const = 0;

  /**
   * @brief Calculate the early exercise boundary
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param num_nodes Number of Chebyshev nodes
   * @param num_iterations Number of fixed point iterations
   * @param fpIntegrator Integrator for fixed point calculations
   * @return Early exercise boundary as Chebyshev interpolation
   */
  virtual std::shared_ptr<num::ChebyshevInterpolation>
  calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::Integrator> fpIntegrator) const = 0;

  /**
   * @brief Calculate the maximum early exercise boundary value
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @return Maximum boundary value
   */
  virtual double xMax(double K, double r, double q) const = 0;

  /**
   * @brief Get the integrator
   *
   * @return Integrator used for calculations
   */
  std::shared_ptr<num::Integrator> getIntegrator() const { return integrator_; }

protected:
  std::shared_ptr<num::Integrator> integrator_;
};

/**
 * @class AmericanPut
 * @brief American put option pricing model
 *
 * This class implements American put option pricing using the ALO algorithm.
 */
class AmericanPut : public AmericanOption {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanPut(std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Calculate the early exercise premium
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param boundary Early exercise boundary
   * @return Early exercise premium
   */
  double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolation> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param num_nodes Number of Chebyshev nodes
   * @param num_iterations Number of fixed point iterations
   * @param fpIntegrator Integrator for fixed point calculations
   * @return Early exercise boundary as Chebyshev interpolation
   */
  std::shared_ptr<num::ChebyshevInterpolation> calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::Integrator> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @return Maximum boundary value
   */
  double xMax(double K, double r, double q) const override;
};

/**
 * @class AmericanCall
 * @brief American call option pricing model
 *
 * This class implements American call option pricing using the ALO algorithm.
 */
class AmericanCall : public AmericanOption {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanCall(std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Calculate the early exercise premium
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param boundary Early exercise boundary
   * @return Early exercise premium
   */
  double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolation> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param num_nodes Number of Chebyshev nodes
   * @param num_iterations Number of fixed point iterations
   * @param fpIntegrator Integrator for fixed point calculations
   * @return Early exercise boundary as Chebyshev interpolation
   */
  std::shared_ptr<num::ChebyshevInterpolation> calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::Integrator> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @return Maximum boundary value
   */
  double xMax(double K, double r, double q) const override;
};

/**
 * @class FixedPointEvaluator
 * @brief Base class for fixed point equation evaluators
 *
 * This class defines the interface for fixed point equation evaluators
 * used in the ALO algorithm to compute early exercise boundaries.
 */
class FixedPointEvaluator {
public:
  /**
   * @brief Constructor
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param B Boundary function
   * @param integrator Integrator for fixed point calculations
   */
  FixedPointEvaluator(double K, double r, double q, double vol,
                      const std::function<double(double)> &B,
                      std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Destructor
   */
  virtual ~FixedPointEvaluator() = default;

  /**
   * @brief Evaluate the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Tuple of (N, D, f(b)) where f(b) is the fixed point equation value
   */
  virtual std::tuple<double, double, double> evaluate(double tau,
                                                      double b) const = 0;

  /**
   * @brief Calculate derivatives of the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Pair of (N', D') derivatives
   */
  virtual std::pair<double, double> derivatives(double tau, double b) const = 0;

protected:
  /**
   * @brief Calculate d1 and d2 terms for Black-Scholes
   *
   * @param t Time
   * @param z Price ratio
   * @return Pair of (d1, d2)
   */
  std::pair<double, double> d(double t, double z) const;

  /**
   * @brief Calculate normal CDF
   *
   * @param x Input value
   * @return Normal CDF at x
   */
  double normalCDF(double x) const;

  /**
   * @brief Calculate normal PDF
   *
   * @param x Input value
   * @return Normal PDF at x
   */
  double normalPDF(double x) const;

  // Member variables
  double K_;
  double r_;
  double q_;
  double vol_;
  double vol2_; // vol^2, precomputed
  std::function<double(double)> B_;
  std::shared_ptr<num::Integrator> integrator_;
};

/**
 * @class EquationA
 * @brief Implementation of Equation A from the ALO paper
 */
class EquationA : public FixedPointEvaluator {
public:
  /**
   * @brief Constructor
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param B Boundary function
   * @param integrator Integrator for fixed point calculations
   */
  EquationA(double K, double r, double q, double vol,
            const std::function<double(double)> &B,
            std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Evaluate the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Tuple of (N, D, f(b)) where f(b) is the fixed point equation value
   */
  std::tuple<double, double, double> evaluate(double tau,
                                              double b) const override;

  /**
   * @brief Calculate derivatives of the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Pair of (N', D') derivatives
   */
  std::pair<double, double> derivatives(double tau, double b) const override;
};

/**
 * @class EquationB
 * @brief Implementation of Equation B from the ALO paper
 */
class EquationB : public FixedPointEvaluator {
public:
  /**
   * @brief Constructor
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param B Boundary function
   * @param integrator Integrator for fixed point calculations
   */
  EquationB(double K, double r, double q, double vol,
            const std::function<double(double)> &B,
            std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Evaluate the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Tuple of (N, D, f(b)) where f(b) is the fixed point equation value
   */
  std::tuple<double, double, double> evaluate(double tau,
                                              double b) const override;

  /**
   * @brief Calculate derivatives of the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Pair of (N', D') derivatives
   */
  std::pair<double, double> derivatives(double tau, double b) const override;
};

/**
 * @brief Create a fixed point evaluator
 *
 * @param equation Equation type (A or B)
 * @param K Strike price
 * @param r Risk-free interest rate
 * @param q Dividend yield
 * @param vol Volatility
 * @param B Boundary function
 * @param integrator Integrator for fixed point calculations
 * @return Shared pointer to fixed point evaluator
 */
std::shared_ptr<FixedPointEvaluator>
createFixedPointEvaluator(char equation, double K, double r, double q,
                          double vol, const std::function<double(double)> &B,
                          std::shared_ptr<num::Integrator> integrator);

/**
 * @class AmericanOptionFloat
 * @brief Base class for single-precision American option pricing models
 *
 * This class provides common functionality for American option pricing
 * using the Anderson-Lake-Offengelden algorithm with single-precision.
 */
class AmericanOptionFloat {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanOptionFloat(
      std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Destructor
   */
  virtual ~AmericanOptionFloat() = default;

  /**
   * @brief Calculate the early exercise premium
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param boundary Early exercise boundary
   * @return Early exercise premium
   */
  virtual float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary)
      const = 0;

  /**
   * @brief Calculate the early exercise boundary
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param num_nodes Number of Chebyshev nodes
   * @param num_iterations Number of fixed point iterations
   * @param fpIntegrator Integrator for fixed point calculations
   * @return Early exercise boundary as Chebyshev interpolation
   */
  virtual std::shared_ptr<num::ChebyshevInterpolationFloat>
  calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegratorFloat> fpIntegrator) const = 0;

  /**
   * @brief Calculate the maximum early exercise boundary value
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @return Maximum boundary value
   */
  virtual float xMax(float K, float r, float q) const = 0;

  /**
   * @brief Get the integrator
   *
   * @return Integrator used for calculations
   */
  std::shared_ptr<num::IntegratorFloat> getIntegrator() const {
    return integrator_;
  }

protected:
  std::shared_ptr<num::IntegratorFloat> integrator_;
};

/**
 * @class AmericanPutFloat
 * @brief Single-precision American put option pricing model
 *
 * This class implements American put option pricing using the ALO algorithm
 * with single-precision and optimized numerical methods.
 */
class AmericanPutFloat : public AmericanOptionFloat {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanPutFloat(std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Calculate the early exercise premium
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param boundary Early exercise boundary
   * @return Early exercise premium
   */
  float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param num_nodes Number of Chebyshev nodes
   * @param num_iterations Number of fixed point iterations
   * @param fpIntegrator Integrator for fixed point calculations
   * @return Early exercise boundary as Chebyshev interpolation
   */
  std::shared_ptr<num::ChebyshevInterpolationFloat> calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegratorFloat> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @return Maximum boundary value
   */
  float xMax(float K, float r, float q) const override;

  /**
   * @brief Approximate American put price with Barone-Adesi-Whaley method
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @return Approximate American put price
   */
  float approximatePrice(float S, float K, float r, float q, float vol,
                         float T) const;

  /**
   * @brief Calculate a batch of American put prices using approximation
   *
   * @param S Current spot price
   * @param strikes Vector of strike prices
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @return Vector of American put prices
   */
  std::vector<float> batchApproximatePrice(float S,
                                           const std::vector<float> &strikes,
                                           float r, float q, float vol,
                                           float T) const;
};

/**
 * @class AmericanCallFloat
 * @brief Single-precision American call option pricing model
 *
 * This class implements American call option pricing using the ALO algorithm
 * with single-precision and optimized numerical methods.
 */
class AmericanCallFloat : public AmericanOptionFloat {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanCallFloat(std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Calculate the early exercise premium
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param boundary Early exercise boundary
   * @return Early exercise premium
   */
  float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @param num_nodes Number of Chebyshev nodes
   * @param num_iterations Number of fixed point iterations
   * @param fpIntegrator Integrator for fixed point calculations
   * @return Early exercise boundary as Chebyshev interpolation
   */
  std::shared_ptr<num::ChebyshevInterpolationFloat> calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegratorFloat> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @return Maximum boundary value
   */
  float xMax(float K, float r, float q) const override;

  /**
   * @brief Approximate American call price with Barone-Adesi-Whaley method
   *
   * @param S Current spot price
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @return Approximate American call price
   */
  float approximatePrice(float S, float K, float r, float q, float vol,
                         float T) const;

  /**
   * @brief Calculate a batch of American call prices using approximation
   *
   * @param S Current spot price
   * @param strikes Vector of strike prices
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param T Time to maturity in years
   * @return Vector of American call prices
   */
  std::vector<float> batchApproximatePrice(float S,
                                           const std::vector<float> &strikes,
                                           float r, float q, float vol,
                                           float T) const;
};

/**
 * @class FixedPointEvaluatorFloat
 * @brief Base class for single-precision fixed point equation evaluators
 *
 * This class defines the interface for fixed point equation evaluators
 * used in the ALO algorithm to compute early exercise boundaries with
 * single-precision.
 */
class FixedPointEvaluatorFloat {
public:
  /**
   * @brief Constructor
   *
   * @param K Strike price
   * @param r Risk-free interest rate
   * @param q Dividend yield
   * @param vol Volatility
   * @param B Boundary function
   * @param integrator Integrator for fixed point calculations
   */
  FixedPointEvaluatorFloat(float K, float r, float q, float vol,
                           const std::function<float(float)> &B,
                           std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Destructor
   */
  virtual ~FixedPointEvaluatorFloat() = default;

  /**
   * @brief Evaluate the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Tuple of (N, D, f(b)) where f(b) is the fixed point equation value
   */
  virtual std::tuple<float, float, float> evaluate(float tau,
                                                   float b) const = 0;

  /**
   * @brief Calculate derivatives of the fixed point equation
   *
   * @param tau Time to maturity
   * @param b Boundary value
   * @return Pair of (N', D') derivatives
   */
  virtual std::pair<float, float> derivatives(float tau, float b) const = 0;

protected:
  /**
   * @brief Calculate d1 and d2 terms for Black-Scholes
   *
   * @param t Time
   * @param z Price ratio
   * @return Pair of (d1, d2)
   */
  std::pair<float, float> d(float t, float z) const;

  /**
   * @brief Calculate normal CDF using fast approximation
   *
   * @param x Input value
   * @return Normal CDF at x
   */
  float normalCDF(float x) const;

  /**
   * @brief Calculate normal PDF using fast approximation
   *
   * @param x Input value
   * @return Normal PDF at x
   */
  float normalPDF(float x) const;

  // Member variables
  float K_;
  float r_;
  float q_;
  float vol_;
  float vol2_; // vol^2, precomputed
  std::function<float(float)> B_;
  std::shared_ptr<num::IntegratorFloat> integrator_;
};

/**
 * @class EquationAFloat
 * @brief Single-precision implementation of Equation A from the ALO paper
 */
class EquationAFloat : public FixedPointEvaluatorFloat {
public:
  /**
   * @brief Constructor
   */
  EquationAFloat(float K, float r, float q, float vol,
                 const std::function<float(float)> &B,
                 std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Evaluate the fixed point equation
   */
  std::tuple<float, float, float> evaluate(float tau, float b) const override;

  /**
   * @brief Calculate derivatives of the fixed point equation
   */
  std::pair<float, float> derivatives(float tau, float b) const override;
};

/**
 * @class EquationBFloat
 * @brief Single-precision implementation of Equation B from the ALO paper
 */
class EquationBFloat : public FixedPointEvaluatorFloat {
public:
  /**
   * @brief Constructor
   */
  EquationBFloat(float K, float r, float q, float vol,
                 const std::function<float(float)> &B,
                 std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Evaluate the fixed point equation
   */
  std::tuple<float, float, float> evaluate(float tau, float b) const override;

  /**
   * @brief Calculate derivatives of the fixed point equation
   */
  std::pair<float, float> derivatives(float tau, float b) const override;
};

/**
 * @brief Create a single-precision fixed point evaluator
 */
std::shared_ptr<FixedPointEvaluatorFloat> createFixedPointEvaluatorFloat(
    char equation, float K, float r, float q, float vol,
    const std::function<float(float)> &B,
    std::shared_ptr<num::IntegratorFloat> integrator);

} // namespace mod
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_MOD_AMERICAN_H
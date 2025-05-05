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
 * @class AmericanOptionDouble
 * @brief Base class for double-precision American option pricing models
 *
 * This class provides common functionality for American option pricing
 * using the Anderson-Lake-Offengelden algorithm.
 */
class AmericanOptionDouble {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanOptionDouble(std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Destructor
   */
  virtual ~AmericanOptionDouble() = default;

  /**
   * @brief Calculate the early exercise premium
   */
  virtual double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolation> &boundary) const = 0;

  /**
   * @brief Calculate the early exercise boundary
   */
  virtual std::shared_ptr<num::ChebyshevInterpolation>
  calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::Integrator> fpIntegrator) const = 0;

  /**
   * @brief Calculate the maximum early exercise boundary value
   */
  virtual double xMax(double K, double r, double q) const = 0;

  /**
   * @brief Get the integrator
   */
  std::shared_ptr<num::Integrator> getIntegrator() const { return integrator_; }

protected:
  std::shared_ptr<num::Integrator> integrator_;
};

/**
 * @class AmericanPutDouble
 * @brief Double-precision American put option pricing model
 *
 * This class implements American put option pricing using the ALO algorithm.
 */
class AmericanPutDouble : public AmericanOptionDouble {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanPutDouble(std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Calculate the early exercise premium
   */
  double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolation> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   */
  std::shared_ptr<num::ChebyshevInterpolation> calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::Integrator> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   */
  double xMax(double K, double r, double q) const override;
};

/**
 * @class AmericanCallDouble
 * @brief Double-precision American call option pricing model
 *
 * This class implements American call option pricing using the ALO algorithm.
 */
class AmericanCallDouble : public AmericanOptionDouble {
public:
  /**
   * @brief Constructor
   *
   * @param integrator Integrator for calculating the early exercise premium
   */
  explicit AmericanCallDouble(std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Calculate the early exercise premium
   */
  double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolation> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   */
  std::shared_ptr<num::ChebyshevInterpolation> calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::Integrator> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   */
  double xMax(double K, double r, double q) const override;
};

/**
 * @class FixedPointEvaluatorDouble
 * @brief Base class for double-precision fixed point equation evaluators
 *
 * This class defines the interface for fixed point equation evaluators
 * used in the ALO algorithm to compute early exercise boundaries.
 */
class FixedPointEvaluatorDouble {
public:
  /**
   * @brief Constructor
   */
  FixedPointEvaluatorDouble(double K, double r, double q, double vol,
                            const std::function<double(double)> &B,
                            std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Destructor
   */
  virtual ~FixedPointEvaluatorDouble() = default;

  /**
   * @brief Evaluate the fixed point equation
   */
  virtual std::tuple<double, double, double> evaluate(double tau,
                                                      double b) const = 0;

  /**
   * @brief Calculate derivatives of the fixed point equation
   */
  virtual std::pair<double, double> derivatives(double tau, double b) const = 0;

protected:
  /**
   * @brief Calculate d1 and d2 terms for Black-Scholes
   */
  std::pair<double, double> d(double t, double z) const;

  /**
   * @brief Calculate normal CDF
   */
  double normalCDF(double x) const;

  /**
   * @brief Calculate normal PDF
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
 * @class EquationADouble
 * @brief Double-precision implementation of Equation A from the ALO paper
 */
class EquationADouble : public FixedPointEvaluatorDouble {
public:
  /**
   * @brief Constructor
   */
  EquationADouble(double K, double r, double q, double vol,
                  const std::function<double(double)> &B,
                  std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Evaluate the fixed point equation
   */
  std::tuple<double, double, double> evaluate(double tau,
                                              double b) const override;

  /**
   * @brief Calculate derivatives of the fixed point equation
   */
  std::pair<double, double> derivatives(double tau, double b) const override;
};

/**
 * @class EquationBDouble
 * @brief Double-precision implementation of Equation B from the ALO paper
 */
class EquationBDouble : public FixedPointEvaluatorDouble {
public:
  /**
   * @brief Constructor
   */
  EquationBDouble(double K, double r, double q, double vol,
                  const std::function<double(double)> &B,
                  std::shared_ptr<num::Integrator> integrator);

  /**
   * @brief Evaluate the fixed point equation
   */
  std::tuple<double, double, double> evaluate(double tau,
                                              double b) const override;

  /**
   * @brief Calculate derivatives of the fixed point equation
   */
  std::pair<double, double> derivatives(double tau, double b) const override;
};

/**
 * @brief Create a double-precision fixed point evaluator
 */
std::shared_ptr<FixedPointEvaluatorDouble>
createFixedPointEvaluatorDouble(char equation, double K, double r, double q,
                                double vol,
                                const std::function<double(double)> &B,
                                std::shared_ptr<num::Integrator> integrator);

/**
 * @class AmericanOptionSingle
 * @brief Base class for single-precision American option pricing models
 *
 * This class provides common functionality for American option pricing
 * using the Anderson-Lake-Offengelden algorithm with single-precision.
 */
class AmericanOptionSingle {
public:
  /**
   * @brief Constructor
   */
  explicit AmericanOptionSingle(
      std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Destructor
   */
  virtual ~AmericanOptionSingle() = default;

  /**
   * @brief Calculate the early exercise premium
   */
  virtual float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary)
      const = 0;

  /**
   * @brief Calculate the early exercise boundary
   */
  virtual std::shared_ptr<num::ChebyshevInterpolationFloat>
  calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegratorFloat> fpIntegrator) const = 0;

  /**
   * @brief Calculate the maximum early exercise boundary value
   */
  virtual float xMax(float K, float r, float q) const = 0;

  /**
   * @brief Get the integrator
   */
  std::shared_ptr<num::IntegratorFloat> getIntegrator() const {
    return integrator_;
  }

protected:
  std::shared_ptr<num::IntegratorFloat> integrator_;
};

/**
 * @class AmericanPutSingle
 * @brief Single-precision American put option pricing model
 *
 * This class implements American put option pricing using the ALO algorithm
 * with single-precision and optimized numerical methods.
 */
class AmericanPutSingle : public AmericanOptionSingle {
public:
  /**
   * @brief Constructor
   */
  explicit AmericanPutSingle(std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Calculate the early exercise premium
   */
  float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   */
  std::shared_ptr<num::ChebyshevInterpolationFloat> calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegratorFloat> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   */
  float xMax(float K, float r, float q) const override;

  /**
   * @brief Approximate American put price with Barone-Adesi-Whaley method
   */
  float approximatePrice(float S, float K, float r, float q, float vol,
                         float T) const;

  /**
   * @brief Calculate a batch of American put prices using approximation
   */
  std::vector<float> batchApproximatePrice(float S,
                                           const std::vector<float> &strikes,
                                           float r, float q, float vol,
                                           float T) const;
};

/**
 * @class AmericanCallSingle
 * @brief Single-precision American call option pricing model
 *
 * This class implements American call option pricing using the ALO algorithm
 * with single-precision and optimized numerical methods.
 */
class AmericanCallSingle : public AmericanOptionSingle {
public:
  /**
   * @brief Constructor
   */
  explicit AmericanCallSingle(std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Calculate the early exercise premium
   */
  float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationFloat> &boundary)
      const override;

  /**
   * @brief Calculate the early exercise boundary
   */
  std::shared_ptr<num::ChebyshevInterpolationFloat> calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegratorFloat> fpIntegrator) const override;

  /**
   * @brief Calculate the maximum early exercise boundary value
   */
  float xMax(float K, float r, float q) const override;

  /**
   * @brief Approximate American call price with Barone-Adesi-Whaley method
   */
  float approximatePrice(float S, float K, float r, float q, float vol,
                         float T) const;

  /**
   * @brief Calculate a batch of American call prices using approximation
   */
  std::vector<float> batchApproximatePrice(float S,
                                           const std::vector<float> &strikes,
                                           float r, float q, float vol,
                                           float T) const;
};

/**
 * @class FixedPointEvaluatorSingle
 * @brief Base class for single-precision fixed point equation evaluators
 *
 * This class defines the interface for fixed point equation evaluators
 * used in the ALO algorithm to compute early exercise boundaries with
 * single-precision.
 */
class FixedPointEvaluatorSingle {
public:
  /**
   * @brief Constructor
   */
  FixedPointEvaluatorSingle(float K, float r, float q, float vol,
                            const std::function<float(float)> &B,
                            std::shared_ptr<num::IntegratorFloat> integrator);

  /**
   * @brief Destructor
   */
  virtual ~FixedPointEvaluatorSingle() = default;

  /**
   * @brief Evaluate the fixed point equation
   */
  virtual std::tuple<float, float, float> evaluate(float tau,
                                                   float b) const = 0;

  /**
   * @brief Calculate derivatives of the fixed point equation
   */
  virtual std::pair<float, float> derivatives(float tau, float b) const = 0;

protected:
  /**
   * @brief Calculate d1 and d2 terms for Black-Scholes
   */
  std::pair<float, float> d(float t, float z) const;

  /**
   * @brief Calculate normal CDF using fast approximation
   */
  float normalCDF(float x) const;

  /**
   * @brief Calculate normal PDF using fast approximation
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
 * @class EquationASingle
 * @brief Single-precision implementation of Equation A from the ALO paper
 */
class EquationASingle : public FixedPointEvaluatorSingle {
public:
  /**
   * @brief Constructor
   */
  EquationASingle(float K, float r, float q, float vol,
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
 * @class EquationBSingle
 * @brief Single-precision implementation of Equation B from the ALO paper
 */
class EquationBSingle : public FixedPointEvaluatorSingle {
public:
  /**
   * @brief Constructor
   */
  EquationBSingle(float K, float r, float q, float vol,
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
std::shared_ptr<FixedPointEvaluatorSingle> createFixedPointEvaluatorSingle(
    char equation, float K, float r, float q, float vol,
    const std::function<float(float)> &B,
    std::shared_ptr<num::IntegratorFloat> integrator);

} // namespace mod
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_MOD_AMERICAN_H
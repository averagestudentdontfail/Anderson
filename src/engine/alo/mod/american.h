#ifndef ENGINE_ALO_MOD_AMERICAN_H
#define ENGINE_ALO_MOD_AMERICAN_H

#include "../num/chebyshev.h" 
#include "../num/integrate.h" 
#include <array>
#include <functional>
#include <memory>
#include <vector>
#include <tuple>

namespace engine {
namespace alo {
namespace mod {

/**
 * @class AmericanOptionDouble
 * @brief Base class for double-precision American option pricing models
 */
class AmericanOptionDouble {
public:
  explicit AmericanOptionDouble(std::shared_ptr<num::IntegrateDouble> integrate_pricing);
  virtual ~AmericanOptionDouble() = default;

  virtual double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolationDouble> &boundary) const = 0;

  virtual std::shared_ptr<num::ChebyshevInterpolationDouble>
  calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::IntegrateDouble> integrate_fp) const = 0; // separate integrator for FP

  virtual double xMax(double K, double r, double q) const = 0;
  std::shared_ptr<num::IntegrateDouble> getPricingIntegrate() const { return integrate_pricing_; }

protected:
  std::shared_ptr<num::IntegrateDouble> integrate_pricing_; // Renamed for clarity
};

/**
 * @class AmericanPutDouble
 */
class AmericanPutDouble : public AmericanOptionDouble {
public:
  explicit AmericanPutDouble(std::shared_ptr<num::IntegrateDouble> integrate_pricing);
  double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolationDouble> &boundary) const override;
  std::shared_ptr<num::ChebyshevInterpolationDouble> calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::IntegrateDouble> integrate_fp) const override;
  double xMax(double K, double r, double q) const override;
};

/**
 * @class AmericanCallDouble
 */
class AmericanCallDouble : public AmericanOptionDouble {
public:
  explicit AmericanCallDouble(std::shared_ptr<num::IntegrateDouble> integrate_pricing);
  double calculateEarlyExercisePremium(
      double S, double K, double r, double q, double vol, double T,
      const std::shared_ptr<num::ChebyshevInterpolationDouble> &boundary) const override;
  std::shared_ptr<num::ChebyshevInterpolationDouble> calculateExerciseBoundary(
      double S, double K, double r, double q, double vol, double T,
      size_t num_nodes, size_t num_iterations,
      std::shared_ptr<num::IntegrateDouble> integrate_fp) const override;
  double xMax(double K, double r, double q) const override;
};

/**
 * @class FixedPointEvaluatorDouble
 */
class FixedPointEvaluatorDouble {
public:
  FixedPointEvaluatorDouble(double K_val, double r_val, double q_val, double vol_val,
                            const std::function<double(double)> &B_func, // Boundary func
                            std::shared_ptr<num::IntegrateDouble> integrate_instance);
  virtual ~FixedPointEvaluatorDouble() = default;
  virtual std::tuple<double, double, double> evaluate(double tau, double b) const = 0; // Returns N, D, f_v
  virtual std::pair<double, double> derivatives(double tau, double b) const = 0;    // Returns N', D'

protected:
  std::pair<double, double> d_black_scholes(double t, double spot_ratio_K) const; // Renamed for clarity
  double normalCDF(double x) const;
  double normalPDF(double x) const;

  double K_value_; // Renamed members to avoid conflict with params
  double r_rate_;
  double q_yield_;
  double volatility_;
  double vol_sq_; 
  std::function<double(double)> B_boundary_func_; // Renamed
  std::shared_ptr<num::IntegrateDouble> integrate_fp_; // Renamed
};

/**
 * @class EquationADouble
 */
class EquationADouble : public FixedPointEvaluatorDouble {
public:
  EquationADouble(double K_val, double r_val, double q_val, double vol_val,
                  const std::function<double(double)> &B_func,
                  std::shared_ptr<num::IntegrateDouble> integrate_instance);
  std::tuple<double, double, double> evaluate(double tau, double b) const override;
  std::pair<double, double> derivatives(double tau, double b) const override;
};

/**
 * @class EquationBDouble
 */
class EquationBDouble : public FixedPointEvaluatorDouble {
public:
  EquationBDouble(double K_val, double r_val, double q_val, double vol_val,
                  const std::function<double(double)> &B_func,
                  std::shared_ptr<num::IntegrateDouble> integrate_instance);
  std::tuple<double, double, double> evaluate(double tau, double b) const override;
  std::pair<double, double> derivatives(double tau, double b) const override;
};

std::shared_ptr<FixedPointEvaluatorDouble>
createFixedPointEvaluatorDouble(char equation_type, double K_val, double r_val, double q_val, // Renamed params
                                double vol_val, const std::function<double(double)> &B_func,
                                std::shared_ptr<num::IntegrateDouble> integrate_instance);


// ========================================================================= //
//                  SINGLE PRECISION AMERICAN OPTIONS                        //
// ========================================================================= //

/**
 * @class AmericanOptionSingle
 */
class AmericanOptionSingle {
public:
  explicit AmericanOptionSingle(std::shared_ptr<num::IntegrateSingle> integrate_pricing);
  virtual ~AmericanOptionSingle() = default;

  virtual float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationSingle> &boundary) const = 0;

  virtual std::shared_ptr<num::ChebyshevInterpolationSingle>
  calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegrateSingle> integrate_fp) const = 0;

  virtual float xMax(float K, float r, float q) const = 0;
  std::shared_ptr<num::IntegrateSingle> getPricingIntegrate() const { return integrate_pricing_; }

protected:
  std::shared_ptr<num::IntegrateSingle> integrate_pricing_;
};

/**
 * @class AmericanPutSingle
 */
class AmericanPutSingle : public AmericanOptionSingle {
public:
  explicit AmericanPutSingle(std::shared_ptr<num::IntegrateSingle> integrate_pricing);
  float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationSingle> &boundary) const override;
  std::shared_ptr<num::ChebyshevInterpolationSingle> calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegrateSingle> integrate_fp) const override;
  float xMax(float K, float r, float q) const override;

  // Fast approximation methods specifically for single precision puts
  float approximatePriceBAW(float S, float K, float r, float q, float vol, float T) const;
  void batchApproximatePriceBAW(const std::vector<float>& S_vec, const std::vector<float>& K_vec,
                                const std::vector<float>& r_vec, const std::vector<float>& q_vec,
                                const std::vector<float>& vol_vec, const std::vector<float>& T_vec,
                                std::vector<float>& results_vec) const;
};

/**
 * @class AmericanCallSingle
 */
class AmericanCallSingle : public AmericanOptionSingle {
public:
  explicit AmericanCallSingle(std::shared_ptr<num::IntegrateSingle> integrate_pricing);
  float calculateEarlyExercisePremium(
      float S, float K, float r, float q, float vol, float T,
      const std::shared_ptr<num::ChebyshevInterpolationSingle> &boundary) const override;
  std::shared_ptr<num::ChebyshevInterpolationSingle> calculateExerciseBoundary(
      float S, float K, float r, float q, float vol, float T, size_t num_nodes,
      size_t num_iterations,
      std::shared_ptr<num::IntegrateSingle> integrate_fp) const override;
  float xMax(float K, float r, float q) const override;

  float approximatePriceBAW(float S, float K, float r, float q, float vol, float T) const;
   void batchApproximatePriceBAW(const std::vector<float>& S_vec, const std::vector<float>& K_vec,
                                const std::vector<float>& r_vec, const std::vector<float>& q_vec,
                                const std::vector<float>& vol_vec, const std::vector<float>& T_vec,
                                std::vector<float>& results_vec) const;
};

/**
 * @class FixedPointEvaluatorSingle
 */
class FixedPointEvaluatorSingle {
public:
  FixedPointEvaluatorSingle(float K_val, float r_val, float q_val, float vol_val,
                            const std::function<float(float)> &B_func,
                            std::shared_ptr<num::IntegrateSingle> integrate_instance);
  virtual ~FixedPointEvaluatorSingle() = default;
  virtual std::tuple<float, float, float> evaluate(float tau, float b) const = 0;
  virtual std::pair<float, float> derivatives(float tau, float b) const = 0;

protected:
  std::pair<float, float> d_black_scholes(float t, float spot_ratio_K) const; // Renamed
  float normalCDF(float x) const; // Uses num::fast_normal_cdf
  float normalPDF(float x) const; // Uses num::fast_normal_pdf

  float K_value_;
  float r_rate_;
  float q_yield_;
  float volatility_;
  float vol_sq_; 
  std::function<float(float)> B_boundary_func_;
  std::shared_ptr<num::IntegrateSingle> integrate_fp_;
};

/**
 * @class EquationASingle
 */
class EquationASingle : public FixedPointEvaluatorSingle {
public:
  EquationASingle(float K_val, float r_val, float q_val, float vol_val,
                  const std::function<float(float)> &B_func,
                  std::shared_ptr<num::IntegrateSingle> integrate_instance);
  std::tuple<float, float, float> evaluate(float tau, float b) const override;
  std::pair<float, float> derivatives(float tau, float b) const override;
};

/**
 * @class EquationBSingle
 */
class EquationBSingle : public FixedPointEvaluatorSingle {
public:
  EquationBSingle(float K_val, float r_val, float q_val, float vol_val,
                  const std::function<float(float)> &B_func,
                  std::shared_ptr<num::IntegrateSingle> integrate_instance);
  std::tuple<float, float, float> evaluate(float tau, float b) const override;
  std::pair<float, float> derivatives(float tau, float b) const override;
};

std::shared_ptr<FixedPointEvaluatorSingle> createFixedPointEvaluatorSingle(
    char equation_type, float K_val, float r_val, float q_val, float vol_val,
    const std::function<float(float)> &B_func,
    std::shared_ptr<num::IntegrateSingle> integrate_instance);

} // namespace mod
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_MOD_AMERICAN_H
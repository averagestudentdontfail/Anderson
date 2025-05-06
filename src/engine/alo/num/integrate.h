#ifndef ENGINE_ALO_NUM_INTEGRATE_H
#define ENGINE_ALO_NUM_INTEGRATE_H

#include <functional>
#include <memory>
#include <string>

namespace engine {
namespace alo {
namespace num {

/**
 * @brief Abstract base class for double-precision numerical integration
 *
 * This class defines the interface for integration methods used
 * in the ALO engine. The implementations must follow deterministic
 * execution principles to ensure consistent results.
 */
class IntegrateDouble {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~IntegrateDouble() = default;

  /**
   * @brief Integrate a function over an interval
   *
   * @param f Function to integrate
   * @param a Lower bound
   * @param b Upper bound
   * @return Approximate integral value
   */
  virtual double integrate(const std::function<double(double)> &f, double a,
                           double b) const = 0;

  /**
   * @brief Get the name of the Integrate
   *
   * @return Integrate name
   */
  virtual std::string name() const = 0;
};

/**
 * @brief Create a double-precision Integrate of the specified type
 *
 * Factory function to create instances of various Integrate implementations.
 *
 * @param scheme_type Type of integration scheme ("GaussLegendre", "TanhSinh",
 * "GSLQAGS")
 * @param order Number of integration points (for fixed-point schemes)
 * @param tolerance Error tolerance (for adaptive schemes)
 * @return Shared pointer to Integrate instance
 */
std::shared_ptr<IntegrateDouble>
createIntegrateDouble(const std::string &scheme_type, size_t order = 0,
                       double tolerance = 0.0);

/**
 * @brief Abstract base class for single-precision numerical integration
 */
class IntegrateSingle {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~IntegrateSingle() = default;

  /**
   * @brief Integrate a function over an interval
   *
   * @param f Function to integrate
   * @param a Lower bound
   * @param b Upper bound
   * @return Approximate integral value
   */
  virtual float integrate(const std::function<float(float)> &f, float a,
                          float b) const = 0;

  /**
   * @brief Get the name of the Integrate
   *
   * @return Integrate name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Batch integrate a function at multiple intervals
   *
   * @param f Function to integrate
   * @param a Vector of lower bounds
   * @param b Vector of upper bounds
   * @param results Vector to store results
   */
  virtual void batchIntegrate(const std::function<float(float)> &f,
                              const std::vector<float> &a,
                              const std::vector<float> &b,
                              std::vector<float> &results) const;
};

/**
 * @brief Create a single-precision Integrate of the specified type
 *
 * @param scheme_type Type of integration scheme ("GaussLegendre", "Adaptive")
 * @param order Number of integration points (for fixed-point schemes)
 * @param tolerance Error tolerance (for adaptive schemes)
 * @return Shared pointer to single-precision Integrate instance
 */
std::shared_ptr<IntegrateSingle>
createIntegrateSingle(const std::string &scheme_type, size_t order = 0,
                       float tolerance = 0.0f);

} // namespace num
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_NUM_INTEGRATE_H
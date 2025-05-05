#ifndef ENGINE_ALO_NUM_INTEGRATOR_H
#define ENGINE_ALO_NUM_INTEGRATOR_H

#include <functional>
#include <memory>
#include <string>

namespace engine {
namespace alo {
namespace num {

/**
 * @brief Abstract base class for numerical integration
 *
 * This class defines the interface for integration methods used
 * in the ALO engine. The implementations must follow deterministic
 * execution principles to ensure consistent results.
 */
class Integrator {
public:
  /**
   * @brief Virtual destructor
   */
  virtual ~Integrator() = default;

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
   * @brief Get the name of the integrator
   *
   * @return Integrator name
   */
  virtual std::string name() const = 0;
};

/**
 * @brief Create an integrator of the specified type
 *
 * Factory function to create instances of various integrator implementations.
 *
 * @param scheme_type Type of integration scheme ("GaussLegendre", "TanhSinh",
 * "GSLQAGS")
 * @param order Number of integration points (for fixed-point schemes)
 * @param tolerance Error tolerance (for adaptive schemes)
 * @return Shared pointer to integrator instance
 */
std::shared_ptr<Integrator> createIntegrator(const std::string &scheme_type,
                                             size_t order = 0,
                                             double tolerance = 0.0);

/**
 * @brief Abstract base class for single-precision numerical integration
 */
class IntegratorFloat {
public:
  virtual ~IntegratorFloat() = default;
  virtual float integrate(const std::function<float(float)> &f, float a,
                          float b) const = 0;
  virtual std::string name() const = 0;

  // Optional batch integration
  virtual void batchIntegrate(const std::function<float(float)> &f,
                              const std::vector<float> &a,
                              const std::vector<float> &b,
                              std::vector<float> &results) const;
};

// Factory function for float integrators
std::shared_ptr<IntegratorFloat>
createIntegratorFloat(const std::string &scheme_type, size_t order = 0,
                      float tolerance = 0.0f);

} // namespace num
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_NUM_INTEGRATOR_H
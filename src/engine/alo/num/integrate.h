#ifndef ENGINE_ALO_NUM_INTEGRATE_H
#define ENGINE_ALO_NUM_INTEGRATE_H

#include <functional>
#include <memory>
#include <string>
#include <vector> 

namespace engine {
namespace alo {
namespace num {

/**
 * @brief Abstract base class for double-precision numerical integration
 */
class IntegrateDouble { 
public:
  virtual ~IntegrateDouble() = default;
  virtual double integrate(const std::function<double(double)> &f, double a,
                           double b) const = 0;
  virtual std::string name() const = 0;
};

/**
 * @brief Create a double-precision Integrator of the specified type
 */
std::shared_ptr<IntegrateDouble> 
createIntegrateDouble(const std::string &scheme_type, size_t order = 0, 
                      double tolerance = 0.0);

/**
 * @brief Abstract base class for single-precision numerical integration
 */
class IntegrateSingle { 
public:
  virtual ~IntegrateSingle() = default;
  virtual float integrate(const std::function<float(float)> &f, float a,
                          float b) const = 0;
  virtual std::string name() const = 0;

  virtual void batchIntegrate(const std::function<float(float)> &f,
                              const std::vector<float> &a,
                              const std::vector<float> &b,
                              std::vector<float> &results) const;
};

/**
 * @brief Create a single-precision Integrator of the specified type
 */
std::shared_ptr<IntegrateSingle> 
createIntegrateSingle(const std::string &scheme_type, size_t order = 0, 
                      float tolerance = 0.0f);

} // namespace num
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_NUM_INTEGRATE_H
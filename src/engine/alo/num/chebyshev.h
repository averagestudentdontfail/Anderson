#ifndef ENGINE_ALO_NUM_CHEBYSHEV_H
#define ENGINE_ALO_NUM_CHEBYSHEV_H

#include <functional>
#include <immintrin.h> 
#include <memory>
#include <vector>

namespace engine {
namespace alo {
namespace num {

/**
 * @brief Chebyshev polynomial kind
 */
enum ChebyshevKind {
  FIRST_KIND, // Chebyshev polynomials of the first kind T_n(x)
  SECOND_KIND // Chebyshev polynomials of the second kind U_n(x)
};

/**
 * @brief Chebyshev polynomial interpolation
 *
 * This class implements Chebyshev polynomial interpolation for
 * representing functions with high accuracy, particularly the
 * early exercise boundary in American option pricing.
 */
class ChebyshevInterpolation {
public:
  /**
   * @brief Constructor from function
   *
   * Creates a Chebyshev interpolation from a function.
   *
   * @param num_points Number of interpolation points
   * @param func Function to interpolate
   * @param kind Chebyshev polynomial kind
   * @param domain_start Start of domain
   * @param domain_end End of domain
   */
  ChebyshevInterpolation(size_t num_points,
                         const std::function<double(double)> &func,
                         ChebyshevKind kind = SECOND_KIND,
                         double domain_start = -1.0, double domain_end = 1.0);

  /**
   * @brief Constructor from nodes and values
   *
   * Creates a Chebyshev interpolation from nodes and function values.
   *
   * @param nodes Interpolation nodes
   * @param values Function values at nodes
   * @param kind Chebyshev polynomial kind
   * @param domain_start Start of domain
   * @param domain_end End of domain
   */
  ChebyshevInterpolation(const std::vector<double> &nodes,
                         const std::vector<double> &values,
                         ChebyshevKind kind = SECOND_KIND,
                         double domain_start = -1.0, double domain_end = 1.0);

  /**
   * @brief Destructor
   */
  ~ChebyshevInterpolation() = default;

  /**
   * @brief Evaluate interpolation at a point
   *
   * @param x Evaluation point
   * @param extrapolate Whether to allow extrapolation
   * @return Interpolated value
   */
  double operator()(double x, bool extrapolate = false) const;

  /**
   * @brief Update the function values at nodes
   *
   * This method allows updating the interpolation with new function
   * values without changing the nodes.
   *
   * @param values New function values at the existing nodes
   */
  void updateValues(const std::vector<double> &values);

  /**
   * @brief Get number of interpolation points
   *
   * @return Number of points
   */
  size_t getNumPoints() const { return num_points_; }

  /**
   * @brief Get Chebyshev nodes
   *
   * @return Vector of nodes
   */
  const std::vector<double> &getNodes() const { return nodes_; }

  /**
   * @brief Get function values at nodes
   *
   * @return Vector of values
   */
  const std::vector<double> &getValues() const { return values_; }

  /**
   * @brief Get Chebyshev coefficients
   *
   * @return Vector of coefficients
   */
  const std::vector<double> &getCoefficients() const { return coefficients_; }

  /**
   * @brief Get domain start
   *
   * @return Start of domain
   */
  double getDomainStart() const { return domain_start_; }

  /**
   * @brief Get domain end
   *
   * @return End of domain
   */
  double getDomainEnd() const { return domain_end_; }

  /**
   * @brief Get Chebyshev kind
   *
   * @return Chebyshev polynomial kind
   */
  ChebyshevKind getKind() const { return kind_; }

private:
  /**
   * @brief Initialize nodes based on Chebyshev kind
   */
  void initializeNodes();

  /**
   * @brief Compute Chebyshev coefficients
   */
  void computeCoefficients();

  /**
   * @brief Map a point from original domain to standard domain [-1, 1]
   *
   * @param x Point in original domain
   * @return Point in standard domain
   */
  double mapToStandardDomain(double x) const;

  /**
   * @brief Map a point from standard domain [-1, 1] to original domain
   *
   * @param t Point in standard domain
   * @return Point in original domain
   */
  double mapFromStandardDomain(double t) const;

  /**
   * @brief Evaluate Chebyshev polynomial
   *
   * @param n Polynomial degree
   * @param x Evaluation point
   * @param kind Chebyshev polynomial kind
   * @return Polynomial value
   */
  double evaluateChebyshev(int n, double x, ChebyshevKind kind) const;

  // Member variables
  size_t num_points_;
  ChebyshevKind kind_;
  double domain_start_;
  double domain_end_;
  std::vector<double> nodes_;
  std::vector<double> values_;
  std::vector<double> coefficients_;
};

/**
 * @brief SIMD-accelerated Chebyshev interpolation
 *
 * This class provides SIMD acceleration for evaluating Chebyshev
 * interpolation at multiple points simultaneously.
 */
class SimdChebyshevInterpolation {
public:
  /**
   * @brief Constructor from standard interpolation
   *
   * @param interp Base Chebyshev interpolation
   */
  explicit SimdChebyshevInterpolation(const ChebyshevInterpolation &interp);

  /**
   * @brief Destructor
   */
  ~SimdChebyshevInterpolation() = default;

  /**
   * @brief Evaluate interpolation at multiple points
   *
   * @param x Vector of evaluation points
   * @param y Vector to store results
   * @param extrapolate Whether to allow extrapolation
   */
  void evaluate(const std::vector<double> &x, std::vector<double> &y,
                bool extrapolate = false) const;

  /**
   * @brief Evaluate interpolation at 4 points using AVX2
   *
   * @param x Array of 4 evaluation points
   * @return Array of 4 interpolated values
   */
  std::array<double, 4> evaluate4(const std::array<double, 4> &x,
                                  bool extrapolate = false) const;

private:
  const ChebyshevInterpolation &interp_;
};

/**
 * @brief Create a Chebyshev interpolation
 *
 * @param num_points Number of interpolation points
 * @param func Function to interpolate
 * @param kind Chebyshev polynomial kind
 * @param domain_start Start of domain
 * @param domain_end End of domain
 * @return Shared pointer to Chebyshev interpolation
 */
std::shared_ptr<ChebyshevInterpolation> createChebyshevInterpolation(
    size_t num_points, const std::function<double(double)> &func,
    ChebyshevKind kind = SECOND_KIND, double domain_start = -1.0,
    double domain_end = 1.0);

/**
 * @brief Create a Chebyshev interpolation from nodes and values
 *
 * @param nodes Interpolation nodes
 * @param values Function values at nodes
 * @param kind Chebyshev polynomial kind
 * @param domain_start Start of domain
 * @param domain_end End of domain
 * @return Shared pointer to Chebyshev interpolation
 */
std::shared_ptr<ChebyshevInterpolation> createChebyshevInterpolation(
    const std::vector<double> &nodes, const std::vector<double> &values,
    ChebyshevKind kind = SECOND_KIND, double domain_start = -1.0,
    double domain_end = 1.0);

/**
 * @brief Single-precision Chebyshev polynomial interpolation
 */
class ChebyshevInterpolationFloat {
public:
  ChebyshevInterpolationFloat(size_t num_points,
                              const std::function<float(float)> &func,
                              ChebyshevKind kind = SECOND_KIND,
                              float domain_start = -1.0f,
                              float domain_end = 1.0f);

  ChebyshevInterpolationFloat(const std::vector<float> &nodes,
                              const std::vector<float> &values,
                              ChebyshevKind kind = SECOND_KIND,
                              float domain_start = -1.0f,
                              float domain_end = 1.0f);

  ~ChebyshevInterpolationFloat() = default;

  float operator()(float x, bool extrapolate = false) const;
  void updateValues(const std::vector<float> &values);

  // Getters
  size_t getNumPoints() const { return num_points_; }
  const std::vector<float> &getNodes() const { return nodes_; }
  const std::vector<float> &getValues() const { return values_; }
  const std::vector<float> &getCoefficients() const { return coefficients_; }
  float getDomainStart() const { return domain_start_; }
  float getDomainEnd() const { return domain_end_; }
  ChebyshevKind getKind() const { return kind_; }

private:
  void initializeNodes();
  void computeCoefficients();
  float mapToStandardDomain(float x) const;
  float mapFromStandardDomain(float t) const;
  float evaluateChebyshev(int n, float x, ChebyshevKind kind) const;

  size_t num_points_;
  ChebyshevKind kind_;
  float domain_start_;
  float domain_end_;
  std::vector<float> nodes_;
  std::vector<float> values_;
  std::vector<float> coefficients_;
};

/**
 * @brief SIMD-accelerated Chebyshev interpolation using single-precision
 */
class SimdChebyshevInterpolationFloat {
public:
  explicit SimdChebyshevInterpolationFloat(
      const ChebyshevInterpolationFloat &interp);
  ~SimdChebyshevInterpolationFloat() = default;

  void evaluate(const std::vector<float> &x, std::vector<float> &y,
                bool extrapolate = false) const;
  std::array<float, 8> evaluate8(const std::array<float, 8> &x,
                                 bool extrapolate = false) const;

private:
  const ChebyshevInterpolationFloat &interp_;
};

// Factory functions for float version
std::shared_ptr<ChebyshevInterpolationFloat> createChebyshevInterpolationFloat(
    size_t num_points, const std::function<float(float)> &func,
    ChebyshevKind kind = SECOND_KIND, float domain_start = -1.0f,
    float domain_end = 1.0f);

std::shared_ptr<ChebyshevInterpolationFloat> createChebyshevInterpolationFloat(
    const std::vector<float> &nodes, const std::vector<float> &values,
    ChebyshevKind kind = SECOND_KIND, float domain_start = -1.0f,
    float domain_end = 1.0f);

} // namespace num
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_NUM_CHEBYSHEV_H
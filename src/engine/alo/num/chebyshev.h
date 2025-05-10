// --- START OF FILE num/chebyshev.h ---

#ifndef ENGINE_ALO_NUM_CHEBYSHEV_H
#define ENGINE_ALO_NUM_CHEBYSHEV_H

#include <functional>
#include <immintrin.h> // For __m256d, __m256
#include <memory>
#include <vector>
#include <array> // For std::array in SIMD versions

namespace engine {
namespace alo {
namespace num {

/**
 * @brief Chebyshev polynomial kind
 */
enum ChebyshevKind {
  FIRST_KIND,  // Chebyshev polynomials of the first kind T_n(x)
  SECOND_KIND  // Chebyshev polynomials of the second kind U_n(x) - More common for interpolation nodes
};

/**
 * @brief Double-precision Chebyshev polynomial interpolation
 */
class ChebyshevInterpolationDouble {
public:
  ChebyshevInterpolationDouble(size_t num_points,
                               const std::function<double(double)> &func,
                               ChebyshevKind kind = SECOND_KIND, // Default to second kind nodes
                               double domain_start = -1.0,
                               double domain_end = 1.0);

  ChebyshevInterpolationDouble(const std::vector<double> &nodes, // Nodes in [-1,1] or original domain? Assuming standard [-1,1]
                               const std::vector<double> &values_at_nodes, // Values at these nodes
                               ChebyshevKind kind = SECOND_KIND,
                               double domain_start = -1.0, // Original domain mapping
                               double domain_end = 1.0);

  ~ChebyshevInterpolationDouble() = default;

  double operator()(double x, bool extrapolate = false) const;
  void updateValues(const std::vector<double> &new_values_at_nodes);

  size_t getNumPoints() const { return num_points_; }
  const std::vector<double> &getNodes() const { return nodes_in_standard_domain_; } // Nodes always in [-1,1]
  const std::vector<double> &getValuesAtNodes() const { return values_at_nodes_; }
  const std::vector<double> &getCoefficients() const { return coefficients_; }
  double getDomainStart() const { return domain_start_; }
  double getDomainEnd() const { return domain_end_; }
  ChebyshevKind getKind() const { return kind_; }

private:
  void initializeNodes();
  void computeCoefficients();
  double mapToStandardDomain(double x) const;
  double mapFromStandardDomain(double t) const;
  // evaluateChebyshev was likely a helper, can be static or private utility if needed
  // static double evaluateNthChebyshev(int n, double x, ChebyshevKind kind);

  size_t num_points_;
  ChebyshevKind kind_;
  double domain_start_;
  double domain_end_;
  std::vector<double> nodes_in_standard_domain_; // Stores nodes in [-1, 1]
  std::vector<double> values_at_nodes_;          // Stores f(mapFromStandardDomain(nodes_[i]))
  std::vector<double> coefficients_;
};

/**
 * @brief SIMD-accelerated double-precision Chebyshev interpolation
 */
class SimdChebyshevInterpolationDouble {
public:
  explicit SimdChebyshevInterpolationDouble(const ChebyshevInterpolationDouble &interp);
  ~SimdChebyshevInterpolationDouble() = default;

  void evaluate(const std::vector<double> &x_vec, std::vector<double> &y_vec, // Changed names for clarity
                bool extrapolate = false) const;
  std::array<double, 4> evaluate4(const std::array<double, 4> &x_arr, // Changed name for clarity
                                  bool extrapolate = false) const;
private:
  const ChebyshevInterpolationDouble &interp_ref_; // Store as reference
};

std::shared_ptr<ChebyshevInterpolationDouble>
createChebyshevInterpolationDouble(size_t num_points,
                                   const std::function<double(double)> &func,
                                   ChebyshevKind kind = SECOND_KIND,
                                   double domain_start = -1.0,
                                   double domain_end = 1.0);

std::shared_ptr<ChebyshevInterpolationDouble>
createChebyshevInterpolationDouble(const std::vector<double> &nodes_in_standard_domain, // Explicitly standard
                                   const std::vector<double> &values_at_nodes,
                                   ChebyshevKind kind = SECOND_KIND,
                                   double domain_start = -1.0,
                                   double domain_end = 1.0);

/**
 * @brief Single-precision Chebyshev polynomial interpolation
 */
class ChebyshevInterpolationSingle {
public:
  ChebyshevInterpolationSingle(size_t num_points,
                               const std::function<float(float)> &func,
                               ChebyshevKind kind = SECOND_KIND,
                               float domain_start = -1.0f,
                               float domain_end = 1.0f);

  ChebyshevInterpolationSingle(const std::vector<float> &nodes, // Assuming nodes in [-1,1]
                               const std::vector<float> &values_at_nodes,
                               ChebyshevKind kind = SECOND_KIND,
                               float domain_start = -1.0f,
                               float domain_end = 1.0f);

  ~ChebyshevInterpolationSingle() = default;

  float operator()(float x, bool extrapolate = false) const;
  void updateValues(const std::vector<float> &new_values_at_nodes);

  size_t getNumPoints() const { return num_points_; }
  const std::vector<float> &getNodes() const { return nodes_in_standard_domain_; }
  const std::vector<float> &getValuesAtNodes() const { return values_at_nodes_; }
  const std::vector<float> &getCoefficients() const { return coefficients_; }
  float getDomainStart() const { return domain_start_; }
  float getDomainEnd() const { return domain_end_; }
  ChebyshevKind getKind() const { return kind_; }

private:
  void initializeNodes();
  void computeCoefficients();
  float mapToStandardDomain(float x) const;
  float mapFromStandardDomain(float t) const;
  // static float evaluateNthChebyshev(int n, float x, ChebyshevKind kind);

  size_t num_points_;
  ChebyshevKind kind_;
  float domain_start_;
  float domain_end_;
  std::vector<float> nodes_in_standard_domain_; 
  std::vector<float> values_at_nodes_;      
  std::vector<float> coefficients_;
};

/**
 * @brief SIMD-accelerated single-precision Chebyshev interpolation
 */
class SimdChebyshevInterpolationSingle {
public:
  explicit SimdChebyshevInterpolationSingle(const ChebyshevInterpolationSingle &interp);
  ~SimdChebyshevInterpolationSingle() = default;

  void evaluate(const std::vector<float> &x_vec, std::vector<float> &y_vec,
                bool extrapolate = false) const;
  std::array<float, 8> evaluate8(const std::array<float, 8> &x_arr, // AVX2 processes 8 floats
                                 bool extrapolate = false) const;
private:
  const ChebyshevInterpolationSingle &interp_ref_; // Store as reference
};

std::shared_ptr<ChebyshevInterpolationSingle>
createChebyshevInterpolationSingle(size_t num_points,
                                   const std::function<float(float)> &func,
                                   ChebyshevKind kind = SECOND_KIND,
                                   float domain_start = -1.0f,
                                   float domain_end = 1.0f);

std::shared_ptr<ChebyshevInterpolationSingle>
createChebyshevInterpolationSingle(const std::vector<float> &nodes_in_standard_domain, // Explicitly standard
                                   const std::vector<float> &values_at_nodes,
                                   ChebyshevKind kind = SECOND_KIND,
                                   float domain_start = -1.0f,
                                   float domain_end = 1.0f);

} // namespace num
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_NUM_CHEBYSHEV_H
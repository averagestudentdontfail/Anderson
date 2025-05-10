#include "chebyshev.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <numeric> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace engine {
namespace alo {
namespace num {

ChebyshevInterpolationDouble::ChebyshevInterpolationDouble(
    size_t num_points, const std::function<double(double)> &func,
    ChebyshevKind kind, double domain_start, double domain_end)
    : num_points_(num_points), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end) {

  if (num_points_ < 2) {
    throw std::invalid_argument("ChebyshevInterpolationDouble: num_points must be at least 2");
  }
  if (domain_end_ <= domain_start_) {
    throw std::invalid_argument("ChebyshevInterpolationDouble: domain_end must be greater than domain_start");
  }

  initializeNodes(); // Initializes nodes_in_standard_domain_

  values_at_nodes_.resize(num_points_);
  for (size_t i = 0; i < num_points_; ++i) {
    values_at_nodes_[i] = func(mapFromStandardDomain(nodes_in_standard_domain_[i]));
  }
  computeCoefficients();
}

ChebyshevInterpolationDouble::ChebyshevInterpolationDouble(
    const std::vector<double> &nodes_in_standard_domain, // Expected to be in [-1,1]
    const std::vector<double> &values_at_nodes,
    ChebyshevKind kind, double domain_start, double domain_end)
    : num_points_(nodes_in_standard_domain.size()), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end), nodes_in_standard_domain_(nodes_in_standard_domain),
      values_at_nodes_(values_at_nodes) {

  if (nodes_in_standard_domain_.size() != values_at_nodes_.size()) {
    throw std::invalid_argument("ChebyshevInterpolationDouble: nodes and values_at_nodes must have the same size");
  }
  if (num_points_ < 2) {
    throw std::invalid_argument("ChebyshevInterpolationDouble: num_points must be at least 2");
  }
  if (domain_end_ <= domain_start_) {
    throw std::invalid_argument("ChebyshevInterpolationDouble: domain_end must be greater than domain_start");
  }
  // TODO: Optionally validate that provided nodes are indeed correct Chebyshev nodes for the kind.
  computeCoefficients();
}

void ChebyshevInterpolationDouble::initializeNodes() {
  nodes_in_standard_domain_.resize(num_points_);
  if (num_points_ == 1) { // Special case, though constructor checks for num_points_ < 2
      nodes_in_standard_domain_[0] = 0.0;
      return;
  }

  if (kind_ == FIRST_KIND) {
    // Roots of T_n(x): cos((2j+1)π / (2n)) for j = 0, ..., n-1
    for (size_t j = 0; j < num_points_; ++j) {
      nodes_in_standard_domain_[j] = std::cos(M_PI * (2.0 * static_cast<double>(j) + 1.0) / (2.0 * static_cast<double>(num_points_)));
    }
  } else { // SECOND_KIND (Extrema of T_n(x) or roots of U_{n-1}(x) scaled)
    // Extrema of T_{n-1}(x): cos(jπ / (n-1)) for j = 0, ..., n-1
    for (size_t j = 0; j < num_points_; ++j) {
      nodes_in_standard_domain_[j] = std::cos(M_PI * static_cast<double>(j) / (static_cast<double>(num_points_) - 1.0));
    }
     // Ensure perfect -1 and 1 at ends for second kind due to potential floating point noise
    if (num_points_ > 1) {
        nodes_in_standard_domain_[0] = 1.0; // cos(0)
        nodes_in_standard_domain_[num_points_-1] = -1.0; // cos(pi)
        // Standard convention is nodes from 1 down to -1 for cos(j*pi/(N-1))
        // If you want them sorted from -1 to 1, you can reverse, but then adjust coefficient calculation or usage.
        // The ALO paper typically implies x_i are sorted. The QuantLib `ChebyshevInterpolation` sorts them.
        // For consistency with standard literature that defines nodes via cos(theta) where theta increases,
        // this order (1 down to -1) is common. If sorted needed, apply std::sort and std::reverse.
        // Let's sort them for general consistency, as Clenshaw expects evaluation on [-1,1] mapped from sorted domain.
        std::sort(nodes_in_standard_domain_.begin(), nodes_in_standard_domain_.end());
    }
  }
}

void ChebyshevInterpolationDouble::computeCoefficients() {
  coefficients_.assign(num_points_, 0.0);
  const size_t N = num_points_;

  // Using formula C_k = (2/N) * sum_{j=0}^{N-1} f(x_j) T_k(x_j)
  // with x_j = cos(theta_j)
  // T_k(x_j) = cos(k * theta_j)
  // For nodes of the SECOND kind (extrema), theta_j = j*pi/(N-1)
  // For nodes of the FIRST kind (roots), theta_j = (2j+1)*pi/(2N)

  // This is a Discrete Cosine Transform (DCT).
  // Standard libraries (like FFTW) can do this efficiently.
  // Here's a direct summation which is O(N^2). For small N, it's fine.

  for (size_t k = 0; k < N; ++k) {
    double sum = 0.0;
    for (size_t j = 0; j < N; ++j) {
      // T_k(nodes_in_standard_domain_[j])
      // nodes_in_standard_domain_[j] = cos(theta_j_std)
      // We need T_k(cos(theta_j_std)) = cos(k * theta_j_std)
      // where theta_j_std is arccos(nodes_in_standard_domain_[j])
      // If nodes_in_standard_domain_ were generated as cos( (pi * j) / (N-1) ) for kind 2,
      // then T_k(nodes_in_standard_domain_[j]) = cos( k * pi * j / (N-1) )
      double T_k_at_node_j;
      if (kind_ == FIRST_KIND) {
           // theta_j for first kind roots: (2j+1)π / (2N)
           // The nodes_in_standard_domain_ already stores cos((2j+1)π / (2N)) assuming original j indexing for roots
           // But to use the coefficient formula correctly, we need the angles.
           // It's simpler if we stick to the definition for the chosen node type.
           // Formula with T_k(x_j) is general.
           // T_k(x) = cos(k * acos(x))
           // For practical computation, direct sum for coefficients of f(cos theta)
           // c_k = (2/N) sum f(cos( (j+0.5)pi/N )) cos( k(j+0.5)pi/N )
           double angle_j = M_PI * (static_cast<double>(j) + 0.5) / static_cast<double>(N);
           T_k_at_node_j = std::cos(static_cast<double>(k) * angle_j);
           // sum += values_at_nodes_[j] * T_k_at_node_j; // if values_at_nodes_ corresponds to this indexing of j
           // Assuming values_at_nodes_ corresponds to the stored nodes_in_standard_domain_
           // and nodes_in_standard_domain_ are already the Chebyshev nodes for the kind.
           T_k_at_node_j = std::cos(k * std::acos(nodes_in_standard_domain_[j])); // General form
      } else { // SECOND_KIND (extrema)
           // theta_j for second kind extrema: jπ / (N-1)
           // nodes_in_standard_domain_[j] is cos(jπ / (N-1)) if sorted descending, or cos((N-1-j)π/(N-1)) if ascending
           // Let's use the sorted ascending nodes: arccos will give the correct angle.
           T_k_at_node_j = std::cos(k * std::acos(nodes_in_standard_domain_[j]));
      }
       sum += values_at_nodes_[j] * T_k_at_node_j;
    }
    coefficients_[k] = (2.0 / static_cast<double>(N)) * sum;
  }

  coefficients_[0] /= 2.0; // c0' = c0/2
  if (N > 1 && kind_ == SECOND_KIND) { // For second kind (extrema) points, last coeff is also halved
    // This halving of c_{N-1} is for specific forms of DCT, often not needed if Clenshaw is used carefully.
    // The standard definition of coefficients based on f(cos theta) usually only halves c0.
    // Let's stick to standard definition unless a specific DCT variant is strictly followed.
    // coefficients_[N-1] /= 2.0; // QuantLib's ChebyshevInterpolation does this if using Barycentric.
                               // For Clenshaw with T_k, standard coeffs are fine.
  }
}


double ChebyshevInterpolationDouble::operator()(double x, bool extrapolate) const {
  double t = mapToStandardDomain(x);

  if (!extrapolate && (t < -1.0 - 1e-12 || t > 1.0 + 1e-12)) { // Add tolerance for floating point
    // If t is very slightly outside due to precision, clamp it.
    if (t < -1.0) t = -1.0;
    if (t > 1.0) t = 1.0;
    // else throw std::domain_error("ChebyshevInterpolationDouble: Evaluation point outside interpolation domain");
  }
   // Clamp t to [-1, 1] for stability if extrapolation is allowed or if it's slightly outside due to precision
  t = std::max(-1.0, std::min(1.0, t));


  // Clenshaw's algorithm for sum c_k T_k(t)
  // Note: coefficients_[0] is c0', which is actual c0/2. Clenshaw usually expects c0.
  // b_N = 0, b_{N+1} = 0
  // b_k = c_k + 2*t*b_{k+1} - b_{k+2} for k = N-1 ... 1
  // result = c_0 + t*b_1 - b_2  (if c_0 is full c0)
  // OR result = (b0 - b2)/2 if recurrence is slightly different.
  // Standard Clenshaw for P(x) = sum_{k=0}^{N-1} coeff_k T_k(x)
  // Let coeff'_0 = coeff_0 / 2, and coeff'_k = coeff_k for k > 0
  // y = coeff'_{N-1}
  // y = coeff'_{N-2} + 2*t*y
  // for k = N-3 down to 0:
  //   y = coeff'_k + 2*t*y_prev - y_prev_prev
  // This form can be unstable.
  // More stable form: (Press et al., Numerical Recipes)
  // d1 = 0, d2 = 0
  // For j = N-1 down to 1:
  //   dj = 2*t*d_{j+1} - d_{j+2} + coeff[j]
  // res = t*d1 - d2 + coeff[0]  (if coeff[0] is true c0)
  // Since our coeff[0] is c0/2 : result = t*d1 - d2 + 2*coeff[0]

  if (num_points_ == 0) return 0.0;
  if (num_points_ == 1) return coefficients_[0] * 2.0; // c0 = 2 * coeff[0]

  double b_kp2 = 0.0; // b_{N+1}
  double b_kp1 = 0.0; // b_{N} (as coefficients are 0 to N-1)

  for (int k = static_cast<int>(num_points_) - 1; k >= 1; --k) {
    double b_k = coefficients_[k] + 2.0 * t * b_kp1 - b_kp2;
    b_kp2 = b_kp1;
    b_kp1 = b_k;
  }
  // Final step with c0 (which is stored as c0/2 in coefficients_[0])
  return (coefficients_[0] * 2.0) + t * b_kp1 - b_kp2; // Correct use of Clenshaw with c0'
}

void ChebyshevInterpolationDouble::updateValues(const std::vector<double> &new_values_at_nodes) {
  if (new_values_at_nodes.size() != num_points_) {
    throw std::invalid_argument("ChebyshevInterpolationDouble::updateValues: new_values_at_nodes size mismatch");
  }
  values_at_nodes_ = new_values_at_nodes;
  computeCoefficients();
}

double ChebyshevInterpolationDouble::mapToStandardDomain(double x) const {
  if (domain_end_ == domain_start_) return 0.0; // Avoid division by zero
  return 2.0 * (x - domain_start_) / (domain_end_ - domain_start_) - 1.0;
}

double ChebyshevInterpolationDouble::mapFromStandardDomain(double t) const {
  return 0.5 * ((domain_end_ - domain_start_) * t + (domain_end_ + domain_start_));
}


// --- SimdChebyshevInterpolationDouble Implementation ---
SimdChebyshevInterpolationDouble::SimdChebyshevInterpolationDouble(
    const ChebyshevInterpolationDouble &interp)
    : interp_ref_(interp) {}

void SimdChebyshevInterpolationDouble::evaluate(const std::vector<double> &x_vec,
                                                std::vector<double> &y_vec,
                                                bool extrapolate) const {
  if (y_vec.size() < x_vec.size()) {
    y_vec.resize(x_vec.size());
  }
  const size_t n = x_vec.size();
  size_t i = 0;
  const size_t avx2_step = 4;

  for (; i + (avx2_step - 1) < n; i += avx2_step) {
    std::array<double, avx2_step> x_block;
    std::array<double, avx2_step> y_block;
    for(size_t j=0; j<avx2_step; ++j) x_block[j] = x_vec[i+j];
    
    y_block = evaluate4(x_block, extrapolate);
    
    for(size_t j=0; j<avx2_step; ++j) y_vec[i+j] = y_block[j];
  }
  for (; i < n; ++i) {
    y_vec[i] = interp_ref_(x_vec[i], extrapolate);
  }
}

std::array<double, 4>
SimdChebyshevInterpolationDouble::evaluate4(const std::array<double, 4> &x_arr,
                                            bool extrapolate) const {
  std::array<double, 4> result_arr;
  __m256d t_vec;
  double t_scalar[4];

  const double domain_start = interp_ref_.getDomainStart();
  const double domain_end = interp_ref_.getDomainEnd();
  const double scale = (domain_end_ == domain_start_) ? 0.0 : 2.0 / (domain_end - domain_start);
  const double offset = -1.0 - scale * domain_start;

  for(int k=0; k<4; ++k) {
      t_scalar[k] = scale * x_arr[k] + offset;
      if (!extrapolate && (t_scalar[k] < -1.0 - 1e-12 || t_scalar[k] > 1.0 + 1e-12)) {
          if (t_scalar[k] < -1.0) t_scalar[k] = -1.0;
          if (t_scalar[k] > 1.0) t_scalar[k] = 1.0;
      }
      t_scalar[k] = std::max(-1.0, std::min(1.0, t_scalar[k]));
  }
  t_vec = _mm256_loadu_pd(t_scalar);

  const auto &coeffs = interp_ref_.getCoefficients();
  const size_t N = coeffs.size();

  if (N == 0) {
    return {0.0, 0.0, 0.0, 0.0};
  }
  if (N == 1) {
    return {coeffs[0] * 2.0, coeffs[0] * 2.0, coeffs[0] * 2.0, coeffs[0] * 2.0};
  }

  __m256d b_kp2_vec = _mm256_setzero_pd();
  __m256d b_kp1_vec = _mm256_setzero_pd();
  __m256d two_t_vec = _mm256_mul_pd(_mm256_set1_pd(2.0), t_vec);

  for (int k = static_cast<int>(N) - 1; k >= 1; --k) {
    __m256d coeff_k_vec = _mm256_set1_pd(coeffs[k]);
    // b_k = coeffs[k] + 2.0 * t * b_kp1 - b_kp2;
    __m256d term_2t_bkp1 = _mm256_mul_pd(two_t_vec, b_kp1_vec);
    __m256d b_k_vec = _mm256_sub_pd(_mm256_add_pd(coeff_k_vec, term_2t_bkp1), b_kp2_vec);
    b_kp2_vec = b_kp1_vec;
    b_kp1_vec = b_k_vec;
  }
  
  __m256d c0_true_vec = _mm256_set1_pd(coeffs[0] * 2.0); // True c0
  __m256d term_t_bkp1 = _mm256_mul_pd(t_vec, b_kp1_vec);
  __m256d result_m256d = _mm256_sub_pd(_mm256_add_pd(c0_true_vec, term_t_bkp1), b_kp2_vec);
  
  _mm256_storeu_pd(result_arr.data(), result_m256d);
  return result_arr;
}


// --- Factory Functions Double ---
std::shared_ptr<ChebyshevInterpolationDouble>
createChebyshevInterpolationDouble(size_t num_points,
                                   const std::function<double(double)> &func,
                                   ChebyshevKind kind, double domain_start,
                                   double domain_end) {
  return std::make_shared<ChebyshevInterpolationDouble>(num_points, func, kind, domain_start, domain_end);
}

std::shared_ptr<ChebyshevInterpolationDouble>
createChebyshevInterpolationDouble(const std::vector<double> &nodes_in_standard_domain,
                                   const std::vector<double> &values_at_nodes,
                                   ChebyshevKind kind, double domain_start,
                                   double domain_end) {
  return std::make_shared<ChebyshevInterpolationDouble>(nodes_in_standard_domain, values_at_nodes, kind, domain_start, domain_end);
}


// --- ChebyshevInterpolationSingle Implementation ---
// (Similar structure to Double, using float and num::simd for single precision)

ChebyshevInterpolationSingle::ChebyshevInterpolationSingle(
    size_t num_points, const std::function<float(float)> &func,
    ChebyshevKind kind, float domain_start, float domain_end)
    : num_points_(num_points), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end) {
    if (num_points_ < 2) throw std::invalid_argument("ChebyshevInterpolationSingle: num_points must be at least 2");
    if (domain_end_ <= domain_start_) throw std::invalid_argument("ChebyshevInterpolationSingle: domain_end must be greater than domain_start");
    initializeNodes();
    values_at_nodes_.resize(num_points_);
    for (size_t i = 0; i < num_points_; ++i) {
        values_at_nodes_[i] = func(mapFromStandardDomain(nodes_in_standard_domain_[i]));
    }
    computeCoefficients();
}

ChebyshevInterpolationSingle::ChebyshevInterpolationSingle(
    const std::vector<float> &nodes_in_standard_domain,
    const std::vector<float> &values_at_nodes,
    ChebyshevKind kind, float domain_start, float domain_end)
    : num_points_(nodes_in_standard_domain.size()), kind_(kind), domain_start_(domain_start),
      domain_end_(domain_end), nodes_in_standard_domain_(nodes_in_standard_domain),
      values_at_nodes_(values_at_nodes) {
    if (nodes_in_standard_domain_.size() != values_at_nodes_.size()) throw std::invalid_argument("Nodes and values size mismatch");
    if (num_points_ < 2) throw std::invalid_argument("num_points must be at least 2");
    if (domain_end_ <= domain_start_) throw std::invalid_argument("domain_end must be greater than domain_start");
    computeCoefficients();
}

void ChebyshevInterpolationSingle::initializeNodes() {
    nodes_in_standard_domain_.resize(num_points_);
    if (num_points_ == 1) {
      nodes_in_standard_domain_[0] = 0.0f;
      return;
    }
    if (kind_ == FIRST_KIND) {
        for (size_t j = 0; j < num_points_; ++j) {
            nodes_in_standard_domain_[j] = std::cos(static_cast<float>(M_PI) * (2.0f * static_cast<float>(j) + 1.0f) / (2.0f * static_cast<float>(num_points_)));
        }
    } else { // SECOND_KIND
        for (size_t j = 0; j < num_points_; ++j) {
            nodes_in_standard_domain_[j] = std::cos(static_cast<float>(M_PI) * static_cast<float>(j) / (static_cast<float>(num_points_) - 1.0f));
        }
        if (num_points_ > 1) {
            nodes_in_standard_domain_[0] = 1.0f;
            nodes_in_standard_domain_[num_points_-1] = -1.0f;
            std::sort(nodes_in_standard_domain_.begin(), nodes_in_standard_domain_.end());
        }
    }
}

void ChebyshevInterpolationSingle::computeCoefficients() {
    coefficients_.assign(num_points_, 0.0f);
    const size_t N = num_points_;
    for (size_t k = 0; k < N; ++k) {
        float sum = 0.0f;
        for (size_t j = 0; j < N; ++j) {
            sum += values_at_nodes_[j] * std::cos(k * std::acos(nodes_in_standard_domain_[j]));
        }
        coefficients_[k] = (2.0f / static_cast<float>(N)) * sum;
    }
    coefficients_[0] /= 2.0f;
}

float ChebyshevInterpolationSingle::operator()(float x, bool extrapolate) const {
    float t = mapToStandardDomain(x);
    if (!extrapolate && (t < -1.0f - 1e-6f || t > 1.0f + 1e-6f)) {
         if (t < -1.0f) t = -1.0f;
         if (t > 1.0f) t = 1.0f;
    }
    t = std::max(-1.0f, std::min(1.0f, t));

    if (num_points_ == 0) return 0.0f;
    if (num_points_ == 1) return coefficients_[0] * 2.0f;

    float b_kp2 = 0.0f;
    float b_kp1 = 0.0f;
    for (int k_int = static_cast<int>(num_points_) - 1; k_int >= 1; --k_int) {
        float b_k = coefficients_[k_int] + 2.0f * t * b_kp1 - b_kp2;
        b_kp2 = b_kp1;
        b_kp1 = b_k;
    }
    return (coefficients_[0] * 2.0f) + t * b_kp1 - b_kp2;
}

void ChebyshevInterpolationSingle::updateValues(const std::vector<float> &new_values_at_nodes) {
    if (new_values_at_nodes.size() != num_points_) throw std::invalid_argument("updateValues size mismatch");
    values_at_nodes_ = new_values_at_nodes;
    computeCoefficients();
}

float ChebyshevInterpolationSingle::mapToStandardDomain(float x) const {
    if (domain_end_ == domain_start_) return 0.0f;
    return 2.0f * (x - domain_start_) / (domain_end_ - domain_start_) - 1.0f;
}

float ChebyshevInterpolationSingle::mapFromStandardDomain(float t) const {
    return 0.5f * ((domain_end_ - domain_start_) * t + (domain_end_ + domain_start_));
}

// --- SimdChebyshevInterpolationSingle Implementation ---
SimdChebyshevInterpolationSingle::SimdChebyshevInterpolationSingle(
    const ChebyshevInterpolationSingle &interp)
    : interp_ref_(interp) {}

void SimdChebyshevInterpolationSingle::evaluate(const std::vector<float> &x_vec,
                                                std::vector<float> &y_vec,
                                                bool extrapolate) const {
    if (y_vec.size() < x_vec.size()) {
        y_vec.resize(x_vec.size());
    }
    const size_t n = x_vec.size();
    size_t i = 0;
    const size_t avx2_step = 8;

    for (; i + (avx2_step - 1) < n; i += avx2_step) {
        std::array<float, avx2_step> x_block;
        std::array<float, avx2_step> y_block;
        for(size_t j=0; j<avx2_step; ++j) x_block[j] = x_vec[i+j];
        
        y_block = evaluate8(x_block, extrapolate);
        
        for(size_t j=0; j<avx2_step; ++j) y_vec[i+j] = y_block[j];
    }
    for (; i < n; ++i) {
        y_vec[i] = interp_ref_(x_vec[i], extrapolate);
    }
}

std::array<float, 8>
SimdChebyshevInterpolationSingle::evaluate8(const std::array<float, 8> &x_arr,
                                             bool extrapolate) const {
    std::array<float, 8> result_arr;
    __m256 t_vec;
    float t_scalar[8];

    const float domain_start = interp_ref_.getDomainStart();
    const float domain_end = interp_ref_.getDomainEnd();
    const float scale = (domain_end_ == domain_start_) ? 0.0f : 2.0f / (domain_end_ - domain_start_);
    const float offset = -1.0f - scale * domain_start;

    for(int k=0; k<8; ++k) {
        t_scalar[k] = scale * x_arr[k] + offset;
        if (!extrapolate && (t_scalar[k] < -1.0f - 1e-6f || t_scalar[k] > 1.0f + 1e-6f)) {
            if (t_scalar[k] < -1.0f) t_scalar[k] = -1.0f;
            if (t_scalar[k] > 1.0f) t_scalar[k] = 1.0f;
        }
        t_scalar[k] = std::max(-1.0f, std::min(1.0f, t_scalar[k]));
    }
    t_vec = _mm256_loadu_ps(t_scalar);

    const auto &coeffs = interp_ref_.getCoefficients();
    const size_t N = coeffs.size();

    if (N == 0) {
        return {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }
    if (N == 1) {
        float val = coeffs[0] * 2.0f;
        return {val, val, val, val, val, val, val, val};
    }
    
    __m256 b_kp2_vec = _mm256_setzero_ps();
    __m256 b_kp1_vec = _mm256_setzero_ps();
    __m256 two_t_vec = _mm256_mul_ps(_mm256_set1_ps(2.0f), t_vec);

    for (int k_int = static_cast<int>(N) - 1; k_int >= 1; --k_int) {
        __m256 coeff_k_vec = _mm256_set1_ps(coeffs[k_int]);
        __m256 term_2t_bkp1 = _mm256_mul_ps(two_t_vec, b_kp1_vec);
        __m256 b_k_vec = _mm256_sub_ps(_mm256_add_ps(coeff_k_vec, term_2t_bkp1), b_kp2_vec);
        b_kp2_vec = b_kp1_vec;
        b_kp1_vec = b_k_vec;
    }
    
    __m256 c0_true_vec = _mm256_set1_ps(coeffs[0] * 2.0f);
    __m256 term_t_bkp1 = _mm256_mul_ps(t_vec, b_kp1_vec);
    __m256 result_m256 = _mm256_sub_ps(_mm256_add_ps(c0_true_vec, term_t_bkp1), b_kp2_vec);
    
    _mm256_storeu_ps(result_arr.data(), result_m256);
    return result_arr;
}

// --- Factory Functions Single ---
std::shared_ptr<ChebyshevInterpolationSingle>
createChebyshevInterpolationSingle(size_t num_points,
                                   const std::function<float(float)> &func,
                                   ChebyshevKind kind, float domain_start,
                                   float domain_end) {
    return std::make_shared<ChebyshevInterpolationSingle>(num_points, func, kind, domain_start, domain_end);
}

std::shared_ptr<ChebyshevInterpolationSingle>
createChebyshevInterpolationSingle(const std::vector<float> &nodes_in_standard_domain,
                                   const std::vector<float> &values_at_nodes,
                                   ChebyshevKind kind, float domain_start,
                                   float domain_end) {
    return std::make_shared<ChebyshevInterpolationSingle>(nodes_in_standard_domain, values_at_nodes, kind, domain_start, domain_end);
}


} // namespace num
} // namespace alo
} // namespace engine
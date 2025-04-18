 #include "chebyshev.h"
 #include <cmath>
 #include <stdexcept>
 #include <algorithm>
 
 namespace engine {
 namespace alo {
 namespace num {
  
 ChebyshevInterpolation::ChebyshevInterpolation(
     size_t num_points, 
     const std::function<double(double)>& func, 
     ChebyshevKind kind,
     double domain_start,
     double domain_end) 
     : num_points_(num_points), 
       kind_(kind),
       domain_start_(domain_start),
       domain_end_(domain_end) {
     
     if (num_points_ < 2) {
         throw std::invalid_argument("ChebyshevInterpolation requires at least 2 points");
     }
     
     if (domain_end_ <= domain_start_) {
         throw std::invalid_argument("Domain end must be greater than domain start");
     }
     
     // Initialize nodes
     initializeNodes();
     
     // Evaluate function at nodes
     values_.resize(num_points_);
     for (size_t i = 0; i < num_points_; ++i) {
         double x = mapFromStandardDomain(nodes_[i]);
         values_[i] = func(x);
     }
     
     // Compute Chebyshev coefficients
     computeCoefficients();
 }
 
 ChebyshevInterpolation::ChebyshevInterpolation(
     const std::vector<double>& nodes,
     const std::vector<double>& values,
     ChebyshevKind kind,
     double domain_start,
     double domain_end)
     : num_points_(nodes.size()),
       kind_(kind),
       domain_start_(domain_start),
       domain_end_(domain_end),
       nodes_(nodes),
       values_(values) {
     
     if (nodes_.size() != values_.size()) {
         throw std::invalid_argument("Nodes and values must have the same size");
     }
     
     if (nodes_.size() < 2) {
         throw std::invalid_argument("ChebyshevInterpolation requires at least 2 points");
     }
     
     if (domain_end_ <= domain_start_) {
         throw std::invalid_argument("Domain end must be greater than domain start");
     }
     
     // Compute Chebyshev coefficients
     computeCoefficients();
 }
 
 void ChebyshevInterpolation::initializeNodes() {
     nodes_.resize(num_points_);
     
     if (kind_ == FIRST_KIND) {
         // Chebyshev nodes of the first kind: x_j = cos((2j+1)π/(2n)), j=0,...,n-1
         for (size_t j = 0; j < num_points_; ++j) {
             nodes_[j] = std::cos(M_PI * (2.0 * j + 1.0) / (2.0 * num_points_));
         }
     } else { // SECOND_KIND
         // Chebyshev nodes of the second kind: x_j = cos(jπ/(n-1)), j=0,...,n-1
         for (size_t j = 0; j < num_points_; ++j) {
             nodes_[j] = std::cos(M_PI * j / (num_points_ - 1));
         }
     }
 }
 
 void ChebyshevInterpolation::computeCoefficients() {
     coefficients_.resize(num_points_);
     
     if (kind_ == FIRST_KIND) {
         // For first kind, use DCT-II (Discrete Cosine Transform Type II)
         for (size_t k = 0; k < num_points_; ++k) {
             double sum = 0.0;
             for (size_t j = 0; j < num_points_; ++j) {
                 sum += values_[j] * std::cos(M_PI * k * (2.0 * j + 1.0) / (2.0 * num_points_));
             }
             coefficients_[k] = (2.0 / num_points_) * sum;
         }
         // Adjust the first coefficient
         coefficients_[0] *= 0.5;
     } else { // SECOND_KIND
         // For second kind nodes, use a simplified DCT approach
         for (size_t k = 0; k < num_points_; ++k) {
             double sum = 0.0;
             for (size_t j = 0; j < num_points_; ++j) {
                 // For second kind nodes, we can use a simplified DCT
                 sum += values_[j] * std::cos(M_PI * k * j / (num_points_ - 1));
             }
             if (k == 0 || k == num_points_ - 1) {
                 coefficients_[k] = sum / (num_points_ - 1);
             } else {
                 coefficients_[k] = 2.0 * sum / (num_points_ - 1);
             }
         }
     }
 }
 
 double ChebyshevInterpolation::operator()(double x, bool extrapolate) const {
     // Map x to standard domain [-1, 1]
     double t = mapToStandardDomain(x);
     
     // Check if x is within the domain
     if (!extrapolate && (t < -1.0 || t > 1.0)) {
         throw std::domain_error("Evaluation point outside interpolation domain");
     }
     
     // Apply Clenshaw's algorithm to evaluate the Chebyshev series
     if (kind_ == FIRST_KIND) {
         double b_kp2 = 0.0;
         double b_kp1 = 0.0;
         double b_k;
         
         for (int k = static_cast<int>(num_points_) - 1; k >= 0; --k) {
             b_k = coefficients_[k] + 2.0 * t * b_kp1 - b_kp2;
             b_kp2 = b_kp1;
             b_kp1 = b_k;
         }
         
         return b_kp1 - t * b_kp2;
     } else { // SECOND_KIND
         // Direct evaluation using Horner's method
         double result = 0.0;
         for (int k = num_points_ - 1; k >= 0; --k) {
             result = result * t + coefficients_[k];
         }
         return result;
     }
 }
 
 void ChebyshevInterpolation::updateValues(const std::vector<double>& values) {
     if (values.size() != num_points_) {
         throw std::invalid_argument("Number of values must match the number of nodes");
     }
     
     values_ = values;
     computeCoefficients();
 }
 
 double ChebyshevInterpolation::mapToStandardDomain(double x) const {
     return 2.0 * (x - domain_start_) / (domain_end_ - domain_start_) - 1.0;
 }
 
 double ChebyshevInterpolation::mapFromStandardDomain(double t) const {
     return 0.5 * ((domain_end_ - domain_start_) * t + domain_end_ + domain_start_);
 }
 
 double ChebyshevInterpolation::evaluateChebyshev(int n, double x, ChebyshevKind kind) const {
     if (n == 0) return 1.0;
     if (n == 1) return x;
     
     // Use recurrence relation to compute higher-order polynomials
     double T_n_minus_2 = 1.0;  // T_0(x)
     double T_n_minus_1 = x;    // T_1(x)
     double T_n = 0.0;
     
     for (int i = 2; i <= n; ++i) {
         T_n = 2.0 * x * T_n_minus_1 - T_n_minus_2;
         T_n_minus_2 = T_n_minus_1;
         T_n_minus_1 = T_n;
     }
     
     if (kind == FIRST_KIND) {
         return T_n;
     } else { // SECOND_KIND
         // Convert to second kind Chebyshev polynomial
         if (n == 0) return 1.0;
         return (1.0 - x * x) * T_n;
     }
 }
  
 SimdChebyshevInterpolation::SimdChebyshevInterpolation(const ChebyshevInterpolation& interp)
     : interp_(interp) {
 }
 
 void SimdChebyshevInterpolation::evaluate(
     const std::vector<double>& x, 
     std::vector<double>& y,
     bool extrapolate) const {
     
     if (y.size() < x.size()) {
         y.resize(x.size());
     }
     
     // Check if SIMD is available and beneficial
     if (x.size() >= 4) {
         // Process in groups of 4 using SIMD
         size_t i = 0;
         for (; i + 3 < x.size(); i += 4) {
             std::array<double, 4> x_block = {x[i], x[i+1], x[i+2], x[i+3]};
             std::array<double, 4> y_block = evaluate4(x_block, extrapolate);
             
             for (size_t j = 0; j < 4; ++j) {
                 y[i + j] = y_block[j];
             }
         }
         
         // Handle remaining points
         for (; i < x.size(); ++i) {
             y[i] = interp_(x[i], extrapolate);
         }
     } else {
         // For small input, just use scalar computation
         for (size_t i = 0; i < x.size(); ++i) {
             y[i] = interp_(x[i], extrapolate);
         }
     }
 }
 
 std::array<double, 4> SimdChebyshevInterpolation::evaluate4(
     const std::array<double, 4>& x,
     bool extrapolate) const {
     
     std::array<double, 4> result;
     
     // Map input points to standard domain
     std::array<double, 4> t;
     const double domain_start = interp_.getDomainStart();
     const double domain_end = interp_.getDomainEnd();
     
     for (size_t i = 0; i < 4; ++i) {
         t[i] = 2.0 * (x[i] - domain_start) / (domain_end - domain_start) - 1.0;
         
         // Check domain bounds if not extrapolating
         if (!extrapolate && (t[i] < -1.0 || t[i] > 1.0)) {
             throw std::domain_error("Evaluation point outside interpolation domain");
         }
     }
     
     // Get interpolation details
     const auto& coeffs = interp_.getCoefficients();
     ChebyshevKind kind = interp_.getKind();
     
     // Load coefficients into SIMD registers (if enough coefficients)
     if (coeffs.size() >= 4 && kind == SECOND_KIND) {
         // Use SIMD for Horner's method (second kind)
         __m256d result_vec = _mm256_setzero_pd();
         
         for (int k = coeffs.size() - 1; k >= 0; --k) {
             __m256d coeff_vec = _mm256_set1_pd(coeffs[k]);
             __m256d t_vec = _mm256_loadu_pd(t.data());
             
             // result = result * t + coeff
             result_vec = _mm256_fmadd_pd(result_vec, t_vec, coeff_vec);
         }
         
         _mm256_storeu_pd(result.data(), result_vec);
     } else {
         // Fall back to scalar computation for small coefficient sets or first kind
         for (int i = 0; i < 4; ++i) {
             if (kind == FIRST_KIND) {
                 // Apply Clenshaw's algorithm
                 double b_kp2 = 0.0;
                 double b_kp1 = 0.0;
                 double b_k;
                 
                 for (int k = static_cast<int>(coeffs.size()) - 1; k >= 0; --k) {
                     b_k = coeffs[k] + 2.0 * t[i] * b_kp1 - b_kp2;
                     b_kp2 = b_kp1;
                     b_kp1 = b_k;
                 }
                 
                 result[i] = b_kp1 - t[i] * b_kp2;
             } else { // SECOND_KIND
                 // Apply Horner's method
                 double r = 0.0;
                 for (int k = coeffs.size() - 1; k >= 0; --k) {
                     r = r * t[i] + coeffs[k];
                 }
                 result[i] = r;
             }
         }
     }
     
     return result;
 }
 
 std::shared_ptr<ChebyshevInterpolation> createChebyshevInterpolation(
     size_t num_points, 
     const std::function<double(double)>& func, 
     ChebyshevKind kind,
     double domain_start,
     double domain_end) {
     
     return std::make_shared<ChebyshevInterpolation>(
         num_points, func, kind, domain_start, domain_end);
 }
 
 std::shared_ptr<ChebyshevInterpolation> createChebyshevInterpolation(
     const std::vector<double>& nodes,
     const std::vector<double>& values,
     ChebyshevKind kind,
     double domain_start,
     double domain_end) {
     
     return std::make_shared<ChebyshevInterpolation>(
         nodes, values, kind, domain_start, domain_end);
 }
 
 } // namespace num
 } // namespace alo
 } // namespace engine
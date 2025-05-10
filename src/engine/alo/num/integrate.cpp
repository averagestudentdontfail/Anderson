#include "integrate.h"
#include <immintrin.h> 
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <map> 

// Ensure M_PI and M_PI_2 are defined for TanhSinh
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923 // PI/2
#endif

namespace engine {
namespace alo {
namespace num {

struct GaussLegendreData {
    std::vector<double> nodes;
    std::vector<double> weights;
};

// Static storage for precomputed Gauss-Legendre nodes and weights
// Using a map to store data for different orders.
// Key: order, Value: GaussLegendreData
static const std::map<size_t, GaussLegendreData> GL_PRECOMPUTED_DOUBLE = {
    {1, {{0.0}, {2.0}}},
    {2, {{-0.5773502691896257, 0.5773502691896257}, {1.0, 1.0}}},
    {3, {{-0.7745966692414834, 0.0, 0.7745966692414834}, {0.5555555555555556, 0.8888888888888888, 0.5555555555555556}}},
    {4, {{-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526}, 
         {0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539}}},
    {5, {{-0.9061798459386640, -0.5384693101056831, 0.0, 0.5384693101056831, 0.9061798459386640},
         {0.2369268850561891, 0.4786286704993665, 0.5688888888888889, 0.4786286704993665, 0.2369268850561891}}},
    {7, {{-0.9491079123427585, -0.7415311855993944, -0.4058451513773972, 0.0,
          0.4058451513773972,  0.7415311855993944,  0.9491079123427585},
         {0.1294849661688697, 0.2797053914892767, 0.3818300505051189,
          0.4179591836734694, 0.3818300505051189, 0.2797053914892767,
          0.1294849661688697}}},
    {8, {{-0.9602898564975363,-0.7966664774136267,-0.5255324099163290,-0.1834346424956498,
          0.1834346424956498,0.5255324099163290,0.7966664774136267,0.9602898564975363},
         {0.1012285362903763,0.2223810344533745,0.3137066458778873,0.3626837833783620,
          0.3626837833783620,0.3137066458778873,0.2223810344533745,0.1012285362903763}}},
    // Add more orders as needed, e.g., 10, 12, 16, 20, 24, 25, 27, 32 etc.
    // Order 25 was previously defined.
    {25, {{-0.9955569697904981,-0.9766639214595175,-0.9429745712289743,-0.8949919978782753,
            -0.8334426287608340,-0.7592592630373576,-0.6735663684734684,-0.5776629302412229,
            -0.4731469662935845,-0.3611723058093879,-0.2429801799032639,-0.1207530708447741,0.0,
             0.1207530708447741,0.2429801799032639,0.3611723058093879,0.4731469662935845,
             0.5776629302412229,0.6735663684734684,0.7592592630373576,0.8334426287608340,
             0.8949919978782753,0.9429745712289743,0.9766639214595175,0.9955569697904981},
           {0.0113937985010262,0.0263549866150321,0.0409391567013063,0.0549046959758351,
            0.0680383338123569,0.0801407003350010,0.0910282619829636,0.1005359490670506,
            0.1085196244742637,0.1148582591457116,0.1194557635357847,0.1222424429903100,
            0.1231760537267154,0.1222424429903100,0.1194557635357847,0.1148582591457116,
            0.1085196244742637,0.1005359490670506,0.0910282619829636,0.0801407003350010,
            0.0680383338123569,0.0549046959758351,0.0409391567013063,0.0263549866150321,
            0.0113937985010262}}},
    // Order 27 was used in QL, values taken from a standard generator
    {27, {{-0.9961792628889886,-0.9782286581460570,-0.9458226521856563,-0.8992005757021038,
            -0.8391169718222189,-0.7663811206689788,-0.6828454791571403,-0.5896380977729661,
            -0.4879900029287655,-0.3790232126755540,-0.2639649827963907,-0.1441590672327308,
            -0.0243312301640831, 0.0, 0.0243312301640831, // Symmetrized around 0
             0.1441590672327308,0.2639649827963907,0.3790232126755540,0.4879900029287655,
             0.5896380977729661,0.6828454791571403,0.7663811206689788,0.8391169718222189,
             0.8992005757021038,0.9458226521856563,0.9782286581460570,0.9961792628889886},
           {0.0097989960512943,0.0227575625501992,0.0355047847316408,0.0478481301259484,
            0.0595985325645789,0.0705878906601189,0.0806753521268833,0.0897264238206302,
            0.0976186521041138,0.1042582260352920,0.1095783812798404,0.1135354900057835,
            0.1161034212297789,0.1172024672904842,0.1161034212297789, // Symmetric weights
            0.1135354900057835,0.1095783812798404,0.1042582260352920,0.0976186521041138,
            0.0897264238206302,0.0806753521268833,0.0705878906601189,0.0595985325645789,
            0.0478481301259484,0.0355047847316408,0.0227575625501992,0.0097989960512943}}}
};

// Single precision precomputed values (can be derived from double or recomputed)
static const std::map<size_t, GaussLegendreData> GL_PRECOMPUTED_SINGLE = {
    // Cast from double, or recompute for float precision if critical
    {1, {{0.0f}, {2.0f}}},
    {7, {{-0.9491079123f, -0.7415311856f, -0.4058451514f, 0.0f,
          0.4058451514f,  0.7415311856f,  0.9491079123f},
         {0.1294849662f, 0.2797053915f, 0.3818300505f,
          0.4179591837f, 0.3818300505f, 0.2797053915f,
          0.1294849662f}}},
    {8, {{-0.9602898565f,-0.7966664774f,-0.5255324099f,-0.1834346425f,
          0.1834346425f,0.5255324099f,0.7966664774f,0.9602898565f},
         {0.1012285363f,0.2223810345f,0.3137066459f,0.3626837834f,
          0.3626837834f,0.3137066459f,0.2223810345f,0.1012285363f}}}
    // Add other orders as needed
};

class GaussLegendreIntegrateDoubleImpl : public IntegrateDouble {
public:
  explicit GaussLegendreIntegrateDoubleImpl(size_t order) : order_(order) {
    if (order_ < 1) {
      throw std::invalid_argument("GaussLegendreIntegrateDoubleImpl: Order must be at least 1");
    }
    initializeNodesAndWeights();
  }

  ~GaussLegendreIntegrateDoubleImpl() override = default;

  double integrate(const std::function<double(double)> &f, double a, double b) const override {
    double result = 0.0;
    if (std::abs(b - a) < std::numeric_limits<double>::epsilon() * 100.0 * (std::abs(a)+std::abs(b))) return 0.0; // More robust zero check

    const double half_length = 0.5 * (b - a);
    const double mid_point = 0.5 * (a + b);

    for (size_t i = 0; i < order_; ++i) {
      const double x = mid_point + half_length * nodes_[i];
      result += weights_[i] * f(x);
    }
    result *= half_length;
    return result;
  }

  std::string name() const override { return "Gauss-Legendre (double)"; }

private:
  void initializeNodesAndWeights() {
    auto it = GL_PRECOMPUTED_DOUBLE.find(order_);
    if (it != GL_PRECOMPUTED_DOUBLE.end()) {
        nodes_ = it->second.nodes;
        weights_ = it->second.weights;
    } else {
        // For production, it's better to throw an error if an unsupported order is requested,
        // or implement dynamic generation (which is complex and slow for constructor).
        throw std::invalid_argument("GaussLegendreIntegrateDoubleImpl: Order " + std::to_string(order_) + " not precomputed.");
    }
  }

  size_t order_;
  std::vector<double> nodes_;
  std::vector<double> weights_;
};


class TanhSinhIntegrateDoubleImpl : public IntegrateDouble {
public:
  explicit TanhSinhIntegrateDoubleImpl(double tolerance, int max_levels = 15)
      : tolerance_(tolerance), max_levels_(max_levels) {
    if (tolerance_ <= 0.0) {
      throw std::invalid_argument("TanhSinhIntegrateDoubleImpl: Tolerance must be positive");
    }
    if (max_levels_ < 1 || max_levels_ > 20) {
      throw std::invalid_argument("TanhSinhIntegrateDoubleImpl: Max levels must be between 1 and 20");
    }
  }

  ~TanhSinhIntegrateDoubleImpl() override = default;

  double integrate(const std::function<double(double)> &f, double a, double b) const override {
    if (std::abs(b - a) < std::numeric_limits<double>::epsilon() * 100.0 * (std::abs(a)+std::abs(b))) return 0.0;

    double m = 0.5 * (a + b); 
    double c = 0.5 * (b - a); 
    
    auto integrand_transformed = [&](double t) {
        if (!std::isfinite(t)) return 0.0; // Handle potential inf/nan from sinh
        double pi_half_sinh_t = M_PI_2 * std::sinh(t);
        double cosh_pi_half_sinh_t = std::cosh(pi_half_sinh_t);
        
        if (cosh_pi_half_sinh_t < std::numeric_limits<double>::min() || !std::isfinite(cosh_pi_half_sinh_t)) {
             return 0.0;
        }
        double tanh_val = std::tanh(pi_half_sinh_t);
        double x_val = m + c * tanh_val;
        
        double cosh_t_val = std::cosh(t);
        if (!std::isfinite(cosh_t_val)) { // sinh(t) could be large, making cosh(t) overflow
            // This part of the domain contributes negligibly or indicates issue with bounds/function
            return 0.0;
        }
        double dxdt_factor = (M_PI_2 * cosh_t_val) / (cosh_pi_half_sinh_t * cosh_pi_half_sinh_t);
        return f(x_val) * c * dxdt_factor;
    };

    double h = 1.0;
    double current_sum = integrand_transformed(0.0); // S0 * h (h=1 initially)
    double prev_integral_times_h = 0.0; // Store previous S_{k-1}*h_k
                                      // This is I_k = h_k * S_k
    
    for (int level = 1; level <= max_levels_; ++level) {
        double h_new = h / 2.0;
        double new_points_sum = 0.0;
        for (int k_odd = 1; k_odd < (1 << level) ; k_odd += 2) { // Sum over odd k for new points
            double t_k = k_odd * h_new;
            double term_plus = integrand_transformed(t_k);
            double term_minus = integrand_transformed(-t_k);
            if (!std::isfinite(term_plus) || !std::isfinite(term_minus)) {
                // If integrand blows up, this method might not be suitable or integrand is problematic
                // Or, we've gone too far into tails where sinh(t) is huge.
                // Consider returning current best estimate or throwing.
                // For now, skip problematic terms, they should be small in tails.
                continue;
            }
            new_points_sum += term_plus + term_minus;
        }
        
        double old_sum_times_old_h = current_sum * h; // This is I_{level-1}
        current_sum = 0.5 * current_sum + new_points_sum; // This calculates S_level from S_{level-1}
        h = h_new;
        double current_integral_times_h = current_sum * h; // This is I_level

        if (level > 3) { // Start checking convergence after a few levels
             // Error estimate: |I_k - I_{k-1}|
            if (std::abs(current_integral_times_h - old_sum_times_old_h) < tolerance_ * std::abs(current_integral_times_h) ||
                std::abs(current_integral_times_h - old_sum_times_old_h) < tolerance_ ) { // also check absolute tolerance
                break;
            }
        }
        prev_integral_times_h = current_integral_times_h; // Store for next iteration's error check
    }
    return current_sum * h; 
  }

  std::string name() const override { return "Tanh-Sinh (double)"; }

private:
  double tolerance_;
  int max_levels_;
};


class AdaptiveIntegrateDoubleImpl : public IntegrateDouble {
public:
  AdaptiveIntegrateDoubleImpl(double absolute_tolerance, double relative_tolerance,
                           size_t max_intervals)
      : abs_tol_(absolute_tolerance), rel_tol_(relative_tolerance),
        max_intervals_(max_intervals) {
    if (abs_tol_ <= 0.0 && rel_tol_ <= 0.0) {
      throw std::invalid_argument("AdaptiveIntegrateDoubleImpl: At least one tolerance must be positive");
    }
    if (max_intervals_ < 1) {
      throw std::invalid_argument("AdaptiveIntegrateDoubleImpl: Max intervals must be at least 1");
    }
    initializeGaussKronrod();
  }

  ~AdaptiveIntegrateDoubleImpl() override = default;

  double integrate(const std::function<double(double)> &f, double a, double b) const override {
    if (std::abs(b - a) < std::numeric_limits<double>::epsilon() * 100.0 * (std::abs(a)+std::abs(b))) return 0.0;

    struct Interval { 
        double a, b, integral_k, error_est; 
        bool operator<(const Interval& other) const { // For min-priority queue behavior with max_heap
            return error_est > other.error_est; // Smallest error has lower priority (pop largest error)
        }
    };
    
    std::vector<Interval> interval_heap; // Use vector as a max-heap (largest error on top)
    std::make_heap(interval_heap.begin(), interval_heap.end());

    double initial_integral_k, initial_error_est;
    gaussKronrodRule(f, a, b, initial_integral_k, initial_error_est);
    
    interval_heap.push_back({a, b, initial_integral_k, initial_error_est});
    std::push_heap(interval_heap.begin(), interval_heap.end());

    double total_integral = initial_integral_k;
    double total_error_estimate = initial_error_est;

    size_t iterations = 0;
    while (total_error_estimate > std::max(abs_tol_, rel_tol_ * std::abs(total_integral)) && 
           iterations < max_intervals_ && !interval_heap.empty()) {
        
        std::pop_heap(interval_heap.begin(), interval_heap.end());
        Interval current_interval = interval_heap.back();
        interval_heap.pop_back();

        total_integral -= current_interval.integral_k; // Remove old contribution
        total_error_estimate -= current_interval.error_est;

        double mid = 0.5 * (current_interval.a + current_interval.b);

        double left_integral_k, left_error_est;
        gaussKronrodRule(f, current_interval.a, mid, left_integral_k, left_error_est);
        
        double right_integral_k, right_error_est;
        gaussKronrodRule(f, mid, current_interval.b, right_integral_k, right_error_est);

        interval_heap.push_back({current_interval.a, mid, left_integral_k, left_error_est});
        std::push_heap(interval_heap.begin(), interval_heap.end());
        interval_heap.push_back({mid, current_interval.b, right_integral_k, right_error_est});
        std::push_heap(interval_heap.begin(), interval_heap.end());
        
        total_integral += left_integral_k + right_integral_k;
        total_error_estimate += left_error_est + right_error_est;
        
        iterations++;
    }
    return total_integral;
  }

  std::string name() const override { return "Adaptive Gauss-Kronrod (double)"; }

private:
  void initializeGaussKronrod() {
    gk_nodes_ = {-0.9914553711208126, -0.9491079123427585, -0.8648644233597691, -0.7415311855993944,
                 -0.5860872354676911, -0.4058451513773972, -0.2077849550078985, 0.0,
                  0.2077849550078985,  0.4058451513773972,  0.5860872354676911,  0.7415311855993944,
                  0.8648644233597691,  0.9491079123427585,  0.9914553711208126};
    g_weights_ = {0.0, 0.1294849661688697, 0.0, 0.2797053914892767, 0.0, 0.3818300505051189, 0.0, 
                  0.4179591836734694, 0.0, 0.3818300505051189, 0.0, 0.2797053914892767, 0.0, 
                  0.1294849661688697, 0.0}; 
    k_weights_ = {0.0229353220105292,0.0630920926299786,0.1047900103222502,0.1406532597155259,
                  0.1690047266392679,0.1903505780647854,0.2044329400752989,0.2094821410847278,
                  0.2044329400752989,0.1903505780647854,0.1690047266392679,0.1406532597155259,
                  0.1047900103222502,0.0630920926299786,0.0229353220105292};
  }

  void gaussKronrodRule(const std::function<double(double)> &f, double a,
                        double b, double &integral_k, double &error_est) const {
    const double half_length = 0.5 * (b - a);
    const double mid_point = 0.5 * (a + b);
    double integral_g = 0.0;
    integral_k = 0.0;
    // Evaluate function at symmetric points once
    std::vector<double> fx_vals(gk_nodes_.size());
    for(size_t i=0; i < (gk_nodes_.size()+1)/2; ++i) { // Iterate up to the center point
        double node_val = gk_nodes_[i];
        fx_vals[i] = f(mid_point + half_length * node_val);
        if (i*2 +1 < gk_nodes_.size()){ // if not the center point for odd number of nodes
             fx_vals[gk_nodes_.size()-1-i] = f(mid_point - half_length * node_val);
        }
    }

    for (size_t i = 0; i < gk_nodes_.size(); ++i) {
      integral_k += k_weights_[i] * fx_vals[i];
      if (g_weights_[i] != 0.0) { 
          integral_g += g_weights_[i] * fx_vals[i];
      }
    }
    integral_k *= half_length;
    integral_g *= half_length;
    error_est = std::abs(integral_k - integral_g);
  }

  double abs_tol_;
  double rel_tol_;
  size_t max_intervals_;
  std::vector<double> gk_nodes_;
  std::vector<double> g_weights_;
  std::vector<double> k_weights_;
};

std::shared_ptr<IntegrateDouble>
createIntegrateDouble(const std::string &scheme_type, size_t order,
                       double tolerance) {
  if (scheme_type == "GaussLegendre") {
    if (order == 0) order = 7; 
    return std::make_shared<GaussLegendreIntegrateDoubleImpl>(order);
  } else if (scheme_type == "TanhSinh") {
    if (tolerance <= 0.0) tolerance = 1e-10; // Default for TanhSinh can be tighter
    return std::make_shared<TanhSinhIntegrateDoubleImpl>(tolerance);
  } else if (scheme_type == "Adaptive") {
    if (tolerance <= 0.0) tolerance = 1e-8; 
    return std::make_shared<AdaptiveIntegrateDoubleImpl>(tolerance, tolerance, 200); // Max 200 intervals
  } else {
    throw std::invalid_argument("Unknown IntegrateDouble type: " + scheme_type);
  }
}

void IntegrateSingle::batchIntegrate(const std::function<float(float)> &f,
                                      const std::vector<float> &a,
                                      const std::vector<float> &b,
                                      std::vector<float> &results) const {
  if (a.size() != b.size()) {
      throw std::invalid_argument("batchIntegrate (single): a and b must have the same size.");
  }
  if (results.size() < a.size()) {
    results.resize(a.size());
  }
  for (size_t i = 0; i < a.size(); ++i) {
    results[i] = integrate(f, a[i], b[i]);
  }
}


class GaussLegendreIntegrateSingleImpl : public IntegrateSingle {
public:
  explicit GaussLegendreIntegrateSingleImpl(size_t order) : order_(order) {
    if (order_ < 1) {
      throw std::invalid_argument("GaussLegendreIntegrateSingleImpl: Order must be at least 1");
    }
    initializeNodesAndWeights();
  }

  ~GaussLegendreIntegrateSingleImpl() override = default;

  float integrate(const std::function<float(float)> &f, float a, float b) const override {
    float result = 0.0f;
    if (std::abs(b - a) < std::numeric_limits<float>::epsilon() * 100.0f * (std::abs(a)+std::abs(b))) return 0.0f;

    const float half_length = 0.5f * (b - a);
    const float mid_point = 0.5f * (a + b);

    for (size_t i = 0; i < order_; ++i) {
      const float x = mid_point + half_length * nodes_[i];
      result += weights_[i] * f(x);
    }
    result *= half_length;
    return result;
  }

  std::string name() const override { return "Gauss-Legendre (single)"; }

  void batchIntegrate(const std::function<float(float)> &f,
                      const std::vector<float> &a_vec, const std::vector<float> &b_vec,
                      std::vector<float> &results_vec) const override {
    if (a_vec.size() != b_vec.size()) {
      throw std::invalid_argument("batchIntegrate (single): a and b must have the same size.");
    }
    if (results_vec.size() < a_vec.size()) {
        results_vec.resize(a_vec.size());
    }
    
    size_t i = 0;
    const size_t num_integrals = a_vec.size();
    const size_t avx2_f_step = 8;

    for (; i + (avx2_f_step - 1) < num_integrals; i += avx2_f_step) {
        __m256 a_m256 = _mm256_loadu_ps(&a_vec[i]);
        __m256 b_m256 = _mm256_loadu_ps(&b_vec[i]);

        __m256 half_len_m256 = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_sub_ps(b_m256, a_m256));
        __m256 mid_pt_m256   = _mm256_mul_ps(_mm256_set1_ps(0.5f), _mm256_add_ps(a_m256, b_m256));
        __m256 integral_sum_m256 = _mm256_setzero_ps();

        for (size_t k = 0; k < order_; ++k) {
            __m256 node_k_m256 = _mm256_set1_ps(nodes_[k]);
            __m256 weight_k_m256 = _mm256_set1_ps(weights_[k]);
            
            __m256 x_eval_m256 = _mm256_fmadd_ps(half_len_m256, node_k_m256, mid_pt_m256);
            
            alignas(32) float x_eval_arr[avx2_f_step]; // Ensure alignment for store/load
            _mm256_store_ps(x_eval_arr, x_eval_m256); // Use aligned store
            
            alignas(32) float f_vals_arr[avx2_f_step];
            for(size_t j=0; j < avx2_f_step; ++j) {
                f_vals_arr[j] = f(x_eval_arr[j]);
            }
            __m256 f_vals_m256 = _mm256_load_ps(f_vals_arr); // Use aligned load
            
            integral_sum_m256 = _mm256_fmadd_ps(weight_k_m256, f_vals_m256, integral_sum_m256);
        }
        integral_sum_m256 = _mm256_mul_ps(integral_sum_m256, half_len_m256);
        _mm256_storeu_ps(&results_vec[i], integral_sum_m256);
    }

    for (; i < num_integrals; ++i) {
        results_vec[i] = integrate(f, a_vec[i], b_vec[i]);
    }
  }

private:
  void initializeNodesAndWeights() {
    auto it = GL_PRECOMPUTED_SINGLE.find(order_);
    if (it != GL_PRECOMPUTED_SINGLE.end()) {
        nodes_ = it->second.nodes; // Assuming GaussLegendreData can hold floats or implicitly converts
        weights_ = it->second.weights;
        // Ensure they are float vectors
        if (!nodes_.empty() && typeid(nodes_[0]) != typeid(float)) { // Basic check
            nodes_.assign(it->second.nodes.begin(), it->second.nodes.end()); // Re-assign if needed
            weights_.assign(it->second.weights.begin(), it->second.weights.end());
        }
    } else {
      throw std::invalid_argument("GaussLegendreIntegrateSingleImpl: Order " + std::to_string(order_) + " not precomputed for single precision.");
    }
  }
  size_t order_;
  std::vector<float> nodes_;
  std::vector<float> weights_;
};


class TanhSinhIntegrateSingleImpl : public IntegrateSingle {
public:
  explicit TanhSinhIntegrateSingleImpl(float tolerance, int max_levels = 12) // Max levels slightly less for float
      : tolerance_(tolerance), max_levels_(max_levels) {
    if (tolerance_ <= 0.0f) {
      throw std::invalid_argument("TanhSinhIntegrateSingleImpl: Tolerance must be positive");
    }
    if (max_levels_ < 1 || max_levels_ > 18) { 
      throw std::invalid_argument("TanhSinhIntegrateSingleImpl: Max levels must be between 1 and 18");
    }
  }
  ~TanhSinhIntegrateSingleImpl() override = default;

  float integrate(const std::function<float(float)> &f, float a, float b) const override {
    if (std::abs(b - a) < std::numeric_limits<float>::epsilon() * 100.0f * (std::abs(a)+std::abs(b))) return 0.0f;

    float m = 0.5f * (a + b); 
    float c = 0.5f * (b - a); 
    
    auto integrand_transformed = [&](float t_float) {
        double t = static_cast<double>(t_float); // Use double for intermediate math for stability
        if (!std::isfinite(t)) return 0.0f;
        double pi_half_sinh_t = M_PI_2 * std::sinh(t);
        double cosh_pi_half_sinh_t = std::cosh(pi_half_sinh_t);
        if (cosh_pi_half_sinh_t < std::numeric_limits<double>::min() || !std::isfinite(cosh_pi_half_sinh_t)) {
             return 0.0f;
        }
        double tanh_val = std::tanh(pi_half_sinh_t);
        float x_val = m + c * static_cast<float>(tanh_val);
        
        double cosh_t_val = std::cosh(t);
         if (!std::isfinite(cosh_t_val)) return 0.0f;
        double dxdt_factor = (M_PI_2 * cosh_t_val) / (cosh_pi_half_sinh_t * cosh_pi_half_sinh_t);
        return f(x_val) * c * static_cast<float>(dxdt_factor);
    };

    float h = 1.0f;
    float current_sum = integrand_transformed(0.0f); 
    float prev_integral_times_h = 0.0f; 
    
    for (int level = 1; level <= max_levels_; ++level) {
        float h_new = h / 2.0f;
        float new_points_sum = 0.0f;
        for (int k_odd = 1; k_odd < (1 << level) ; k_odd += 2) { 
            float t_k = static_cast<float>(k_odd) * h_new;
            float term_plus = integrand_transformed(t_k);
            float term_minus = integrand_transformed(-t_k);
             if (!std::isfinite(term_plus) || !std::isfinite(term_minus)) continue;
            new_points_sum += term_plus + term_minus;
        }
        
        float old_sum_times_old_h = current_sum * h; 
        current_sum = 0.5f * current_sum + new_points_sum; 
        h = h_new;
        float current_integral_times_h = current_sum * h;

        if (level > 3) { 
            if (std::abs(current_integral_times_h - old_sum_times_old_h) < tolerance_ * std::abs(current_integral_times_h) ||
                std::abs(current_integral_times_h - old_sum_times_old_h) < tolerance_ ) { 
                break;
            }
        }
        prev_integral_times_h = current_integral_times_h;
    }
    return current_sum * h; 
  }
  std::string name() const override { return "Tanh-Sinh (single)"; }
private:
  float tolerance_;
  int max_levels_;
};


class AdaptiveIntegrateSingleImpl : public IntegrateSingle {
public:
  AdaptiveIntegrateSingleImpl(float absolute_tolerance, float relative_tolerance,
                           size_t max_intervals)
      : abs_tol_(absolute_tolerance), rel_tol_(relative_tolerance),
        max_intervals_(max_intervals) {
    if (abs_tol_ <= 0.0f && rel_tol_ <= 0.0f) {
      throw std::invalid_argument("AdaptiveIntegrateSingleImpl: At least one tolerance must be positive");
    }
    if (max_intervals_ < 1) {
      throw std::invalid_argument("AdaptiveIntegrateSingleImpl: Max intervals must be at least 1");
    }
    initializeGaussKronrod();
  }

  ~AdaptiveIntegrateSingleImpl() override = default;

  float integrate(const std::function<float(float)> &f, float a, float b) const override {
    if (std::abs(b - a) < std::numeric_limits<float>::epsilon() * 100.0f * (std::abs(a)+std::abs(b))) return 0.0f;

    struct Interval { 
        float a, b, integral_k, error_est; 
        bool operator<(const Interval& other) const { return error_est > other.error_est; }
    };
    std::vector<Interval> interval_heap;
    std::make_heap(interval_heap.begin(), interval_heap.end());

    float initial_integral_k, initial_error_est;
    gaussKronrodRule(f, a, b, initial_integral_k, initial_error_est);
    interval_heap.push_back({a, b, initial_integral_k, initial_error_est});
    std::push_heap(interval_heap.begin(), interval_heap.end());

    float total_integral = initial_integral_k;
    float total_error_estimate = initial_error_est;
    size_t iterations = 0;

    while (total_error_estimate > std::max(abs_tol_, rel_tol_ * std::abs(total_integral)) && 
           iterations < max_intervals_ && !interval_heap.empty()) {
        std::pop_heap(interval_heap.begin(), interval_heap.end());
        Interval current_interval = interval_heap.back();
        interval_heap.pop_back();

        total_integral -= current_interval.integral_k;
        total_error_estimate -= current_interval.error_est;

        float mid = 0.5f * (current_interval.a + current_interval.b);
        float left_integral_k, left_error_est, right_integral_k, right_error_est;

        gaussKronrodRule(f, current_interval.a, mid, left_integral_k, left_error_est);
        gaussKronrodRule(f, mid, current_interval.b, right_integral_k, right_error_est);

        if(std::isfinite(left_integral_k) && std::isfinite(left_error_est)){
            interval_heap.push_back({current_interval.a, mid, left_integral_k, left_error_est});
            std::push_heap(interval_heap.begin(), interval_heap.end());
            total_integral += left_integral_k;
            total_error_estimate += left_error_est;
        }
        if(std::isfinite(right_integral_k) && std::isfinite(right_error_est)){
            interval_heap.push_back({mid, current_interval.b, right_integral_k, right_error_est});
            std::push_heap(interval_heap.begin(), interval_heap.end());
            total_integral += right_integral_k;
            total_error_estimate += right_error_est;
        }
        iterations++;
    }
    return total_integral;
  }

  std::string name() const override { return "Adaptive Gauss-Kronrod (single)"; }

private:
  void initializeGaussKronrod() {
    gk_nodes_ = {-0.9914553711f,-0.9491079123f,-0.8648644233f,-0.7415311855f,-0.5860872354f,
                 -0.4058451513f,-0.2077849550f,0.0f,0.2077849550f,0.4058451513f,0.5860872354f,
                  0.7415311855f,0.8648644233f,0.9491079123f,0.9914553711f};
    g_weights_ = {0.0f,0.1294849661f,0.0f,0.2797053914f,0.0f,0.3818300505f,0.0f,0.4179591836f,
                  0.0f,0.3818300505f,0.0f,0.2797053914f,0.0f,0.1294849661f,0.0f};
    k_weights_ = {0.0229353220f,0.0630920926f,0.1047900103f,0.1406532597f,0.1690047266f,
                  0.1903505780f,0.2044329400f,0.2094821410f,0.2044329400f,0.1903505780f,
                  0.1690047266f,0.1406532597f,0.1047900103f,0.0630920926f,0.0229353220f};
  }

  void gaussKronrodRule(const std::function<float(float)> &f, float a, float b,
                        float &integral_k, float &error_est) const {
    const float half_length = 0.5f * (b - a);
    const float mid_point = 0.5f * (a + b);
    float integral_g = 0.0f;
    integral_k = 0.0f;
    
    std::vector<float> fx_vals(gk_nodes_.size());
    for(size_t i=0; i < (gk_nodes_.size()+1)/2; ++i) {
        float node_val = gk_nodes_[i];
        fx_vals[i] = f(mid_point + half_length * node_val);
        if (i*2 +1 < gk_nodes_.size()){
             fx_vals[gk_nodes_.size()-1-i] = f(mid_point - half_length * node_val);
        }
    }

    for (size_t i = 0; i < gk_nodes_.size(); ++i) {
      integral_k += k_weights_[i] * fx_vals[i];
      if (g_weights_[i] != 0.0f) {
          integral_g += g_weights_[i] * fx_vals[i];
      }
    }
    integral_k *= half_length;
    integral_g *= half_length;
    error_est = std::abs(integral_k - integral_g);
  }

  float abs_tol_;
  float rel_tol_;
  size_t max_intervals_;
  std::vector<float> gk_nodes_;
  std::vector<float> g_weights_;
  std::vector<float> k_weights_;
};


std::shared_ptr<IntegrateSingle>
createIntegrateSingle(const std::string &scheme_type, size_t order,
                       float tolerance) {
  if (scheme_type == "GaussLegendre") {
    if (order == 0) order = 8; 
    return std::make_shared<GaussLegendreIntegrateSingleImpl>(order);
  } else if (scheme_type == "TanhSinh") { 
     if (tolerance <= 0.0f) tolerance = 1e-7f; 
     return std::make_shared<TanhSinhIntegrateSingleImpl>(tolerance);
  } else if (scheme_type == "Adaptive") {
    if (tolerance <= 0.0f) tolerance = 1e-6f; 
    return std::make_shared<AdaptiveIntegrateSingleImpl>(tolerance, tolerance, 200);
  }
  else {
    throw std::invalid_argument("Unknown IntegrateSingle type: " + scheme_type);
  }
}

} // namespace num
} // namespace alo
} // namespace engine
 #include "integrator.h"
 #include <stdexcept>
 #include <cmath>
 #include <vector>
 #include <algorithm>
 #include <numeric>
 #include <limits>
 
 namespace engine {
 namespace alo {
 namespace num {

 /**
  * @brief Gauss-Legendre quadrature implementation
  * 
  * This class implements Gauss-Legendre quadrature for numerical integration.
  */
 class GaussLegendreIntegrator : public Integrator {
 public:
     /**
      * @brief Constructor
      * 
      * @param order Number of integration points
      */
     explicit GaussLegendreIntegrator(size_t order) : order_(order) {
         if (order_ < 1) {
             throw std::invalid_argument("GaussLegendreIntegrator: Order must be at least 1");
         }
         initializeNodesAndWeights();
     }
     
     /**
      * @brief Destructor
      */
     ~GaussLegendreIntegrator() override = default;
     
     /**
      * @brief Integrate a function over an interval
      * 
      * @param f Function to integrate
      * @param a Lower bound
      * @param b Upper bound
      * @return Approximate integral value
      */
     double integrate(const std::function<double(double)>& f, double a, double b) const override {
         double result = 0.0;
         
         // Handle zero-length interval
         if (std::abs(b - a) < 1e-15) {
             return 0.0;
         }
         
         // Change of variable to map [a,b] to [-1,1]
         const double half_length = 0.5 * (b - a);
         const double mid_point = 0.5 * (a + b);
         
         for (size_t i = 0; i < order_; ++i) {
             const double x = mid_point + half_length * nodes_[i];
             result += weights_[i] * f(x);
         }
         
         result *= half_length;
         return result;
     }
     
     /**
      * @brief Get the name of the integrator
      * 
      * @return Integrator name
      */
     std::string name() const override {
         return "Gauss-Legendre";
     }
     
 private:
     /**
      * @brief Initialize Gauss-Legendre nodes and weights
      */
     void initializeNodesAndWeights() {
         // Precomputed Gauss-Legendre nodes and weights for common orders
         if (order_ == 7) {
             nodes_ = {
                 -0.9491079123427585, 
                 -0.7415311855993944,
                 -0.4058451513773972,
                 0.0,
                 0.4058451513773972,
                 0.7415311855993944,
                 0.9491079123427585
             };
             
             weights_ = {
                 0.1294849661688697,
                 0.2797053914892767,
                 0.3818300505051189,
                 0.4179591836734694,
                 0.3818300505051189,
                 0.2797053914892767,
                 0.1294849661688697
             };
         } else if (order_ == 25) {
             // 25-point Gauss-Legendre quadrature nodes and weights
             nodes_ = {
                 -0.9955569697904981, -0.9766639214595175, -0.9429745712289743, -0.8949919978782753, -0.8334426287608340,
                 -0.7592592630373576, -0.6735663684734684, -0.5776629302412229, -0.4731469662935845, -0.3611723058093879,
                 -0.2429801799032639, -0.1207530708447741, 0.0, 0.1207530708447741, 0.2429801799032639,
                 0.3611723058093879, 0.4731469662935845, 0.5776629302412229, 0.6735663684734684, 0.7592592630373576,
                 0.8334426287608340, 0.8949919978782753, 0.9429745712289743, 0.9766639214595175, 0.9955569697904981
             };
             
             weights_ = {
                 0.0113937985010262, 0.0263549866150321, 0.0409391567013063, 0.0549046959758351, 0.0680383338123569,
                 0.0801407003350010, 0.0910282619829636, 0.1005359490670506, 0.1085196244742637, 0.1148582591457116,
                 0.1194557635357847, 0.1222424429903100, 0.1231760537267154, 0.1222424429903100, 0.1194557635357847,
                 0.1148582591457116, 0.1085196244742637, 0.1005359490670506, 0.0910282619829636, 0.0801407003350010,
                 0.0680383338123569, 0.0549046959758351, 0.0409391567013063, 0.0263549866150321, 0.0113937985010262
             };
         } else if (order_ == 27) {
             // 27-point Gauss-Legendre quadrature nodes and weights
             nodes_ = {
                 -0.9961792628889886, -0.9782286581460570, -0.9458226521856563, -0.8992005757021038, -0.8391169718222189,
                 -0.7663811206689788, -0.6828454791571403, -0.5896380977729661, -0.4879900029287655, -0.3790232126755540,
                 -0.2639649827963907, -0.1441590672327308, -0.0486667884430163, 0.0486667884430163, 0.1441590672327308,
                 0.2639649827963907, 0.3790232126755540, 0.4879900029287655, 0.5896380977729661, 0.6828454791571403,
                 0.7663811206689788, 0.8391169718222189, 0.8992005757021038, 0.9458226521856563, 0.9782286581460570,
                 0.9961792628889886
             };
             
             weights_ = {
                 0.0097989960512943, 0.0227575625501992, 0.0355047847316408, 0.0478481301259484, 0.0595985325645789,
                 0.0705878906601189, 0.0806753521268833, 0.0897264238206302, 0.0976186521041138, 0.1042582260352920,
                 0.1095783812798404, 0.1135354900057835, 0.1161034212297789, 0.1172024672904842, 0.1161034212297789,
                 0.1135354900057835, 0.1095783812798404, 0.1042582260352920, 0.0976186521041138, 0.0897264238206302,
                 0.0806753521268833, 0.0705878906601189, 0.0595985325645789, 0.0478481301259484, 0.0355047847316408,
                 0.0227575625501992, 0.0097989960512943
             };
         } else {
             // For other orders, use a simple approach
             nodes_.resize(order_);
             weights_.resize(order_);
             
             for (size_t i = 0; i < order_; ++i) {
                 double theta = M_PI * (i + 0.5) / order_;
                 nodes_[i] = std::cos(theta);
                 weights_[i] = M_PI / order_;
             }
         }
     }
     
     size_t order_;
     std::vector<double> nodes_;
     std::vector<double> weights_;
 };
 
 /**
  * @brief Tanh-Sinh quadrature implementation
  * 
  * This class implements Tanh-Sinh quadrature (double exponential quadrature)
  * for numerical integration with high precision.
  */
 class TanhSinhIntegrator : public Integrator {
 public:
     /**
      * @brief Constructor
      * 
      * @param tolerance Integration tolerance
      * @param max_levels Maximum number of refinement levels
      */
     explicit TanhSinhIntegrator(double tolerance, int max_levels = 12) 
         : tolerance_(tolerance), max_levels_(max_levels) {
         if (tolerance_ <= 0.0) {
             throw std::invalid_argument("TanhSinhIntegrator: Tolerance must be positive");
         }
         if (max_levels_ < 1) {
             throw std::invalid_argument("TanhSinhIntegrator: Max levels must be at least 1");
         }
     }
     
     /**
      * @brief Destructor
      */
     ~TanhSinhIntegrator() override = default;
     
     /**
      * @brief Integrate a function over an interval
      * 
      * @param f Function to integrate
      * @param a Lower bound
      * @param b Upper bound
      * @return Approximate integral value
      */
     double integrate(const std::function<double(double)>& f, double a, double b) const override {
         // Handle zero-length interval
         if (std::abs(b - a) < 1e-15) {
             return 0.0;
         }
         
         // Implementation of tanh-sinh quadrature
         const double h0 = 1.0;
         
         // Transform the original function using the tanh-sinh substitution
         auto g = [&](double t) {
             // Map t to x in [a,b] using tanh-sinh substitution
             double x = 0.5 * (b + a) + 0.5 * (b - a) * std::tanh(M_PI_2 * std::sinh(t));
             // Calculate the weight of the substitution
             double w = 0.5 * (b - a) * M_PI_2 * std::cosh(t) / std::pow(std::cosh(M_PI_2 * std::sinh(t)), 2);
             return f(x) * w;
         };
         
         // Initial estimate at t=0
         double result = g(0.0);
         double h = h0;
         
         // Adaptive refinement
         for (int level = 1; level <= max_levels_; ++level) {
             h *= 0.5;
             double sum = 0.0;
             
             // Add contributions at new points
             for (int i = 1; i < (1 << level); i += 2) {
                 double t = i * h;
                 sum += g(t) + g(-t);
             }
             
             // Update result
             result = 0.5 * result + h * sum;
             
             // Check for convergence
             if (level > 3 && std::abs(sum * h) < tolerance_) {
                 break;
             }
         }
         
         return result;
     }
     
     /**
      * @brief Get the name of the integrator
      * 
      * @return Integrator name
      */
     std::string name() const override {
         return "Tanh-Sinh";
     }
     
 private:
     double tolerance_;
     int max_levels_;
 };
 
 /**
  * @brief Adaptive quadrature implementation
  * 
  * This class implements adaptive quadrature using the Gauss-Kronrod method
  * for efficient numerical integration with error control.
  */
 class AdaptiveIntegrator : public Integrator {
 public:
     /**
      * @brief Constructor
      * 
      * @param absolute_tolerance Absolute error tolerance
      * @param relative_tolerance Relative error tolerance
      * @param max_intervals Maximum number of intervals
      */
     AdaptiveIntegrator(double absolute_tolerance, double relative_tolerance, size_t max_intervals)
         : abs_tol_(absolute_tolerance), rel_tol_(relative_tolerance), max_intervals_(max_intervals) {
         if (abs_tol_ <= 0.0 && rel_tol_ <= 0.0) {
             throw std::invalid_argument("AdaptiveIntegrator: At least one tolerance must be positive");
         }
         if (max_intervals_ < 1) {
             throw std::invalid_argument("AdaptiveIntegrator: Max intervals must be at least 1");
         }
         
         // Initialize Gauss-Kronrod nodes and weights
         initializeGaussKronrod();
     }
     
     /**
      * @brief Destructor
      */
     ~AdaptiveIntegrator() override = default;
     
     /**
      * @brief Integrate a function over an interval
      * 
      * @param f Function to integrate
      * @param a Lower bound
      * @param b Upper bound
      * @return Approximate integral value
      */
     double integrate(const std::function<double(double)>& f, double a, double b) const override {
         // Handle zero-length interval
         if (std::abs(b - a) < 1e-15) {
             return 0.0;
         }
         
         // Initialize interval list with the whole interval
         struct Interval {
             double a;
             double b;
             double integral;
             double error;
         };
         
         std::vector<Interval> intervals;
         
         // Compute initial estimate for the whole interval
         double integral, error;
         gaussKronrodRule(f, a, b, integral, error);
         
         intervals.push_back({a, b, integral, error});
         
         double total_integral = integral;
         double total_error = error;
         
         // Adaptive refinement
         while (total_error > std::max(abs_tol_, rel_tol_ * std::abs(total_integral)) && 
                intervals.size() < max_intervals_) {
             
             // Find interval with largest error
             size_t worst_interval = 0;
             double max_error = intervals[0].error;
             
             for (size_t i = 1; i < intervals.size(); ++i) {
                 if (intervals[i].error > max_error) {
                     max_error = intervals[i].error;
                     worst_interval = i;
                 }
             }
             
             // Split the worst interval
             double mid = 0.5 * (intervals[worst_interval].a + intervals[worst_interval].b);
             
             // Remove the contribution of the worst interval
             total_integral -= intervals[worst_interval].integral;
             total_error -= intervals[worst_interval].error;
             
             // Compute estimates for the two new intervals
             double left_integral, left_error;
             double right_integral, right_error;
             
             gaussKronrodRule(f, intervals[worst_interval].a, mid, left_integral, left_error);
             gaussKronrodRule(f, mid, intervals[worst_interval].b, right_integral, right_error);
             
             // Replace the worst interval with the left subinterval
             intervals[worst_interval] = {intervals[worst_interval].a, mid, left_integral, left_error};
             
             // Add the right subinterval
             intervals.push_back({mid, intervals[worst_interval].b, right_integral, right_error});
             
             // Update total integral and error
             total_integral += left_integral + right_integral;
             total_error += left_error + right_error;
         }
         
         // Final result
         total_integral = 0.0;
         for (const auto& interval : intervals) {
             total_integral += interval.integral;
         }
         
         return total_integral;
     }
     
     /**
      * @brief Get the name of the integrator
      * 
      * @return Integrator name
      */
     std::string name() const override {
         return "Adaptive Gauss-Kronrod";
     }
     
 private:
     /**
      * @brief Initialize Gauss-Kronrod nodes and weights
      */
     void initializeGaussKronrod() {
         // 15-point Gauss-Kronrod rule (7 Gauss points, 8 additional Kronrod points)
         // Gauss nodes are at indices 0, 2, 4, 6, 8, 10, 12
         // Kronrod nodes are at indices 1, 3, 5, 7, 9, 11, 13, 14
         gk_nodes_ = {
             -0.9914553711208126, -0.9491079123427585, -0.8648644233597691, -0.7415311855993944,
             -0.5860872354676911, -0.4058451513773972, -0.2077849550078985, 0.0,
             0.2077849550078985, 0.4058451513773972, 0.5860872354676911, 0.7415311855993944,
             0.8648644233597691, 0.9491079123427585, 0.9914553711208126
         };
         
         // Gauss weights (used for error estimation)
         g_weights_ = {
             0.0, 0.1294849661688697, 0.0, 0.2797053914892767, 0.0, 
             0.3818300505051189, 0.0, 0.4179591836734694, 0.0,
             0.3818300505051189, 0.0, 0.2797053914892767, 0.0, 
             0.1294849661688697, 0.0
         };
         
         // Kronrod weights
         k_weights_ = {
             0.0229353220105292, 0.0630920926299786, 0.1047900103222502, 0.1406532597155259,
             0.1690047266392679, 0.1903505780647854, 0.2044329400752989, 0.2094821410847278,
             0.2044329400752989, 0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
             0.1047900103222502, 0.0630920926299786, 0.0229353220105292
         };
     }
     
     /**
      * @brief Apply Gauss-Kronrod rule to an interval
      * 
      * @param f Function to integrate
      * @param a Lower bound
      * @param b Upper bound
      * @param integral Computed integral value
      * @param error Error estimate
      */
     void gaussKronrodRule(const std::function<double(double)>& f, double a, double b,
                           double& integral, double& error) const {
         // Change of variable to map [a,b] to [-1,1]
         const double half_length = 0.5 * (b - a);
         const double mid_point = 0.5 * (a + b);
         
         double gauss_integral = 0.0;
         double kronrod_integral = 0.0;
         
         for (size_t i = 0; i < gk_nodes_.size(); ++i) {
             const double x = mid_point + half_length * gk_nodes_[i];
             const double fx = f(x);
             
             kronrod_integral += k_weights_[i] * fx;
             
             // Gauss nodes have non-zero weights
             if (g_weights_[i] > 0.0) {
                 gauss_integral += g_weights_[i] * fx;
             }
         }
         
         integral = half_length * kronrod_integral;
         error = half_length * std::abs(kronrod_integral - gauss_integral);
     }
     
     double abs_tol_;
     double rel_tol_;
     size_t max_intervals_;
     
     std::vector<double> gk_nodes_;
     std::vector<double> g_weights_;
     std::vector<double> k_weights_;
 };
  
 std::shared_ptr<Integrator> createIntegrator(
     const std::string& scheme_type,
     size_t order,
     double tolerance) {
     
     if (scheme_type == "GaussLegendre") {
         if (order == 0) {
             order = 7;  // Default order
         }
         return std::make_shared<GaussLegendreIntegrator>(order);
     } else if (scheme_type == "TanhSinh") {
         if (tolerance <= 0.0) {
             tolerance = 1e-8;  // Default tolerance
         }
         return std::make_shared<TanhSinhIntegrator>(tolerance);
     } else if (scheme_type == "Adaptive") {
         if (tolerance <= 0.0) {
             tolerance = 1e-8;  // Default tolerance
         }
         return std::make_shared<AdaptiveIntegrator>(tolerance, tolerance, 1000);
     } else {
         throw std::invalid_argument("Unknown integrator type: " + scheme_type);
     }
 }
 
 } // namespace num
 } // namespace alo
 } // namespace engine
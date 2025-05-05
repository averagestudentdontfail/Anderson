 #include "american.h"
 #include "../num/chebyshev.h"
 #include "../num/integrate.h"
 #include <cmath>
 #include <stdexcept>
 #include <limits>
 
 namespace engine {
 namespace alo {
 namespace mod {
 
 AmericanOption::AmericanOption(std::shared_ptr<num::Integrator> integrator)
     : integrator_(integrator) {
     if (!integrator_) {
         throw std::invalid_argument("AmericanOption: Integrator cannot be null");
     }
 }
  
 AmericanPut::AmericanPut(std::shared_ptr<num::Integrator> integrator)
     : AmericanOption(integrator) {
 }
 
 double AmericanPut::xMax(double K, double r, double q) const {
     // Table 2 from the ALO paper for puts
     if (r > 0.0 && q > 0.0)
         return K * std::min(1.0, r / q);
     else if (r > 0.0 && q <= 0.0)
         return K;
     else if (r == 0.0 && q < 0.0)
         return K;
     else if (r == 0.0 && q >= 0.0)
         return 0.0; // European case
     else if (r < 0.0 && q >= 0.0)
         return 0.0; // European case
     else if (r < 0.0 && q < r)
         return K; // double boundary case
     else if (r < 0.0 && r <= q && q < 0.0)
         return 0.0; // European case
     else
         throw std::runtime_error("Internal error in xMax calculation for put");
 }
 
 double AmericanPut::calculateEarlyExercisePremium(
     double S, double K, double r, double q, double vol, double T,
     const std::shared_ptr<num::ChebyshevInterpolation>& boundary) const {
     
     if (!boundary) {
         throw std::invalid_argument("AmericanPut: Boundary interpolation cannot be null");
     }
     
     const double xmax = xMax(K, r, q);
     
     // Function to map tau to z in [-1, 1]
     auto tauToZ = [T](double tau) -> double {
         return 2.0 * std::sqrt(tau / T) - 1.0;
     };
     
     // Function to get boundary value at tau
     auto B = [xmax, boundary, tauToZ](double tau) -> double {
         if (tau <= 0.0) return xmax;
         
         const double z = tauToZ(tau);
         return xmax * std::exp(-std::sqrt(std::max(0.0, (*boundary)(z, true))));
     };
     
     // Integrand for put early exercise premium
     auto integrand = [S, K, r, q, vol, B](double z) -> double {
         const double t = z * z; // tau = z^2
         const double b_t = B(t);
         
         if (b_t <= 0.0) return 0.0;
         
         const double dr = std::exp(-r * t);
         const double dq = std::exp(-q * t);
         const double v = vol * std::sqrt(t);
         
         if (v < 1e-10) {
             if (std::abs(S * dq - b_t * dr) < 1e-10)
                 return z * (r * K * dr - q * S * dq);
             else if (b_t * dr > S * dq)
                 return 2.0 * z * (r * K * dr - q * S * dq);
             else
                 return 0.0;
         }
         
         const double dp = (std::log(S * dq / (b_t * dr)) / v) + 0.5 * v;
         
         return 2.0 * z * (r * K * dr * (0.5 * (1.0 + std::erf((-dp + v) / std::sqrt(2.0)))) - 
                           q * S * dq * (0.5 * (1.0 + std::erf(-dp / std::sqrt(2.0)))));
     };
     
     // Integrate to get early exercise premium
     return integrator_->integrate(integrand, 0.0, std::sqrt(T));
 }
 
 std::shared_ptr<num::ChebyshevInterpolation> AmericanPut::calculateExerciseBoundary(
     double S, double K, double r, double q, double vol, double T,
     size_t num_nodes, size_t num_iterations,
     std::shared_ptr<num::Integrator> fpIntegrator) const {
     
     if (!fpIntegrator) {
         throw std::invalid_argument("AmericanPut: Fixed point integrator cannot be null");
     }
     
     const double xmax = xMax(K, r, q);
     
     // Initialize interpolation nodes
     std::vector<double> nodes(num_nodes);
     std::vector<double> y(num_nodes, 0.0); // Boundary function values at nodes
     
     // Chebyshev nodes of the second kind in [-1, 1]
     for (size_t i = 0; i < num_nodes; ++i) {
         nodes[i] = std::cos(M_PI * i / (num_nodes - 1)); // x_i in [-1, 1]
     }
     
     // Create initial interpolation
     auto interp = num::createChebyshevInterpolation(
         nodes, y, num::SECOND_KIND, -1.0, 1.0);
     
     // Function to map tau to z in [-1, 1]
     auto tauToZ = [T](double tau) -> double {
         return 2.0 * std::sqrt(tau / T) - 1.0;
     };
     
     // Function to get boundary value at tau
     auto B = [xmax, &interp, tauToZ](double tau) -> double {
         if (tau <= 0.0) return xmax;
         
         const double z = tauToZ(tau);
         return xmax * std::exp(-std::sqrt(std::max(0.0, (*interp)(z, true))));
     };
     
     // Function to map boundary value to interpolation y-value
     auto h = [xmax](double fv) -> double {
         return std::pow(std::log(fv / xmax), 2);
     };
     
     // Create fixed point evaluator
     auto evaluator = createFixedPointEvaluator('A', K, r, q, vol, B, fpIntegrator);
     
     // Perform fixed point iterations
     // First is a Jacobi-Newton step
     for (size_t k = 0; k < 1; ++k) {
         for (size_t i = 1; i < num_nodes; ++i) { // Skip first node (corresponds to tau=0)
             const double z = nodes[i];
             const double tau = T * std::pow(0.5 * (1.0 + z), 2);
             const double b = B(tau);
             
             const auto [N, D, fv] = evaluator->evaluate(tau, b);
             
             if (tau < 1e-10) {
                 y[i] = h(fv);
             } else {
                 const auto [Nd, Dd] = evaluator->derivatives(tau, b);
                 
                 // Newton step
                 const double fd = K * std::exp(-(r - q) * tau) * (Nd / D - Dd * N / (D * D));
                 const double b_new = b - (fv - b) / (fd - 1.0);
                 
                 y[i] = h(b_new);
             }
         }
         
         // Update interpolation
         interp->updateValues(y);
     }
     
     // Remaining iterations are standard fixed point
     for (size_t k = 1; k < num_iterations; ++k) {
         for (size_t i = 1; i < num_nodes; ++i) { // Skip first node (corresponds to tau=0)
             const double z = nodes[i];
             const double tau = T * std::pow(0.5 * (1.0 + z), 2);
             const double b = B(tau);
             
             const auto [N, D, fv] = evaluator->evaluate(tau, b);
             
             y[i] = h(fv);
         }
         
         // Update interpolation
         interp->updateValues(y);
     }
     
     return interp;
 }
  
 AmericanCall::AmericanCall(std::shared_ptr<num::Integrator> integrator)
     : AmericanOption(integrator) {
 }
 
 double AmericanCall::xMax(double K, double r, double q) const {
     // For call options, the early exercise boundary is different
     if (q > 0.0 && r >= 0.0)
         return K * std::max(1.0, r / q);
     else if (q <= 0.0 && r >= 0.0)
         return std::numeric_limits<double>::infinity(); // Effectively infinite, early exercise never optimal
     else if (q >= r && r < 0.0)
         return std::numeric_limits<double>::infinity(); // European case
     else if (q < r && r < 0.0)
         return K; // Double boundary case
     else
         throw std::runtime_error("Internal error in xMax calculation for call");
 }
 
 double AmericanCall::calculateEarlyExercisePremium(
     double S, double K, double r, double q, double vol, double T,
     const std::shared_ptr<num::ChebyshevInterpolation>& boundary) const {
     
     if (!boundary) {
         throw std::invalid_argument("AmericanCall: Boundary interpolation cannot be null");
     }
     
     // If zero dividend, calls should never be exercised early
     if (q <= 0.0) {
         return 0.0;
     }
     
     const double xmax = xMax(K, r, q);
     
     // For infinite boundary (no early exercise), return zero premium
     if (std::isinf(xmax) || xmax > 1e12) {
         return 0.0;
     }
     
     // Function to map tau to z in [-1, 1]
     auto tauToZ = [T](double tau) -> double {
         return 2.0 * std::sqrt(tau / T) - 1.0;
     };
     
     // Function to get boundary value at tau
     auto B = [xmax, boundary, tauToZ](double tau) -> double {
         if (tau <= 0.0) return xmax;
         
         const double z = tauToZ(tau);
         return xmax * std::exp(-std::sqrt(std::max(0.0, (*boundary)(z, true))));
     };
     
     // Integrand for call early exercise premium
     auto integrand = [S, K, r, q, vol, B](double z) -> double {
         const double t = z * z; // tau = z^2
         const double b_t = B(t);
         
         if (b_t <= 0.0 || std::isinf(b_t)) return 0.0;
         
         const double dr = std::exp(-r * t);
         const double dq = std::exp(-q * t);
         const double v = vol * std::sqrt(t);
         
         if (v < 1e-10) {
             if (std::abs(b_t * dq - K * dr) < 1e-10)
                 return z * (q * b_t * dq - r * K * dr);
             else if (b_t * dq > K * dr)
                 return 2.0 * z * (q * b_t * dq - r * K * dr);
             else
                 return 0.0;
         }
         
         const double dp = (std::log(b_t * dq / (K * dr)) / v) + 0.5 * v;
         
         return 2.0 * z * (q * b_t * dq * (0.5 * (1.0 + std::erf(dp / std::sqrt(2.0)))) - 
                           r * K * dr * (0.5 * (1.0 + std::erf((dp - v) / std::sqrt(2.0)))));
     };
     
     // Integrate to get early exercise premium
     return integrator_->integrate(integrand, 0.0, std::sqrt(T));
 }
 
 std::shared_ptr<num::ChebyshevInterpolation> AmericanCall::calculateExerciseBoundary(
     double S, double K, double r, double q, double vol, double T,
     size_t num_nodes, size_t num_iterations,
     std::shared_ptr<num::Integrator> fpIntegrator) const {
     
     if (!fpIntegrator) {
         throw std::invalid_argument("AmericanCall: Fixed point integrator cannot be null");
     }
     
     const double xmax = xMax(K, r, q);
     
     // For infinite boundary (no early exercise), return an empty boundary
     if (std::isinf(xmax) || xmax > 1e12) {
         std::vector<double> nodes(num_nodes);
         std::vector<double> y(num_nodes, 0.0);
         
         for (size_t i = 0; i < num_nodes; ++i) {
             nodes[i] = std::cos(M_PI * i / (num_nodes - 1));
         }
         
         return num::createChebyshevInterpolation(
             nodes, y, num::SECOND_KIND, -1.0, 1.0);
     }
     
     // Initialize interpolation nodes
     std::vector<double> nodes(num_nodes);
     std::vector<double> y(num_nodes, 0.0); // Boundary function values at nodes
     
     // Chebyshev nodes of the second kind in [-1, 1]
     for (size_t i = 0; i < num_nodes; ++i) {
         nodes[i] = std::cos(M_PI * i / (num_nodes - 1)); // x_i in [-1, 1]
     }
     
     // Create initial interpolation
     auto interp = num::createChebyshevInterpolation(
         nodes, y, num::SECOND_KIND, -1.0, 1.0);
     
     // Function to map tau to z in [-1, 1]
     auto tauToZ = [T](double tau) -> double {
         return 2.0 * std::sqrt(tau / T) - 1.0;
     };
     
     // Function to get boundary value at tau
     auto B = [xmax, &interp, tauToZ](double tau) -> double {
         if (tau <= 0.0) return xmax;
         
         const double z = tauToZ(tau);
         return xmax * std::exp(-std::sqrt(std::max(0.0, (*interp)(z, true))));
     };
     
     // Function to map boundary value to interpolation y-value
     auto h = [xmax](double fv) -> double {
         return std::pow(std::log(fv / xmax), 2);
     };
     
     // For calls, we swap r and q in the fixed point equation (put-call symmetry)
     auto evaluator = createFixedPointEvaluator('A', K, q, r, vol, B, fpIntegrator);
     
     // Perform fixed point iterations
     // First is a Jacobi-Newton step
     for (size_t k = 0; k < 1; ++k) {
         for (size_t i = 1; i < num_nodes; ++i) { // Skip first node (corresponds to tau=0)
             const double z = nodes[i];
             const double tau = T * std::pow(0.5 * (1.0 + z), 2);
             const double b = B(tau);
             
             const auto [N, D, fv] = evaluator->evaluate(tau, b);
             
             if (tau < 1e-10) {
                 y[i] = h(fv);
             } else {
                 const auto [Nd, Dd] = evaluator->derivatives(tau, b);
                 
                 // Newton step
                 const double fd = K * std::exp(-(q - r) * tau) * (Nd / D - Dd * N / (D * D));
                 const double b_new = b - (fv - b) / (fd - 1.0);
                 
                 y[i] = h(b_new);
             }
         }
         
         // Update interpolation
         interp->updateValues(y);
     }
     
     // Remaining iterations are standard fixed point
     for (size_t k = 1; k < num_iterations; ++k) {
         for (size_t i = 1; i < num_nodes; ++i) { // Skip first node (corresponds to tau=0)
             const double z = nodes[i];
             const double tau = T * std::pow(0.5 * (1.0 + z), 2);
             const double b = B(tau);
             
             const auto [N, D, fv] = evaluator->evaluate(tau, b);
             
             y[i] = h(fv);
         }
         
         // Update interpolation
         interp->updateValues(y);
     }
     
     return interp;
 }
  
 FixedPointEvaluator::FixedPointEvaluator(
     double K, double r, double q, double vol, 
     const std::function<double(double)>& B,
     std::shared_ptr<num::Integrator> integrator)
     : K_(K), r_(r), q_(q), vol_(vol), vol2_(vol * vol), 
       B_(B), integrator_(integrator) {}
 
 std::pair<double, double> FixedPointEvaluator::d(double t, double z) const {
     if (t <= 0.0 || z <= 0.0) {
         return {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
     }
     
     const double v = vol_ * std::sqrt(t);
     const double m = (std::log(z) + (r_ - q_) * t) / v + 0.5 * v;
     
     return {m, m - v};
 }
 
 double FixedPointEvaluator::normalCDF(double x) const {
     // Using standard error function implementation
     return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
 }
 
 double FixedPointEvaluator::normalPDF(double x) const {
     return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
 }
  
 EquationA::EquationA(
     double K, double r, double q, double vol, 
     const std::function<double(double)>& B,
     std::shared_ptr<num::Integrator> integrator)
     : FixedPointEvaluator(K, r, q, vol, B, integrator) {}
 
 std::tuple<double, double, double> EquationA::evaluate(double tau, double b) const {
     double N, D;
     const double eps = 1e-10;
     
     if (tau < eps * eps) {
         // For very small tau, use limit values
         if (std::abs(b - K_) < eps) {
             N = D = 0.5;
         } else {
             N = D = (b > K_) ? 1.0 : 0.0;
         }
     } else {
         const double stv = std::sqrt(tau) / vol_;
         
         // Integrate to get K1 + K2
         auto K12_integrand = [&](double y) -> double {
             const double m = 0.25 * tau * (1.0 + y) * (1.0 + y);
             const double df = std::exp(q_ * tau - q_ * m);
             
             if (y <= 5.0 * eps - 1.0) {
                 if (std::abs(b - B_(tau - m)) < eps) {
                     return df * stv / (std::sqrt(2.0 * M_PI));
                 } else {
                     return 0.0;
                 }
             } else {
                 const auto dp = d(m, b / B_(tau - m)).first;
                 return df * (0.5 * tau * (y + 1.0) * normalCDF(dp) + stv * normalPDF(dp));
             }
         };
         
         // Integrate to get K3
         auto K3_integrand = [&](double y) -> double {
             const double m = 0.25 * tau * (1.0 + y) * (1.0 + y);
             const double df = std::exp(r_ * tau - r_ * m);
             
             if (y <= 5.0 * eps - 1.0) {
                 if (std::abs(b - B_(tau - m)) < eps) {
                     return df * stv / (std::sqrt(2.0 * M_PI));
                 } else {
                     return 0.0;
                 }
             } else {
                 return df * stv * normalPDF(d(m, b / B_(tau - m)).second);
             }
         };
         
         double K12 = integrator_->integrate(K12_integrand, -1.0, 1.0);
         double K3 = integrator_->integrate(K3_integrand, -1.0, 1.0);
         
         const auto dpm = d(tau, b / K_);
         N = normalPDF(dpm.second) / vol_ / std::sqrt(tau) + r_ * K3;
         D = normalPDF(dpm.first) / vol_ / std::sqrt(tau) + normalCDF(dpm.first) + q_ * K12;
     }
     
     // Calculate function value
     const double alpha = K_ * std::exp(-(r_ - q_) * tau);
     double fv;
     
     if (tau < eps * eps) {
         if (std::abs(b - K_) < eps || b > K_) {
             fv = alpha;
         } else {
             if (std::abs(q_) < eps) {
                 fv = alpha * r_ * ((q_ < 0.0) ? -1.0 : 1.0) / eps;
             } else {
                 fv = alpha * r_ / q_;
             }
         }
     } else {
         fv = alpha * N / D;
     }
     
     return {N, D, fv};
 }
 
 std::pair<double, double> EquationA::derivatives(double tau, double b) const {
     double Nd, Dd;
     const double eps = 1e-10;
     
     if (tau < eps * eps) {
         if (std::abs(b - K_) < eps) {
             const double sqTau = std::sqrt(tau);
             Dd = M_2_SQRTPI * M_SQRT1_2 * (
                 -(0.5 * vol2_ + r_ - q_) / (b * vol_ * vol2_ * sqTau) + 1.0 / (b * vol_ * sqTau));
             Nd = M_2_SQRTPI * M_SQRT1_2 * (-0.5 * vol2_ + r_ - q_) / (b * vol_ * vol2_ * sqTau);
         } else {
             Dd = Nd = 0.0;
         }
     } else {
         const auto dpm = d(tau, b / K_);
         
         Dd = -normalPDF(dpm.first) * dpm.first / (b * vol2_ * tau) +
              normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau));
         Nd = -normalPDF(dpm.second) * dpm.second / (b * vol2_ * tau);
     }
     
     return {Nd, Dd};
 }
 
 EquationB::EquationB(
     double K, double r, double q, double vol, 
     const std::function<double(double)>& B,
     std::shared_ptr<num::Integrator> integrator)
     : FixedPointEvaluator(K, r, q, vol, B, integrator) {}
 
 std::tuple<double, double, double> EquationB::evaluate(double tau, double b) const {
     double N, D;
     const double eps = 1e-10;
     
     if (tau < eps * eps) {
         // For very small tau, use limit values
         if (std::abs(b - K_) < eps) {
             N = D = 0.5;
         } else if (b < K_) {
             N = D = 0.0;
         } else {
             N = D = 1.0;
         }
     } else {
         // Integrate for N and D
         auto N_integrand = [&](double u) -> double {
             const double df = std::exp(r_ * u);
             if (u >= tau * (1.0 - 5.0 * eps)) {
                 if (std::abs(b - B_(u)) < eps) {
                     return 0.5 * df;
                 } else {
                     return df * ((b < B_(u)) ? 0.0 : 1.0);
                 }
             } else {
                 return df * normalCDF(d(tau - u, b / B_(u)).second);
             }
         };
         
         auto D_integrand = [&](double u) -> double {
             const double df = std::exp(q_ * u);
             if (u >= tau * (1.0 - 5.0 * eps)) {
                 if (std::abs(b - B_(u)) < eps) {
                     return 0.5 * df;
                 } else {
                     return df * ((b < B_(u)) ? 0.0 : 1.0);
                 }
             } else {
                 return df * normalCDF(d(tau - u, b / B_(u)).first);
             }
         };
         
         double ni = integrator_->integrate(N_integrand, 0.0, tau);
         double di = integrator_->integrate(D_integrand, 0.0, tau);
         
         const auto dpm = d(tau, b / K_);
         
         N = normalCDF(dpm.second) + r_ * ni;
         D = normalCDF(dpm.first) + q_ * di;
     }
     
     // Calculate function value
     const double alpha = K_ * std::exp(-(r_ - q_) * tau);
     double fv;
     
     if (tau < eps * eps) {
         if (std::abs(b - K_) < eps || b > K_) {
             fv = alpha;
         } else {
             if (std::abs(q_) < eps) {
                 fv = alpha * r_ * ((q_ < 0.0) ? -1.0 : 1.0) / eps;
             } else {
                 fv = alpha * r_ / q_;
             }
         }
     } else {
         fv = alpha * N / D;
     }
     
     return {N, D, fv};
 }
 
 std::pair<double, double> EquationB::derivatives(double tau, double b) const {
     const double eps = 1e-10;
     
     if (tau < eps * eps) {
         return {0.0, 0.0};
     }
     
     const auto dpm = d(tau, b / K_);
     
     return {
         normalPDF(dpm.second) / (b * vol_ * std::sqrt(tau)),
         normalPDF(dpm.first) / (b * vol_ * std::sqrt(tau))
     };
 }
  
 std::shared_ptr<FixedPointEvaluator> createFixedPointEvaluator(
     char equation,
     double K, double r, double q, double vol, 
     const std::function<double(double)>& B,
     std::shared_ptr<num::Integrator> integrator) {
     
     if (equation == 'A') {
         return std::make_shared<EquationA>(K, r, q, vol, B, integrator);
     } else if (equation == 'B') {
         return std::make_shared<EquationB>(K, r, q, vol, B, integrator);
     } else {
         // Auto-select equation based on r and q
         if (std::abs(r - q) < 0.001) {
             return std::make_shared<EquationA>(K, r, q, vol, B, integrator);
         } else {
             return std::make_shared<EquationB>(K, r, q, vol, B, integrator);
         }
     }
 }
 
 } // namespace mod
 } // namespace alo
 } // namespace engine
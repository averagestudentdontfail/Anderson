// --- START OF FILE mod/american.cpp ---

#include "american.h"
#include "../num/chebyshev.h" 
#include "../num/integrate.h" 
#include "../num/float.h"     
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#include <algorithm> 

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT2
#define M_SQRT2 1.41421356237309504880
#endif
#ifndef INV_SQRT_2PI // 1/sqrt(2*PI)
#define INV_SQRT_2PI 0.39894228040143267794
#endif
#ifndef INV_SQRT_2PI_F // 1/sqrt(2*PI) float
#define INV_SQRT_2PI_F 0.3989422804f
#endif


namespace engine {
namespace alo {
namespace mod {

// Anonymous namespace for robust scalar helper functions
namespace {

// --- Robust Scalar Black-Scholes (double) ---
double scalar_bs_d1_dbl(double S, double K, double r, double q, double vol, double T) {
    if (vol <= 1e-12 || T <= 1e-12) { // Handle degenerate cases
        if (S == K) return (r - q) * std::sqrt(T) / (vol < 1e-9 ? 1e-9 : vol); // Avoid NaN if S=K, vol=0
        return (S > K) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity();
    }
    return (std::log(S / K) + (r - q + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
}

double scalar_normal_cdf_dbl(double x) {
    return 0.5 * (1.0 + std::erf(x / M_SQRT2));
}

double scalar_normal_pdf_dbl(double x) {
    return INV_SQRT_2PI * std::exp(-0.5 * x * x);
}

// --- Robust Scalar Black-Scholes (float) ---
float scalar_bs_d1_sgl(float S, float K, float r, float q, float vol, float T) {
    if (vol <= 1e-7f || T <= 1e-7f) {
         if (S == K) return (r - q) * std::sqrt(T) / (vol < 1e-7f ? 1e-7f : vol);
        return (S > K) ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity();
    }
    return (std::log(S / K) + (r - q + 0.5f * vol * vol) * T) / (vol * std::sqrt(T));
}
// For float N(x) and n(x), we use num::fast_normal_cdf and num::fast_normal_pdf from num/float.h


// --- Robust Scalar Barone-Adesi-Whaley Put Approximation (double) ---
double robust_scalar_baw_put_approx_dbl(double S, double K, double r, double q, double vol, double T) {
    if (T <= 1e-9) return std::max(0.0, K - S);
    if (vol <= 1e-9) return std::max(0.0, K * std::exp(-r * T) - S * std::exp(-q * T));

    double euro_put_price = K * std::exp(-r * T) * scalar_normal_cdf_dbl(-scalar_bs_d1_dbl(S, K, r, q, vol, T) + vol * std::sqrt(T)) -
                            S * std::exp(-q * T) * scalar_normal_cdf_dbl(-scalar_bs_d1_dbl(S, K, r, q, vol, T));

    if (r <= q || r <= 1e-9) {
        return std::max(euro_put_price, K - S);
    }

    double b_baw = r - q; // Cost of carry for BAW model
    double M_baw = 2.0 * r / (vol * vol);
    double N_baw = 2.0 * b_baw / (vol * vol);
    
    double q2_discriminant = (N_baw - 1.0) * (N_baw - 1.0) + 4.0 * M_baw;
    if (q2_discriminant < 0) return std::max(euro_put_price, K - S); // Should not happen for r > 0

    double q2 = 0.5 * (-(N_baw - 1.0) + std::sqrt(q2_discriminant));

    if (q2 <= 1.0 + 1e-9) { // Condition for S_star to be problematic or infinite
        return std::max(euro_put_price, K - S);
    }
    
    double S_star = K * q2 / (q2 - 1.0);

    if (S <= S_star) {
        return std::max(0.0, K-S); // Exercise if S is at or below critical price
    } else {
        // Calculate A2 term for premium (d1 here is for the European part of A2)
        // d1_S_star is d1(S, S_star, r, b_baw, vol, T) where b_baw is cost of carry used in BAW for dividends
        double d1_S_star = (std::log(S / S_star) + (b_baw + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        double A2 = (S_star / q2) * (1.0 - std::exp((b_baw - r) * T) * scalar_normal_cdf_dbl(-d1_S_star));
        
        double american_price = euro_put_price + A2 * std::pow(S / S_star, -q2);
        return std::max(american_price, K - S); // Cannot be less than intrinsic
    }
}

// --- Robust Scalar Barone-Adesi-Whaley Call Approximation (double) ---
double robust_scalar_baw_call_approx_dbl(double S, double K, double r, double q, double vol, double T) {
    if (q <= 1e-9) { // No (or negligible) dividend, American Call = European Call
        if (T <= 1e-9 || vol <= 1e-9) return std::max(0.0, S - K);
        double d1 = scalar_bs_d1_dbl(S,K,r,q,vol,T);
        double d2 = d1 - vol*std::sqrt(T);
        return S * std::exp(-q*T) * scalar_normal_cdf_dbl(d1) - K * std::exp(-r*T) * scalar_normal_cdf_dbl(d2);
    }
    if (T <= 1e-9) return std::max(0.0, S - K);
    if (vol <= 1e-9) return std::max(0.0, S * std::exp(-q * T) - K * std::exp(-r * T));

    double euro_call_price = S * std::exp(-q * T) * scalar_normal_cdf_dbl(scalar_bs_d1_dbl(S, K, r, q, vol, T)) -
                             K * std::exp(-r * T) * scalar_normal_cdf_dbl(scalar_bs_d1_dbl(S, K, r, q, vol, T) - vol * std::sqrt(T));

    double b_baw = r - q;
    double M_baw = 2.0 * r / (vol * vol);
    double N_baw = 2.0 * b_baw / (vol * vol);

    double q1_discriminant = (N_baw - 1.0) * (N_baw - 1.0) + 4.0 * M_baw;
     if (q1_discriminant < 0) return std::max(euro_call_price, S - K);

    double q1 = 0.5 * (-(N_baw - 1.0) - std::sqrt(q1_discriminant)); // Note: minus before sqrt for call's q1

    if (q1 >= -1e-9 || std::abs(q1 - 1.0) < 1e-9) { // S_star problematic
         return std::max(euro_call_price, S - K);
    }

    double S_star = K * q1 / (q1 - 1.0);
    
    if (S >= S_star) {
        return std::max(0.0, S-K);
    } else {
        double d1_S_star = (std::log(S / S_star) + (b_baw + 0.5 * vol * vol) * T) / (vol * std::sqrt(T));
        double A1 = -(S_star / q1) * (1.0 - std::exp((b_baw - r) * T) * scalar_normal_cdf_dbl(d1_S_star));
        
        double american_price = euro_call_price + A1 * std::pow(S / S_star, q1);
        return std::max(american_price, S - K);
    }
}


// --- Robust Scalar Barone-Adesi-Whaley Put Approximation (float) ---
float robust_scalar_baw_put_approx_sgl(float S, float K, float r, float q, float vol, float T) {
    if (T <= 1e-7f) return std::max(0.0f, K - S);
    if (vol <= 1e-7f) return std::max(0.0f, K * std::exp(-r * T) - S * std::exp(-q * T));

    float euro_put_price = K * std::exp(-r * T) * num::fast_normal_cdf(-scalar_bs_d1_sgl(S, K, r, q, vol, T) + vol * std::sqrt(T)) -
                           S * std::exp(-q * T) * num::fast_normal_cdf(-scalar_bs_d1_sgl(S, K, r, q, vol, T));

    if (r <= q || r <= 1e-7f) {
        return std::max(euro_put_price, K - S);
    }

    float b_baw = r - q;
    float M_baw = 2.0f * r / (vol * vol);
    float N_baw = 2.0f * b_baw / (vol * vol);
    
    float q2_discriminant = (N_baw - 1.0f) * (N_baw - 1.0f) + 4.0f * M_baw;
    if (q2_discriminant < 0.0f) return std::max(euro_put_price, K - S);

    float q2 = 0.5f * (-(N_baw - 1.0f) + std::sqrt(q2_discriminant));

    if (q2 <= 1.0f + 1e-7f) {
        return std::max(euro_put_price, K - S);
    }
    
    float S_star = K * q2 / (q2 - 1.0f);

    if (S <= S_star) {
        return std::max(0.0f, K-S);
    } else {
        float d1_S_star = (std::log(S / S_star) + (b_baw + 0.5f * vol * vol) * T) / (vol * std::sqrt(T));
        float A2 = (S_star / q2) * (1.0f - std::exp((b_baw - r) * T) * num::fast_normal_cdf(-d1_S_star));
        
        float american_price = euro_put_price + A2 * std::pow(S / S_star, -q2);
        return std::max(american_price, K-S);
    }
}

// --- Robust Scalar Barone-Adesi-Whaley Call Approximation (float) ---
float robust_scalar_baw_call_approx_sgl(float S, float K, float r, float q, float vol, float T) {
    if (q <= 1e-7f) { 
        if (T <= 1e-7f || vol <= 1e-7f) return std::max(0.0f, S - K);
        float d1 = scalar_bs_d1_sgl(S,K,r,q,vol,T);
        float d2 = d1 - vol*std::sqrt(T);
        return S * std::exp(-q*T) * num::fast_normal_cdf(d1) - K * std::exp(-r*T) * num::fast_normal_cdf(d2);
    }
    if (T <= 1e-7f) return std::max(0.0f, S - K);
    if (vol <= 1e-7f) return std::max(0.0f, S * std::exp(-q * T) - K * std::exp(-r * T));

    float euro_call_price = S * std::exp(-q * T) * num::fast_normal_cdf(scalar_bs_d1_sgl(S, K, r, q, vol, T)) -
                            K * std::exp(-r * T) * num::fast_normal_cdf(scalar_bs_d1_sgl(S, K, r, q, vol, T) - vol * std::sqrt(T));
    
    float b_baw = r - q;
    float M_baw = 2.0f * r / (vol*vol);
    float N_baw = 2.0f * b_baw / (vol*vol);

    float q1_discriminant = (N_baw - 1.0f) * (N_baw - 1.0f) + 4.0f * M_baw;
    if (q1_discriminant < 0.0f) return std::max(euro_call_price, S - K);

    float q1 = 0.5f * (-(N_baw - 1.0f) - std::sqrt(q1_discriminant));

    if (q1 >= -1e-7f || std::abs(q1 - 1.0f) < 1e-7f ) { 
         return std::max(euro_call_price, S - K);
    }

    float S_star = K * q1 / (q1 - 1.0f);
    
    if (S >= S_star) {
        return std::max(0.0f, S-K);
    } else {
        float d1_S_star = (std::log(S / S_star) + (b_baw + 0.5f * vol * vol) * T) / (vol * std::sqrt(T));
        float A1 = -(S_star / q1) * (1.0f - std::exp((b_baw - r) * T) * num::fast_normal_cdf(d1_S_star));
        
        float american_price = euro_call_price + A1 * std::pow(S / S_star, q1);
        return std::max(american_price, S - K);
    }
}

} // anonymous namespace


// ========================================================================= //
//                  DOUBLE PRECISION CLASS IMPLEMENTATIONS                   //
// ========================================================================= //

AmericanOptionDouble::AmericanOptionDouble(std::shared_ptr<num::IntegrateDouble> integrate_pricing)
    : integrate_pricing_(std::move(integrate_pricing)) { // Use std::move
  if (!integrate_pricing_) {
    throw std::invalid_argument("AmericanOptionDouble: Pricing integrator cannot be null");
  }
}

// --- AmericanPutDouble ---
AmericanPutDouble::AmericanPutDouble(std::shared_ptr<num::IntegrateDouble> integrate_pricing)
    : AmericanOptionDouble(std::move(integrate_pricing)) {}

double AmericanPutDouble::xMax(double K, double r, double q) const {
  return (r > 0.0 && r >= q && q > 1e-9) ? (K * r / q) : K; // Avoid division by zero for q
}

double AmericanPutDouble::calculateEarlyExercisePremium(
    double S, double K, double r, double q, double vol, double T,
    const std::shared_ptr<num::ChebyshevInterpolationDouble> &boundary) const {
  if (!boundary) {
    throw std::invalid_argument("AmericanPutDouble: Boundary interpolation cannot be null");
  }
  if (T <= 1e-9) return 0.0; // No time, no premium

  const double xmax_val = xMax(K, r, q); 

  auto B_func = [&](double tau_prime) -> double {
    if (tau_prime <= 1e-12) return K; 
    double z_interp = 2.0 * std::sqrt(tau_prime / T) - 1.0;
    z_interp = std::max(-1.0, std::min(1.0, z_interp)); 
    double h_B = (*boundary)(z_interp, true); 
    return xmax_val * std::exp(-std::sqrt(std::max(0.0, h_B)));
  };
  
  auto integrand = [&](double z_var) -> double { 
    if (z_var < 1e-9) return 0.0; 
    double t_prime = z_var * z_var * T; 
    double B_at_t_prime = B_func(t_prime);

    if (B_at_t_prime <= 1e-9 || vol <= 1e-9 || t_prime <= 1e-12) return 0.0;

    double sqrt_t_prime = z_var * std::sqrt(T); // == std::sqrt(t_prime)
    double d1_val = scalar_bs_d1_dbl(S, B_at_t_prime, r, q, vol, t_prime);
    double d2_val = d1_val - vol * sqrt_t_prime;

    double term1 = r * K * std::exp(-r * t_prime) * scalar_normal_cdf_dbl(-d2_val);
    double term2 = q * S * std::exp(-q * t_prime) * scalar_normal_cdf_dbl(-d1_val);
    
    return (term1 - term2) * 2.0 * T * z_var; 
  };
  return integrate_pricing_->integrate(integrand, 0.0, 1.0);
}

std::shared_ptr<num::ChebyshevInterpolationDouble>
AmericanPutDouble::calculateExerciseBoundary(
    double S_unused, double K, double r, double q, double vol, double T,
    size_t num_nodes, size_t num_iterations,
    std::shared_ptr<num::IntegrateDouble> integrate_fp) const {
  if (!integrate_fp) {
    throw std::invalid_argument("AmericanPutDouble: Fixed point integrator cannot be null");
  }
   if (T <= 1e-9) { 
    std::vector<double> nodes_std(num_nodes);
    const double xmax_val_temp = xMax(K,r,q);
    double h_K_val = (xmax_val_temp > 1e-9 && K > 1e-9) ? std::pow(std::log(K/xmax_val_temp),2.0) : 0.0;
    if (K <= 1e-9 && xmax_val_temp <= 1e-9) h_K_val = 0.0;
    else if (K > 1e-9 && xmax_val_temp <= 1e-9) h_K_val = 100.0;

    std::vector<double> y_vals(num_nodes, h_K_val); 
     for (size_t i = 0; i < num_nodes; ++i) {
        nodes_std[i] = (num_nodes == 1) ? 0.0 : (-1.0 + 2.0 * static_cast<double>(i) / (num_nodes -1.0));
    }
    std::sort(nodes_std.begin(), nodes_std.end()); // Ensure sorted for Chebyshev
    return std::make_shared<num::ChebyshevInterpolationDouble>(nodes_std, y_vals, num::SECOND_KIND, -1.0, 1.0);
  }

  const double xmax_val = xMax(K, r, q); 
  std::vector<double> nodes_std(num_nodes); 
  std::vector<double> y_boundary_transformed(num_nodes); 

  for (size_t i = 0; i < num_nodes; ++i) {
    nodes_std[i] = (num_nodes == 1) ? 0.0 : -std::cos(M_PI * static_cast<double>(i) / (static_cast<double>(num_nodes) - 1.0));
  }
   if (num_nodes > 1) {
        nodes_std[0] = -1.0; 
        nodes_std[num_nodes-1] = 1.0;
        // The formula -cos(i*pi/(N-1)) generates nodes in [-1, 1] in increasing order if i goes from N-1 down to 0.
        // Or cos(i*pi/(N-1)) generates nodes [1, -1].
        // Let's ensure it's sorted ascending for consistency.
        std::sort(nodes_std.begin(), nodes_std.end()); 
    }

  double initial_h_B = (xmax_val > 1e-9 && K > 1e-9) ? std::pow(std::log(K / xmax_val), 2.0) : 0.0;
  if (K <= 1e-9 && xmax_val <= 1e-9) initial_h_B = 0.0;
  else if (K > 1e-9 && xmax_val <= 1e-9) initial_h_B = 100.0; 
  
  std::fill(y_boundary_transformed.begin(), y_boundary_transformed.end(), initial_h_B);
  
  // Find index for tau=0 (z=-1)
  size_t tau0_node_idx = 0; 
  for(size_t k=0; k<num_nodes; ++k) { if(std::abs(nodes_std[k] - (-1.0)) < 1e-9) { tau0_node_idx = k; break; } }

  double K_div_xmax = (xmax_val > 1e-9) ? K / xmax_val : 1.0; // Avoid division by zero
  if (K_div_xmax <= 1e-9) y_boundary_transformed[tau0_node_idx] = 100.0; 
  else y_boundary_transformed[tau0_node_idx] = std::pow(std::log(K_div_xmax), 2.0);

  auto interp = std::make_shared<num::ChebyshevInterpolationDouble>(nodes_std, y_boundary_transformed, num::SECOND_KIND, -1.0, 1.0);

  auto B_func = [&](double tau_prime) -> double {
    if (tau_prime <= 1e-12) return K; 
    double z_interp = 2.0 * std::sqrt(tau_prime / T) - 1.0;
    z_interp = std::max(-1.0, std::min(1.0, z_interp));
    double h_B = (*interp)(z_interp, true);
    return xmax_val * std::exp(-std::sqrt(std::max(0.0, h_B)));
  };

  auto h_transform = [&](double fv) -> double {
    if (fv <= 1e-9 || xmax_val <= 1e-9) return 100.0; 
    double ratio = fv / xmax_val;
    if (ratio <= 1e-9) return 100.0; // log of near zero is very negative
    return std::pow(std::log(ratio), 2.0);
  };

  char eq_type = 'A'; 
  if (std::abs(r-q) >= 0.001 && !(r < 0.0 && q < r) ) eq_type = 'B'; 

  auto evaluator = createFixedPointEvaluatorDouble(eq_type, K, r, q, vol, B_func, integrate_fp);

  for (size_t iter_count = 0; iter_count < num_iterations; ++iter_count) {
      bool is_newton_step = (iter_count == 0); // Only first step is Newton
      for (size_t i = 0; i < num_nodes; ++i) {
        double z_node = nodes_std[i];
        double tau_node = T * 0.25 * (z_node + 1.0) * (z_node + 1.0);
        
        if (i == tau0_node_idx && std::abs(z_node - (-1.0)) < 1e-9) {
             y_boundary_transformed[i] = h_transform(K); 
             continue;
        }

        double b_current = B_func(tau_node); // Uses interp from previous iteration
        auto [N_val, D_val, fv_val] = evaluator->evaluate(tau_node, b_current);

        if(fv_val <= 1e-9) { // Boundary effectively zero
            y_boundary_transformed[i] = h_transform(1e-9); // Max h value
            continue;
        }

        if (is_newton_step) {
            if (tau_node < 1e-10) {
              y_boundary_transformed[i] = h_transform(fv_val);
            } else {
              auto [Nd_val, Dd_val] = evaluator->derivatives(tau_node, b_current);
              double fd_val = 0.0;
              if (std::abs(D_val) > 1e-12) {
                  fd_val = K * std::exp(-(r - q) * tau_node) * (Nd_val / D_val - Dd_val * N_val / (D_val * D_val));
              }
              
              if (std::abs(fd_val - 1.0) < 1e-9) { 
                  y_boundary_transformed[i] = h_transform(fv_val);
              } else {
                  double b_new = b_current - (fv_val - b_current) / (fd_val - 1.0);
                  if (!std::isfinite(b_new) || b_new <= 0) b_new = fv_val; 
                   y_boundary_transformed[i] = h_transform(std::max(1e-9, b_new)); 
              }
            }
        } else { // Richardson step
             y_boundary_transformed[i] = h_transform(std::max(1e-9, fv_val));
        }
      }
      interp->updateValues(y_boundary_transformed);
  }
  return interp;
}


// --- AmericanCallDouble ---
AmericanCallDouble::AmericanCallDouble(std::shared_ptr<num::IntegrateDouble> integrate_pricing)
    : AmericanOptionDouble(std::move(integrate_pricing)) {}

double AmericanCallDouble::xMax(double K, double r, double q) const {
  return (q > 0.0 && q >=r && r > 1e-9 ) ? (K*q/r) : K;
}

double AmericanCallDouble::calculateEarlyExercisePremium(
    double S, double K, double r, double q, double vol, double T,
    const std::shared_ptr<num::ChebyshevInterpolationDouble> &boundary) const {
  if (q <= 1e-9) return 0.0; 
  if (!boundary) throw std::invalid_argument("AmericanCallDouble: Boundary null");
  if (T <= 1e-9) return 0.0;

  const double xmax_val = xMax(K, r, q);
  if (std::isinf(xmax_val)) return 0.0; // Should be caught by q <= 1e-9

  auto B_func = [&](double tau_prime) -> double {
    if (tau_prime <= 1e-12) return K; // Call boundary B(0)=K (or S if already optimal to exercise)
    double z_interp = 2.0 * std::sqrt(tau_prime / T) - 1.0;
    z_interp = std::max(-1.0, std::min(1.0, z_interp));
    double h_B = (*boundary)(z_interp, true);
    // For calls, upper boundary B > xmax, so B = xmax * exp(sqrt(h(B)))
    // However, the boundary is calculated via effective put, which uses exp(-sqrt(h_eff)).
    // The final h_B for the call is (ln(B_call/xmax_call))^2.
    // If the stored h_B is for B_call > xmax_call, then B_call = xmax_call * exp(sqrt(h_B))
    // If the stored h_B means B_call < xmax_call (unlikely for typical xmax_call=K), it would be exp(-sqrt).
    // The boundary passed here is the direct interpolation of h_call.
    return xmax_val * std::exp(std::sqrt(std::max(0.0, h_B)));
  };

  auto integrand = [&](double z_var) -> double { 
    if (z_var < 1e-9) return 0.0;
    double t_prime = z_var * z_var * T;
    double B_at_t_prime = B_func(t_prime);

    if (B_at_t_prime <= K + 1e-9 || !std::isfinite(B_at_t_prime) || vol <= 1e-9 || t_prime <= 1e-12) return 0.0; 

    double sqrt_t_prime = z_var * std::sqrt(T);
    double d1_val = scalar_bs_d1_dbl(S, B_at_t_prime, r, q, vol, t_prime); // S is spot, B is strike
    double d2_val = d1_val - vol * sqrt_t_prime;

    double term1 = q * S * std::exp(-q * t_prime) * scalar_normal_cdf_dbl(d1_val);    
    double term2 = r * K * std::exp(-r * t_prime) * scalar_normal_cdf_dbl(d2_val);    
    
    return (term1 - term2) * 2.0 * T * z_var; 
  };
  return integrate_pricing_->integrate(integrand, 0.0, 1.0);
}


std::shared_ptr<num::ChebyshevInterpolationDouble>
AmericanCallDouble::calculateExerciseBoundary(
    double S, double K, double r, double q, double vol, double T,
    size_t num_nodes, size_t num_iterations,
    std::shared_ptr<num::IntegrateDouble> integrate_fp) const {

  if (q <= 1e-9 || T <= 1e-9) { 
    std::vector<double> nodes_std(num_nodes);
    // h(B) = (ln(B/xmax))^2. If B is infinite (no early exercise), h(B) -> large.
    // xmax for call is K. If B_call->inf, ln(inf/K)^2 -> inf.
    std::vector<double> y_vals(num_nodes, 1000.0); // Large h(B) -> boundary far from K
    for (size_t i = 0; i < num_nodes; ++i) {
        nodes_std[i] = (num_nodes==1)? 0.0 : (-1.0 + 2.0 * static_cast<double>(i) / (num_nodes -1.0));
    }
     std::sort(nodes_std.begin(), nodes_std.end());
    return std::make_shared<num::ChebyshevInterpolationDouble>(nodes_std, y_vals, num::SECOND_KIND, -1.0, 1.0);
  }

  // Use effective put parameters
  double K_eff = S; // Strike for effective put is original Spot
  double S_eff = K; // Spot for effective put is original Strike
  double r_eff = q; // Risk-free rate for effective put is original dividend yield
  double q_eff = r; // Dividend yield for effective put is original risk-free rate

  AmericanPutDouble effective_put_pricer(integrate_fp); // Need any valid integrator for xMax
  // Calculate boundary for this effective put
  auto interp_eff_put_h = effective_put_pricer.calculateExerciseBoundary(
                                S_eff, K_eff, r_eff, q_eff, vol, T,
                                num_nodes, num_iterations, integrate_fp);

  // Now transform B_eff_put(tau) to B_call(tau) and then to h_call(tau)
  // B_call(tau) = S * K / B_eff_put(tau)
  // y_call_transformed[i] = h_call( B_call(tau_i) )
  
  const double xmax_call = xMax(K, r, q); // xmax for the actual call option
  const double xmax_eff_put_val = effective_put_pricer.xMax(K_eff, r_eff, q_eff);


  std::vector<double> y_call_transformed(num_nodes);
  const auto& nodes_std = interp_eff_put_h->getNodes(); // Get nodes from the computed effective put boundary

  auto h_transform_call = [&](double fv_call) -> double {
    if (fv_call <= 1e-9 || xmax_call <= 1e-9 || std::isinf(xmax_call)) return 1000.0; 
    double ratio = fv_call / xmax_call;
    if (ratio <= 1e-9) return 1000.0; 
    return std::pow(std::log(ratio), 2.0);
  };

  for(size_t i=0; i<num_nodes; ++i) {
    double z_node = nodes_std[i];
    double tau_node = T * 0.25 * (z_node + 1.0) * (z_node + 1.0);
    
    double h_eff_put = (*interp_eff_put_h)(z_node, true);
    double b_eff_put_val = xmax_eff_put_val * std::exp(-std::sqrt(std::max(0.0, h_eff_put)));
    
    if (tau_node <= 1e-12) { // At tau=0, B_call should be K (if S<=K initially for call) or S (if S>K)
                             // The ALO framework boundary is typically B*(0)=K
        y_call_transformed[i] = h_transform_call(K);
        continue;
    }

    if (b_eff_put_val < 1e-9) { 
        // This means B_eff_put is very small, so B_call = SK/B_eff_put will be very large
        y_call_transformed[i] = 1000.0; // Large h value
    } else {
        double b_call_val = (S * K) / b_eff_put_val;
        // For calls, boundary is typically B > K. If xmax_call = K, then B_call/xmax_call > 1.
        // The transformation h(B) = (ln(B/xmax))^2 works for B>xmax or B<xmax.
        // For calls, we expect B_call > xmax_call (if xmax_call = K), so ln term is positive.
        y_call_transformed[i] = h_transform_call(std::max(K + 1e-9, b_call_val)); // Ensure B_call > K
    }
  }
  return std::make_shared<num::ChebyshevInterpolationDouble>(nodes_std, y_call_transformed, num::SECOND_KIND, -1.0, 1.0);
}


// --- FixedPointEvaluatorDouble ---
// (Constructor, d_black_scholes, normalCDF, normalPDF are the same as before)
FixedPointEvaluatorDouble::FixedPointEvaluatorDouble(
    double K_val, double r_val, double q_val, double vol_val,
    const std::function<double(double)> &B_func,
    std::shared_ptr<num::IntegrateDouble> integrate_instance)
    : K_value_(K_val), r_rate_(r_val), q_yield_(q_val), volatility_(vol_val),
      vol_sq_(vol_val * vol_val), B_boundary_func_(B_func), integrate_fp_(std::move(integrate_instance)) {}

std::pair<double, double> FixedPointEvaluatorDouble::d_black_scholes(double t, double spot_ratio_K) const {
  if (t <= 1e-12 || spot_ratio_K <= 1e-12 || volatility_ <= 1e-9) {
    bool spot_is_large = spot_ratio_K > 1.0/ (1e-9); 
    bool spot_is_small = spot_ratio_K < 1e-9;      
    if (spot_is_large) return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    if (spot_is_small) return {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    if (volatility_ <= 1e-9 || t <= 1e-12) { // If vol or t is zero
      if (std::log(spot_ratio_K) > 0) // Simplified check, more robust might be needed
          return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
      else
          return {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
    }
  }
  const double v_sqrt_t = volatility_ * std::sqrt(t);
  if (std::abs(v_sqrt_t) < 1e-12) { // Denominator is zero
    if (std::log(spot_ratio_K) + (r_rate_ - q_yield_ + 0.5 * vol_sq_) * t > 0)
        return {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
    else
        return {-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
  }
  const double d1 = (std::log(spot_ratio_K) + (r_rate_ - q_yield_ + 0.5 * vol_sq_) * t) / v_sqrt_t;
  const double d2 = d1 - v_sqrt_t;
  return {d1, d2};
}

double FixedPointEvaluatorDouble::normalCDF(double x) const { return scalar_normal_cdf_dbl(x); }
double FixedPointEvaluatorDouble::normalPDF(double x) const { return scalar_normal_pdf_dbl(x); }


// --- EquationADouble & EquationBDouble implementations are the same as provided previously ---
// ... (assuming they are correct based on prior fixes)
EquationADouble::EquationADouble(
    double K_val, double r_val, double q_val, double vol_val,
    const std::function<double(double)> &B_func,
    std::shared_ptr<num::IntegrateDouble> integrate_instance)
    : FixedPointEvaluatorDouble(K_val, r_val, q_val, vol_val, B_func, std::move(integrate_instance)) {}

std::tuple<double, double, double> EquationADouble::evaluate(double tau, double b_boundary) const {
    double N_val, D_val;
    const double eps = 1e-10; 

    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps) { 
            const double v_eff = volatility_ * std::sqrt(std::max(tau, eps*eps)); 
            if (std::abs(v_eff) < 1e-12) { // Avoid division by zero if v_eff is zero
                 N_val = (b_boundary > K_value_) ? 0.0 : ((b_boundary < K_value_) ? 0.0 : 0.5); // Simplified based on payoff
                 D_val = (b_boundary > K_value_) ? 1.0 : ((b_boundary < K_value_) ? 0.0 : 0.5);
            } else {
                N_val = scalar_normal_pdf_dbl(0.0) / v_eff; // Approx phi(0)/v
                D_val = N_val + 0.5; // Approx phi(0)/v + N(0)
            }
        } else {
            N_val = 0.0;
            D_val = (b_boundary > K_value_) ? 1.0 : 0.0;
        }
    } else { // tau is not small
        const double stv_inv = 1.0 / (volatility_ * std::sqrt(tau));

        auto integrand_K12 = [&](double y_cheby) -> double {
            const double m_integral = 0.25 * tau * (1.0 + y_cheby) * (1.0 + y_cheby);
            if (tau - m_integral < 1e-12) return 0.0;
            const double B_tau_minus_m = B_boundary_func_(tau - m_integral);
            if (B_tau_minus_m <= 1e-9) return 0.0;
            const double df_q = std::exp(q_yield_ * (tau - m_integral));

            if (m_integral < eps * eps) { // If m is very small
                 if (std::abs(b_boundary - B_tau_minus_m) < eps)
                    return df_q * stv_inv * scalar_normal_pdf_dbl(0.0); // Approx stv*phi(0)
                else return 0.0;
            } else {
                const auto d_pair_integral = d_black_scholes(m_integral, b_boundary / B_tau_minus_m);
                return df_q * (0.5 * tau * (y_cheby + 1.0) * normalCDF(d_pair_integral.first) + 
                               (1.0/volatility_)*std::sqrt(tau)*normalPDF(d_pair_integral.first));
            }
        };
        auto integrand_K3 = [&](double y_cheby) -> double {
            const double m_integral = 0.25 * tau * (1.0 + y_cheby) * (1.0 + y_cheby);
            if (tau - m_integral < 1e-12) return 0.0;
            const double B_tau_minus_m = B_boundary_func_(tau - m_integral);
            if (B_tau_minus_m <= 1e-9) return 0.0;
            const double df_r = std::exp(r_rate_ * (tau - m_integral));
             if (m_integral < eps*eps) {
                if (std::abs(b_boundary - B_tau_minus_m) < eps)
                    return df_r * stv_inv * scalar_normal_pdf_dbl(0.0);
                else return 0.0;
            } else {
                 const auto d_pair_integral = d_black_scholes(m_integral, b_boundary / B_tau_minus_m);
                return df_r * (1.0/volatility_)*std::sqrt(tau) * normalPDF(d_pair_integral.second);
            }
        };
        double K12_val = integrate_fp_->integrate(integrand_K12, -1.0, 1.0);
        double K3_val = integrate_fp_->integrate(integrand_K3, -1.0, 1.0);
        const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
        N_val = normalPDF(d_pair_direct.second) * stv_inv + r_rate_ * K3_val;
        D_val = normalPDF(d_pair_direct.first) * stv_inv + normalCDF(d_pair_direct.first) + q_yield_ * K12_val;
    }

    const double alpha = K_value_ * std::exp(-(r_rate_ - q_yield_) * tau);
    double fv_val;
    if (tau < eps * eps) {
        // This logic is for American Puts from QuantLib, seems reasonable for fixed point target
        if (std::abs(b_boundary - K_value_) < eps) fv_val = alpha; // if b=K, fv=K*df_r_q
        else if (b_boundary > K_value_) fv_val = alpha;          // if b>K (OTM put), hold value is positive
        else { // b_boundary < K_value_ (ITM put)
             if (std::abs(q_yield_) < eps) {
                fv_val = alpha * r_rate_ * ((q_yield_ < 0.0) ? -1.0 : 1.0) / eps; 
             } else {
                fv_val = alpha * r_rate_ / q_yield_; 
             }
             // If r_rate_ <= 0 and q_yield_ > 0 for a put, early exercise is generally not optimal near expiry, value tends to European
             // The fv here is the target for the boundary iteration, not the option price itself.
        }
    } else {
        if (std::abs(D_val) < 1e-12) {
            // If D is zero, it implies singular behavior.
            // If N is also zero, fv might be K*alpha. If N is non-zero, fv is infinite.
            // This needs careful handling based on ALO paper's limiting cases.
            // For now, a large value if N > 0, or K*alpha if N also small.
            fv_val = (std::abs(N_val) < 1e-9) ? alpha : ( (N_val > 0) ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity() );
        }
        else fv_val = alpha * N_val / D_val;
    }
    return {N_val, D_val, std::max(0.0, fv_val)}; // Ensure fv is not negative
}

std::pair<double, double> EquationADouble::derivatives(double tau, double b_boundary) const {
    double Nd_prime, Dd_prime;
    const double eps = 1e-10;

    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps) {
            const double sqTau_eff = std::sqrt(std::max(tau, eps*eps)); 
            const double common_denom_part = (b_boundary * volatility_ * vol_sq_ * sqTau_eff);
            if (std::abs(common_denom_part) < 1e-15) { 
                 Dd_prime = Nd_prime = std::numeric_limits<double>::quiet_NaN(); // Indicates problem
            } else {
                Dd_prime = INV_SQRT_2PI * M_SQRT1_2 * 
                       (-(0.5 * vol_sq_ + r_rate_ - q_yield_) / common_denom_part + 1.0 / (b_boundary * volatility_ * sqTau_eff));
                Nd_prime = INV_SQRT_2PI * M_SQRT1_2 * (-0.5 * vol_sq_ + r_rate_ - q_yield_) / common_denom_part;
            }
        } else { Dd_prime = Nd_prime = 0.0; }
    } else {
        const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
        const double common_denom_tau = (b_boundary * vol_sq_ * tau);
        const double common_denom_sqrt_tau = (b_boundary * volatility_ * std::sqrt(tau));

        if (std::abs(common_denom_tau) < 1e-15 || std::abs(common_denom_sqrt_tau) < 1e-15 ) {
            Dd_prime = Nd_prime = std::numeric_limits<double>::quiet_NaN();
        } else {
            Dd_prime = -normalPDF(d_pair_direct.first) * d_pair_direct.first / common_denom_tau +
                       normalPDF(d_pair_direct.first) / common_denom_sqrt_tau;
            Nd_prime = -normalPDF(d_pair_direct.second) * d_pair_direct.second / common_denom_tau;
        }
    }
    return {Nd_prime, Dd_prime};
}

EquationBDouble::EquationBDouble(
    double K_val, double r_val, double q_val, double vol_val,
    const std::function<double(double)> &B_func,
    std::shared_ptr<num::IntegrateDouble> integrate_instance)
    : FixedPointEvaluatorDouble(K_val, r_val, q_val, vol_val, B_func, std::move(integrate_instance)) {}

std::tuple<double, double, double> EquationBDouble::evaluate(double tau, double b_boundary) const {
    double N_val, D_val;
    const double eps = 1e-10;

    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps) N_val = D_val = 0.5;
        else if (b_boundary < K_value_) N_val = D_val = 0.0;
        else N_val = D_val = 1.0;
    } else {
        auto integrand_N_contrib = [&](double u_integral) -> double { /* ... as before ... */ 
            const double df_r = std::exp(r_rate_ * u_integral);
            if (u_integral >= tau * (1.0 - 5.0 * eps)) { 
                if (std::abs(b_boundary - B_boundary_func_(u_integral)) < eps) return 0.5 * df_r;
                else return df_r * ((b_boundary < B_boundary_func_(u_integral)) ? 0.0 : 1.0);
            } else {
                const double B_val_at_u = B_boundary_func_(u_integral);
                if (B_val_at_u <= 1e-9) return (b_boundary > 0 ? df_r : 0.0);
                return df_r * normalCDF(d_black_scholes(tau - u_integral, b_boundary / B_val_at_u).second);
            }
        };
        auto integrand_D_contrib = [&](double u_integral) -> double { /* ... as before ... */ 
            const double df_q = std::exp(q_yield_ * u_integral);
            if (u_integral >= tau * (1.0 - 5.0 * eps)) {
                 if (std::abs(b_boundary - B_boundary_func_(u_integral)) < eps) return 0.5 * df_q;
                else return df_q * ((b_boundary < B_boundary_func_(u_integral)) ? 0.0 : 1.0);
            } else {
                 const double B_val_at_u = B_boundary_func_(u_integral);
                 if (B_val_at_u <= 1e-9) return (b_boundary > 0 ? df_q : 0.0);
                return df_q * normalCDF(d_black_scholes(tau - u_integral, b_boundary / B_val_at_u).first);
            }
        };
        double ni_val = integrate_fp_->integrate(integrand_N_contrib, 0.0, tau);
        double di_val = integrate_fp_->integrate(integrand_D_contrib, 0.0, tau);

        const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
        N_val = normalCDF(d_pair_direct.second) + r_rate_ * ni_val;
        D_val = normalCDF(d_pair_direct.first) + q_yield_ * di_val;
    }
    
    const double alpha = K_value_ * std::exp(-(r_rate_ - q_yield_) * tau);
    double fv_val;
    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps || b_boundary > K_value_) {
            fv_val = alpha;
        } else { 
             if (std::abs(q_yield_) < eps) {
                fv_val = alpha * r_rate_ * ((q_yield_ < 0.0) ? -1.0 : 1.0) / eps;
             } else {
                fv_val = alpha * r_rate_ / q_yield_;
             }
             if (r_rate_ <= 0 && q_yield_ > 0) fv_val = 0.0;
        }
    } else {
         if (std::abs(D_val) < 1e-12) fv_val = (N_val > 0 ? std::numeric_limits<double>::infinity() : -std::numeric_limits<double>::infinity());
         else fv_val = alpha * N_val / D_val;
    }
    return {N_val, D_val, std::max(0.0, fv_val)};
}

std::pair<double, double> EquationBDouble::derivatives(double tau, double b_boundary) const {
    const double eps = 1e-10;
    if (tau < eps * eps || b_boundary <= 1e-9 || volatility_ <= 1e-9) {
        return {0.0, 0.0};
    }
    const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
    const double common_denom = (b_boundary * volatility_ * std::sqrt(tau));
    if (std::abs(common_denom) < 1e-15) return { std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};

    return {normalPDF(d_pair_direct.second) / common_denom,
            normalPDF(d_pair_direct.first) / common_denom};
}

// --- Factory for FixedPointEvaluatorDouble ---
std::shared_ptr<FixedPointEvaluatorDouble>
createFixedPointEvaluatorDouble(char equation_type, double K_val, double r_val, double q_val,
                                double vol_val, const std::function<double(double)> &B_func,
                                std::shared_ptr<num::IntegrateDouble> integrate_instance) {
  if (equation_type == 'A') {
    return std::make_shared<EquationADouble>(K_val, r_val, q_val, vol_val, B_func, integrate_instance);
  } else if (equation_type == 'B') {
    return std::make_shared<EquationBDouble>(K_val, r_val, q_val, vol_val, B_func, integrate_instance);
  } else { 
    throw std::invalid_argument("Unknown equation type for FixedPointEvaluatorDouble: " + std::string(1,equation_type));
  }
}

AmericanPutSingle::AmericanPutSingle(std::shared_ptr<num::IntegrateSingle> integrate_pricing)
    : AmericanOptionSingle(std::move(integrate_pricing)) {}

float AmericanPutSingle::xMax(float K, float r, float q) const {
  return (r > 0.0f && r >= q && q > 1e-7f) ? (K * r / q) : K; // Avoid division by zero
}

float AmericanPutSingle::calculateEarlyExercisePremium(
    float S, float K, float r, float q, float vol, float T,
    const std::shared_ptr<num::ChebyshevInterpolationSingle> &boundary) const {
  if (!boundary) {
    throw std::invalid_argument("AmericanPutSingle: Boundary interpolation cannot be null");
  }
  if (T <= 1e-7f) return 0.0f;

  const float xmax_val = xMax(K, r, q); 

  auto B_func = [&](float tau_prime) -> float {
    if (tau_prime <= 1e-9f) return K; 
    float z_interp = 2.0f * std::sqrt(tau_prime / T) - 1.0f;
    z_interp = std::max(-1.0f, std::min(1.0f, z_interp)); 
    float h_B = (*boundary)(z_interp, true); 
    return xmax_val * std::exp(-std::sqrt(std::max(0.0f, h_B)));
  };
  
  auto integrand = [&](float z_var) -> float { 
    if (z_var < 1e-7f) return 0.0f; 
    float t_prime = z_var * z_var * T; 
    float B_at_t_prime = B_func(t_prime);

    if (B_at_t_prime <= 1e-7f || vol <= 1e-7f || t_prime <= 1e-9f) return 0.0f;

    float sqrt_t_prime = z_var * std::sqrt(T);
    float d1_val = scalar_bs_d1_sgl(S, B_at_t_prime, r, q, vol, t_prime);
    float d2_val = d1_val - vol * sqrt_t_prime;

    float term1 = r * K * std::exp(-r * t_prime) * num::fast_normal_cdf(-d2_val);
    float term2 = q * S * std::exp(-q * t_prime) * num::fast_normal_cdf(-d1_val);
    
    return (term1 - term2) * 2.0f * T * z_var; 
  };
  return integrate_pricing_->integrate(integrand, 0.0f, 1.0f);
}

std::shared_ptr<num::ChebyshevInterpolationSingle>
AmericanPutSingle::calculateExerciseBoundary(
    float S_unused, float K, float r, float q, float vol, float T,
    size_t num_nodes, size_t num_iterations,
    std::shared_ptr<num::IntegrateSingle> integrate_fp) const {
  if (!integrate_fp) {
    throw std::invalid_argument("AmericanPutSingle: Fixed point integrator cannot be null");
  }
   if (T <= 1e-7f) { 
    std::vector<float> nodes_std(num_nodes);
    const float xmax_val_temp = xMax(K,r,q);
    float h_K_val = (xmax_val_temp > 1e-7f && K > 1e-7f) ? std::pow(std::log(K/xmax_val_temp),2.0f) : 0.0f;
     if (K <= 1e-7f && xmax_val_temp <= 1e-7f) h_K_val = 0.0f;
    else if (K > 1e-7f && xmax_val_temp <= 1e-7f) h_K_val = 100.0f;

    std::vector<float> y_vals(num_nodes, h_K_val); 
     for (size_t i = 0; i < num_nodes; ++i) {
        nodes_std[i] = (num_nodes == 1) ? 0.0f : (-1.0f + 2.0f * static_cast<float>(i) / (num_nodes -1.0f));
    }
     std::sort(nodes_std.begin(), nodes_std.end());
    return std::make_shared<num::ChebyshevInterpolationSingle>(nodes_std, y_vals, num::SECOND_KIND, -1.0f, 1.0f);
  }

  const float xmax_val = xMax(K, r, q); 
  std::vector<float> nodes_std(num_nodes); 
  std::vector<float> y_boundary_transformed(num_nodes); 

  for (size_t i = 0; i < num_nodes; ++i) {
    nodes_std[i] = (num_nodes == 1) ? 0.0f : -std::cos(static_cast<float>(M_PI) * static_cast<float>(i) / (static_cast<float>(num_nodes) - 1.0f));
  }
   if (num_nodes > 1) {
        nodes_std[0] = -1.0f; 
        nodes_std[num_nodes-1] = 1.0f;
        std::sort(nodes_std.begin(), nodes_std.end()); 
    }

  float initial_h_B = (xmax_val > 1e-7f && K > 1e-7f) ? std::pow(std::log(K / xmax_val), 2.0f) : 0.0f;
  if (K <= 1e-7f && xmax_val <= 1e-7f) initial_h_B = 0.0f;
  else if (K > 1e-7f && xmax_val <= 1e-7f) initial_h_B = 100.0f; 
  
  std::fill(y_boundary_transformed.begin(), y_boundary_transformed.end(), initial_h_B);
  
  size_t tau0_node_idx = 0; 
  for(size_t k_idx=0; k_idx<num_nodes; ++k_idx) { if(std::abs(nodes_std[k_idx] - (-1.0f)) < 1e-7f) { tau0_node_idx = k_idx; break; } }

  float K_div_xmax = (xmax_val > 1e-7f) ? K / xmax_val : 1.0f; 
  if (K_div_xmax <= 1e-7f) y_boundary_transformed[tau0_node_idx] = 100.0f; 
  else y_boundary_transformed[tau0_node_idx] = std::pow(std::log(K_div_xmax), 2.0f);

  auto interp = std::make_shared<num::ChebyshevInterpolationSingle>(nodes_std, y_boundary_transformed, num::SECOND_KIND, -1.0f, 1.0f);

  auto B_func = [&](float tau_prime) -> float {
    if (tau_prime <= 1e-9f) return K; 
    float z_interp = 2.0f * std::sqrt(tau_prime / T) - 1.0f;
    z_interp = std::max(-1.0f, std::min(1.0f, z_interp));
    float h_B = (*interp)(z_interp, true);
    return xmax_val * std::exp(-std::sqrt(std::max(0.0f, h_B)));
  };

  auto h_transform = [&](float fv) -> float {
    if (fv <= 1e-7f || xmax_val <= 1e-7f) return 100.0f; 
    float ratio = fv / xmax_val;
    if (ratio <= 1e-7f) return 100.0f; 
    return std::pow(std::log(ratio), 2.0f);
  };

  char eq_type = 'A'; 
  if (std::abs(r-q) >= 0.001f && !(r < 0.0f && q < r) ) eq_type = 'B'; 

  auto evaluator = createFixedPointEvaluatorSingle(eq_type, K, r, q, vol, B_func, integrate_fp);

  for (size_t iter_count = 0; iter_count < num_iterations; ++iter_count) {
      bool is_newton_step = (iter_count == 0);
      for (size_t i = 0; i < num_nodes; ++i) {
        float z_node = nodes_std[i];
        float tau_node = T * 0.25f * (z_node + 1.0f) * (z_node + 1.0f);
        
        if (i == tau0_node_idx && std::abs(z_node - (-1.0f)) < 1e-7f) {
             y_boundary_transformed[i] = h_transform(K); 
             continue;
        }

        float b_current = B_func(tau_node); 
        auto [N_val, D_val, fv_val] = evaluator->evaluate(tau_node, b_current);

        if(fv_val <= 1e-7f) {
            y_boundary_transformed[i] = h_transform(1e-7f);
            continue;
        }

        if (is_newton_step) {
            if (tau_node < 1e-8f) {
              y_boundary_transformed[i] = h_transform(fv_val);
            } else {
              auto [Nd_val, Dd_val] = evaluator->derivatives(tau_node, b_current);
              float fd_val = 0.0f;
              if (std::abs(D_val) > 1e-9f) { 
                  fd_val = K * std::exp(-(r - q) * tau_node) * (Nd_val / D_val - Dd_val * N_val / (D_val * D_val));
              }
              
              if (std::abs(fd_val - 1.0f) < 1e-7f) { 
                  y_boundary_transformed[i] = h_transform(fv_val);
              } else {
                  float b_new = b_current - (fv_val - b_current) / (fd_val - 1.0f);
                  if (!std::isfinite(b_new) || b_new <= 0.0f) b_new = fv_val; 
                   y_boundary_transformed[i] = h_transform(std::max(1e-7f, b_new)); 
              }
            }
        } else { 
             y_boundary_transformed[i] = h_transform(std::max(1e-7f, fv_val));
        }
      }
      interp->updateValues(y_boundary_transformed);
  }
  return interp;
}

float AmericanPutSingle::approximatePriceBAW(float S, float K, float r, float q, float vol, float T) const {
    return robust_scalar_baw_put_approx_sgl(S, K, r, q, vol, T);
}

void AmericanPutSingle::batchApproximatePriceBAW(const std::vector<float>& S_vec, const std::vector<float>& K_vec,
                                const std::vector<float>& r_vec, const std::vector<float>& q_vec,
                                const std::vector<float>& vol_vec, const std::vector<float>& T_vec,
                                std::vector<float>& results_vec) const {
    if (S_vec.size() != K_vec.size() || S_vec.size() != results_vec.size() ||
        S_vec.size() != r_vec.size() || S_vec.size() != q_vec.size() ||
        S_vec.size() != vol_vec.size() || S_vec.size() != T_vec.size()) {
        throw std::invalid_argument("BAW Batch (Put Single): input/output size mismatch");
    }
    for(size_t i=0; i<S_vec.size(); ++i) {
        results_vec[i] = robust_scalar_baw_put_approx_sgl(S_vec[i], K_vec[i], r_vec[i], q_vec[i], vol_vec[i], T_vec[i]);
    }
}


// --- AmericanCallSingle ---
AmericanCallSingle::AmericanCallSingle(std::shared_ptr<num::IntegrateSingle> integrate_pricing)
    : AmericanOptionSingle(std::move(integrate_pricing)) {}

float AmericanCallSingle::xMax(float K, float r, float q) const {
    return (q > 0.0f && q >=r && r > 1e-7f ) ? (K*q/r) : K;
}

float AmericanCallSingle::calculateEarlyExercisePremium(
    float S, float K, float r, float q, float vol, float T,
    const std::shared_ptr<num::ChebyshevInterpolationSingle> &boundary) const {
    if (q <= 1e-7f) return 0.0f; 
    if (!boundary) throw std::invalid_argument("AmericanCallSingle: Boundary null");
    if (T <= 1e-7f) return 0.0f;

    const float xmax_val = xMax(K, r, q);
    if (std::isinf(xmax_val)) return 0.0f;

    auto B_func = [&](float tau_prime) -> float {
        if (tau_prime <= 1e-9f) return K; 
        float z_interp = 2.0f * std::sqrt(tau_prime / T) - 1.0f;
        z_interp = std::max(-1.0f, std::min(1.0f, z_interp));
        float h_B = (*boundary)(z_interp, true);
        return xmax_val * std::exp(std::sqrt(std::max(0.0f, h_B)));
    };

    auto integrand = [&](float z_var) -> float { 
        if (z_var < 1e-7f) return 0.0f;
        float t_prime = z_var * z_var * T;
        float B_at_t_prime = B_func(t_prime);

        if (B_at_t_prime <= K + 1e-7f || !std::isfinite(B_at_t_prime) || vol <= 1e-7f || t_prime <= 1e-9f) return 0.0f; 

        float sqrt_t_prime = z_var * std::sqrt(T);
        float d1_val = scalar_bs_d1_sgl(S, B_at_t_prime, r, q, vol, t_prime); 
        float d2_val = d1_val - vol * sqrt_t_prime;

        float term1 = q * S * std::exp(-q * t_prime) * num::fast_normal_cdf(d1_val);    
        float term2 = r * K * std::exp(-r * t_prime) * num::fast_normal_cdf(d2_val);    
        
        return (term1 - term2) * 2.0f * T * z_var; 
    };
    return integrate_pricing_->integrate(integrand, 0.0f, 1.0f);
}

std::shared_ptr<num::ChebyshevInterpolationSingle>
AmericanCallSingle::calculateExerciseBoundary(
    float S, float K, float r, float q, float vol, float T,
    size_t num_nodes, size_t num_iterations,
    std::shared_ptr<num::IntegrateSingle> integrate_fp) const {
    if (q <= 1e-7f || T <= 1e-7f) { 
        std::vector<float> nodes_std(num_nodes);
        std::vector<float> y_vals(num_nodes, 1000.0f); 
        for (size_t i = 0; i < num_nodes; ++i) {
            nodes_std[i] = (num_nodes == 1) ? 0.0f : (-1.0f + 2.0f * static_cast<float>(i) / (num_nodes -1.0f));
        }
        std::sort(nodes_std.begin(), nodes_std.end());
        return std::make_shared<num::ChebyshevInterpolationSingle>(nodes_std, y_vals, num::SECOND_KIND, -1.0f, 1.0f);
    }

    float K_eff = S; float S_eff = K; float r_eff = q; float q_eff = r;

    AmericanPutSingle effective_put_pricer(integrate_fp); 
    auto interp_eff_put_h = effective_put_pricer.calculateExerciseBoundary(
                                S_eff, K_eff, r_eff, q_eff, vol, T,
                                num_nodes, num_iterations, integrate_fp);

    const float xmax_call = xMax(K, r, q); 
    const float xmax_eff_put_val = effective_put_pricer.xMax(K_eff, r_eff, q_eff);

    std::vector<float> y_call_transformed(num_nodes);
    const auto& nodes_std = interp_eff_put_h->getNodes();

    auto h_transform_call = [&](float fv_call) -> float {
        if (fv_call <= 1e-7f || xmax_call <= 1e-7f || std::isinf(xmax_call)) return 1000.0f; 
        float ratio = fv_call / xmax_call;
        if (ratio <= 1e-7f) return 1000.0f; 
        return std::pow(std::log(ratio), 2.0f);
    };

    for(size_t i=0; i<num_nodes; ++i) {
        float z_node = nodes_std[i];
        float tau_node = T * 0.25f * (z_node + 1.0f) * (z_node + 1.0f);
        
        float h_eff_put = (*interp_eff_put_h)(z_node, true);
        float b_eff_put_val = xmax_eff_put_val * std::exp(-std::sqrt(std::max(0.0f, h_eff_put)));
        
        if (tau_node <= 1e-9f) {
            y_call_transformed[i] = h_transform_call(K);
            continue;
        }
        if (b_eff_put_val < 1e-7f) { 
            y_call_transformed[i] = 1000.0f; 
        } else {
            float b_call_val = (S * K) / b_eff_put_val;
            y_call_transformed[i] = h_transform_call(std::max(K + 1e-7f, b_call_val)); 
        }
    }
    return std::make_shared<num::ChebyshevInterpolationSingle>(nodes_std, y_call_transformed, num::SECOND_KIND, -1.0f, 1.0f);
}

float AmericanCallSingle::approximatePriceBAW(float S, float K, float r, float q, float vol, float T) const {
    return robust_scalar_baw_call_approx_sgl(S, K, r, q, vol, T);
}

void AmericanCallSingle::batchApproximatePriceBAW(const std::vector<float>& S_vec, const std::vector<float>& K_vec,
                                const std::vector<float>& r_vec, const std::vector<float>& q_vec,
                                const std::vector<float>& vol_vec, const std::vector<float>& T_vec,
                                std::vector<float>& results_vec) const {
    if (S_vec.size() != K_vec.size() || S_vec.size() != results_vec.size() ||
        S_vec.size() != r_vec.size() || S_vec.size() != q_vec.size() ||
        S_vec.size() != vol_vec.size() || S_vec.size() != T_vec.size()) {
        throw std::invalid_argument("BAW Batch (Call Single): input/output size mismatch");
    }
    for(size_t i=0; i<S_vec.size(); ++i) {
        results_vec[i] = robust_scalar_baw_call_approx_sgl(S_vec[i], K_vec[i], r_vec[i], q_vec[i], vol_vec[i], T_vec[i]);
    }
}


// --- FixedPointEvaluatorSingle ---
FixedPointEvaluatorSingle::FixedPointEvaluatorSingle(
    float K_val, float r_val, float q_val, float vol_val,
    const std::function<float(float)> &B_func,
    std::shared_ptr<num::IntegrateSingle> integrate_instance)
    : K_value_(K_val), r_rate_(r_val), q_yield_(q_val), volatility_(vol_val),
      vol_sq_(vol_val * vol_val), B_boundary_func_(B_func), integrate_fp_(std::move(integrate_instance)) {}

std::pair<float, float> FixedPointEvaluatorSingle::d_black_scholes(float t, float spot_ratio_K) const {
  if (t <= 1e-9f || spot_ratio_K <= 1e-9f || volatility_ <= 1e-7f) {
    bool spot_is_large = spot_ratio_K > 1.0f/ (1e-7f); 
    bool spot_is_small = spot_ratio_K < 1e-7f;      
    if (spot_is_large) return {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    if (spot_is_small) return {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    if (volatility_ <= 1e-7f || t <= 1e-9f) {
      if (std::log(spot_ratio_K) > 0.0f)
          return {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
      else
          return {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
    }
  }
  const float v_sqrt_t = volatility_ * std::sqrt(t);
  if (std::abs(v_sqrt_t) < 1e-9f) { 
    if (std::log(spot_ratio_K) + (r_rate_ - q_yield_ + 0.5f * vol_sq_) * t > 0.0f)
        return {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity()};
    else
        return {-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity()};
  }
  const float d1 = (std::log(spot_ratio_K) + (r_rate_ - q_yield_ + 0.5f * vol_sq_) * t) / v_sqrt_t;
  const float d2 = d1 - v_sqrt_t;
  return {d1, d2};
}

float FixedPointEvaluatorSingle::normalCDF(float x) const { return num::fast_normal_cdf(x); }
float FixedPointEvaluatorSingle::normalPDF(float x) const { return num::fast_normal_pdf(x); }


// --- EquationASingle & EquationBSingle implementations are the same as previously provided ---
// ... (assuming they are correct based on prior fixes and use float types throughout)
EquationASingle::EquationASingle(
    float K_val, float r_val, float q_val, float vol_val,
    const std::function<float(float)> &B_func,
    std::shared_ptr<num::IntegrateSingle> integrate_instance)
    : FixedPointEvaluatorSingle(K_val, r_val, q_val, vol_val, B_func, std::move(integrate_instance)) {}

std::tuple<float, float, float> EquationASingle::evaluate(float tau, float b_boundary) const {
    float N_val, D_val;
    const float eps = 1e-7f; 

    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps) { 
            const float v_eff = volatility_ * std::sqrt(std::max(tau, eps*eps)); 
            if (std::abs(v_eff) < 1e-9f) { 
                 N_val = (b_boundary > K_value_) ? 0.0f : ((b_boundary < K_value_) ? 0.0f : 0.5f); 
                 D_val = (b_boundary > K_value_) ? 1.0f : ((b_boundary < K_value_) ? 0.0f : 0.5f);
            } else {
                N_val = normalPDF(0.0f) / v_eff; 
                D_val = N_val + 0.5f; 
            }
        } else {
            N_val = 0.0f;
            D_val = (b_boundary > K_value_) ? 1.0f : 0.0f;
        }
    } else { 
        const float stv_inv = 1.0f / (volatility_ * std::sqrt(tau)); 

        auto integrand_K12 = [&](float y_cheby) -> float { 
            const float m_integral = 0.25f * tau * (1.0f + y_cheby) * (1.0f + y_cheby);
            if (tau - m_integral < 1e-9f) return 0.0f;
            const float B_tau_minus_m = B_boundary_func_(tau - m_integral);
            if (B_tau_minus_m <= 1e-7f) return 0.0f;
            const float df_q = std::exp(q_yield_ * (tau - m_integral));

            if (m_integral < eps * eps) { 
                 if (std::abs(b_boundary - B_tau_minus_m) < eps)
                    return df_q * stv_inv * normalPDF(0.0f); 
                else return 0.0f;
            } else {
                const auto d_pair_integral = d_black_scholes(m_integral, b_boundary / B_tau_minus_m);
                return df_q * (0.5f * tau * (y_cheby + 1.0f) * normalCDF(d_pair_integral.first) + 
                               (1.0f/volatility_)*std::sqrt(tau)*normalPDF(d_pair_integral.first));
            }
        };
        auto integrand_K3 = [&](float y_cheby) -> float {
            const float m_integral = 0.25f * tau * (1.0f + y_cheby) * (1.0f + y_cheby);
            if (tau - m_integral < 1e-9f) return 0.0f;
            const float B_tau_minus_m = B_boundary_func_(tau - m_integral);
            if (B_tau_minus_m <= 1e-7f) return 0.0f;
            const float df_r = std::exp(r_rate_ * (tau - m_integral)); 
             if (m_integral < eps*eps) {
                if (std::abs(b_boundary - B_tau_minus_m) < eps)
                    return df_r * stv_inv * normalPDF(0.0f);
                else return 0.0f;
            } else {
                 const auto d_pair_integral = d_black_scholes(m_integral, b_boundary / B_tau_minus_m);
                return df_r * (1.0f/volatility_)*std::sqrt(tau) * normalPDF(d_pair_integral.second);
            }
        };
        float K12_val = integrate_fp_->integrate(integrand_K12, -1.0f, 1.0f);
        float K3_val = integrate_fp_->integrate(integrand_K3, -1.0f, 1.0f);
        const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
        N_val = normalPDF(d_pair_direct.second) * stv_inv + r_rate_ * K3_val;
        D_val = normalPDF(d_pair_direct.first) * stv_inv + normalCDF(d_pair_direct.first) + q_yield_ * K12_val;
    }

    const float alpha = K_value_ * std::exp(-(r_rate_ - q_yield_) * tau);
    float fv_val;
     if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps || b_boundary > K_value_) {
            fv_val = alpha;
        } else { 
             if (std::abs(q_yield_) < eps) {
                fv_val = alpha * r_rate_ * ((q_yield_ < 0.0f) ? -1.0f : 1.0f) / eps;
             } else {
                fv_val = alpha * r_rate_ / q_yield_;
             }
             if (r_rate_ <= 0.0f && q_yield_ > 0.0f) fv_val = 0.0f;
        }
    } else {
        if (std::abs(D_val) < 1e-9f) fv_val = (N_val > 0.0f ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity());
        else fv_val = alpha * N_val / D_val;
    }
    return {N_val, D_val, std::max(0.0f, fv_val)}; 
}

std::pair<float, float> EquationASingle::derivatives(float tau, float b_boundary) const {
    float Nd_prime, Dd_prime;
    const float eps = 1e-7f;
    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps) {
            const float sqTau_eff = std::sqrt(std::max(tau, eps*eps)); 
            const float common_denom_part = (b_boundary * volatility_ * vol_sq_ * sqTau_eff);
            if (std::abs(common_denom_part) < 1e-12f) { 
                 Dd_prime = Nd_prime = std::numeric_limits<float>::quiet_NaN();
            } else {
                Dd_prime = static_cast<float>(INV_SQRT_2PI * M_SQRT1_2) * 
                       (-(0.5f * vol_sq_ + r_rate_ - q_yield_) / common_denom_part + 1.0f / (b_boundary * volatility_ * sqTau_eff));
                Nd_prime = static_cast<float>(INV_SQRT_2PI * M_SQRT1_2) * (-0.5f * vol_sq_ + r_rate_ - q_yield_) / common_denom_part;
            }
        } else { Dd_prime = Nd_prime = 0.0f; }
    } else {
        const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
        const float common_denom_tau = (b_boundary * vol_sq_ * tau);
        const float common_denom_sqrt_tau = (b_boundary * volatility_ * std::sqrt(tau));

        if (std::abs(common_denom_tau) < 1e-12f || std::abs(common_denom_sqrt_tau) < 1e-12f ) {
            Dd_prime = Nd_prime = std::numeric_limits<float>::quiet_NaN();
        } else {
            Dd_prime = -normalPDF(d_pair_direct.first) * d_pair_direct.first / common_denom_tau +
                       normalPDF(d_pair_direct.first) / common_denom_sqrt_tau;
            Nd_prime = -normalPDF(d_pair_direct.second) * d_pair_direct.second / common_denom_tau;
        }
    }
    return {Nd_prime, Dd_prime};
}

EquationBSingle::EquationBSingle(
    float K_val, float r_val, float q_val, float vol_val,
    const std::function<float(float)> &B_func,
    std::shared_ptr<num::IntegrateSingle> integrate_instance)
    : FixedPointEvaluatorSingle(K_val, r_val, q_val, vol_val, B_func, std::move(integrate_instance)) {}

std::tuple<float, float, float> EquationBSingle::evaluate(float tau, float b_boundary) const {
    float N_val, D_val;
    const float eps = 1e-7f;

    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps) N_val = D_val = 0.5f;
        else if (b_boundary < K_value_) N_val = D_val = 0.0f;
        else N_val = D_val = 1.0f;
    } else {
        auto integrand_N_contrib = [&](float u_integral) -> float { 
            const float df_r = std::exp(r_rate_ * u_integral);
            if (u_integral >= tau * (1.0f - 5.0f * eps)) { 
                if (std::abs(b_boundary - B_boundary_func_(u_integral)) < eps) return 0.5f * df_r;
                else return df_r * ((b_boundary < B_boundary_func_(u_integral)) ? 0.0f : 1.0f);
            } else {
                const float B_val_at_u = B_boundary_func_(u_integral);
                if (B_val_at_u <= 1e-7f) return (b_boundary > 0.0f ? df_r : 0.0f);
                return df_r * normalCDF(d_black_scholes(tau - u_integral, b_boundary / B_val_at_u).second);
            }
        };
        auto integrand_D_contrib = [&](float u_integral) -> float { 
            const float df_q = std::exp(q_yield_ * u_integral);
            if (u_integral >= tau * (1.0f - 5.0f * eps)) {
                 if (std::abs(b_boundary - B_boundary_func_(u_integral)) < eps) return 0.5f * df_q;
                else return df_q * ((b_boundary < B_boundary_func_(u_integral)) ? 0.0f : 1.0f);
            } else {
                 const float B_val_at_u = B_boundary_func_(u_integral);
                 if (B_val_at_u <= 1e-7f) return (b_boundary > 0.0f ? df_q : 0.0f);
                return df_q * normalCDF(d_black_scholes(tau - u_integral, b_boundary / B_val_at_u).first);
            }
        };
        float ni_val = integrate_fp_->integrate(integrand_N_contrib, 0.0f, tau);
        float di_val = integrate_fp_->integrate(integrand_D_contrib, 0.0f, tau);

        const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
        N_val = normalCDF(d_pair_direct.second) + r_rate_ * ni_val;
        D_val = normalCDF(d_pair_direct.first) + q_yield_ * di_val;
    }
    
    const float alpha = K_value_ * std::exp(-(r_rate_ - q_yield_) * tau);
    float fv_val;
    if (tau < eps * eps) {
        if (std::abs(b_boundary - K_value_) < eps || b_boundary > K_value_) {
            fv_val = alpha;
        } else { 
             if (std::abs(q_yield_) < eps) {
                fv_val = alpha * r_rate_ * ((q_yield_ < 0.0f) ? -1.0f : 1.0f) / eps;
             } else {
                fv_val = alpha * r_rate_ / q_yield_;
             }
             if (r_rate_ <= 0.0f && q_yield_ > 0.0f) fv_val = 0.0f;
        }
    } else {
         if (std::abs(D_val) < 1e-9f) fv_val = (N_val > 0.0f ? std::numeric_limits<float>::infinity() : -std::numeric_limits<float>::infinity());
         else fv_val = alpha * N_val / D_val;
    }
    return {N_val, D_val, std::max(0.0f, fv_val)};
}

std::pair<float, float> EquationBSingle::derivatives(float tau, float b_boundary) const {
    const float eps = 1e-7f;
    if (tau < eps * eps || b_boundary <= 1e-7f || volatility_ <= 1e-7f) {
        return {0.0f, 0.0f};
    }
    const auto d_pair_direct = d_black_scholes(tau, b_boundary / K_value_);
    const float common_denom = (b_boundary * volatility_ * std::sqrt(tau));
    if (std::abs(common_denom) < 1e-12f) return { std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::quiet_NaN()};

    return {normalPDF(d_pair_direct.second) / common_denom,
            normalPDF(d_pair_direct.first) / common_denom};
}


// --- Factory for FixedPointEvaluatorSingle ---
std::shared_ptr<FixedPointEvaluatorSingle> createFixedPointEvaluatorSingle(
    char equation_type, float K_val, float r_val, float q_val, float vol_val,
    const std::function<float(float)> &B_func,
    std::shared_ptr<num::IntegrateSingle> integrate_instance) {
  if (equation_type == 'A') {
    return std::make_shared<EquationASingle>(K_val, r_val, q_val, vol_val, B_func, integrate_instance);
  } else if (equation_type == 'B') {
    return std::make_shared<EquationBSingle>(K_val, r_val, q_val, vol_val, B_func, integrate_instance);
  } else {
    throw std::invalid_argument("Unknown equation type for FixedPointEvaluatorSingle: " + std::string(1,equation_type));
  }
}


} // namespace mod
} // namespace alo
} // namespace engine
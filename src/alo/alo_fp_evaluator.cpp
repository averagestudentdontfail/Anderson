#include "alo_engine.h"
#include <cmath>
#include <algorithm>
#include <iostream>

// Fixed Point Evaluator base class implementation
ALOEngine::FixedPointEvaluator::FixedPointEvaluator(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B,
    std::shared_ptr<numerics::Integrator> integrator)
    : K_(K), r_(r), q_(q), vol_(vol), vol2_(vol * vol), 
      B_(B), integrator_(integrator) {}

std::pair<double, double> ALOEngine::FixedPointEvaluator::d(double t, double z) const {
    if (t <= 0.0 || z <= 0.0) {
        return {-HUGE_VAL, -HUGE_VAL};
    }
    
    const double v = vol_ * std::sqrt(t);
    const double m = (std::log(z) + (r_ - q_) * t) / v + 0.5 * v;
    
    return {m, m - v};
}

double ALOEngine::FixedPointEvaluator::normalCDF(double x) const {
    // Using standard error function implementation
    return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
}

double ALOEngine::FixedPointEvaluator::normalPDF(double x) const {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

// Equation A implementation
ALOEngine::EquationA::EquationA(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B,
    std::shared_ptr<numerics::Integrator> integrator)
    : FixedPointEvaluator(K, r, q, vol, B, integrator) {}

std::tuple<double, double, double> ALOEngine::EquationA::evaluate(double tau, double b) const {
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

std::pair<double, double> ALOEngine::EquationA::derivatives(double tau, double b) const {
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

// Equation B implementation
ALOEngine::EquationB::EquationB(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B,
    std::shared_ptr<numerics::Integrator> integrator)
    : FixedPointEvaluator(K, r, q, vol, B, integrator) {}

std::tuple<double, double, double> ALOEngine::EquationB::evaluate(double tau, double b) const {
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

std::pair<double, double> ALOEngine::EquationB::derivatives(double tau, double b) const {
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

// Create the appropriate fixed point evaluator based on parameters
std::shared_ptr<ALOEngine::FixedPointEvaluator> ALOEngine::createFixedPointEvaluator(
    double K, double r, double q, double vol, 
    const std::function<double(double)>& B) const {
    
    // Choose equation type
    FixedPointEquation eq = equation_;
    if (eq == AUTO) {
        // Automatically select equation based on r and q
        eq = (std::abs(r - q) < 0.001) ? FP_A : FP_B;
    }
    
    // Create appropriate evaluator
    if (eq == FP_A) {
        return std::make_shared<EquationA>(K, r, q, vol, B, scheme_->getFixedPointIntegrator());
    } else {
        return std::make_shared<EquationB>(K, r, q, vol, B, scheme_->getFixedPointIntegrator());
    }
}
#ifndef INTEGRATION_H
#define INTEGRATION_H

#include <functional>
#include <vector>
#include <memory>
#include <cmath>
#include <string> 
#include <gsl/gsl_integration.h>

namespace numerics {

// Abstract base class for integration methods
class Integrator {
public:
    virtual ~Integrator() = default;
    virtual double integrate(const std::function<double(double)>& f, double a, double b) const = 0;
    virtual std::string name() const = 0;
};

// Gauss-Legendre quadrature implementation with predefined weights and nodes
class GaussLegendreIntegrator : public Integrator {
public:
    explicit GaussLegendreIntegrator(size_t order);
    ~GaussLegendreIntegrator() override = default;
    
    double integrate(const std::function<double(double)>& f, double a, double b) const override;
    std::string name() const override { return "Gauss-Legendre"; }
    
private:
    size_t order_;
    std::vector<double> nodes_;
    std::vector<double> weights_;
    
    // Precomputed values for common orders
    void initializeNodesAndWeights();
};

// Tanh-Sinh quadrature implementation
class TanhSinhIntegrator : public Integrator {
public:
    explicit TanhSinhIntegrator(double tolerance);
    ~TanhSinhIntegrator() override = default;
    
    double integrate(const std::function<double(double)>& f, double a, double b) const override;
    std::string name() const override { return "Tanh-Sinh"; }
    
private:
    double tolerance_;
    size_t max_refinements_;
};

// GSL QAGS adaptive integration implementation for validation and robustness
class GSLQAGSIntegrator : public Integrator {
public:
    GSLQAGSIntegrator(double abs_error, double rel_error, size_t max_intervals);
    ~GSLQAGSIntegrator() override;
    
    double integrate(const std::function<double(double)>& f, double a, double b) const override;
    std::string name() const override { return "GSL-QAGS"; }
    
private:
    double abs_error_;
    double rel_error_;
    size_t max_intervals_;
    gsl_integration_workspace* workspace_;
};

// Factory function to create appropriate integrator based on scheme
std::shared_ptr<Integrator> createIntegrator(const std::string& scheme_type, 
                                            size_t order = 0, 
                                            double tolerance = 0.0);

} // namespace numerics

#endif // INTEGRATION_H
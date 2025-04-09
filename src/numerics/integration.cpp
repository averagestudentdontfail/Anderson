#include "integration.h"
#include <stdexcept>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h> 

namespace numerics {

// Helper function to wrap std::function for GSL
struct GSLFunctionWrapper {
    const std::function<double(double)>* f;
    static double evaluate(double x, void* params) {
        auto& wrapper = *reinterpret_cast<GSLFunctionWrapper*>(params);
        return (*(wrapper.f))(x);
    }
};

// Gauss-Legendre implementation
GaussLegendreIntegrator::GaussLegendreIntegrator(size_t order) : order_(order) {
    if (order_ < 1) {
        throw std::invalid_argument("GaussLegendreIntegrator: Order must be at least 1");
    }
    initializeNodesAndWeights();
}

void GaussLegendreIntegrator::initializeNodesAndWeights() {
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
        // For other orders, use a simple but less accurate approach
        nodes_.resize(order_);
        weights_.resize(order_);
        
        for (size_t i = 0; i < order_; ++i) {
            double theta = M_PI * (i + 0.5) / order_;
            nodes_[i] = std::cos(theta);
            weights_[i] = M_PI / order_;
        }
    }
}

double GaussLegendreIntegrator::integrate(const std::function<double(double)>& f, double a, double b) const {
    double result = 0.0;
    
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

// Tanh-Sinh implementation (custom implementation since we're not using Boost)
TanhSinhIntegrator::TanhSinhIntegrator(double tolerance) 
    : tolerance_(tolerance), max_refinements_(15) {
    if (tolerance_ <= 0.0) {
        throw std::invalid_argument("TanhSinhIntegrator: Tolerance must be positive");
    }
}

double TanhSinhIntegrator::integrate(const std::function<double(double)>& f, double a, double b) const {
    if (std::abs(b - a) < 1e-15) {
        return 0.0;
    }
    
    // Implementation of tanh-sinh quadrature without using Boost
    const int MAX_LEVEL = 12;
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
    for (int level = 1; level <= MAX_LEVEL; ++level) {
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

// GSL QAGS implementation
GSLQAGSIntegrator::GSLQAGSIntegrator(double abs_error, double rel_error, size_t max_intervals)
    : abs_error_(abs_error), rel_error_(rel_error), max_intervals_(max_intervals) {
    workspace_ = gsl_integration_workspace_alloc(max_intervals_);
    if (!workspace_) {
        throw std::runtime_error("GSLQAGSIntegrator: Failed to allocate GSL workspace");
    }
}

GSLQAGSIntegrator::~GSLQAGSIntegrator() {
    if (workspace_) {
        gsl_integration_workspace_free(workspace_);
    }
}

double GSLQAGSIntegrator::integrate(const std::function<double(double)>& f, double a, double b) const {
    GSLFunctionWrapper wrapper;
    wrapper.f = &f;
    
    gsl_function gsl_f;
    gsl_f.function = &GSLFunctionWrapper::evaluate;
    gsl_f.params = &wrapper;
    
    double result, error;
    int status = gsl_integration_qags(&gsl_f, a, b, abs_error_, rel_error_, max_intervals_, workspace_, &result, &error);
    
    if (status) {
        std::cerr << "GSLQAGSIntegrator warning: " << gsl_strerror(status) << ", error estimate: " << error << std::endl;
    }
    
    return result;
}

// Factory function implementation
std::shared_ptr<Integrator> createIntegrator(const std::string& scheme_type, 
                                           size_t order, 
                                           double tolerance) {
    if (scheme_type == "GaussLegendre") {
        return std::make_shared<GaussLegendreIntegrator>(order);
    } else if (scheme_type == "TanhSinh") {
        return std::make_shared<TanhSinhIntegrator>(tolerance);
    } else if (scheme_type == "GSLQAGS") {
        return std::make_shared<GSLQAGSIntegrator>(tolerance, tolerance, 1000);
    } else {
        throw std::invalid_argument("Unknown integrator type: " + scheme_type);
    }
}

} // namespace numerics
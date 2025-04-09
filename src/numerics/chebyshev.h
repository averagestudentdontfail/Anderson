#ifndef CHEBYSHEV_H
#define CHEBYSHEV_H

#include <vector>
#include <functional>
#include <eigen3/Eigen/Dense>

namespace numerics {

enum ChebyshevKind {
    FIRST_KIND,   // Chebyshev polynomials of the first kind (T_n)
    SECOND_KIND   // Chebyshev polynomials of the second kind (U_n)
};

class ChebyshevInterpolation {
public:
    // Constructor for constructing interpolation from a function
    ChebyshevInterpolation(size_t num_points, 
                          const std::function<double(double)>& func, 
                          ChebyshevKind kind = SECOND_KIND,
                          double domain_start = -1.0,
                          double domain_end = 1.0);
    
    // Constructor for constructing from existing nodes and values
    ChebyshevInterpolation(const std::vector<double>& nodes,
                          const std::vector<double>& values,
                          ChebyshevKind kind = SECOND_KIND,
                          double domain_start = -1.0,
                          double domain_end = 1.0);
    
    // Evaluate the interpolation at a point
    double operator()(double x, bool extrapolate = false) const;
    
    // Update the function values at the nodes
    void updateValues(const std::vector<double>& values);
    
    // Access to nodes and coefficients
    const std::vector<double>& nodes() const { return nodes_; }
    const std::vector<double>& values() const { return values_; }
    const std::vector<double>& coefficients() const { return coefficients_; }
    
    // Domain information
    double domainStart() const { return domain_start_; }
    double domainEnd() const { return domain_end_; }
    
private:
    // Initialize nodes based on kind
    void initializeNodes();
    
    // Compute Chebyshev coefficients from function values
    void computeCoefficients();
    
    // Map x from [a,b] to [-1,1]
    double mapToStandardDomain(double x) const;
    
    // Map x from [-1,1] to [a,b]
    double mapFromStandardDomain(double x) const;
    
    // Evaluate Chebyshev polynomial of specified kind
    double evaluateChebyshev(int n, double x, ChebyshevKind kind) const;
    
    size_t num_points_;
    ChebyshevKind kind_;
    double domain_start_;
    double domain_end_;
    std::vector<double> nodes_;      // Chebyshev nodes in [-1,1]
    std::vector<double> values_;     // Function values at the mapped nodes
    std::vector<double> coefficients_; // Chebyshev coefficients
};

} // namespace numerics

#endif // CHEBYSHEV_H
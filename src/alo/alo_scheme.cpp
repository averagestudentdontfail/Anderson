#include "alo_engine.h"
#include <stdexcept>
#include <iostream>

// Implementation of ALOIterationScheme
ALOIterationScheme::ALOIterationScheme(
    size_t n, size_t m, 
    std::shared_ptr<numerics::Integrator> fpIntegrator,
    std::shared_ptr<numerics::Integrator> pricingIntegrator)
    : n_(n), m_(m), 
      fpIntegrator_(fpIntegrator),
      pricingIntegrator_(pricingIntegrator) {
    
    if (n_ < 2) {
        throw std::invalid_argument("ALOIterationScheme: Number of Chebyshev nodes must be at least 2");
    }
    
    if (m_ < 1) {
        throw std::invalid_argument("ALOIterationScheme: Number of fixed point iterations must be at least 1");
    }
    
    if (!fpIntegrator_) {
        throw std::invalid_argument("ALOIterationScheme: Fixed point integrator cannot be null");
    }
    
    if (!pricingIntegrator_) {
        throw std::invalid_argument("ALOIterationScheme: Pricing integrator cannot be null");
    }
}

// Static factory methods for ALOEngine
std::shared_ptr<ALOIterationScheme> ALOEngine::createFastScheme() {
    // Legendre-Legendre (7,2,7)-27 scheme
    // This scheme prioritizes speed over accuracy
    try {
        auto fpIntegrator = numerics::createIntegrator("GaussLegendre", 7);
        auto pricingIntegrator = numerics::createIntegrator("GaussLegendre", 27);
        
        return std::make_shared<ALOIterationScheme>(7, 2, fpIntegrator, pricingIntegrator);
    } catch (const std::exception& e) {
        std::cerr << "Error creating fast scheme: " << e.what() << std::endl;
        throw;
    }
}

std::shared_ptr<ALOIterationScheme> ALOEngine::createAccurateScheme() {
    // Legendre-TanhSinh (25,5,13)-1e-8 scheme
    // This scheme provides a good balance between accuracy and performance
    try {
        auto fpIntegrator = numerics::createIntegrator("GaussLegendre", 25);
        auto pricingIntegrator = numerics::createIntegrator("TanhSinh", 0, 1e-8);
        
        return std::make_shared<ALOIterationScheme>(13, 5, fpIntegrator, pricingIntegrator);
    } catch (const std::exception& e) {
        std::cerr << "Error creating accurate scheme: " << e.what() << std::endl;
        throw;
    }
}

std::shared_ptr<ALOIterationScheme> ALOEngine::createHighPrecisionScheme() {
    // TanhSinh-TanhSinh (10,30)-1e-10 scheme
    // This scheme prioritizes accuracy over speed
    try {
        auto fpIntegrator = numerics::createIntegrator("TanhSinh", 0, 1e-10);
        auto pricingIntegrator = numerics::createIntegrator("TanhSinh", 0, 1e-10);
        
        return std::make_shared<ALOIterationScheme>(30, 10, fpIntegrator, pricingIntegrator);
    } catch (const std::exception& e) {
        std::cerr << "Error creating high precision scheme: " << e.what() << std::endl;
        throw;
    }
}
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <chrono>
#include <functional>
#include <string>
#include <memory>
#include "alo/alo_engine.h"
#include "deterministic_pricing_system.h" // Include the deterministic pricing system header

// Structure to represent a test case
struct TestCase {
    std::string name;
    double S;    // Spot price
    double K;    // Strike price
    double r;    // Risk-free rate
    double q;    // Dividend yield
    double vol;  // Volatility
    double T;    // Time to maturity
};

// Function to print a table row with option prices
void printPriceRow(const std::string& name, double european, double fast, double accurate, 
                  double highPrecision, double premium, double duration) {
    std::cout << "| " << std::left << std::setw(4) << name << " | " 
              << std::fixed << std::setprecision(6) << std::right
              << std::setw(11) << european << " | " 
              << std::setw(11) << fast << " | " 
              << std::setw(11) << accurate << " | " 
              << std::setw(11) << highPrecision << " | " 
              << std::setw(11) << premium << " | " 
              << std::setw(10) << duration << " |\n";
}

// Function to run benchmarks
void runBenchmarks(const std::vector<TestCase>& testCases) {
    // Create engines with different schemes
    ALOEngine fast(FAST);
    ALOEngine accurate(ACCURATE);
    ALOEngine highPrecision(HIGH_PRECISION);
    
    std::cout << "+----------------------------------------------------------------------------------------+\n";
    std::cout << "| American Put Option Prices using ALO Algorithm                                          |\n";
    std::cout << "+------+-------------+-------------+-------------+-------------+-------------+------------+\n";
    std::cout << "| Case |   European  |    Fast     |  Accurate   |High Precision|   Premium  |   Time(ms) |\n";
    std::cout << "+------+-------------+-------------+-------------+-------------+-------------+------------+\n";
    
    for (const auto& test : testCases) {
        // Calculate European price for reference
        double europeanPrice = ALOEngine::blackScholesPut(
            test.S, test.K, test.r, test.q, test.vol, test.T);
        
        // Time the calculation of the accurate price
        auto start = std::chrono::high_resolution_clock::now();
        double accuratePrice = accurate.calculatePut(
            test.S, test.K, test.r, test.q, test.vol, test.T);
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        
        // Calculate other prices
        double fastPrice = fast.calculatePut(
            test.S, test.K, test.r, test.q, test.vol, test.T);
        double highPrecisionPrice = highPrecision.calculatePut(
            test.S, test.K, test.r, test.q, test.vol, test.T);
        
        // Calculate early exercise premium
        double premium = accuratePrice - europeanPrice;
        
        // Print results
        printPriceRow(test.name, europeanPrice, fastPrice, accuratePrice, 
                     highPrecisionPrice, premium, duration);
    }
    
    std::cout << "+------+-------------+-------------+-------------+-------------+-------------+------------+\n";
}

// Function to analyze convergence for a specific test case
void analyzeConvergence(const TestCase& tc) {
    std::cout << "\nConvergence Analysis for Test Case " << tc.name << ":\n";
    std::cout << "-----------------------------------------------\n";
    
    std::vector<int> nodes_list = {5, 7, 9, 11, 13, 15, 20, 25, 30};
    std::vector<int> iter_list = {1, 2, 3, 5, 7, 10};
    
    // Reference value with high precision
    ALOEngine reference(HIGH_PRECISION);
    double ref_price = reference.calculatePut(tc.S, tc.K, tc.r, tc.q, tc.vol, tc.T);
    
    std::cout << "Reference price (high precision): " << std::fixed << std::setprecision(10) << ref_price << "\n\n";
    
    std::cout << "Effect of number of Chebyshev nodes (with 5 iterations):\n";
    std::cout << "+-------+----------------+----------------+\n";
    std::cout << "| Nodes |     Price      |     Error      |\n";
    std::cout << "+-------+----------------+----------------+\n";
    
    // Analyze effect of different number of nodes
    for (int nodes : nodes_list) {
        // Create custom scheme
        auto fpIntegrator = numerics::createIntegrator("GaussLegendre", 25);
        auto pricingIntegrator = numerics::createIntegrator("TanhSinh", 0, 1e-8);
        auto scheme = std::make_shared<ALOIterationScheme>(nodes, 5, fpIntegrator, pricingIntegrator);
        
        // Create engine with custom scheme
        ALOEngine engine(ACCURATE);
        engine.setScheme(ACCURATE); // This will be overridden
        
        // Calculate price and error
        double price = engine.calculatePut(tc.S, tc.K, tc.r, tc.q, tc.vol, tc.T);
        double error = std::abs(price - ref_price);
        
        std::cout << "| " << std::setw(5) << nodes 
                  << " | " << std::setw(14) << std::fixed << std::setprecision(10) << price 
                  << " | " << std::setw(14) << std::scientific << std::setprecision(6) << error 
                  << " |\n";
    }
    
    std::cout << "+-------+----------------+----------------+\n\n";
    
    std::cout << "Effect of number of fixed point iterations (with 13 nodes):\n";
    std::cout << "+-------+----------------+----------------+\n";
    std::cout << "| Iters |     Price      |     Error      |\n";
    std::cout << "+-------+----------------+----------------+\n";
    
    // Analyze effect of different number of iterations
    for (int iters : iter_list) {
        // Create custom scheme
        auto fpIntegrator = numerics::createIntegrator("GaussLegendre", 25);
        auto pricingIntegrator = numerics::createIntegrator("TanhSinh", 0, 1e-8);
        auto scheme = std::make_shared<ALOIterationScheme>(13, iters, fpIntegrator, pricingIntegrator);
        
        // Create engine with custom scheme
        ALOEngine engine(ACCURATE);
        engine.setScheme(ACCURATE); // This will be overridden
        
        // Calculate price and error
        double price = engine.calculatePut(tc.S, tc.K, tc.r, tc.q, tc.vol, tc.T);
        double error = std::abs(price - ref_price);
        
        std::cout << "| " << std::setw(5) << iters 
                  << " | " << std::setw(14) << std::fixed << std::setprecision(10) << price 
                  << " | " << std::setw(14) << std::scientific << std::setprecision(6) << error 
                  << " |\n";
    }
    
    std::cout << "+-------+----------------+----------------+\n";
}

// Function to run a sensitivity analysis on parameters
void runSensitivityAnalysis(const TestCase& baseCase) {
    std::cout << "\nSensitivity Analysis for Base Case:\n";
    std::cout << "-------------------------------------\n";
    
    // Create engine with accurate scheme for sensitivity analysis
    ALOEngine engine(ACCURATE);
    
    // Base case price
    double basePrice = engine.calculatePut(
        baseCase.S, baseCase.K, baseCase.r, baseCase.q, baseCase.vol, baseCase.T);
    
    // Spot price sensitivity
    std::cout << "Spot Price Sensitivity:\n";
    std::cout << "+--------+-------------+---------------+\n";
    std::cout << "|  Spot  |    Price    |  % Change     |\n";
    std::cout << "+--------+-------------+---------------+\n";
    
    std::vector<double> spotFactors = {0.8, 0.9, 0.95, 1.0, 1.05, 1.10, 1.20};
    for (double factor : spotFactors) {
        double spot = baseCase.S * factor;
        double price = engine.calculatePut(
            spot, baseCase.K, baseCase.r, baseCase.q, baseCase.vol, baseCase.T);
        double percentChange = (price - basePrice) / basePrice * 100.0;
        
        std::cout << "| " << std::setw(6) << std::fixed << std::setprecision(2) << spot 
                  << " | " << std::setw(11) << std::fixed << std::setprecision(6) << price 
                  << " | " << std::setw(11) << std::showpos << std::fixed << std::setprecision(2) << percentChange 
                  << "% |\n";
    }
    std::cout << "+--------+-------------+---------------+\n\n";
    
    // Volatility sensitivity
    std::cout << "Volatility Sensitivity:\n";
    std::cout << "+--------+-------------+---------------+\n";
    std::cout << "|  Vol   |    Price    |  % Change     |\n";
    std::cout << "+--------+-------------+---------------+\n";
    
    std::vector<double> volFactors = {0.5, 0.75, 0.9, 1.0, 1.1, 1.25, 1.5};
    for (double factor : volFactors) {
        double vol = baseCase.vol * factor;
        double price = engine.calculatePut(
            baseCase.S, baseCase.K, baseCase.r, baseCase.q, vol, baseCase.T);
        double percentChange = (price - basePrice) / basePrice * 100.0;
        
        std::cout << "| " << std::setw(6) << std::fixed << std::setprecision(2) << vol 
                  << " | " << std::setw(11) << std::fixed << std::setprecision(6) << price 
                  << " | " << std::setw(11) << std::showpos << std::fixed << std::setprecision(2) << percentChange 
                  << "% |\n";
    }
    std::cout << "+--------+-------------+---------------+\n\n";
    
    // Interest rate sensitivity
    std::cout << "Interest Rate Sensitivity:\n";
    std::cout << "+--------+-------------+---------------+\n";
    std::cout << "|  Rate  |    Price    |  % Change     |\n";
    std::cout << "+--------+-------------+---------------+\n";
    
    std::vector<double> rateValues = {0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08};
    for (double rate : rateValues) {
        double price = engine.calculatePut(
            baseCase.S, baseCase.K, rate, baseCase.q, baseCase.vol, baseCase.T);
        double percentChange = (price - basePrice) / basePrice * 100.0;
        
        std::cout << "| " << std::setw(6) << std::fixed << std::setprecision(2) << rate 
                  << " | " << std::setw(11) << std::fixed << std::setprecision(6) << price 
                  << " | " << std::setw(11) << std::showpos << std::fixed << std::setprecision(2) << percentChange 
                  << "% |\n";
    }
    std::cout << "+--------+-------------+---------------+\n\n";
    
    // Time to maturity sensitivity
    std::cout << "Time to Maturity Sensitivity:\n";
    std::cout << "+--------+-------------+---------------+\n";
    std::cout << "|  Time  |    Price    |  % Change     |\n";
    std::cout << "+--------+-------------+---------------+\n";
    
    std::vector<double> timeValues = {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0};
    for (double time : timeValues) {
        double price = engine.calculatePut(
            baseCase.S, baseCase.K, baseCase.r, baseCase.q, baseCase.vol, time);
        double percentChange = (price - basePrice) / basePrice * 100.0;
        
        std::cout << "| " << std::setw(6) << std::fixed << std::setprecision(2) << time 
                  << " | " << std::setw(11) << std::fixed << std::setprecision(6) << price 
                  << " | " << std::setw(11) << std::showpos << std::fixed << std::setprecision(2) << percentChange 
                  << "% |\n";
    }
    std::cout << "+--------+-------------+---------------+\n";
}

// Main function
int main(int argc, char** argv) {
    // Define test cases
    std::vector<TestCase> testCases = {
        // Standard test cases
        {"TC1", 100.0, 100.0, 0.05, 0.01, 0.20, 1.0},     // At-the-money, medium vol
        {"TC2", 100.0, 110.0, 0.05, 0.01, 0.20, 1.0},     // In-the-money
        {"TC3", 100.0,  90.0, 0.05, 0.01, 0.20, 1.0},     // Out-of-the-money
        
        // Higher volatility
        {"TC4", 100.0, 100.0, 0.05, 0.01, 0.40, 1.0},     // Higher volatility
        
        // Different interest rates and dividend yields
        {"TC5", 100.0, 100.0, 0.08, 0.03, 0.20, 1.0},     // Higher rates
        {"TC6", 100.0, 100.0, 0.03, 0.05, 0.20, 1.0},     // Dividend > interest
        
        // Different maturities
        {"TC7", 100.0, 100.0, 0.05, 0.01, 0.20, 0.25},    // Short maturity
        {"TC8", 100.0, 100.0, 0.05, 0.01, 0.20, 3.0},     // Long maturity
        
        // Example from literature
        {"TC9",  36.0,  40.0, 0.06, 0.00, 0.20, 1.0},     // Common benchmark case
    };
    
    // Print program header
    std::cout << "================================================================================\n";
    std::cout << "Anderson-Lake-Offengenden Derivative Pricing Algorithm\n";
    std::cout << "================================================================================\n\n";
    
    // Display basic program info
    std::cout << "This program implements the Andersen-Lake-Offengenden algorithm for\n";
    std::cout << "pricing American put options with high accuracy and performance.\n\n";
    
    // Run benchmarks
    std::cout << "Running Benchmarks...\n";
    runBenchmarks(testCases);
    
    // Run convergence analysis on the first test case
    analyzeConvergence(testCases[0]);
    
    // Run sensitivity analysis on the first test case
    runSensitivityAnalysis(testCases[0]);
    
    std::cout << "\n================================================================================\n";
    std::cout << "Starting Deterministic Pricing System...\n";
    std::cout << "================================================================================\n\n";
    std::cout << "The system is now running in deterministic mode. You can run the monitor tool\n";
    std::cout << "in another terminal with 'xmake run monitor_tool' to view performance metrics.\n";
    std::cout << "Press Ctrl+C to exit.\n\n";
    
    // Initialize and run the deterministic pricing system
    return initializeSystem(argc, argv);
}
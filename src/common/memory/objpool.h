#ifndef OBJ_POOL_H
#define OBJ_POOL_H

#include "mempool.h"
#include <memory>
#include <vector>

/**
 * @brief Specialized object pool for frequently allocated objects
 * 
 * This object pool uses the memory pool under the hood but provides
 * a more specialized interface for managing specific types of objects.
 * 
 * @tparam T Type of objects to allocate
 * @tparam BlockSize Size of each memory block in bytes
 * @tparam Alignment Memory alignment requirement
 */
template <typename T, size_t BlockSize = 4096, size_t Alignment = 64>
class ObjPool {
private:
    MemPool<T, BlockSize, Alignment> mem_pool_;
    
public:
    /**
     * @brief Constructor
     */
    ObjPool() = default;
    
    /**
     * @brief Destructor
     */
    ~ObjPool() = default;
    
    // Prevent copying
    ObjPool(const ObjPool&) = delete;
    ObjPool& operator=(const ObjPool&) = delete;
    
    // Allow moving
    ObjPool(ObjPool&&) = default;
    ObjPool& operator=(ObjPool&&) = default;
    
    /**
     * @brief Get a new object from the pool
     * 
     * @return Smart pointer to the object
     */
    std::shared_ptr<T> get() {
        return mem_pool_.create();
    }
    
    /**
     * @brief Create a new object with constructor arguments
     * 
     * @tparam Args Constructor argument types
     * @param args Constructor arguments
     * @return Smart pointer to the object
     */
    template <typename... Args>
    std::shared_ptr<T> create(Args&&... args) {
        return mem_pool_.create(std::forward<Args>(args)...);
    }
    
    /**
     * @brief Get multiple objects from the pool
     * 
     * @param count Number of objects to get
     * @return Vector of smart pointers to objects
     */
    std::vector<std::shared_ptr<T>> getBatch(size_t count) {
        std::vector<std::shared_ptr<T>> result;
        result.reserve(count);
        
        for (size_t i = 0; i < count; ++i) {
            result.push_back(get());
        }
        
        return result;
    }
    
    /**
     * @brief Get pool statistics
     * 
     * @return Tuple of (allocated, deallocated, active, blocks, utilization)
     */
    std::tuple<size_t, size_t, size_t, size_t, double> getStats() const {
        return {
            mem_pool_.getAllocatedCount(),
            mem_pool_.getDeallocatedCount(),
            mem_pool_.getActiveCount(),
            mem_pool_.getBlockCount(),
            mem_pool_.getUtilization()
        };
    }
    
    /**
     * @brief Clear the pool
     * 
     * Note: This will invalidate any objects obtained from this pool
     */
    void clear() {
        mem_pool_.clear();
    }
};

/**
 * @brief Option pricing result structure
 */
struct PricingResult {
    double price;
    double delta;
    double gamma;
    double vega;
    double theta;
    double rho;
    double early_exercise_premium;
    bool is_american;
    int iterations;
    
    PricingResult() : 
        price(0.0), delta(0.0), gamma(0.0), vega(0.0), 
        theta(0.0), rho(0.0), early_exercise_premium(0.0),
        is_american(true), iterations(0) {}
    
    PricingResult(double p, double d, double g, double v, double t, double r, double eep, bool a, int i) :
        price(p), delta(d), gamma(g), vega(v), 
        theta(t), rho(r), early_exercise_premium(eep),
        is_american(a), iterations(i) {}
};

/**
 * @brief Option pricing request structure
 */
struct PricingRequest {
    double S;        // Spot price
    double K;        // Strike price
    double r;        // Risk-free rate
    double q;        // Dividend yield
    double vol;      // Volatility
    double T;        // Time to maturity
    bool is_put;     // True for put, false for call
    bool is_american; // True for American, false for European
    
    PricingRequest() :
        S(0.0), K(0.0), r(0.0), q(0.0), vol(0.0), T(0.0),
        is_put(true), is_american(true) {}
    
    PricingRequest(double s, double k, double rate, double div, double volatility, double time, 
                  bool put = true, bool american = true) :
        S(s), K(k), r(rate), q(div), vol(volatility), T(time),
        is_put(put), is_american(american) {}
};

// Specialized object pools for common types
using PricingResultPool = ObjPool<PricingResult, 4096>;
using PricingRequestPool = ObjPool<PricingRequest, 4096>;

#endif 
 #ifndef ENGINE_ALO_MOD_EUROPEAN_H
 #define ENGINE_ALO_MOD_EUROPEAN_H
 
 #include <array>
 #include <vector>
 #include <immintrin.h>  // For SIMD operations
 
 namespace engine {
 namespace alo {
 namespace mod {
 
 /**
  * @class EuropeanOption
  * @brief Base class for European option pricing models
  * 
  * This class provides common functionality for European option pricing
  * using the Black-Scholes formula with deterministic execution.
  */
 class EuropeanOption {
 public:
     /**
      * @brief Constructor
      */
     EuropeanOption() = default;
     
     /**
      * @brief Destructor
      */
     virtual ~EuropeanOption() = default;
     
     /**
      * @brief Calculate option price
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Option price
      */
     virtual double calculatePrice(double S, double K, double r, double q, double vol, double T) const = 0;
     
     /**
      * @brief Calculate option delta
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Option delta
      */
     virtual double calculateDelta(double S, double K, double r, double q, double vol, double T) const = 0;
     
     /**
      * @brief Calculate option gamma
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Option gamma
      */
     virtual double calculateGamma(double S, double K, double r, double q, double vol, double T) const = 0;
     
     /**
      * @brief Calculate option vega
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Option vega
      */
     virtual double calculateVega(double S, double K, double r, double q, double vol, double T) const = 0;
     
     /**
      * @brief Calculate option theta
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Option theta
      */
     virtual double calculateTheta(double S, double K, double r, double q, double vol, double T) const = 0;
     
     /**
      * @brief Calculate option rho
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Option rho
      */
     virtual double calculateRho(double S, double K, double r, double q, double vol, double T) const = 0;
     
 protected:
     /**
      * @brief Calculate Black-Scholes d1 term
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return d1 term
      */
     static double d1(double S, double K, double r, double q, double vol, double T);
     
     /**
      * @brief Calculate Black-Scholes d2 term
      * 
      * @param d1 d1 term
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return d2 term
      */
     static double d2(double d1, double vol, double T);
     
     /**
      * @brief Calculate normal CDF
      * 
      * @param x Input value
      * @return Normal CDF at x
      */
     static double normalCDF(double x);
     
     /**
      * @brief Calculate normal PDF
      * 
      * @param x Input value
      * @return Normal PDF at x
      */
     static double normalPDF(double x);
 };
 
 /**
  * @class EuropeanPut
  * @brief European put option pricing model
  * 
  * This class implements European put option pricing using the
  * Black-Scholes formula with deterministic execution.
  */
 class EuropeanPut : public EuropeanOption {
 public:
     /**
      * @brief Constructor
      */
     EuropeanPut() = default;
     
     /**
      * @brief Calculate put option price
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Put option price
      */
     double calculatePrice(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate put option delta
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Put option delta
      */
     double calculateDelta(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate put option gamma
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Put option gamma
      */
     double calculateGamma(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate put option vega
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Put option vega
      */
     double calculateVega(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate put option theta
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Put option theta
      */
     double calculateTheta(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate put option rho
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Put option rho
      */
     double calculateRho(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate prices for multiple put options with the same parameters except strikes
      * 
      * @param S Current spot price
      * @param strikes Vector of strike prices
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Vector of put option prices
      */
     std::vector<double> batchCalculatePrice(double S, const std::vector<double>& strikes,
                                            double r, double q, double vol, double T) const;
     
     /**
      * @brief SIMD-accelerated pricing for 4 put options at once with AVX2
      * 
      * @param spots Array of 4 spot prices
      * @param strikes Array of 4 strike prices
      * @param rs Array of 4 risk-free rates
      * @param qs Array of 4 dividend yields
      * @param vols Array of 4 volatilities
      * @param Ts Array of 4 times to maturity
      * @return Array of 4 put option prices
      */
     std::array<double, 4> calculatePrice4(
         const std::array<double, 4>& spots,
         const std::array<double, 4>& strikes,
         const std::array<double, 4>& rs,
         const std::array<double, 4>& qs,
         const std::array<double, 4>& vols,
         const std::array<double, 4>& Ts) const;
 };
 
 /**
  * @class EuropeanCall
  * @brief European call option pricing model
  * 
  * This class implements European call option pricing using the
  * Black-Scholes formula with deterministic execution.
  */
 class EuropeanCall : public EuropeanOption {
 public:
     /**
      * @brief Constructor
      */
     EuropeanCall() = default;
     
     /**
      * @brief Calculate call option price
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Call option price
      */
     double calculatePrice(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate call option delta
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Call option delta
      */
     double calculateDelta(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate call option gamma
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Call option gamma
      */
     double calculateGamma(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate call option vega
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Call option vega
      */
     double calculateVega(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate call option theta
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Call option theta
      */
     double calculateTheta(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate call option rho
      * 
      * @param S Current spot price
      * @param K Strike price
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Call option rho
      */
     double calculateRho(double S, double K, double r, double q, double vol, double T) const override;
     
     /**
      * @brief Calculate prices for multiple call options with the same parameters except strikes
      * 
      * @param S Current spot price
      * @param strikes Vector of strike prices
      * @param r Risk-free interest rate
      * @param q Dividend yield
      * @param vol Volatility
      * @param T Time to maturity in years
      * @return Vector of call option prices
      */
     std::vector<double> batchCalculatePrice(double S, const std::vector<double>& strikes,
                                            double r, double q, double vol, double T) const;
     
     /**
      * @brief SIMD-accelerated pricing for 4 call options at once with AVX2
      * 
      * @param spots Array of 4 spot prices
      * @param strikes Array of 4 strike prices
      * @param rs Array of 4 risk-free rates
      * @param qs Array of 4 dividend yields
      * @param vols Array of 4 volatilities
      * @param Ts Array of 4 times to maturity
      * @return Array of 4 call option prices
      */
     std::array<double, 4> calculatePrice4(
         const std::array<double, 4>& spots,
         const std::array<double, 4>& strikes,
         const std::array<double, 4>& rs,
         const std::array<double, 4>& qs,
         const std::array<double, 4>& vols,
         const std::array<double, 4>& Ts) const;
 };
 
 /**
  * @brief Apply put-call parity to convert between put and call prices
  * 
  * @param isPut Whether to convert from call to put (true) or put to call (false)
  * @param price Input option price
  * @param S Current spot price
  * @param K Strike price
  * @param r Risk-free interest rate
  * @param q Dividend yield
  * @param T Time to maturity in years
  * @return Converted option price
  */
 double putCallParity(bool isPut, double price, double S, double K, double r, double q, double T);
 
 } // namespace mod
 } // namespace alo
 } // namespace engine
 
 #endif // ENGINE_ALO_MOD_EUROPEAN_H
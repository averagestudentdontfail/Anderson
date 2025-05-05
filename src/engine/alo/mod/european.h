#ifndef ENGINE_ALO_MOD_EUROPEAN_H
#define ENGINE_ALO_MOD_EUROPEAN_H

#include <array>
#include <immintrin.h>
#include <vector>

namespace engine {
namespace alo {
namespace mod {

/**
 * @class EuropeanOptionDouble
 * @brief Base class for double-precision European option pricing models
 *
 * This class provides common functionality for European option pricing
 * using the Black-Scholes formula with deterministic execution.
 */
class EuropeanOptionDouble {
public:
  /**
   * @brief Constructor
   */
  EuropeanOptionDouble() = default;

  /**
   * @brief Destructor
   */
  virtual ~EuropeanOptionDouble() = default;

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
  virtual double calculatePrice(double S, double K, double r, double q,
                                double vol, double T) const = 0;

  /**
   * @brief Calculate option delta
   */
  virtual double calculateDelta(double S, double K, double r, double q,
                                double vol, double T) const = 0;

  /**
   * @brief Calculate option gamma
   */
  virtual double calculateGamma(double S, double K, double r, double q,
                                double vol, double T) const = 0;

  /**
   * @brief Calculate option vega
   */
  virtual double calculateVega(double S, double K, double r, double q,
                               double vol, double T) const = 0;

  /**
   * @brief Calculate option theta
   */
  virtual double calculateTheta(double S, double K, double r, double q,
                                double vol, double T) const = 0;

  /**
   * @brief Calculate option rho
   */
  virtual double calculateRho(double S, double K, double r, double q,
                              double vol, double T) const = 0;

protected:
  /**
   * @brief Calculate Black-Scholes d1 term
   */
  static double d1(double S, double K, double r, double q, double vol,
                   double T);

  /**
   * @brief Calculate Black-Scholes d2 term
   */
  static double d2(double d1, double vol, double T);

  /**
   * @brief Calculate normal CDF
   */
  static double normalCDF(double x);

  /**
   * @brief Calculate normal PDF
   */
  static double normalPDF(double x);
};

/**
 * @class EuropeanPutDouble
 * @brief Double-precision European put option pricing model
 */
class EuropeanPutDouble : public EuropeanOptionDouble {
public:
  /**
   * @brief Constructor
   */
  EuropeanPutDouble() = default;

  /**
   * @brief Calculate put option price
   */
  double calculatePrice(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate put option delta
   */
  double calculateDelta(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate put option gamma
   */
  double calculateGamma(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate put option vega
   */
  double calculateVega(double S, double K, double r, double q, double vol,
                       double T) const override;

  /**
   * @brief Calculate put option theta
   */
  double calculateTheta(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate put option rho
   */
  double calculateRho(double S, double K, double r, double q, double vol,
                      double T) const override;

  /**
   * @brief Calculate prices for multiple put options with the same parameters
   * except strikes
   */
  std::vector<double> batchCalculatePrice(double S,
                                          const std::vector<double> &strikes,
                                          double r, double q, double vol,
                                          double T) const;

  /**
   * @brief SIMD-accelerated pricing for 4 put options at once with AVX2
   */
  std::array<double, 4> calculatePrice4(const std::array<double, 4> &spots,
                                        const std::array<double, 4> &strikes,
                                        const std::array<double, 4> &rs,
                                        const std::array<double, 4> &qs,
                                        const std::array<double, 4> &vols,
                                        const std::array<double, 4> &Ts) const;
};

/**
 * @class EuropeanCallDouble
 * @brief Double-precision European call option pricing model
 */
class EuropeanCallDouble : public EuropeanOptionDouble {
public:
  /**
   * @brief Constructor
   */
  EuropeanCallDouble() = default;

  /**
   * @brief Calculate call option price
   */
  double calculatePrice(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate call option delta
   */
  double calculateDelta(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate call option gamma
   */
  double calculateGamma(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate call option vega
   */
  double calculateVega(double S, double K, double r, double q, double vol,
                       double T) const override;

  /**
   * @brief Calculate call option theta
   */
  double calculateTheta(double S, double K, double r, double q, double vol,
                        double T) const override;

  /**
   * @brief Calculate call option rho
   */
  double calculateRho(double S, double K, double r, double q, double vol,
                      double T) const override;

  /**
   * @brief Calculate prices for multiple call options with the same parameters
   * except strikes
   */
  std::vector<double> batchCalculatePrice(double S,
                                          const std::vector<double> &strikes,
                                          double r, double q, double vol,
                                          double T) const;

  /**
   * @brief SIMD-accelerated pricing for 4 call options at once with AVX2
   */
  std::array<double, 4> calculatePrice4(const std::array<double, 4> &spots,
                                        const std::array<double, 4> &strikes,
                                        const std::array<double, 4> &rs,
                                        const std::array<double, 4> &qs,
                                        const std::array<double, 4> &vols,
                                        const std::array<double, 4> &Ts) const;
};

/**
 * @brief Apply put-call parity for double precision
 */
double putCallParityDouble(bool isPut, double price, double S, double K,
                           double r, double q, double T);

/**
 * @class EuropeanOptionSingle
 * @brief Base class for single-precision European option pricing models
 *
 * This class provides common functionality for European option pricing
 * using the Black-Scholes formula with single-precision and SIMD optimization.
 */
class EuropeanOptionSingle {
public:
  /**
   * @brief Constructor
   */
  EuropeanOptionSingle() = default;

  /**
   * @brief Destructor
   */
  virtual ~EuropeanOptionSingle() = default;

  /**
   * @brief Calculate option price
   */
  virtual float calculatePrice(float S, float K, float r, float q, float vol,
                               float T) const = 0;

  /**
   * @brief Calculate option delta
   */
  virtual float calculateDelta(float S, float K, float r, float q, float vol,
                               float T) const = 0;

  /**
   * @brief Calculate option gamma
   */
  virtual float calculateGamma(float S, float K, float r, float q, float vol,
                               float T) const = 0;

  /**
   * @brief Calculate option vega
   */
  virtual float calculateVega(float S, float K, float r, float q, float vol,
                              float T) const = 0;

  /**
   * @brief Calculate option theta
   */
  virtual float calculateTheta(float S, float K, float r, float q, float vol,
                               float T) const = 0;

  /**
   * @brief Calculate option rho
   */
  virtual float calculateRho(float S, float K, float r, float q, float vol,
                             float T) const = 0;

protected:
  /**
   * @brief Calculate Black-Scholes d1 term
   */
  static float d1(float S, float K, float r, float q, float vol, float T);

  /**
   * @brief Calculate Black-Scholes d2 term
   */
  static float d2(float d1, float vol, float T);

  /**
   * @brief Calculate normal CDF using fast approximation
   */
  static float normalCDF(float x);

  /**
   * @brief Calculate normal PDF using fast approximation
   */
  static float normalPDF(float x);
};

/**
 * @class EuropeanPutSingle
 * @brief Single-precision European put option pricing model
 */
class EuropeanPutSingle : public EuropeanOptionSingle {
public:
  /**
   * @brief Constructor
   */
  EuropeanPutSingle() = default;

  /**
   * @brief Calculate put option price
   */
  float calculatePrice(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate put option delta
   */
  float calculateDelta(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate put option gamma
   */
  float calculateGamma(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate put option vega
   */
  float calculateVega(float S, float K, float r, float q, float vol,
                      float T) const override;

  /**
   * @brief Calculate put option theta
   */
  float calculateTheta(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate put option rho
   */
  float calculateRho(float S, float K, float r, float q, float vol,
                     float T) const override;

  /**
   * @brief Calculate prices for multiple put options with the same parameters
   * except strikes
   */
  std::vector<float> batchCalculatePrice(float S,
                                         const std::vector<float> &strikes,
                                         float r, float q, float vol,
                                         float T) const;

  /**
   * @brief SIMD-accelerated pricing for 8 put options at once with AVX2
   */
  std::array<float, 8> calculatePrice8(const std::array<float, 8> &spots,
                                       const std::array<float, 8> &strikes,
                                       const std::array<float, 8> &rs,
                                       const std::array<float, 8> &qs,
                                       const std::array<float, 8> &vols,
                                       const std::array<float, 8> &Ts) const;
};

/**
 * @class EuropeanCallSingle
 * @brief Single-precision European call option pricing model
 */
class EuropeanCallSingle : public EuropeanOptionSingle {
public:
  /**
   * @brief Constructor
   */
  EuropeanCallSingle() = default;

  /**
   * @brief Calculate call option price
   */
  float calculatePrice(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate call option delta
   */
  float calculateDelta(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate call option gamma
   */
  float calculateGamma(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate call option vega
   */
  float calculateVega(float S, float K, float r, float q, float vol,
                      float T) const override;

  /**
   * @brief Calculate call option theta
   */
  float calculateTheta(float S, float K, float r, float q, float vol,
                       float T) const override;

  /**
   * @brief Calculate call option rho
   */
  float calculateRho(float S, float K, float r, float q, float vol,
                     float T) const override;

  /**
   * @brief Calculate prices for multiple call options with the same parameters
   * except strikes
   */
  std::vector<float> batchCalculatePrice(float S,
                                         const std::vector<float> &strikes,
                                         float r, float q, float vol,
                                         float T) const;

  /**
   * @brief SIMD-accelerated pricing for 8 call options at once with AVX2
   */
  std::array<float, 8> calculatePrice8(const std::array<float, 8> &spots,
                                       const std::array<float, 8> &strikes,
                                       const std::array<float, 8> &rs,
                                       const std::array<float, 8> &qs,
                                       const std::array<float, 8> &vols,
                                       const std::array<float, 8> &Ts) const;
};

/**
 * @brief Apply put-call parity with single-precision
 */
float putCallParitySingle(bool isPut, float price, float S, float K, float r,
                          float q, float T);

} // namespace mod
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_MOD_EUROPEAN_H
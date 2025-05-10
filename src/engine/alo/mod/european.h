#ifndef ENGINE_ALO_MOD_EUROPEAN_H
#define ENGINE_ALO_MOD_EUROPEAN_H

#include <array>
#include <vector>
#include "../num/float.h" // For num::fast_normal_cdf/pdf for single precision defaults

namespace engine {
namespace alo {
namespace mod {

/**
 * @class EuropeanOptionDouble
 */
class EuropeanOptionDouble {
public:
  EuropeanOptionDouble() = default;
  virtual ~EuropeanOptionDouble() = default;

  virtual double calculatePrice(double S, double K, double r, double q,
                                double vol, double T) const = 0;
  virtual double calculateDelta(double S, double K, double r, double q,
                                double vol, double T) const = 0;
  virtual double calculateGamma(double S, double K, double r, double q,
                                double vol, double T) const = 0;
  virtual double calculateVega(double S, double K, double r, double q,
                               double vol, double T) const = 0;
  virtual double calculateTheta(double S, double K, double r, double q,
                                double vol, double T) const = 0;
  virtual double calculateRho(double S, double K, double r, double q,
                              double vol, double T) const = 0;
protected:
  // These helpers are fundamental Black-Scholes components
  static double d1(double S, double K, double r, double q, double vol, double T);
  static double d2(double d1_val, double vol, double T); // Renamed d1 to d1_val
  static double normalCDF(double x); // Standard N(x)
  static double normalPDF(double x); // Standard n(x)
};

/**
 * @class EuropeanPutDouble
 */
class EuropeanPutDouble : public EuropeanOptionDouble {
public:
  EuropeanPutDouble() = default;
  double calculatePrice(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateDelta(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateGamma(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateVega(double S, double K, double r, double q, double vol,
                       double T) const override;
  double calculateTheta(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateRho(double S, double K, double r, double q, double vol,
                      double T) const override;

  std::vector<double> batchCalculatePrice(double S,
                                          const std::vector<double> &strikes,
                                          double r, double q, double vol,
                                          double T) const;
  std::array<double, 4> calculatePrice4(const std::array<double, 4> &spots, // Corresponds to AVX2 double
                                        const std::array<double, 4> &strikes,
                                        const std::array<double, 4> &rs,
                                        const std::array<double, 4> &qs,
                                        const std::array<double, 4> &vols,
                                        const std::array<double, 4> &Ts) const;
};

/**
 * @class EuropeanCallDouble
 */
class EuropeanCallDouble : public EuropeanOptionDouble {
public:
  EuropeanCallDouble() = default;
  double calculatePrice(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateDelta(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateGamma(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateVega(double S, double K, double r, double q, double vol,
                       double T) const override;
  double calculateTheta(double S, double K, double r, double q, double vol,
                        double T) const override;
  double calculateRho(double S, double K, double r, double q, double vol,
                      double T) const override;

  std::vector<double> batchCalculatePrice(double S,
                                          const std::vector<double> &strikes,
                                          double r, double q, double vol,
                                          double T) const;
  std::array<double, 4> calculatePrice4(const std::array<double, 4> &spots,
                                        const std::array<double, 4> &strikes,
                                        const std::array<double, 4> &rs,
                                        const std::array<double, 4> &qs,
                                        const std::array<double, 4> &vols,
                                        const std::array<double, 4> &Ts) const;
};

double putCallParityDouble(bool isCall, double option_price, double S, double K, // Changed first param to isCall
                           double r, double q, double T);


/**
 * @class EuropeanOptionSingle
 */
class EuropeanOptionSingle {
public:
  EuropeanOptionSingle() = default;
  virtual ~EuropeanOptionSingle() = default;

  virtual float calculatePrice(float S, float K, float r, float q, float vol,
                               float T) const = 0;
  virtual float calculateDelta(float S, float K, float r, float q, float vol,
                               float T) const = 0;
  virtual float calculateGamma(float S, float K, float r, float q, float vol,
                               float T) const = 0;
  virtual float calculateVega(float S, float K, float r, float q, float vol,
                              float T) const = 0;
  virtual float calculateTheta(float S, float K, float r, float q, float vol,
                               float T) const = 0;
  virtual float calculateRho(float S, float K, float r, float q, float vol,
                             float T) const = 0;
protected:
  static float d1(float S, float K, float r, float q, float vol, float T);
  static float d2(float d1_val, float vol, float T); // Renamed d1 to d1_val
  static float normalCDF(float x); // Uses num::fast_normal_cdf
  static float normalPDF(float x); // Uses num::fast_normal_pdf
};

/**
 * @class EuropeanPutSingle
 */
class EuropeanPutSingle : public EuropeanOptionSingle {
public:
  EuropeanPutSingle() = default;
  float calculatePrice(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateDelta(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateGamma(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateVega(float S, float K, float r, float q, float vol,
                      float T) const override;
  float calculateTheta(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateRho(float S, float K, float r, float q, float vol,
                     float T) const override;

  std::vector<float> batchCalculatePrice(float S,
                                         const std::vector<float> &strikes,
                                         float r, float q, float vol,
                                         float T) const;
  std::array<float, 8> calculatePrice8(const std::array<float, 8> &spots, // Corresponds to AVX2 float
                                       const std::array<float, 8> &strikes,
                                       const std::array<float, 8> &rs,
                                       const std::array<float, 8> &qs,
                                       const std::array<float, 8> &vols,
                                       const std::array<float, 8> &Ts) const;
};

/**
 * @class EuropeanCallSingle
 */
class EuropeanCallSingle : public EuropeanOptionSingle {
public:
  EuropeanCallSingle() = default;
  float calculatePrice(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateDelta(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateGamma(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateVega(float S, float K, float r, float q, float vol,
                      float T) const override;
  float calculateTheta(float S, float K, float r, float q, float vol,
                       float T) const override;
  float calculateRho(float S, float K, float r, float q, float vol,
                     float T) const override;

  std::vector<float> batchCalculatePrice(float S,
                                         const std::vector<float> &strikes,
                                         float r, float q, float vol,
                                         float T) const;
  std::array<float, 8> calculatePrice8(const std::array<float, 8> &spots,
                                       const std::array<float, 8> &strikes,
                                       const std::array<float, 8> &rs,
                                       const std::array<float, 8> &qs,
                                       const std::array<float, 8> &vols,
                                       const std::array<float, 8> &Ts) const;
};

float putCallParitySingle(bool isCall, float option_price, float S, float K, float r, // Changed first param
                          float q, float T);

} // namespace mod
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_MOD_EUROPEAN_H
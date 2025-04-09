// hmm.h
// Hidden Markov Model implementation for market regime detection

#ifndef HMM_H
#define HMM_H

#include <vector>
#include <deque>
#include <string>
#include <memory>
#include <random>
#include <cmath>
#include <unordered_map>
#include <eigen3/Eigen/Dense>

namespace vol_arb {

// Forward declaration
class MarketFeatures;

// Define matrix types using Eigen for efficient operations
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

// Structure for emission distributions (observations given states)
class EmissionDistribution {
public:
    enum DistType {
        GAUSSIAN,         // Single gaussian distribution
        GAUSSIAN_MIXTURE, // Mixture of gaussians
        STUDENT_T         // Student's t-distribution
    };
    
    // Constructor for different distribution types
    EmissionDistribution(DistType type = GAUSSIAN, int dimension = 1);
    
    // Set parameters for Gaussian distribution
    void setGaussianParameters(const Vector& mean, const Matrix& covariance);
    
    // Set parameters for mixture of Gaussians
    void setGaussianMixtureParameters(const std::vector<Vector>& means, 
                                   const std::vector<Matrix>& covariances,
                                   const Vector& weights);
    
    // Set parameters for Student's t-distribution
    void setStudentTParameters(const Vector& location, const Matrix& scale, double dof);
    
    // Calculate probability density for observation
    double pdf(const Vector& observation) const;
    
    // Generate a random sample from this distribution
    Vector sample(std::mt19937& rng) const;
    
    // Get distribution type
    DistType getType() const { return type_; }
    
    // Get distribution dimension
    int getDimension() const { return dimension_; }

private:
    DistType type_;
    int dimension_;
    
    // Gaussian parameters
    Vector mean_;
    Matrix covariance_;
    Matrix covarianceInverse_;
    double normalizationFactor_;
    
    // Gaussian mixture parameters
    std::vector<Vector> mixtureMeans_;
    std::vector<Matrix> mixtureCovariances_;
    std::vector<Matrix> mixtureCovarianceInverses_;
    Vector mixtureWeights_;
    std::vector<double> mixtureNormFactors_;
    
    // Student's t parameters
    Vector location_;
    Matrix scale_;
    Matrix scaleInverse_;
    double degreesOfFreedom_;
    double tNormalizationFactor_;
    
    // Internal helper methods
    void computeGaussianNormalization();
    void computeMixtureNormalizations();
    void computeTNormalization();
};

// Structure for market regime parameters
struct MarketRegimeParams {
    std::string name;           // Descriptive name (e.g., "Low Volatility")
    EmissionDistribution distribution; // Emission distribution for this regime
    
    // Default constructor
    MarketRegimeParams() : name("Undefined") {}
    
    // Constructor with name and distribution
    MarketRegimeParams(const std::string& name, const EmissionDistribution& dist)
        : name(name), distribution(dist) {}
};

// Enum for predefined market regimes
enum MarketRegime {
    LOW_VOLATILITY,
    MEDIUM_VOLATILITY,
    HIGH_VOLATILITY,
    TRANSITION_REGIME,
    CUSTOM_REGIME
};

// Class for Hidden Markov Model
class HiddenMarkovModel {
public:
    // Constructor with number of states
    explicit HiddenMarkovModel(int numStates = 3);
    
    // Initialize the model with predefined regime parameters
    void initializeWithDefaultRegimes(int featureDimension = 1);
    
    // Set transition matrix
    void setTransitionMatrix(const Matrix& transitionMatrix);
    
    // Set emission distributions
    void setEmissionDistributions(const std::vector<EmissionDistribution>& distributions);
    
    // Set initial state probabilities
    void setInitialProbabilities(const Vector& initialProbs);
    
    // Update state estimates with new observation
    void update(const Vector& observation);
    
    // Update with multiple observations
    void update(const std::vector<Vector>& observations);
    
    // Get current state probabilities
    Vector getStateProbabilities() const;
    
    // Get probability of specific state
    double getStateProbability(int state) const;
    
    // Get the most likely current state
    int getMostLikelyState() const;
    
    // Get the regime type of the most likely state
    MarketRegime getCurrentRegime() const;
    
    // Predict state probabilities n steps ahead
    Vector predictStateProbabilities(int stepsAhead) const;
    
    // Predict probability of regime change within n steps
    double predictRegimeChangeProbability(int stepsAhead) const;
    
    // Train the model using Baum-Welch algorithm
    void train(const std::vector<Vector>& observations, 
              int maxIterations = 100, 
              double tolerance = 1e-4);
    
    // Decode the most likely state sequence for given observations (Viterbi algorithm)
    std::vector<int> decode(const std::vector<Vector>& observations) const;
    
    // Calculate log-likelihood of observations
    double logLikelihood(const std::vector<Vector>& observations) const;
    
    // Generate a sequence of observations
    std::vector<Vector> generateSequence(int length, std::mt19937& rng) const;
    
    // Get the number of states
    int getNumStates() const { return numStates_; }
    
    // Get the transition matrix
    const Matrix& getTransitionMatrix() const { return transitionMatrix_; }
    
    // Get emission distributions
    const std::vector<EmissionDistribution>& getEmissionDistributions() const { 
        return emissionDistributions_; 
    }
    
    // Set regime names
    void setRegimeNames(const std::vector<std::string>& names);
    
    // Get regime name for a state
    std::string getRegimeName(int state) const;
    
    // Map a state to a regime type
    MarketRegime mapStateToRegime(int state) const;
    
    // Set mapping from states to regime types
    void setStateRegimeMapping(const std::vector<MarketRegime>& mapping);
    
    // Force regime probabilities for testing purposes
    void forceRegimeProbabilities(const Vector& probabilities);

private:
    int numStates_;
    Matrix transitionMatrix_;
    std::vector<EmissionDistribution> emissionDistributions_;
    Vector initialStateProbs_;
    Vector currentStateProbs_;
    std::vector<std::string> regimeNames_;
    std::vector<MarketRegime> stateRegimeMapping_;
    
    // Forward-backward algorithm helper methods
    std::pair<Matrix, double> forward(const std::vector<Vector>& observations) const;
    Matrix backward(const std::vector<Vector>& observations, double scalingFactor) const;
    
    // Helper for Baum-Welch algorithm
    void baumWelchIteration(const std::vector<Vector>& observations, 
                          Matrix& transitionMatrix, 
                          std::vector<EmissionDistribution>& emissionDists,
                          Vector& initialProbs, 
                          double& logLikelihood);
    
    // Initialize with random parameters
    void initializeRandomly(int featureDimension);
};

// Utility class for feature extraction from market data
class MarketFeatures {
public:
    MarketFeatures() = default;
    
    // Extract features from price and volume data
    static Vector extract(const std::vector<double>& prices, 
                        const std::vector<double>& volumes = {});
    
    // Normalize features using z-score
    static Vector normalize(const Vector& features, 
                          const Vector& mean, 
                          const Vector& stdDev);
    
    // Calculate mean and standard deviation for a set of feature vectors
    static std::pair<Vector, Vector> calculateStats(const std::vector<Vector>& featureVectors);
    
    // Feature extraction for volatility regimes
    static Vector extractVolatilityFeatures(const std::vector<double>& returns, 
                                          int window = 22);
    
    // Create predefined feature set for volatility regimes
    static std::vector<Vector> createVolatilityRegimeFeatures();
};

} // namespace vol_arb

#endif // HMM_H
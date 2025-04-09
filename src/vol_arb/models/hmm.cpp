#include "hmm.h"
#include <cmath>
#include <random>
#include <algorithm>
#include <stdexcept>
#include <iostream>
#include <chrono>
#include <limits>

namespace vol_arb {

//------------------------------------------------------------------------------
// EmissionDistribution Implementation
//------------------------------------------------------------------------------

EmissionDistribution::EmissionDistribution(DistType type, int dimension)
    : type_(type), dimension_(dimension), 
      mean_(Vector::Zero(dimension)), 
      covariance_(Matrix::Identity(dimension, dimension)),
      covarianceInverse_(Matrix::Identity(dimension, dimension)),
      normalizationFactor_(1.0),
      location_(Vector::Zero(dimension)),
      scale_(Matrix::Identity(dimension, dimension)),
      scaleInverse_(Matrix::Identity(dimension, dimension)),
      degreesOfFreedom_(5.0) {
    
    if (dimension < 1) {
        throw std::invalid_argument("Distribution dimension must be positive");
    }
    
    // Initialize based on distribution type
    switch (type_) {
        case GAUSSIAN:
            computeGaussianNormalization();
            break;
            
        case GAUSSIAN_MIXTURE:
            // Default to a single component mixture (equivalent to Gaussian)
            mixtureMeans_.push_back(Vector::Zero(dimension));
            mixtureCovariances_.push_back(Matrix::Identity(dimension, dimension));
            mixtureCovarianceInverses_.push_back(Matrix::Identity(dimension, dimension));
            mixtureWeights_ = Vector::Ones(1);
            mixtureNormFactors_.push_back(1.0);
            computeMixtureNormalizations();
            break;
            
        case STUDENT_T:
            computeTNormalization();
            break;
    }
}

void EmissionDistribution::setGaussianParameters(const Vector& mean, const Matrix& covariance) {
    if (type_ != GAUSSIAN) {
        throw std::invalid_argument("Distribution is not Gaussian");
    }
    
    if (mean.size() != dimension_ || covariance.rows() != dimension_ || covariance.cols() != dimension_) {
        throw std::invalid_argument("Parameter dimensions do not match distribution dimension");
    }
    
    mean_ = mean;
    covariance_ = covariance;
    
    // Compute inverse and normalization factor
    Eigen::LLT<Matrix> llt(covariance_);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument("Covariance matrix is not positive definite");
    }
    
    covarianceInverse_ = covariance_.inverse();
    computeGaussianNormalization();
}

void EmissionDistribution::setGaussianMixtureParameters(const std::vector<Vector>& means, 
                                                      const std::vector<Matrix>& covariances,
                                                      const Vector& weights) {
    if (type_ != GAUSSIAN_MIXTURE) {
        throw std::invalid_argument("Distribution is not Gaussian mixture");
    }
    
    if (means.size() != covariances.size() || static_cast<size_t>(weights.size()) != means.size()) {
        throw std::invalid_argument("Inconsistent number of components in mixture");
    }
    
    for (size_t i = 0; i < means.size(); ++i) {
        if (means[i].size() != dimension_ || 
            covariances[i].rows() != dimension_ || 
            covariances[i].cols() != dimension_) {
            throw std::invalid_argument("Parameter dimensions do not match distribution dimension");
        }
    }
    
    // Normalize weights to sum to 1
    double weightSum = weights.sum();
    if (weightSum <= 0) {
        throw std::invalid_argument("Mixture weights must be positive and sum to a positive value");
    }
    
    mixtureMeans_ = means;
    mixtureCovariances_ = covariances;
    mixtureWeights_ = weights / weightSum;
    
    // Compute inverses and normalization factors
    mixtureCovarianceInverses_.resize(covariances.size());
    mixtureNormFactors_.resize(covariances.size());
    
    for (size_t i = 0; i < covariances.size(); ++i) {
        Eigen::LLT<Matrix> llt(mixtureCovariances_[i]);
        if (llt.info() != Eigen::Success) {
            throw std::invalid_argument("Covariance matrix for component " + 
                                      std::to_string(i) + " is not positive definite");
        }
        
        mixtureCovarianceInverses_[i] = mixtureCovariances_[i].inverse();
    }
    
    computeMixtureNormalizations();
}

void EmissionDistribution::setStudentTParameters(const Vector& location, const Matrix& scale, double dof) {
    if (type_ != STUDENT_T) {
        throw std::invalid_argument("Distribution is not Student's t");
    }
    
    if (location.size() != dimension_ || 
        scale.rows() != dimension_ || 
        scale.cols() != dimension_) {
        throw std::invalid_argument("Parameter dimensions do not match distribution dimension");
    }
    
    if (dof <= 2.0) {
        throw std::invalid_argument("Degrees of freedom must be greater than 2 for finite variance");
    }
    
    location_ = location;
    scale_ = scale;
    degreesOfFreedom_ = dof;
    
    // Compute inverse and normalization factor
    Eigen::LLT<Matrix> llt(scale_);
    if (llt.info() != Eigen::Success) {
        throw std::invalid_argument("Scale matrix is not positive definite");
    }
    
    scaleInverse_ = scale_.inverse();
    computeTNormalization();
}

double EmissionDistribution::pdf(const Vector& observation) const {
    if (observation.size() != dimension_) {
        throw std::invalid_argument("Observation dimension does not match distribution dimension");
    }
    
    switch (type_) {
        case GAUSSIAN: {
            // Gaussian PDF: (2π)^(-d/2) |Σ|^(-1/2) exp(-0.5 (x-μ)' Σ^(-1) (x-μ))
            Vector centered = observation - mean_;
            double quadForm = centered.dot(covarianceInverse_ * centered);
            return normalizationFactor_ * std::exp(-0.5 * quadForm);
        }
            
        case GAUSSIAN_MIXTURE: {
            // Mixture PDF: sum_i w_i * N(x | μ_i, Σ_i)
            double result = 0.0;
            
            for (size_t i = 0; i < mixtureMeans_.size(); ++i) {
                Vector centered = observation - mixtureMeans_[i];
                double quadForm = centered.dot(mixtureCovarianceInverses_[i] * centered);
                double componentDensity = mixtureNormFactors_[i] * std::exp(-0.5 * quadForm);
                result += mixtureWeights_[i] * componentDensity;
            }
            
            return result;
        }
            
        case STUDENT_T: {
            // Student's t PDF
            Vector centered = observation - location_;
            double quadForm = centered.dot(scaleInverse_ * centered);
            return tNormalizationFactor_ * 
                  std::pow(1.0 + quadForm / degreesOfFreedom_, 
                         -(degreesOfFreedom_ + dimension_) / 2.0);
        }
    }
    
    return 0.0; // Should never reach here
}

Vector EmissionDistribution::sample(std::mt19937& rng) const {
    // Random number generators
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);
    
    switch (type_) {
        case GAUSSIAN: {
            // Use Cholesky decomposition for sampling
            Matrix L = covariance_.llt().matrixL();
            
            // Generate standard normal samples
            Vector z(dimension_);
            for (int i = 0; i < dimension_; ++i) {
                z(i) = normal_dist(rng);
            }
            
            // Transform to correlated multivariate normal
            return mean_ + L * z;
        }
            
        case GAUSSIAN_MIXTURE: {
            // First select a component based on weights
            double u = uniform_dist(rng);
            double cumSum = 0.0;
            size_t selectedComponent = 0;
            
            for (size_t i = 0; i < static_cast<size_t>(mixtureWeights_.size()); ++i) {
                cumSum += mixtureWeights_[i];
                if (u <= cumSum) {
                    selectedComponent = i;
                    break;
                }
            }
            
            // Sample from the selected component
            Matrix L = mixtureCovariances_[selectedComponent].llt().matrixL();
            
            // Generate standard normal samples
            Vector z(dimension_);
            for (int i = 0; i < dimension_; ++i) {
                z(i) = normal_dist(rng);
            }
            
            // Transform to correlated multivariate normal
            return mixtureMeans_[selectedComponent] + L * z;
        }
            
        case STUDENT_T: {
            // Student's t via Gaussian and Chi-squared
            std::chi_squared_distribution<double> chi_sq_dist(degreesOfFreedom_);
            
            // Generate standard normal samples
            Vector z(dimension_);
            for (int i = 0; i < dimension_; ++i) {
                z(i) = normal_dist(rng);
            }
            
            // Generate chi-squared sample
            double w = chi_sq_dist(rng) / degreesOfFreedom_;
            
            // Use Cholesky decomposition of scale matrix
            Matrix L = scale_.llt().matrixL();
            
            // Transform to multivariate t
            return location_ + (L * z) / std::sqrt(w);
        }
    }
    
    return Vector::Zero(dimension_); // Should never reach here
}

void EmissionDistribution::computeGaussianNormalization() {
    double det = covariance_.determinant();
    if (det <= 0) {
        throw std::invalid_argument("Covariance matrix must have positive determinant");
    }
    
    normalizationFactor_ = 1.0 / std::sqrt(std::pow(2.0 * M_PI, dimension_) * det);
}

void EmissionDistribution::computeMixtureNormalizations() {
    mixtureNormFactors_.resize(mixtureCovariances_.size());
    
    for (size_t i = 0; i < mixtureCovariances_.size(); ++i) {
        double det = mixtureCovariances_[i].determinant();
        if (det <= 0) {
            throw std::invalid_argument("Mixture covariance matrix must have positive determinant");
        }
        
        mixtureNormFactors_[i] = 1.0 / std::sqrt(std::pow(2.0 * M_PI, dimension_) * det);
    }
}

void EmissionDistribution::computeTNormalization() {
    double det = scale_.determinant();
    if (det <= 0) {
        throw std::invalid_argument("Scale matrix must have positive determinant");
    }
    
    tNormalizationFactor_ = std::tgamma((degreesOfFreedom_ + dimension_) / 2.0) /
                           (std::tgamma(degreesOfFreedom_ / 2.0) *
                            std::pow(degreesOfFreedom_ * M_PI, dimension_ / 2.0) *
                            std::sqrt(det));
}

//------------------------------------------------------------------------------
// HiddenMarkovModel Implementation
//------------------------------------------------------------------------------

HiddenMarkovModel::HiddenMarkovModel(int numStates)
    : numStates_(numStates),
      transitionMatrix_(Matrix::Zero(numStates, numStates)),
      initialStateProbs_(Vector::Ones(numStates) / numStates),
      currentStateProbs_(Vector::Ones(numStates) / numStates) {
    
    if (numStates < 1) {
        throw std::invalid_argument("Number of states must be positive");
    }
    
    // Initialize with random emissions and uniform transitions
    emissionDistributions_.resize(numStates);
    regimeNames_.resize(numStates);
    stateRegimeMapping_.resize(numStates);
    
    // Initialize transition matrix to uniform transitions
    transitionMatrix_ = Matrix::Ones(numStates, numStates) / numStates;
    
    // Set default regime names and mappings
    for (int i = 0; i < numStates; ++i) {
        regimeNames_[i] = "State " + std::to_string(i);
        
        // Map states to regimes based on number of states
        if (numStates <= 3) {
            if (i == 0) stateRegimeMapping_[i] = LOW_VOLATILITY;
            else if (i == 1) stateRegimeMapping_[i] = MEDIUM_VOLATILITY;
            else stateRegimeMapping_[i] = HIGH_VOLATILITY;
        } else {
            if (i == 0) stateRegimeMapping_[i] = LOW_VOLATILITY;
            else if (i == numStates - 1) stateRegimeMapping_[i] = HIGH_VOLATILITY;
            else if (i == 1 || i == numStates - 2) stateRegimeMapping_[i] = TRANSITION_REGIME;
            else stateRegimeMapping_[i] = MEDIUM_VOLATILITY;
        }
    }
}

void HiddenMarkovModel::initializeWithDefaultRegimes(int featureDimension) {
    // Reset state and ensure we have the right number of states
    if (numStates_ != 3) {
        numStates_ = 3;
        transitionMatrix_ = Matrix::Zero(numStates_, numStates_);
        emissionDistributions_.resize(numStates_);
        regimeNames_.resize(numStates_);
        stateRegimeMapping_.resize(numStates_);
        initialStateProbs_ = Vector::Ones(numStates_) / numStates_;
        currentStateProbs_ = Vector::Ones(numStates_) / numStates_;
    }
    
    // Set regime names
    regimeNames_[0] = "Low Volatility";
    regimeNames_[1] = "Medium Volatility";
    regimeNames_[2] = "High Volatility";
    
    // Set regime mappings
    stateRegimeMapping_[0] = LOW_VOLATILITY;
    stateRegimeMapping_[1] = MEDIUM_VOLATILITY;
    stateRegimeMapping_[2] = HIGH_VOLATILITY;
    
    // Set transition matrix (with high persistence)
    // Low vol -> Low vol: 0.95, Low vol -> Med vol: 0.05
    // Med vol -> Low vol: 0.03, Med vol -> Med vol: 0.94, Med vol -> High vol: 0.03
    // High vol -> Med vol: 0.10, High vol -> High vol: 0.90
    Matrix transitions(numStates_, numStates_);
    transitions << 0.95, 0.05, 0.00,
                  0.03, 0.94, 0.03,
                  0.00, 0.10, 0.90;
    setTransitionMatrix(transitions);
    
    // Create emission distributions for volatility features
    // For 1D features (e.g., volatility):
    if (featureDimension == 1) {
        // Low volatility state: mean around 0.15 (15% annualized)
        Vector meanLow(1);
        meanLow << 0.15;
        Matrix covLow(1, 1);
        covLow << 0.02*0.02; // standard deviation of 2%
        
        // Medium volatility state: mean around 0.25 (25% annualized)
        Vector meanMed(1);
        meanMed << 0.25;
        Matrix covMed(1, 1);
        covMed << 0.04*0.04; // standard deviation of 4%
        
        // High volatility state: mean around 0.40 (40% annualized)
        Vector meanHigh(1);
        meanHigh << 0.40;
        Matrix covHigh(1, 1);
        covHigh << 0.08*0.08; // standard deviation of 8%
        
        emissionDistributions_[0] = EmissionDistribution(EmissionDistribution::GAUSSIAN, 1);
        emissionDistributions_[0].setGaussianParameters(meanLow, covLow);
        
        emissionDistributions_[1] = EmissionDistribution(EmissionDistribution::GAUSSIAN, 1);
        emissionDistributions_[1].setGaussianParameters(meanMed, covMed);
        
        emissionDistributions_[2] = EmissionDistribution(EmissionDistribution::GAUSSIAN, 1);
        emissionDistributions_[2].setGaussianParameters(meanHigh, covHigh);
    }
    else if (featureDimension == 2) {
        // For 2D features (e.g., volatility and volume ratio):
        // Low volatility state
        Vector meanLow(2);
        meanLow << 0.15, 0.8; // 15% vol, 0.8 volume ratio
        Matrix covLow(2, 2);
        covLow << 0.02*0.02, 0.005,
                  0.005, 0.1*0.1;
        
        // Medium volatility state
        Vector meanMed(2);
        meanMed << 0.25, 1.0; // 25% vol, normal volume
        Matrix covMed(2, 2);
        covMed << 0.04*0.04, 0.01,
                  0.01, 0.15*0.15;
        
        // High volatility state
        Vector meanHigh(2);
        meanHigh << 0.40, 1.5; // 40% vol, 50% higher volume
        Matrix covHigh(2, 2);
        covHigh << 0.08*0.08, 0.02,
                   0.02, 0.3*0.3;
        
        emissionDistributions_[0] = EmissionDistribution(EmissionDistribution::GAUSSIAN, 2);
        emissionDistributions_[0].setGaussianParameters(meanLow, covLow);
        
        emissionDistributions_[1] = EmissionDistribution(EmissionDistribution::GAUSSIAN, 2);
        emissionDistributions_[1].setGaussianParameters(meanMed, covMed);
        
        emissionDistributions_[2] = EmissionDistribution(EmissionDistribution::GAUSSIAN, 2);
        emissionDistributions_[2].setGaussianParameters(meanHigh, covHigh);
    }
    else {
        // For other dimensions, initialize with random parameters
        initializeRandomly(featureDimension);
    }
}

void HiddenMarkovModel::setTransitionMatrix(const Matrix& transitionMatrix) {
    if (transitionMatrix.rows() != numStates_ || transitionMatrix.cols() != numStates_) {
        throw std::invalid_argument("Transition matrix dimensions do not match number of states");
    }
    
    // Check that rows sum to 1
    for (int i = 0; i < numStates_; ++i) {
        double rowSum = transitionMatrix.row(i).sum();
        if (std::abs(rowSum - 1.0) > 1e-10) {
            throw std::invalid_argument("Row " + std::to_string(i) + 
                                      " of transition matrix does not sum to 1 (sum = " + 
                                      std::to_string(rowSum) + ")");
        }
    }
    
    transitionMatrix_ = transitionMatrix;
}

void HiddenMarkovModel::setEmissionDistributions(const std::vector<EmissionDistribution>& distributions) {
    if (static_cast<int>(distributions.size()) != numStates_) {
        throw std::invalid_argument("Number of emission distributions does not match number of states");
    }
    
    emissionDistributions_ = distributions;
}

void HiddenMarkovModel::setInitialProbabilities(const Vector& initialProbs) {
    if (initialProbs.size() != numStates_) {
        throw std::invalid_argument("Initial probability vector size does not match number of states");
    }
    
    // Check that probabilities sum to 1
    double sum = initialProbs.sum();
    if (std::abs(sum - 1.0) > 1e-10) {
        throw std::invalid_argument("Initial probabilities do not sum to 1 (sum = " + 
                                   std::to_string(sum) + ")");
    }
    
    initialStateProbs_ = initialProbs;
    currentStateProbs_ = initialProbs;
}

void HiddenMarkovModel::update(const Vector& observation) {
    // Compute emission probabilities for each state
    Vector emissionProbs(numStates_);
    for (int j = 0; j < numStates_; ++j) {
        emissionProbs(j) = emissionDistributions_[j].pdf(observation);
    }
    
    // Forward algorithm update step (prediction + update)
    // Prediction step: α_t+1|t = T' * α_t
    Vector predictedProbs = transitionMatrix_.transpose() * currentStateProbs_;
    
    // Update step: α_t+1 = normalize(b_t+1 .* α_t+1|t)
    Vector unnormalizedProbs = emissionProbs.cwiseProduct(predictedProbs);
    double normConstant = unnormalizedProbs.sum();
    
    if (normConstant > 0) {
        currentStateProbs_ = unnormalizedProbs / normConstant;
    } else {
        // If normalization constant is zero, retain previous probabilities
        // This can happen if the observation is very unlikely under all states
        std::cerr << "Warning: Zero normalization constant in HMM update" << std::endl;
    }
}

void HiddenMarkovModel::update(const std::vector<Vector>& observations) {
    for (const auto& obs : observations) {
        update(obs);
    }
}

void HiddenMarkovModel::forceRegimeProbabilities(const Vector& probabilities) {
    if (probabilities.size() != numStates_) {
        throw std::invalid_argument("Probability vector size does not match number of states");
    }
    
    double sum = probabilities.sum();
    if (std::abs(sum - 1.0) > 1e-10) {
        currentStateProbs_ = probabilities / sum;
    } else {
        currentStateProbs_ = probabilities;
    }
}

Vector HiddenMarkovModel::getStateProbabilities() const {
    return currentStateProbs_;
}

double HiddenMarkovModel::getStateProbability(int state) const {
    if (state < 0 || state >= numStates_) {
        throw std::out_of_range("State index out of range");
    }
    return currentStateProbs_(state);
}

int HiddenMarkovModel::getMostLikelyState() const {
    int maxIdx;
    currentStateProbs_.maxCoeff(&maxIdx);
    return maxIdx;
}

MarketRegime HiddenMarkovModel::getCurrentRegime() const {
    return mapStateToRegime(getMostLikelyState());
}

Vector HiddenMarkovModel::predictStateProbabilities(int stepsAhead) const {
    if (stepsAhead <= 0) {
        return currentStateProbs_;
    }
    
    // Compute power of transition matrix for multi-step prediction
    Matrix transitionPower = transitionMatrix_;
    for (int i = 1; i < stepsAhead; ++i) {
        transitionPower = transitionPower * transitionMatrix_;
    }
    
    // Predict future state probabilities
    return transitionPower.transpose() * currentStateProbs_;
}

double HiddenMarkovModel::predictRegimeChangeProbability(int stepsAhead) const {
    if (stepsAhead <= 0) {
        return 0.0;
    }
    
    // Get current most likely state/regime
    int currentState = getMostLikelyState();
    MarketRegime currentRegime = mapStateToRegime(currentState);
    
    // Predict future state probabilities
    Vector futureProbs = predictStateProbabilities(stepsAhead);
    
    // Sum probabilities of states that map to different regimes
    double regimeChangeProb = 0.0;
    for (int j = 0; j < numStates_; ++j) {
        if (mapStateToRegime(j) != currentRegime) {
            regimeChangeProb += futureProbs(j);
        }
    }
    
    return regimeChangeProb;
}

void HiddenMarkovModel::train(const std::vector<Vector>& observations, 
                          int maxIterations, 
                          double tolerance) {
    if (observations.empty()) {
        throw std::invalid_argument("Observation sequence cannot be empty for training");
    }
    
    int featureDimension = observations[0].size();
    for (const auto& obs : observations) {
        if (obs.size() != featureDimension) {
            throw std::invalid_argument("All observations must have the same dimension");
        }
    }
    
    // Initialize model parameters if needed
    if (emissionDistributions_.empty()) {
        initializeRandomly(featureDimension);
    }
    
    // Use Baum-Welch algorithm for training
    double prevLogLikelihood = -std::numeric_limits<double>::infinity();
    double currLogLikelihood = logLikelihood(observations);
    
    int iter = 0;
    while (iter < maxIterations && 
           (iter < 3 || std::abs(currLogLikelihood - prevLogLikelihood) > tolerance)) {
        
        // Create copies of current parameters for update
        Matrix newTransitionMatrix = transitionMatrix_;
        std::vector<EmissionDistribution> newEmissionDists = emissionDistributions_;
        Vector newInitialProbs = initialStateProbs_;
        
        // Perform one iteration of Baum-Welch
        prevLogLikelihood = currLogLikelihood;
        baumWelchIteration(observations, newTransitionMatrix, newEmissionDists, 
                         newInitialProbs, currLogLikelihood);
        
        // Update model parameters
        transitionMatrix_ = newTransitionMatrix;
        emissionDistributions_ = newEmissionDists;
        initialStateProbs_ = newInitialProbs;
        
        ++iter;
    }
    
    // Reset current state probabilities to initial
    currentStateProbs_ = initialStateProbs_;
}

void HiddenMarkovModel::baumWelchIteration(const std::vector<Vector>& observations, 
                                       Matrix& transitionMatrix, 
                                       std::vector<EmissionDistribution>& emissionDists,
                                       Vector& initialProbs, 
                                       double& logLikelihood) {
    size_t T = observations.size();
    
    // Calculate forward and backward variables
    auto [alpha, scalingFactor] = forward(observations);
    Matrix beta = backward(observations, scalingFactor);
    
    // Calculate state posterior probabilities (gamma) and transition posteriors (xi)
    std::vector<Vector> gamma(T, Vector::Zero(numStates_));
    std::vector<Matrix> xi(T-1, Matrix::Zero(numStates_, numStates_));
    
    for (size_t t = 0; t < T; ++t) {
        // Compute gamma_t(i) = P(q_t = i | observations)
        gamma[t] = alpha.col(t).cwiseProduct(beta.col(t));
        gamma[t] /= gamma[t].sum(); // Normalize
        
        if (t < T - 1) {
            // Compute xi_t(i,j) = P(q_t = i, q_{t+1} = j | observations)
            for (int i = 0; i < numStates_; ++i) {
                for (int j = 0; j < numStates_; ++j) {
                    // Formula: xi_t(i,j) = alpha_t(i) * a_ij * b_j(o_{t+1}) * beta_{t+1}(j) / scaling
                    xi[t](i, j) = alpha(i, t) * transitionMatrix(i, j) *
                                 emissionDists[j].pdf(observations[t+1]) * beta(j, t+1);
                }
            }
            xi[t] /= xi[t].sum(); // Normalize
        }
    }
    
    // Update model parameters
    
    // 1. Update initial state distribution
    initialProbs = gamma[0];
    
    // 2. Update transition matrix
    for (int i = 0; i < numStates_; ++i) {
        double denominator = 0.0;
        for (size_t t = 0; t < T - 1; ++t) {
            denominator += gamma[t](i);
        }
        
        for (int j = 0; j < numStates_; ++j) {
            double numerator = 0.0;
            for (size_t t = 0; t < T - 1; ++t) {
                numerator += xi[t](i, j);
            }
            
            if (denominator > 0) {
                transitionMatrix(i, j) = numerator / denominator;
            }
        }
        
        // Ensure row sums to 1 (numerical stability)
        double rowSum = transitionMatrix.row(i).sum();
        if (rowSum > 0) {
            transitionMatrix.row(i) /= rowSum;
        } else {
            // If row sum is zero, initialize with uniform transition
            transitionMatrix.row(i).setConstant(1.0 / numStates_);
        }
    }
    
    // 3. Update emission distributions
    // This depends on the specific emission distribution type
    // For simplicity, we'll focus on Gaussian emissions
    for (int j = 0; j < numStates_; ++j) {
        if (emissionDists[j].getType() == EmissionDistribution::GAUSSIAN) {
            Vector mean = Vector::Zero(emissionDists[j].getDimension());
            Matrix covariance = Matrix::Zero(emissionDists[j].getDimension(), 
                                         emissionDists[j].getDimension());
            double totalWeight = 0.0;
            
            // Compute weighted mean
            for (size_t t = 0; t < T; ++t) {
                mean += gamma[t](j) * observations[t];
                totalWeight += gamma[t](j);
            }
            
            if (totalWeight > 0) {
                mean /= totalWeight;
                
                // Compute weighted covariance
                for (size_t t = 0; t < T; ++t) {
                    Vector centered = observations[t] - mean;
                    covariance += gamma[t](j) * centered * centered.transpose();
                }
                covariance /= totalWeight;
                
                // Add small regularization to ensure positive definiteness
                covariance += Matrix::Identity(covariance.rows(), covariance.cols()) * 1e-6;
                
                // Update emission distribution
                emissionDists[j].setGaussianParameters(mean, covariance);
            }
        }
    }
    
    // Compute log-likelihood
    logLikelihood = -std::log(scalingFactor);
}

std::pair<Matrix, double> HiddenMarkovModel::forward(const std::vector<Vector>& observations) const {
    size_t T = observations.size();
    Matrix alpha(numStates_, T);
    std::vector<double> c(T, 0.0); // Scaling factors
    
    // Initialize
    for (int i = 0; i < numStates_; ++i) {
        alpha(i, 0) = initialStateProbs_(i) * emissionDistributions_[i].pdf(observations[0]);
    }
    
    // Scale alpha_0
    c[0] = alpha.col(0).sum();
    if (c[0] > 0) {
        alpha.col(0) /= c[0];
    }
    
    // Induction
    for (size_t t = 1; t < T; ++t) {
        for (int j = 0; j < numStates_; ++j) {
            double sum = 0.0;
            for (int i = 0; i < numStates_; ++i) {
                sum += alpha(i, t-1) * transitionMatrix_(i, j);
            }
            alpha(j, t) = sum * emissionDistributions_[j].pdf(observations[t]);
        }
        
        // Scale alpha_t
        c[t] = alpha.col(t).sum();
        if (c[t] > 0) {
            alpha.col(t) /= c[t];
        }
    }
    
    // Compute total scaling factor for likelihood
    double totalScalingFactor = 1.0;
    for (double factor : c) {
        if (factor > 0) {
            totalScalingFactor *= factor;
        }
    }
    
    return {alpha, totalScalingFactor};
}

Matrix HiddenMarkovModel::backward(const std::vector<Vector>& observations, double scalingFactor) const {
    size_t T = observations.size();
    Matrix beta(numStates_, T);
    
    // Initialize
    beta.col(T-1).setOnes();
    
    // Induction
    for (int t = T-2; t >= 0; --t) {
        for (int i = 0; i < numStates_; ++i) {
            double sum = 0.0;
            for (int j = 0; j < numStates_; ++j) {
                sum += transitionMatrix_(i, j) * 
                      emissionDistributions_[j].pdf(observations[t+1]) * 
                      beta(j, t+1);
            }
            beta(i, t) = sum;
        }
        
        // Scale beta_t using same scaling factors as forward variables
        double c = beta.col(t).sum();
        if (c > 0) {
            beta.col(t) /= c;
        }
    }
    
    return beta;
}

std::vector<int> HiddenMarkovModel::decode(const std::vector<Vector>& observations) const {
    // Implement Viterbi algorithm
    size_t T = observations.size();
    if (T == 0) {
        return {};
    }
    
    // Initialize delta and psi matrices
    Matrix delta(numStates_, T);
    Matrix psi(numStates_, T);
    
    // Initialization step
    for (int i = 0; i < numStates_; ++i) {
        delta(i, 0) = std::log(initialStateProbs_(i)) + 
                     std::log(std::max(1e-10, emissionDistributions_[i].pdf(observations[0])));
        psi(i, 0) = 0;
    }
    
    // Recursion step
    for (size_t t = 1; t < T; ++t) {
        for (int j = 0; j < numStates_; ++j) {
            double maxVal = -std::numeric_limits<double>::infinity();
            int maxIdx = 0;
            
            for (int i = 0; i < numStates_; ++i) {
                double val = delta(i, t-1) + std::log(std::max(1e-10, transitionMatrix_(i, j)));
                if (val > maxVal) {
                    maxVal = val;
                    maxIdx = i;
                }
            }
            
            delta(j, t) = maxVal + std::log(std::max(1e-10, emissionDistributions_[j].pdf(observations[t])));
            psi(j, t) = maxIdx;
        }
    }
    
    // Termination step
    std::vector<int> stateSequence(T);
    delta.col(T-1).maxCoeff(&stateSequence[T-1]);
    
    // Backtracking step
    for (int t = T-2; t >= 0; --t) {
        stateSequence[t] = psi(stateSequence[t+1], t+1);
    }
    
    return stateSequence;
}

double HiddenMarkovModel::logLikelihood(const std::vector<Vector>& observations) const {
    auto [_, scalingFactor] = forward(observations);
    return -std::log(scalingFactor);
}

std::vector<Vector> HiddenMarkovModel::generateSequence(int length, std::mt19937& rng) const {
    if (length <= 0) {
        return {};
    }
    
    std::vector<Vector> sequence(length);
    
    // Generate initial state
    std::discrete_distribution<int> initialDist(initialStateProbs_.data(), 
                                             initialStateProbs_.data() + numStates_);
    int currentState = initialDist(rng);
    
    // Generate first observation
    sequence[0] = emissionDistributions_[currentState].sample(rng);
    
    // Generate remaining states and observations
    for (int t = 1; t < length; ++t) {
        // Transition to next state
        std::discrete_distribution<int> transitionDist(transitionMatrix_.row(currentState).data(),
                                                     transitionMatrix_.row(currentState).data() + numStates_);
        currentState = transitionDist(rng);
        
        // Generate observation from new state
        sequence[t] = emissionDistributions_[currentState].sample(rng);
    }
    
    return sequence;
}

void HiddenMarkovModel::setRegimeNames(const std::vector<std::string>& names) {
    if (static_cast<int>(names.size()) != numStates_) {
        throw std::invalid_argument("Number of regime names does not match number of states");
    }
    regimeNames_ = names;
}

std::string HiddenMarkovModel::getRegimeName(int state) const {
    if (state < 0 || state >= numStates_) {
        throw std::out_of_range("State index out of range");
    }
    return regimeNames_[state];
}

MarketRegime HiddenMarkovModel::mapStateToRegime(int state) const {
    if (state < 0 || state >= numStates_) {
        throw std::out_of_range("State index out of range");
    }
    return stateRegimeMapping_[state];
}

void HiddenMarkovModel::setStateRegimeMapping(const std::vector<MarketRegime>& mapping) {
    if (static_cast<int>(mapping.size()) != numStates_) {
        throw std::invalid_argument("Size of regime mapping does not match number of states");
    }
    stateRegimeMapping_ = mapping;
}

void HiddenMarkovModel::initializeRandomly(int featureDimension) {
    // Initialize transition matrix
    transitionMatrix_ = Matrix::Zero(numStates_, numStates_);
    
    // Use higher self-transition probabilities for stability
    for (int i = 0; i < numStates_; ++i) {
        for (int j = 0; j < numStates_; ++j) {
            if (i == j) {
                transitionMatrix_(i, j) = 0.7 + 0.2 * (static_cast<double>(std::rand()) / RAND_MAX);
            } else {
                transitionMatrix_(i, j) = static_cast<double>(std::rand()) / RAND_MAX;
            }
        }
        
        // Normalize row to sum to 1
        transitionMatrix_.row(i) /= transitionMatrix_.row(i).sum();
    }
    
    // Initialize emission distributions
    emissionDistributions_.resize(numStates_);
    
    // Create random but distinct emission distributions
    for (int i = 0; i < numStates_; ++i) {
        emissionDistributions_[i] = EmissionDistribution(EmissionDistribution::GAUSSIAN, featureDimension);
        
        // Create random mean vector, with larger values for higher state indices
        Vector mean = Vector::Zero(featureDimension);
        for (int d = 0; d < featureDimension; ++d) {
            mean(d) = 0.1 + 0.1 * i + 0.05 * (static_cast<double>(std::rand()) / RAND_MAX);
        }
        
        // Create random covariance matrix
        Matrix cov = Matrix::Identity(featureDimension, featureDimension);
        for (int d = 0; d < featureDimension; ++d) {
            cov(d, d) = 0.01 + 0.02 * (static_cast<double>(std::rand()) / RAND_MAX);
        }
        
        emissionDistributions_[i].setGaussianParameters(mean, cov);
    }
}

//------------------------------------------------------------------------------
// MarketFeatures Implementation
//------------------------------------------------------------------------------

Vector MarketFeatures::extract(const std::vector<double>& prices, 
                             const std::vector<double>& volumes) {
    if (prices.empty()) {
        throw std::invalid_argument("Price vector cannot be empty");
    }
    
    // Determine output dimension based on available data
    int dimension = 1;
    if (!volumes.empty() && volumes.size() == prices.size()) {
        dimension++;
    }
    
    Vector features(dimension);
    int featureIdx = 0;
    
    // Calculate returns and volatility (first feature)
    if (prices.size() >= 2) {
        std::vector<double> returns;
        for (size_t i = 1; i < prices.size(); ++i) {
            returns.push_back((prices[i] / prices[i-1]) - 1.0);
        }
        
        // Calculate standard deviation of returns as volatility
        double sum = 0.0, sumSq = 0.0;
        for (double ret : returns) {
            sum += ret;
            sumSq += ret * ret;
        }
        double mean = sum / returns.size();
        double variance = (sumSq / returns.size()) - (mean * mean);
        double volatility = std::sqrt(std::max(0.0, variance));
        
        // Convert to annualized (assuming daily data with 252 trading days)
        features(featureIdx++) = volatility * std::sqrt(252.0);
    } else {
        features(featureIdx++) = 0.0; // Default if not enough data
    }
    
    // Add volume feature if available
    if (!volumes.empty() && volumes.size() == prices.size()) {
        // Calculate average volume
        double sumVol = 0.0;
        for (double vol : volumes) {
            sumVol += vol;
        }
        double avgVol = sumVol / volumes.size();
        
        // Use ratio of current volume to average as feature
        features(featureIdx++) = volumes.back() / avgVol;
    }
    
    return features;
}

Vector MarketFeatures::normalize(const Vector& features, 
                               const Vector& mean, 
                               const Vector& stdDev) {
    if (features.size() != mean.size() || features.size() != stdDev.size()) {
        throw std::invalid_argument("Dimension mismatch in normalization vectors");
    }
    
    Vector normalized = features;
    
    for (int i = 0; i < features.size(); ++i) {
        if (stdDev(i) > 0) {
            normalized(i) = (features(i) - mean(i)) / stdDev(i);
        } else {
            normalized(i) = 0.0; // Default if standard deviation is zero
        }
    }
    
    return normalized;
}

std::pair<Vector, Vector> MarketFeatures::calculateStats(const std::vector<Vector>& featureVectors) {
    if (featureVectors.empty()) {
        throw std::invalid_argument("Feature vector list cannot be empty");
    }
    
    int dimension = featureVectors[0].size();
    Vector mean = Vector::Zero(dimension);
    Vector stdDev = Vector::Zero(dimension);
    
    // Calculate mean
    for (const auto& features : featureVectors) {
        if (features.size() != dimension) {
            throw std::invalid_argument("All feature vectors must have the same dimension");
        }
        mean += features;
    }
    mean /= featureVectors.size();
    
    // Calculate standard deviation
    for (const auto& features : featureVectors) {
        stdDev += (features - mean).cwiseProduct(features - mean);
    }
    stdDev = (stdDev / featureVectors.size()).cwiseSqrt();
    
    // Handle zero standard deviations
    for (int i = 0; i < dimension; ++i) {
        if (stdDev(i) < 1e-10) {
            stdDev(i) = 1.0; // Default to 1.0 to avoid division by zero
        }
    }
    
    return {mean, stdDev};
}

Vector MarketFeatures::extractVolatilityFeatures(const std::vector<double>& returns, int window) {
    if (returns.empty()) {
        throw std::invalid_argument("Returns vector cannot be empty");
    }
    
    // Determine effective window size
    int effectiveWindow = std::min(static_cast<int>(returns.size()), window);
    
    // Feature vector with realized volatility, skewness, and kurtosis
    Vector features(3);
    
    // Calculate realized volatility (standard deviation of returns)
    double sum = 0.0, sumSq = 0.0;
    for (size_t i = returns.size() - static_cast<size_t>(effectiveWindow); i < returns.size(); ++i) {
        sum += returns[i];
        sumSq += returns[i] * returns[i];
    }
    double mean = sum / effectiveWindow;
    double variance = (sumSq / effectiveWindow) - (mean * mean);
    double volatility = std::sqrt(std::max(0.0, variance));
    
    // Annualize volatility (assuming daily returns)
    features(0) = volatility * std::sqrt(252.0);
    
    // Calculate skewness
    double sumCubed = 0.0;
    for (size_t i = returns.size() - static_cast<size_t>(effectiveWindow); i < returns.size(); ++i) {
        double centered = returns[i] - mean;
        sumCubed += centered * centered * centered;
    }
    double skewness = 0.0;
    if (volatility > 0) {
        skewness = (sumCubed / effectiveWindow) / std::pow(variance, 1.5);
    }
    features(1) = skewness;
    
    // Calculate kurtosis
    double sumQuad = 0.0;
    for (size_t i = returns.size() - static_cast<size_t>(effectiveWindow); i < returns.size(); ++i) {
        double centered = returns[i] - mean;
        sumQuad += centered * centered * centered * centered;
    }
    double kurtosis = 0.0;
    if (variance > 0) {
        kurtosis = (sumQuad / effectiveWindow) / (variance * variance) - 3.0; // Excess kurtosis
    }
    features(2) = kurtosis;
    
    return features;
}

std::vector<Vector> MarketFeatures::createVolatilityRegimeFeatures() {
    // Create example feature vectors for different volatility regimes
    std::vector<Vector> regimeFeatures;
    
    // Feature vector: [annualized_volatility, skewness, kurtosis]
    
    // Low volatility regime
    Vector lowVol(3);
    lowVol << 0.12, 0.1, 1.0;  // 12% vol, slight positive skew, moderate kurtosis
    regimeFeatures.push_back(lowVol);
    
    // Medium volatility regime
    Vector medVol(3);
    medVol << 0.25, -0.2, 2.0;  // 25% vol, slight negative skew, higher kurtosis
    regimeFeatures.push_back(medVol);
    
    // High volatility regime
    Vector highVol(3);
    highVol << 0.40, -0.5, 5.0;  // 40% vol, significant negative skew, high kurtosis
    regimeFeatures.push_back(highVol);
    
    return regimeFeatures;
}

} // namespace vol_arb
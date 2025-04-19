#ifndef ENGINE_TRADING_RISKMAN_H
#define ENGINE_TRADING_RISKMAN_H

#include "portman.h"
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <functional>
#include <memory>
#include <mutex>
#include <atomic>

namespace engine {
namespace trading {

/**
 * @brief Risk rule types
 */
enum class RiskRuleType {
    POSITION_LIMIT,         // Maximum position size
    POSITION_VALUE_LIMIT,   // Maximum position value
    PORTFOLIO_VALUE_LIMIT,  // Maximum portfolio value
    EXPOSURE_LIMIT,         // Maximum exposure to an asset class
    LOSS_LIMIT,             // Maximum loss (stop loss)
    VAR_LIMIT,              // Value at Risk limit
    DELTA_LIMIT,            // Delta exposure limit
    GAMMA_LIMIT,            // Gamma exposure limit
    VEGA_LIMIT,             // Vega exposure limit
    THETA_LIMIT,            // Theta exposure limit
    CONCENTRATION_LIMIT,    // Concentration limit
    DRAWDOWN_LIMIT,         // Maximum drawdown limit
    CUSTOM                  // Custom risk rule
};

/**
 * @brief Risk limit scope
 */
enum class RiskScope {
    INSTRUMENT,       // Applies to a specific instrument
    INSTRUMENT_TYPE,  // Applies to an instrument type
    STRATEGY,         // Applies to a specific strategy
    PORTFOLIO         // Applies to the entire portfolio
};

/**
 * @brief Risk rule action when triggered
 */
enum class RiskAction {
    LOG,           // Log the violation
    ALERT,         // Send an alert
    WARN,          // Issue a warning
    BLOCK_TRADE,   // Block the trade
    REDUCE,        // Reduce position
    LIQUIDATE,     // Liquidate position
    STOP_STRATEGY, // Stop the strategy
    CUSTOM         // Custom action
};

/**
 * @brief Risk rule definition
 */
struct RiskRule {
    std::string id;               // Unique rule ID
    std::string name;             // Rule name
    std::string description;      // Rule description
    RiskRuleType type;            // Rule type
    RiskScope scope;              // Rule scope
    RiskAction action;            // Action when triggered
    double threshold;             // Threshold value
    double warning_threshold;     // Warning threshold
    bool enabled;                 // Whether the rule is enabled
    std::string scope_value;      // Value for the scope (e.g., instrument ID, strategy ID)
    std::function<bool(const void*)> custom_check;  // Custom check function
    std::function<void(const void*)> custom_action; // Custom action function
    
    RiskRule() 
        : type(RiskRuleType::POSITION_LIMIT), 
          scope(RiskScope::PORTFOLIO), 
          action(RiskAction::LOG),
          threshold(0.0),
          warning_threshold(0.0),
          enabled(true) {}
};

/**
 * @brief Risk violation information
 */
struct RiskViolation {
    std::string rule_id;           // ID of the violated rule
    int64_t timestamp;             // Timestamp of the violation
    std::string message;           // Violation message
    double actual_value;           // Actual value that triggered the violation
    double threshold;              // Rule threshold
    RiskAction action_taken;       // Action taken
    bool resolved;                 // Whether the violation has been resolved
    int64_t resolution_timestamp;  // Timestamp of resolution
    std::string resolution_action; // Action taken to resolve the violation
    
    RiskViolation() 
        : timestamp(0), actual_value(0.0), threshold(0.0),
          action_taken(RiskAction::LOG), resolved(false),
          resolution_timestamp(0) {}
};

/**
 * @brief Callback for risk violations
 */
using RiskViolationCallback = std::function<void(const RiskViolation&)>;

/**
 * @brief Manager for risk rules and violations
 * 
 * This class manages risk rules, checks for violations, and executes
 * appropriate actions when violations occur.
 */
class RiskManager {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the manager
     */
    static RiskManager& getInstance();
    
    /**
     * @brief Initialize the risk manager
     * 
     * @param portfolio_manager Pointer to the portfolio manager
     * @return true if initialization succeeded
     */
    bool initialize(PortfolioManager* portfolio_manager = nullptr);
    
    /**
     * @brief Add a risk rule
     * 
     * @param rule Risk rule to add
     * @return true if the rule was added successfully
     */
    bool addRule(const RiskRule& rule);
    
    /**
     * @brief Remove a risk rule
     * 
     * @param rule_id ID of the rule to remove
     * @return true if the rule was removed successfully
     */
    bool removeRule(const std::string& rule_id);
    
    /**
     * @brief Update a risk rule
     * 
     * @param rule Updated risk rule
     * @return true if the rule was updated successfully
     */
    bool updateRule(const RiskRule& rule);
    
    /**
     * @brief Enable a risk rule
     * 
     * @param rule_id ID of the rule to enable
     * @return true if the rule was enabled successfully
     */
    bool enableRule(const std::string& rule_id);
    
    /**
     * @brief Disable a risk rule
     * 
     * @param rule_id ID of the rule to disable
     * @return true if the rule was disabled successfully
     */
    bool disableRule(const std::string& rule_id);
    
    /**
     * @brief Get a risk rule
     * 
     * @param rule_id ID of the rule to get
     * @return Risk rule (empty if not found)
     */
    RiskRule getRule(const std::string& rule_id) const;
    
    /**
     * @brief Get all risk rules
     * 
     * @return Map of rule IDs to rules
     */
    std::unordered_map<std::string, RiskRule> getAllRules() const;
    
    /**
     * @brief Check for violations of all rules
     * 
     * @return Vector of violations
     */
    std::vector<RiskViolation> checkAllRules();
    
    /**
     * @brief Check for violations of a specific rule
     * 
     * @param rule_id ID of the rule to check
     * @return Vector of violations
     */
    std::vector<RiskViolation> checkRule(const std::string& rule_id);
    
    /**
     * @brief Check if a trade would violate any rules
     * 
     * @param instrument_id Instrument ID
     * @param quantity Quantity to trade
     * @param price Trade price
     * @param strategy_id Strategy ID
     * @return Vector of potential violations
     */
    std::vector<RiskViolation> checkTrade(uint64_t instrument_id, double quantity, 
                                        double price, const std::string& strategy_id);
    
    /**
     * @brief Register a callback for risk violations
     * 
     * @param callback Callback function
     * @return Callback ID
     */
    uint64_t registerViolationCallback(RiskViolationCallback callback);
    
    /**
     * @brief Unregister a violation callback
     * 
     * @param callback_id Callback ID
     * @return true if unregistered successfully
     */
    bool unregisterViolationCallback(uint64_t callback_id);
    
    /**
     * @brief Get all unresolved violations
     * 
     * @return Vector of unresolved violations
     */
    std::vector<RiskViolation> getUnresolvedViolations() const;
    
    /**
     * @brief Get all violations within a time range
     * 
     * @param start_time Start timestamp
     * @param end_time End timestamp
     * @return Vector of violations
     */
    std::vector<RiskViolation> getViolations(int64_t start_time, int64_t end_time) const;
    
    /**
     * @brief Resolve a violation
     * 
     * @param violation_id Violation ID
     * @param resolution_action Action taken to resolve the violation
     * @return true if resolved successfully
     */
    bool resolveViolation(uint64_t violation_id, const std::string& resolution_action);
    
    /**
     * @brief Load rules from a configuration file
     * 
     * @param filename Configuration file path
     * @return true if loaded successfully
     */
    bool loadRulesFromFile(const std::string& filename);
    
    /**
     * @brief Save rules to a configuration file
     * 
     * @param filename Configuration file path
     * @return true if saved successfully
     */
    bool saveRulesToFile(const std::string& filename) const;
    
private:
    // Private constructor for singleton
    RiskManager() = default;
    
    // Private destructor
    ~RiskManager() = default;
    
    // Disallow copying
    RiskManager(const RiskManager&) = delete;
    RiskManager& operator=(const RiskManager&) = delete;
    
    // Risk rules and violations
    std::unordered_map<std::string, RiskRule> rules_;
    std::vector<RiskViolation> violations_;
    
    // Violation callbacks
    std::unordered_map<uint64_t, RiskViolationCallback> violation_callbacks_;
    std::atomic<uint64_t> next_callback_id_{1};
    
    // Portfolio manager reference
    PortfolioManager* portfolio_manager_;
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
    
    // Helper methods
    std::vector<RiskViolation> checkPositionLimits();
    std::vector<RiskViolation> checkValueLimits();
    std::vector<RiskViolation> checkExposureLimits();
    std::vector<RiskViolation> checkLossLimits();
    std::vector<RiskViolation> checkVaRLimits();
    std::vector<RiskViolation> checkGreeksLimits();
    std::vector<RiskViolation> checkConcentrationLimits();
    std::vector<RiskViolation> checkDrawdownLimits();
    std::vector<RiskViolation> checkCustomRules();
    
    void notifyViolation(const RiskViolation& violation);
    void executeRiskAction(const RiskRule& rule, const RiskViolation& violation);
    int64_t getCurrentTimestamp() const;
    bool isRuleApplicable(const RiskRule& rule, uint64_t instrument_id, const std::string& strategy_id) const;
    RiskViolation createViolation(const RiskRule& rule, double actual_value, const std::string& message);
};

/**
 * @brief Factory for creating common risk rules
 */
class RiskRuleFactory {
public:
    /**
     * @brief Create a position limit rule
     * 
     * @param name Rule name
     * @param instrument_id Instrument ID (0 for all)
     * @param max_quantity Maximum position quantity
     * @param action Action when triggered
     * @return Risk rule
     */
    static RiskRule createPositionLimitRule(
        const std::string& name, uint64_t instrument_id, 
        double max_quantity, RiskAction action = RiskAction::BLOCK_TRADE);
    
    /**
     * @brief Create a position value limit rule
     * 
     * @param name Rule name
     * @param instrument_id Instrument ID (0 for all)
     * @param max_value Maximum position value
     * @param action Action when triggered
     * @return Risk rule
     */
    static RiskRule createPositionValueLimitRule(
        const std::string& name, uint64_t instrument_id, 
        double max_value, RiskAction action = RiskAction::BLOCK_TRADE);
    
    /**
     * @brief Create a portfolio value limit rule
     * 
     * @param name Rule name
     * @param max_value Maximum portfolio value
     * @param action Action when triggered
     * @return Risk rule
     */
    static RiskRule createPortfolioValueLimitRule(
        const std::string& name, double max_value, 
        RiskAction action = RiskAction::ALERT);
    
    /**
     * @brief Create an exposure limit rule
     * 
     * @param name Rule name
     * @param instrument_type Instrument type
     * @param max_exposure Maximum exposure percentage (0-1)
     * @param action Action when triggered
     * @return Risk rule
     */
    static RiskRule createExposureLimitRule(
        const std::string& name, InstrumentType instrument_type, 
        double max_exposure, RiskAction action = RiskAction::WARN);
    
    /**
     * @brief Create a loss limit rule (stop loss)
     * 
     * @param name Rule name
     * @param instrument_id Instrument ID (0 for all)
     * @param max_loss Maximum loss amount
     * @param action Action when triggered
     * @return Risk rule
     */
    static RiskRule createLossLimitRule(
        const std::string& name, uint64_t instrument_id, 
        double max_loss, RiskAction action = RiskAction::LIQUIDATE);
    
    /**
     * @brief Create a VaR limit rule
     * 
     * @param name Rule name
     * @param max_var Maximum Value at Risk
     * @param confidence_level Confidence level (e.g., 0.95, 0.99)
     * @param action Action when triggered
     * @return Risk rule
     */
    static RiskRule createVaRLimitRule(
        const std::string& name, double max_var, 
        double confidence_level, RiskAction action = RiskAction::ALERT);
    
    /**
     * @brief Create a delta limit rule
     * 
     * @param name Rule name
     * @param instrument_id Instrument ID (0 for all)
     * @param max_delta Maximum delta exposure
     * @param action Action when triggered
     * @return Risk rule
     */
    static RiskRule createDeltaLimitRule(
        const std::string& name, uint64_t instrument_id, 
        double max_delta, RiskAction action = RiskAction::WARN);
    
    /**
     * @brief Create a custom risk rule
     * 
     * @param name Rule name
     * @param description Rule description
     * @param scope Rule scope
     * @param scope_value Value for the scope
     * @param custom_check Custom check function
     * @param custom_action Custom action function
     * @return Risk rule
     */
    static RiskRule createCustomRule(
        const std::string& name, const std::string& description,
        RiskScope scope, const std::string& scope_value,
        std::function<bool(const void*)> custom_check,
        std::function<void(const void*)> custom_action = nullptr);
};

/**
 * @brief RAII class for temporarily disabling risk rules
 */
class ScopedRiskRuleDisable {
public:
    /**
     * @brief Constructor - disables specific risk rules
     * 
     * @param rule_ids Vector of rule IDs to disable
     */
    explicit ScopedRiskRuleDisable(const std::vector<std::string>& rule_ids);
    
    /**
     * @brief Constructor - disables all rules of a specific type
     * 
     * @param rule_type Rule type to disable
     */
    explicit ScopedRiskRuleDisable(RiskRuleType rule_type);
    
    /**
     * @brief Destructor - re-enables the rules
     */
    ~ScopedRiskRuleDisable();
    
private:
    std::vector<std::string> disabled_rule_ids_;
};

} // namespace trading
} // namespace engine

#endif // ENGINE_TRADING_RISKMAN_H
#ifndef ENGINE_TRADING_PORTMAN_H
#define ENGINE_TRADING_PORTMAN_H

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <functional>
#include <atomic>
#include <chrono>

namespace engine {
namespace trading {

/**
 * @brief Type of trade position
 */
enum class PositionType {
    LONG,    // Long position
    SHORT,   // Short position
    NEUTRAL  // Neither long nor short (e.g., butterfly spread)
};

/**
 * @brief Type of financial instrument
 */
enum class InstrumentType {
    STOCK,             // Equity stock
    OPTION,            // Option contract
    OPTION_STRATEGY,   // Option strategy (combination of options)
    FUTURE,            // Futures contract
    ETF,               // Exchange-traded fund
    BOND,              // Bond
    FOREX,             // Foreign exchange
    CRYPTO,            // Cryptocurrency
    OTHER              // Other instrument type
};

/**
 * @brief Status of a trade
 */
enum class TradeStatus {
    PENDING,     // Pending submission
    SUBMITTED,   // Submitted to exchange
    PARTIAL,     // Partially filled
    FILLED,      // Completely filled
    CANCELED,    // Canceled
    REJECTED,    // Rejected by exchange
    EXPIRED      // Expired (e.g., GTC order exceeded time limit)
};

/**
 * @brief Market data for an instrument
 */
struct MarketData {
    double bid;              // Best bid price
    double ask;              // Best ask price
    double last;             // Last trade price
    double volume;           // Trading volume
    double open_interest;    // Open interest (for derivatives)
    double implied_vol;      // Implied volatility (for options)
    double theo_price;       // Theoretical price
    double delta;            // Delta (for options)
    double gamma;            // Gamma (for options)
    double vega;             // Vega (for options)
    double theta;            // Theta (for options)
    double rho;              // Rho (for options)
    int64_t timestamp;       // Timestamp in nanoseconds
    
    MarketData() 
        : bid(0.0), ask(0.0), last(0.0), volume(0.0), open_interest(0.0),
          implied_vol(0.0), theo_price(0.0), delta(0.0), gamma(0.0),
          vega(0.0), theta(0.0), rho(0.0), timestamp(0) {}
};

/**
 * @brief Trading order information
 */
struct OrderInfo {
    uint64_t order_id;           // Unique order ID
    uint64_t instrument_id;      // Instrument ID
    std::string symbol;          // Instrument symbol
    double price;                // Order price
    double quantity;             // Order quantity
    double filled_quantity;      // Filled quantity
    double avg_fill_price;       // Average fill price
    TradeStatus status;          // Order status
    bool is_buy;                 // True for buy, false for sell
    std::string order_type;      // Market, limit, etc.
    std::string time_in_force;   // GTC, IOC, FOK, etc.
    int64_t submit_time;         // Submission timestamp
    int64_t last_update_time;    // Last update timestamp
    std::string strategy_id;     // ID of the strategy that created the order
    
    OrderInfo() 
        : order_id(0), instrument_id(0), price(0.0), quantity(0.0),
          filled_quantity(0.0), avg_fill_price(0.0), status(TradeStatus::PENDING),
          is_buy(true), submit_time(0), last_update_time(0) {}
};

/**
 * @brief Position in an instrument
 */
struct Position {
    uint64_t instrument_id;      // Instrument ID
    std::string symbol;          // Instrument symbol
    InstrumentType type;         // Instrument type
    double quantity;             // Position quantity
    double avg_price;            // Average price
    double market_price;         // Current market price
    double unrealized_pnl;       // Unrealized profit/loss
    double realized_pnl;         // Realized profit/loss
    double commission;           // Total commission paid
    int64_t open_time;           // Position open timestamp
    int64_t last_update_time;    // Last update timestamp
    std::string strategy_id;     // ID of the strategy that created the position
    
    // Option-specific fields
    double delta;                // Position delta
    double gamma;                // Position gamma
    double vega;                 // Position vega
    double theta;                // Position theta
    double rho;                  // Position rho
    double implied_vol;          // Implied volatility
    
    Position() 
        : instrument_id(0), type(InstrumentType::OTHER), quantity(0.0),
          avg_price(0.0), market_price(0.0), unrealized_pnl(0.0),
          realized_pnl(0.0), commission(0.0), open_time(0), last_update_time(0),
          delta(0.0), gamma(0.0), vega(0.0), theta(0.0), rho(0.0),
          implied_vol(0.0) {}
    
    /**
     * @brief Get the position type (long/short/neutral)
     * 
     * @return Position type
     */
    PositionType getPositionType() const {
        if (quantity > 0.0) {
            return PositionType::LONG;
        } else if (quantity < 0.0) {
            return PositionType::SHORT;
        } else {
            return PositionType::NEUTRAL;
        }
    }
    
    /**
     * @brief Get the position value
     * 
     * @return Position value (quantity * market_price)
     */
    double getValue() const {
        return std::abs(quantity) * market_price;
    }
    
    /**
     * @brief Get the position cost basis
     * 
     * @return Cost basis (quantity * avg_price)
     */
    double getCostBasis() const {
        return std::abs(quantity) * avg_price;
    }
};

/**
 * @brief Portfolio statistics
 */
struct PortfolioStats {
    double total_value;               // Total portfolio value
    double cash_balance;              // Cash balance
    double unrealized_pnl;            // Total unrealized P&L
    double realized_pnl;              // Total realized P&L
    double daily_pnl;                 // Daily P&L
    double net_delta;                 // Net portfolio delta
    double net_gamma;                 // Net portfolio gamma
    double net_vega;                  // Net portfolio vega
    double net_theta;                 // Net portfolio theta
    double net_rho;                   // Net portfolio rho
    double margin_used;               // Margin used
    double margin_available;          // Margin available
    double highest_value;             // Highest portfolio value
    double lowest_value;              // Lowest portfolio value
    int64_t last_update_time;         // Last update timestamp
    
    PortfolioStats() 
        : total_value(0.0), cash_balance(0.0), unrealized_pnl(0.0),
          realized_pnl(0.0), daily_pnl(0.0), net_delta(0.0), net_gamma(0.0),
          net_vega(0.0), net_theta(0.0), net_rho(0.0), margin_used(0.0),
          margin_available(0.0), highest_value(0.0), lowest_value(0.0),
          last_update_time(0) {}
};

/**
 * @brief Risk metrics for the portfolio
 */
struct RiskMetrics {
    double var_95;               // 95% Value at Risk
    double var_99;               // 99% Value at Risk
    double expected_shortfall;   // Expected shortfall
    double sharpe_ratio;         // Sharpe ratio
    double sortino_ratio;        // Sortino ratio
    double max_drawdown;         // Maximum drawdown
    double beta;                 // Portfolio beta
    double volatility;           // Portfolio volatility
    double correlation;          // Correlation with benchmark
    int64_t last_update_time;    // Last update timestamp
    
    RiskMetrics() 
        : var_95(0.0), var_99(0.0), expected_shortfall(0.0),
          sharpe_ratio(0.0), sortino_ratio(0.0), max_drawdown(0.0),
          beta(0.0), volatility(0.0), correlation(0.0),
          last_update_time(0) {}
};

/**
 * @brief Callback function types for portfolio events
 */
using PositionUpdateCallback = std::function<void(const Position&)>;
using OrderUpdateCallback = std::function<void(const OrderInfo&)>;
using PortfolioStatsCallback = std::function<void(const PortfolioStats&)>;
using RiskMetricsCallback = std::function<void(const RiskMetrics&)>;

/**
 * @brief Manager for portfolio positions and orders
 * 
 * This class manages portfolio positions, orders, and risk metrics,
 * providing a centralized view of the trading portfolio.
 */
class PortfolioManager {
public:
    /**
     * @brief Get the singleton instance
     * 
     * @return Reference to the manager
     */
    static PortfolioManager& getInstance();
    
    /**
     * @brief Initialize the portfolio manager
     * 
     * @param initial_cash Initial cash balance
     * @param account_id Account ID
     * @return true if initialization succeeded
     */
    bool initialize(double initial_cash, const std::string& account_id);
    
    /**
     * @brief Update a position
     * 
     * @param position Position to update
     * @return true if update succeeded
     */
    bool updatePosition(const Position& position);
    
    /**
     * @brief Update an order
     * 
     * @param order Order to update
     * @return true if update succeeded
     */
    bool updateOrder(const OrderInfo& order);
    
    /**
     * @brief Update market data for an instrument
     * 
     * @param instrument_id Instrument ID
     * @param market_data Market data
     * @return true if update succeeded
     */
    bool updateMarketData(uint64_t instrument_id, const MarketData& market_data);
    
    /**
     * @brief Get a position
     * 
     * @param instrument_id Instrument ID
     * @return Position (empty if not found)
     */
    Position getPosition(uint64_t instrument_id) const;
    
    /**
     * @brief Get an order
     * 
     * @param order_id Order ID
     * @return Order information (empty if not found)
     */
    OrderInfo getOrder(uint64_t order_id) const;
    
    /**
     * @brief Get market data for an instrument
     * 
     * @param instrument_id Instrument ID
     * @return Market data (empty if not found)
     */
    MarketData getMarketData(uint64_t instrument_id) const;
    
    /**
     * @brief Get all positions
     * 
     * @return Map of instrument IDs to positions
     */
    std::unordered_map<uint64_t, Position> getAllPositions() const;
    
    /**
     * @brief Get all orders
     * 
     * @return Map of order IDs to orders
     */
    std::unordered_map<uint64_t, OrderInfo> getAllOrders() const;
    
    /**
     * @brief Get active orders (not filled, canceled, or rejected)
     * 
     * @return Map of order IDs to orders
     */
    std::unordered_map<uint64_t, OrderInfo> getActiveOrders() const;
    
    /**
     * @brief Get portfolio statistics
     * 
     * @return Portfolio statistics
     */
    PortfolioStats getPortfolioStats() const;
    
    /**
     * @brief Get risk metrics
     * 
     * @return Risk metrics
     */
    RiskMetrics getRiskMetrics() const;
    
    /**
     * @brief Register a callback for position updates
     * 
     * @param callback Callback function
     * @return Callback ID
     */
    uint64_t registerPositionCallback(PositionUpdateCallback callback);
    
    /**
     * @brief Register a callback for order updates
     * 
     * @param callback Callback function
     * @return Callback ID
     */
    uint64_t registerOrderCallback(OrderUpdateCallback callback);
    
    /**
     * @brief Register a callback for portfolio stats updates
     * 
     * @param callback Callback function
     * @return Callback ID
     */
    uint64_t registerStatsCallback(PortfolioStatsCallback callback);
    
    /**
     * @brief Register a callback for risk metrics updates
     * 
     * @param callback Callback function
     * @return Callback ID
     */
    uint64_t registerRiskCallback(RiskMetricsCallback callback);
    
    /**
     * @brief Unregister a callback
     * 
     * @param callback_id Callback ID
     * @return true if unregistered successfully
     */
    bool unregisterCallback(uint64_t callback_id);
    
    /**
     * @brief Update portfolio statistics
     * 
     * Recalculates portfolio statistics based on current positions and market data
     * 
     * @return Updated portfolio statistics
     */
    PortfolioStats updatePortfolioStats();
    
    /**
     * @brief Update risk metrics
     * 
     * Recalculates risk metrics based on current portfolio and market conditions
     * 
     * @return Updated risk metrics
     */
    RiskMetrics updateRiskMetrics();
    
    /**
     * @brief Get positions for a specific strategy
     * 
     * @param strategy_id Strategy ID
     * @return Map of instrument IDs to positions
     */
    std::unordered_map<uint64_t, Position> getPositionsByStrategy(const std::string& strategy_id) const;
    
    /**
     * @brief Get orders for a specific strategy
     * 
     * @param strategy_id Strategy ID
     * @return Map of order IDs to orders
     */
    std::unordered_map<uint64_t, OrderInfo> getOrdersByStrategy(const std::string& strategy_id) const;
    
    /**
     * @brief Get the account ID
     * 
     * @return Account ID
     */
    const std::string& getAccountId() const { return account_id_; }
    
    /**
     * @brief Get the cash balance
     * 
     * @return Cash balance
     */
    double getCashBalance() const { return portfolio_stats_.cash_balance; }
    
    /**
     * @brief Set the cash balance
     * 
     * @param cash_balance New cash balance
     */
    void setCashBalance(double cash_balance);
    
    /**
     * @brief Get the last update time
     * 
     * @return Last update timestamp
     */
    int64_t getLastUpdateTime() const { return portfolio_stats_.last_update_time; }
    
private:
    // Private constructor for singleton
    PortfolioManager() = default;
    
    // Private destructor
    ~PortfolioManager() = default;
    
    // Disallow copying
    PortfolioManager(const PortfolioManager&) = delete;
    PortfolioManager& operator=(const PortfolioManager&) = delete;
    
    // Portfolio data
    std::unordered_map<uint64_t, Position> positions_;
    std::unordered_map<uint64_t, OrderInfo> orders_;
    std::unordered_map<uint64_t, MarketData> market_data_;
    PortfolioStats portfolio_stats_;
    RiskMetrics risk_metrics_;
    std::string account_id_;
    
    // Callbacks
    struct CallbackEntry {
        uint64_t id;
        std::function<void()> reset;
    };
    
    std::unordered_map<uint64_t, PositionUpdateCallback> position_callbacks_;
    std::unordered_map<uint64_t, OrderUpdateCallback> order_callbacks_;
    std::unordered_map<uint64_t, PortfolioStatsCallback> stats_callbacks_;
    std::unordered_map<uint64_t, RiskMetricsCallback> risk_callbacks_;
    std::atomic<uint64_t> next_callback_id_{1};
    
    // Mutex for thread safety
    mutable std::mutex mutex_;
    
    // Helper methods
    void recalculatePortfolioStats();
    void recalculateRiskMetrics();
    void notifyPositionCallbacks(const Position& position);
    void notifyOrderCallbacks(const OrderInfo& order);
    void notifyStatsCallbacks(const PortfolioStats& stats);
    void notifyRiskCallbacks(const RiskMetrics& metrics);
    int64_t getCurrentTimestamp() const;
};

/**
 * @brief RAII class for portfolio snapshots
 * 
 * This class captures and restores portfolio state for simulation
 * or what-if analysis.
 */
class PortfolioSnapshot {
public:
    /**
     * @brief Constructor - captures current portfolio state
     */
    PortfolioSnapshot();
    
    /**
     * @brief Destructor - optionally restores original state
     */
    ~PortfolioSnapshot();
    
    /**
     * @brief Apply a position change
     * 
     * @param instrument_id Instrument ID
     * @param quantity Change in quantity
     * @param price Execution price
     * @return true if change was applied
     */
    bool applyPositionChange(uint64_t instrument_id, double quantity, double price);
    
    /**
     * @brief Calculate portfolio statistics after changes
     * 
     * @return Updated portfolio statistics
     */
    PortfolioStats calculateStats() const;
    
    /**
     * @brief Calculate risk metrics after changes
     * 
     * @return Updated risk metrics
     */
    RiskMetrics calculateRisk() const;
    
    /**
     * @brief Commit changes to the actual portfolio
     * 
     * @return true if changes were committed
     */
    bool commit();
    
    /**
     * @brief Revert to the original portfolio state
     * 
     * @return true if reversion was successful
     */
    bool revert();
    
private:
    std::unordered_map<uint64_t, Position> original_positions_;
    std::unordered_map<uint64_t, Position> modified_positions_;
    PortfolioStats original_stats_;
    bool committed_;
};

} // namespace trading
} // namespace engine

#endif // ENGINE_TRADING_PORTMAN_H
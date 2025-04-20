#ifndef ENGINE_ALO_DIST_FRAMEWORK_H
#define ENGINE_ALO_DIST_FRAMEWORK_H

#include <vector>
#include <future>
#include <queue>
#include <atomic>
#include <condition_variable>
#include <mpi.h>
#include <memory>

namespace engine {
namespace alo {

// Forward declaration of ALOEngine
class ALOEngine;

// Define ALOScheme here to avoid circular dependency
enum ALOScheme {
    FAST,           ///< Legendre-Legendre (7,2,7)-27 - Fastest but less accurate
    ACCURATE,       ///< Legendre-TanhSinh (25,5,13)-1e-8 - Good balance of speed and accuracy
    HIGH_PRECISION  ///< TanhSinh-TanhSinh (10,30)-1e-10 - Highest accuracy but slower
};

namespace dist {

/**
 * @brief Tags for MPI messages
 */
enum MessageTags {
    TAG_PARAMETERS = 0,     // Parameters packet
    TAG_CHUNK_COUNT = 1,    // Number of chunks
    TAG_CHUNK_SIZE = 2,     // Size of a chunk
    TAG_CHUNK_DATA = 3,     // Chunk data
    TAG_RESULTS = 4,        // Results data
    TAG_READY = 5,          // Worker ready signal
    TAG_WORK_REQUEST = 6,   // Work stealing request
    TAG_WORK_RESPONSE = 7,  // Work stealing response
    TAG_TERMINATE = 8       // Termination signal
};

/**
 * @brief Distributed work item for option pricing
 */
struct WorkItem {
    double S;                       // Spot price
    std::vector<double> strikes;    // Strike prices
    double r;                       // Risk-free rate
    double q;                       // Dividend yield
    double vol;                     // Volatility
    double T;                       // Time to maturity
    size_t startIdx;                // Starting index in global result array
};

/**
 * @class TaskDispatcher
 * @brief Distributes option pricing tasks across multiple compute nodes
 * 
 * This class implements a distributed work-stealing framework for parallel
 * option pricing across compute nodes. It uses MPI for communication and
 * manages work distribution, load balancing, and result aggregation.
 */
class TaskDispatcher {
public:
    /**
     * @brief Constructor
     * 
     * @param engineScheme ALO scheme to use for local computations
     * @param chunkSize Size of work chunks for distribution
     */
    TaskDispatcher(ALOScheme engineScheme, size_t chunkSize = 1024);
    
    /**
     * @brief Destructor
     */
    ~TaskDispatcher();
    
    /**
     * @brief Distribute a batch of put option pricing tasks across nodes
     * 
     * @param S Spot price
     * @param strikes Vector of strike prices
     * @param r Risk-free rate
     * @param q Dividend yield
     * @param vol Volatility
     * @param T Time to maturity
     * @return Vector of put option prices
     */
    std::vector<double> distributedBatchCalculatePut(
        double S, const std::vector<double>& strikes,
        double r, double q, double vol, double T);
    
private:
    // Member variables
    std::unique_ptr<ALOEngine> localEngine_;
    size_t chunkSize_;
    int rank_;
    int worldSize_;
    std::atomic<bool> terminated_;
    std::thread workerThread_;
    std::mutex workQueueMutex_;
    std::condition_variable workQueueCV_;
    std::queue<WorkItem> workQueue_;
    
    /**
     * @brief Process work on the master node
     */
    std::vector<double> masterNodeProcessing(
        double S, const std::vector<double>& strikes,
        double r, double q, double vol, double T);
    
    /**
     * @brief Send work to a worker node
     */
    void sendWorkToNode(int node, const std::vector<size_t>& workload, 
                       double S, const std::vector<double>& strikes,
                       double r, double q, double vol, double T, size_t n);
    
    /**
     * @brief Process work locally on this node
     */
    void processLocalWork(int nodeIdx, const std::vector<size_t>& workload, 
                         double S, const std::vector<double>& strikes,
                         double r, double q, double vol, double T, 
                         size_t n, std::vector<double>& results);
    
    /**
     * @brief Receive results from a worker node
     */
    void receiveResultsFromNode(int node, const std::vector<size_t>& workload, 
                              std::vector<double>& results, size_t n);
    
    /**
     * @brief Worker node processing function
     */
    void workerNodeProcessing();
    
    /**
     * @brief Worker thread function for handling asynchronous work
     */
    void workerThreadFunc();
    
    /**
     * @brief Signal readiness for work stealing
     */
    void signalReadiness(double S, double r, double q, double vol, double T);
    
    /**
     * @brief Implement work stealing for load balancing
     */
    void implementWorkStealing(
        double S, const std::vector<double>& strikes,
        double r, double q, double vol, double T,
        std::vector<double>& results);
};

/**
 * @brief Create a distributed task dispatcher
 * 
 * @param scheme Numerical scheme to use
 * @param chunkSize Size of work chunks
 * @return Shared pointer to task dispatcher
 */
inline std::shared_ptr<TaskDispatcher> createTaskDispatcher(
    ALOScheme scheme, size_t chunkSize = 1024) {
    return std::make_shared<TaskDispatcher>(scheme, chunkSize);
}

} // namespace dist
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_DIST_FRAMEWORK_H
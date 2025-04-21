#ifndef ENGINE_ALO_HYBRID_PARALLEL_H
#define ENGINE_ALO_HYBRID_PARALLEL_H

#include "mpiwrapper.h"
#include <omp.h>
#include <vector>
#include <functional>
#include <type_traits>

namespace engine {
namespace alo {
namespace hybrid {

/**
 * @brief Hybrid MPI+OpenMP execution framework
 */
class HybridExecutor {
public:
    /**
     * @brief Initialize hybrid environment
     */
    static void initialize(int* argc, char*** argv) {
        mpi::MPIWrapper::init(argc, argv);
        
        // Ensure proper thread affinity
        #ifdef _OPENMP
        omp_set_nested(1);  // Enable nested parallelism
        omp_set_dynamic(0); // Disable dynamic thread adjustment
        
        // Set threads based on cores per node
        int cores_per_node = omp_get_num_procs() / mpi::MPIWrapper::size();
        if (cores_per_node > 0) {
            omp_set_num_threads(cores_per_node);
        }
        #endif
    }
    
    /**
     * @brief Process work items in hybrid mode (specialized for double)
     */
    static std::vector<double> processDistributed(
        const std::vector<double>& items,
        std::function<double(double)> processor,
        size_t chunkSize = 1000) {
        
        int rank = mpi::MPIWrapper::rank();
        int size = mpi::MPIWrapper::size();
        
        // Determine work distribution
        size_t totalWork = items.size();
        size_t workPerNode = (totalWork + size - 1) / size;
        size_t nodeStart = rank * workPerNode;
        size_t nodeEnd = std::min(nodeStart + workPerNode, totalWork);
        
        // Process local work with OpenMP
        std::vector<double> localResults(nodeEnd - nodeStart);
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, chunkSize)
        #endif
        for (size_t i = nodeStart; i < nodeEnd; ++i) {
            localResults[i - nodeStart] = processor(items[i]);
        }
        
        // Gather results from all nodes
        std::vector<double> globalResults;
        gatherResults(localResults, globalResults);
        
        return globalResults;
    }
    
    /**
     * @brief Process work items in hybrid mode (generic version)
     * Note: This version uses serialization which may be slower for large objects
     */
    template<typename T, typename Func, 
             typename = std::enable_if_t<!std::is_same_v<T, double>>>
    static std::vector<T> processDistributed(
        const std::vector<T>& items,
        Func processor,
        size_t chunkSize = 1000) {
        
        int rank = mpi::MPIWrapper::rank();
        int size = mpi::MPIWrapper::size();
        
        // Determine work distribution
        size_t totalWork = items.size();
        size_t workPerNode = (totalWork + size - 1) / size;
        size_t nodeStart = rank * workPerNode;
        size_t nodeEnd = std::min(nodeStart + workPerNode, totalWork);
        
        // Process local work with OpenMP
        std::vector<T> localResults(nodeEnd - nodeStart);
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(dynamic, chunkSize)
        #endif
        for (size_t i = nodeStart; i < nodeEnd; ++i) {
            localResults[i - nodeStart] = processor(items[i]);
        }
        
        // For non-double types, we need a safer way to gather results
        std::vector<T> globalResults;
        gatherResultsGeneric(localResults, globalResults);
        
        return globalResults;
    }
    
private:
    // Specialized version for double that can use MPI_DOUBLE
    static void gatherResults(const std::vector<double>& localResults,
                             std::vector<double>& globalResults) {
        int rank = mpi::MPIWrapper::rank();
        int size = mpi::MPIWrapper::size();
        
        // First, gather sizes from all nodes
        int localSize = static_cast<int>(localResults.size());
        std::vector<int> sizes(size);
        
        MPI_Gather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, 
                   0, MPI_COMM_WORLD);
        
        // Calculate displacements
        std::vector<int> displacements(size);
        int total = 0;
        if (rank == 0) {
            for (int i = 0; i < size; ++i) {
                displacements[i] = total;
                total += sizes[i];
            }
            globalResults.resize(total);
        } else {
            globalResults.clear(); // Make sure globalResults is empty on non-root ranks
        }
        
        // Gather all results at root
        MPI_Gatherv(localResults.data(), localSize, MPI_DOUBLE,
                    rank == 0 ? globalResults.data() : nullptr, 
                    rank == 0 ? sizes.data() : nullptr, 
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
    
    // Generic version that works with any type but uses serialization
    template<typename T>
    static void gatherResultsGeneric(const std::vector<T>& localResults,
                                   std::vector<T>& globalResults) {
        int rank = mpi::MPIWrapper::rank();
        int size = mpi::MPIWrapper::size();
        
        // For simplicity, we'll just send each element individually
        // This is less efficient but safer for arbitrary types
        
        if (rank == 0) {
            // Root process collects results
            globalResults = localResults; // Start with local results
            
            // Receive from other processes
            for (int sender = 1; sender < size; ++sender) {
                // First receive the size
                int count;
                MPI_Recv(&count, 1, MPI_INT, sender, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Receive each element
                for (int i = 0; i < count; ++i) {
                    T element;
                    MPI_Recv(&element, sizeof(T), MPI_BYTE, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    globalResults.push_back(element);
                }
            }
        } else {
            // Non-root processes send their results
            int count = static_cast<int>(localResults.size());
            MPI_Send(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
            
            // Send each element
            for (const auto& element : localResults) {
                MPI_Send(&element, sizeof(T), MPI_BYTE, 0, 1, MPI_COMM_WORLD);
            }
        }
    }
};

} // namespace hybrid
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_HYBRID_PARALLEL_H
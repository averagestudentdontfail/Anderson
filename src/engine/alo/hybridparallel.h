#ifndef ENGINE_ALO_HYBRID_PARALLEL_H
#define ENGINE_ALO_HYBRID_PARALLEL_H

#include "mpiwrapper.h"
#include <omp.h>
#include <vector>
#include <functional>

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
     * @brief Process work items in hybrid mode
     */
    template<typename T, typename Func>
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
        
        // Gather results from all nodes
        std::vector<T> globalResults;
        gatherResults(localResults, globalResults);
        
        return globalResults;
    }
    
private:
    template<typename T>
    static void gatherResults(const std::vector<T>& localResults,
                            std::vector<T>& globalResults) {
        int rank = mpi::MPIWrapper::rank();
        int size = mpi::MPIWrapper::size();
        
        // First, gather sizes from all nodes
        int localSize = localResults.size();
        std::vector<int> sizes(size);
        
        MPI_Gather(&localSize, 1, MPI_INT, sizes.data(), 1, MPI_INT, 
                   0, MPI_COMM_WORLD);
        
        // Calculate displacements
        std::vector<int> displacements(size);
        if (rank == 0) {
            int total = 0;
            for (int i = 0; i < size; ++i) {
                displacements[i] = total;
                total += sizes[i];
            }
            globalResults.resize(total);
        }
        
        // Gather all results at root
        MPI_Gatherv(localResults.data(), localSize, MPI_DOUBLE,
                    globalResults.data(), sizes.data(), displacements.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
};

} // namespace hybrid
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_HYBRID_PARALLEL_H
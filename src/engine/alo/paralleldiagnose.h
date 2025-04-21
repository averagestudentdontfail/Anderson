#ifndef ENGINE_ALO_PARALLEL_DIAGNOSTICS_H
#define ENGINE_ALO_PARALLEL_DIAGNOSTICS_H

#include "mpiwrapper.h"
#include "hybridparallel.h"
#include <omp.h>
#include <iostream>
#include <iomanip>

namespace engine {
namespace alo {
namespace diag {

/**
 * @brief Diagnose parallel execution environment
 */
class ParallelDiagnostics {
public:
    static void diagnose() {
        diagnoseMPI();
        diagnoseOpenMP();
        diagnoseHybrid();
    }
    
private:
    static void diagnoseMPI() {
        int rank = mpi::MPIWrapper::rank();
        int size = mpi::MPIWrapper::size();
        
        if (rank == 0) {
            std::cout << "=== MPI Diagnostics ===\n";
            std::cout << "  MPI Version: " << getMPIVersion() << "\n";
            std::cout << "  World Size: " << size << "\n";
            std::cout << "  Thread Support: " << getThreadSupport() << "\n";
            std::cout << "\n";
            std::cout.flush();
        }
        
        // Ensure all processes complete this diagnosis phase
        mpi::MPIWrapper::barrier();
    }
    
    static void diagnoseOpenMP() {
        #ifdef _OPENMP
        if (mpi::MPIWrapper::rank() == 0) {
            std::cout << "=== OpenMP Diagnostics ===\n";
            std::cout << "  OpenMP Version: " << _OPENMP << "\n";
            std::cout << "  Max Threads: " << omp_get_max_threads() << "\n";
            std::cout << "  Nested Enabled: " << omp_get_nested() << "\n";
            std::cout << "\n";
            std::cout.flush();
        }
        #else
        if (mpi::MPIWrapper::rank() == 0) {
            std::cout << "=== OpenMP Diagnostics ===\n";
            std::cout << "  OpenMP NOT ENABLED (compile with -fopenmp)\n";
            std::cout << "\n";
            std::cout.flush();
        }
        #endif
        
        // Ensure all processes complete this diagnosis phase
        mpi::MPIWrapper::barrier();
    }
    
    static void diagnoseHybrid() {
        // Test hybrid execution with doubles (the specialized implementation)
        const size_t TEST_SIZE = 100;  // Smaller size for diagnostics
        std::vector<double> testData(TEST_SIZE);
        
        // Initialize data
        for (size_t i = 0; i < TEST_SIZE; i++) {
            testData[i] = static_cast<double>(i);
        }
        
        // Simple square function
        auto result = hybrid::HybridExecutor::processDistributed(
            testData, 
            [](double x) { return x * x; },
            10  // Small chunk size for testing
        );
        
        if (mpi::MPIWrapper::rank() == 0) {
            std::cout << "=== Hybrid Execution Test ===\n";
            std::cout << "  Data Size: " << testData.size() << "\n";
            std::cout << "  Result Size: " << result.size() << "\n";
            std::cout << "  First 5 Results: ";
            for (int i = 0; i < std::min(5, static_cast<int>(result.size())); ++i) {
                std::cout << result[i] << " ";
            }
            std::cout << "\n\n";
            std::cout.flush();
        }
        
        // Ensure all processes complete this diagnosis phase
        mpi::MPIWrapper::barrier();
    }
    
    static std::string getMPIVersion() {
        int version, subversion;
        MPI_Get_version(&version, &subversion);
        return std::to_string(version) + "." + std::to_string(subversion);
    }
    
    static std::string getThreadSupport() {
        int provided;
        MPI_Query_thread(&provided);
        switch (provided) {
            case MPI_THREAD_SINGLE: return "SINGLE";
            case MPI_THREAD_FUNNELED: return "FUNNELED";
            case MPI_THREAD_SERIALIZED: return "SERIALIZED";
            case MPI_THREAD_MULTIPLE: return "MULTIPLE";
            default: return "UNKNOWN";
        }
    }
};

} // namespace diag
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_PARALLEL_DIAGNOSTICS_H
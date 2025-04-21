#ifndef ENGINE_ALO_MPI_WRAPPER_H
#define ENGINE_ALO_MPI_WRAPPER_H

#include <mpi.h>
#include <stdexcept>
#include <vector>

namespace engine {
namespace alo {
namespace mpi {

class MPIWrapper {
public:
    static void init(int* argc, char*** argv) {
        int initialized;
        MPI_Initialized(&initialized);
        if (!initialized) {
            int provided;
            MPI_Init_thread(argc, argv, MPI_THREAD_MULTIPLE, &provided);
            if (provided < MPI_THREAD_MULTIPLE) {
                throw std::runtime_error("MPI does not provide required thread support");
            }
        }
    }
    
    static void finalize() {
        int finalized;
        MPI_Finalized(&finalized);
        if (!finalized) {
            MPI_Finalize();
        }
    }
    
    static int rank() {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        return rank;
    }
    
    static int size() {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        return size;
    }
    
    // Add other necessary MPI operations using C API
    static void send(const void* data, int count, MPI_Datatype datatype, 
                    int dest, int tag) {
        MPI_Send(data, count, datatype, dest, tag, MPI_COMM_WORLD);
    }
    
    static void recv(void* data, int count, MPI_Datatype datatype, 
                    int source, int tag, MPI_Status* status = MPI_STATUS_IGNORE) {
        MPI_Recv(data, count, datatype, source, tag, MPI_COMM_WORLD, status);
    }
    
    // Non-blocking operations
    static void isend(const void* data, int count, MPI_Datatype datatype, 
                     int dest, int tag, MPI_Request* request) {
        MPI_Isend(data, count, datatype, dest, tag, MPI_COMM_WORLD, request);
    }
    
    static void irecv(void* data, int count, MPI_Datatype datatype, 
                     int source, int tag, MPI_Request* request) {
        MPI_Irecv(data, count, datatype, source, tag, MPI_COMM_WORLD, request);
    }
    
    static void wait(MPI_Request* request, MPI_Status* status = MPI_STATUS_IGNORE) {
        MPI_Wait(request, status);
    }
    
    static void barrier() {
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    static void bcast(void* data, int count, MPI_Datatype datatype, int root) {
        MPI_Bcast(data, count, datatype, root, MPI_COMM_WORLD);
    }
};

} // namespace mpi
} // namespace alo
} // namespace engine

#endif // ENGINE_ALO_MPI_WRAPPER_H
{
    files = {
        "build/.objs/alo_engine/linux/x86_64/release/src/main.cpp.o",
        "build/.objs/alo_engine/linux/x86_64/release/src/alo/alo_fp_evaluator.cpp.o",
        "build/.objs/alo_engine/linux/x86_64/release/src/alo/alo_scheme.cpp.o",
        "build/.objs/alo_engine/linux/x86_64/release/src/alo/alo_engine.cpp.o",
        "build/.objs/alo_engine/linux/x86_64/release/src/numerics/integration.cpp.o",
        "build/.objs/alo_engine/linux/x86_64/release/src/numerics/chebyshev.cpp.o"
    },
    values = {
        "/usr/bin/g++",
        {
            "-m64",
            "-L/usr/lib/x86_64-linux-gnu",
            "-L/home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/lib",
            "-lgsl",
            "-lgslcblas",
            "-llapack",
            "-lblas",
            "-fopenmp"
        }
    }
}
{
    files = {
        "src/numerics/integrate.cpp"
    },
    depfiles_format = "gcc",
    depfiles = "integrate.o: src/numerics/integrate.cpp src/numerics/integrator.h  src/numerics/integrate.h src/numerics/../common/simd/simdops.h  src/numerics/../common/simd/vectmth.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_integration.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_math.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_sys.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_inline.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_machine.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_precision.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_types.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_nan.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_pow_int.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_minmax.h  /home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include/gsl/gsl_errno.h\
",
    values = {
        "/usr/bin/gcc",
        {
            "-m64",
            "-std=c++17",
            "-I/usr/include",
            "-I/usr/include/suitesparse",
            "-I/home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/eigen-3.4.0-enqwlmq352incu7he6lkthjyvgs2st3s/include",
            "-I/home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/eigen-3.4.0-enqwlmq352incu7he6lkthjyvgs2st3s/include/eigen3",
            "-I/home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu/include",
            "-Isrc",
            "-DNDEBUG",
            "-Wall",
            "-O3",
            "-march=native",
            "-fno-strict-aliasing",
            "-fno-omit-frame-pointer",
            "-fno-math-errno"
        }
    }
}
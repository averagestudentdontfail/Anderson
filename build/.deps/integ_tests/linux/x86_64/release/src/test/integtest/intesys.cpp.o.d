{
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
    },
    files = {
        "src/test/integtest/intesys.cpp"
    },
    depfiles = "intesys.o: src/test/integtest/intesys.cpp\
",
    depfiles_format = "gcc"
}
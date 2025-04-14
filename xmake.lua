set_project("Anderson")
set_version("1.0.0")
set_languages("c++17")

-- Spack dependencies paths
local eigen_dir = "/home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/eigen-3.4.0-enqwlmq352incu7he6lkthjyvgs2st3s"
local gsl_dir = "/home/send2/spack/opt/spack/linux-ubuntu24.04-x86_64_v4/gcc-13.3.0/gsl-2.8-cwzroqmwgnfmq6lqj3kwmiuwcpuuvnuu"

-- Common include directories
local include_dirs = {
    "/usr/include",
    "/usr/include/suitesparse",
    eigen_dir .. "/include",
    eigen_dir .. "/include/eigen3",
    gsl_dir .. "/include",
    "src"
}

-- Common link directories
local link_dirs = {
    "/usr/lib/x86_64-linux-gnu",
    gsl_dir .. "/lib"
}

-- Common libraries
local libraries = {
    "gsl", "gslcblas", "lapack", "blas", "pthread", "rt"
}

-- Runtime library paths
local run_lib_paths = {
    gsl_dir .. "/lib"
}

-- Define source files
local alo_sources = {
    "src/alo/aloengine.cpp",
    "src/alo/alofpeval.cpp",
    "src/alo/aloscheme.cpp"
}

local numerics_sources = {
    "src/numerics/chebyshev.cpp",
    "src/numerics/integrate.cpp"
}

-- Common components
local common_sources = {
    -- No cpp files for now, just headers
}

-- Engine components - deterministic execution framework
local engine_determine_sources = {
    "src/engine/determine/jourman.cpp",
    "src/engine/determine/manmem.cpp",
    "src/engine/determine/priceman.cpp",
    "src/engine/determine/schedman.cpp",
    "src/engine/determine/shmem.cpp"
}

-- System components
local engine_system_sources = {
    "src/engine/system/harcount.cpp",
    "src/engine/system/latmon.cpp",
    "src/engine/system/procore.cpp"
}

-- Add common configurations to all targets
function add_common_config()
    -- Include directories
    for _, dir in ipairs(include_dirs) do
        add_includedirs(dir)
    end
   
    -- Link directories
    for _, dir in ipairs(link_dirs) do
        add_linkdirs(dir)
    end
   
    -- Libraries
    for _, lib in ipairs(libraries) do
        add_links(lib)
    end
   
    -- Compiler Flags
    add_cxflags("-Wall")
   
    if is_mode("debug") then
        add_cxflags("-g", "-O0")
        add_defines("DEBUG")
    else
        add_cxflags("-O3")
        add_defines("NDEBUG")
        add_cxflags("-march=native", "-fno-strict-aliasing", "-fno-omit-frame-pointer", "-fno-math-errno")
    end
   
    -- Set runtime library path
    add_runenvs("LD_LIBRARY_PATH", table.concat(run_lib_paths, ":") .. ":${LD_LIBRARY_PATH}")
end

-- Main pricing system
target("pricing_system")
    set_kind("binary")
    add_files("src/main.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_files(engine_determine_sources)
    add_files(engine_system_sources)
    add_common_config()

-- Monitor tool
target("monitor_tool")
    set_kind("binary")
    add_files("src/tools/monitor/montool.cpp")
    add_files(engine_determine_sources)
    add_common_config()

-- Unit tests
target("unit_tests")
    set_kind("binary")
    add_files("src/test/unitest/*.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_files(engine_determine_sources)
    add_files(engine_system_sources)
    add_common_config()
   
    -- Add test rule
    on_test(function (target)
        os.execv(target:targetfile())
    end)

-- Performance tests
target("perf_tests")
    set_kind("binary")
    add_files("src/test/perfortest/*.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_files(engine_determine_sources)
    add_files(engine_system_sources)
    add_common_config()

-- Integration tests
target("integ_tests")
    set_kind("binary")
    add_files("src/test/integtest/*.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_files(engine_determine_sources)
    add_files(engine_system_sources)
    add_common_config()

-- Replay system
target("replay_system")
    set_kind("binary")
    add_files("src/replaysys.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_files(engine_determine_sources)
    add_common_config()

-- Benchmark tool
target("bench_tool")
    set_kind("binary")
    add_files("src/tools/benchmark/bentool.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_files(engine_determine_sources)
    add_files(engine_system_sources)
    add_common_config()

-- Replay tool
target("replay_tool")
    set_kind("binary")
    add_files("src/tools/replay/reptool.cpp")
    add_files(engine_determine_sources)
    add_common_config()
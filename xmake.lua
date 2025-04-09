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
    "src/alo/alo_engine.cpp",
    "src/alo/alo_fp_evaluator.cpp",
    "src/alo/alo_scheme.cpp"
}

local numerics_sources = {
    "src/numerics/chebyshev.cpp",
    "src/numerics/integration.cpp"
}

-- Add volatility arbitrage sources
local vol_arb_sources = {
    "src/vol_arb/models/gjr_garch.cpp",
    "src/vol_arb/models/hmm.cpp",
    "src/vol_arb/models/hybrid_model.cpp",
    "src/vol_arb/strategy/vol_arb_strategy.cpp",
    "src/vol_arb/strategy/opportunity_scanner.cpp"
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

-- Main pricing engine
target("pricing_system")
    set_kind("binary")
    add_files("src/main.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_common_config()

-- Replay system
target("replay_system")
    set_kind("binary")
    add_files("src/replay_system.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_common_config()

-- Monitoring tool
target("monitor_tool")
    set_kind("binary")
    add_files("src/monitor.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_common_config()

-- Unit tests
target("unit_tests")
    set_kind("binary")
    add_files("src/unit_tests.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_common_config()
    
    -- Add test rule
    on_test(function (target)
        os.execv(target:targetfile())
    end)

-- Volatility arbitrage test
target("vol_arb_test")
    set_kind("binary")
    add_files("src/test_cases/volatility_arbitrage_test.cpp")
    add_files(alo_sources)
    add_files(numerics_sources)
    add_files(vol_arb_sources)
    add_common_config()
    
    -- Add test rule
    on_test(function (target)
        os.execv(target:targetfile())
    end)
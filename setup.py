import os, glob
import subprocess
import sys
from setuptools import setup, find_packages

this_dir = os.path.dirname(os.path.abspath(__file__))

def print_build_config_help():
    """Print available environment variables for build configuration"""
    help_text = """
=== CPM.cu Build Configuration ===

Available environment variables to customize the build:

COMPILATION CONTROL:
  CPMCU_DEBUG=1          Enable debug mode (default: 0)
                         - Adds debug symbols (-g3)
                         - Disables optimization (-O0)
                         - Enables debug macros
                         
  CPMCU_PERF=1           Enable performance monitoring (default: 0)
                         - Adds -DENABLE_PERF flag
                         
  CPMCU_DTYPE=fp16,bf16  Data types to support (default: fp16)
                         - Options: fp16, bf16, or fp16,bf16
                         - Controls which kernels are compiled

CUDA ARCHITECTURE:
  CPMCU_CUDA_ARCH=80,86  Target CUDA compute capabilities (auto-detected if not set)
                         - Example: 80 for A100, 86 for RTX 3090
                         - Multiple values: 80,86 for mixed GPU environments

EXAMPLES:
  # Debug build with both data types:
  export CPMCU_DEBUG=1 CPMCU_DTYPE=fp16,bf16
  python setup.py build_ext --inplace

=======================================
"""
    print(help_text)

def show_current_config():
    """Show current build configuration"""
    print("=== Build Configuration ===")
    
    config_items = []
    
    # CUDA Architecture
    env_arch = os.getenv("CPMCU_CUDA_ARCH")
    if env_arch:
        config_items.append(f"CUDA Arch: {env_arch}")
    else:
        config_items.append("CUDA Arch: Auto-detect")
    
    # Build mode
    debug_mode = os.getenv("CPMCU_DEBUG", "0").lower() in ("1", "true", "yes")
    config_items.append(f"Mode: {'Debug' if debug_mode else 'Release'}")
    
    # Data types
    dtype_env = os.getenv("CPMCU_DTYPE", "fp16")
    config_items.append(f"Data Types: {dtype_env.upper()}")
    
    # Performance monitoring
    perf_mode = os.getenv("CPMCU_PERF", "0").lower() in ("1", "true", "yes")
    config_items.append(f"Performance Monitoring: {'Enabled' if perf_mode else 'Disabled'}")
    
    # Compilation performance
    max_jobs = os.getenv("MAX_JOBS")
    if max_jobs:
        config_items.append(f"Max Jobs: {max_jobs}")
    
    nvcc_threads = os.getenv("NVCC_THREADS")
    if nvcc_threads and nvcc_threads != "8":
        config_items.append(f"NVCC Threads: {nvcc_threads}")
    
    print(" | ".join(config_items))
    print("============================")

# Check for help request
if "--help-config" in sys.argv:
    print_build_config_help()
    sys.exit(0)

def detect_cuda_arch():
    """Detect CUDA architecture from environment or devices"""
    # 1. Check environment variable first
    env_arch = os.getenv("CPMCU_CUDA_ARCH")
    if env_arch:
        arch_list = []
        for token in env_arch.split(','):
            token = token.strip()
            if token and token.isdigit():
                arch_list.append(token)
            elif token:
                raise ValueError(
                    f"Invalid CUDA architecture format: '{token}'. "
                    f"CPMCU_CUDA_ARCH should only contain comma-separated numbers like '80,86'"
                )
        
        if arch_list:
            print(f"Using CUDA architectures from environment variable: {arch_list}")
            return arch_list
    
    # 2. Auto-detect from CUDA devices
    try:
        import torch
        if torch.cuda.is_available():
            arch_set = set()
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                major, minor = torch.cuda.get_device_capability(i)
                arch_set.add(f"{major}{minor}")
            
            arch_list = sorted(list(arch_set))
            print(f"Detected CUDA architectures: {arch_list} (from {device_count} GPU devices)")
            return arch_list
        else:
            print("No CUDA devices detected. Cannot determine CUDA architecture.")
            return []
    except ImportError:
        print("PyTorch not available. Cannot determine CUDA architecture.")
        return []
    except Exception as e:
        print(f"CUDA architecture detection failed: {e}. Cannot determine CUDA architecture.")
        return []

def get_compile_config():
    """Get compilation configuration based on environment variables"""
    debug_mode = os.getenv("CPMCU_DEBUG", "0").lower() in ("1", "true", "yes")
    perf_mode = os.getenv("CPMCU_PERF", "0").lower() in ("1", "true", "yes")
    
    # Parse data types
    dtype_env = os.getenv("CPMCU_DTYPE", "fp16").lower()
    dtype_list = [dtype.strip() for dtype in dtype_env.split(',') if dtype.strip()]
    
    valid_dtypes = {"fp16", "bf16"}
    invalid_dtypes = [dtype for dtype in dtype_list if dtype not in valid_dtypes]
    if invalid_dtypes:
        raise ValueError(
            f"Invalid CPMCU_DTYPE values: {invalid_dtypes}. "
            f"Supported values: 'fp16', 'bf16', 'fp16,bf16'"
        )
    
    dtype_set = set(dtype_list)
    dtype_defines = []
    if "fp16" in dtype_set:
        dtype_defines.append("-DENABLE_DTYPE_FP16")
    if "bf16" in dtype_set:
        dtype_defines.append("-DENABLE_DTYPE_BF16")
    
    # Base arguments
    common_args = ["-std=c++17"] + dtype_defines
    nvcc_base = common_args + [
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__", 
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
    ]
    
    if debug_mode:
        cxx_args = common_args + ["-g3", "-O0", "-DDISABLE_MEMPOOL", "-DDEBUG", "-fno-inline", "-fno-omit-frame-pointer"]
        nvcc_args = nvcc_base + ["-O0", "-g", "-lineinfo", "-DDISABLE_MEMPOOL", "-DDEBUG", "-DCUDA_DEBUG",
                                "-Xcompiler", "-g3", "-Xcompiler", "-fno-inline", "-Xcompiler", "-fno-omit-frame-pointer"]
        link_args = ["-g", "-rdynamic"]
    else:
        cxx_args = common_args + ["-O3"]
        nvcc_args = nvcc_base + ["-O3", "--use_fast_math"]
        link_args = []
    
    if perf_mode:
        cxx_args.append("-DENABLE_PERF")
        nvcc_args.append("-DENABLE_PERF")
    
    return cxx_args, nvcc_args, link_args, dtype_set

def get_sources_and_headers(dtype_set):
    """Get source files and headers based on enabled data types"""
    # Get flash attention sources
    flash_sources = []
    for dtype in dtype_set:
        if dtype == "fp16":
            flash_sources.extend(glob.glob("src/flash_attn/src/*hdim128_fp16*.cu"))
        elif dtype == "bf16":
            flash_sources.extend(glob.glob("src/flash_attn/src/*hdim128_bf16*.cu"))
    
    # All sources
    sources = [
        "src/entry.cu",
        "src/utils.cu", 
        "src/signal_handler.cu",
        "src/perf.cu",
        *glob.glob("src/qgemm/gptq_marlin/*cu"),
        *flash_sources,
    ]
    
    # Get headers
    header_patterns = [
        "src/**/*.h", "src/**/*.hpp", "src/**/*.cuh",
        "src/cutlass/include/**/*.h", "src/cutlass/include/**/*.hpp",
        "src/flash_attn/**/*.h", "src/flash_attn/**/*.hpp", "src/flash_attn/**/*.cuh",
    ]
    
    headers = []
    for pattern in header_patterns:
        abs_headers = glob.glob(os.path.join(this_dir, pattern), recursive=True)
        rel_headers = [os.path.relpath(h, this_dir) for h in abs_headers]
        headers.extend([h for h in rel_headers if os.path.exists(os.path.join(this_dir, h))])
    
    return sources, headers

def create_build_extension():
    """Create custom build extension class"""
    try:
        from torch.utils.cpp_extension import BuildExtension
    except ImportError:
        return None
        
    class NinjaBuildExtension(BuildExtension):
        def __init__(self, *args, **kwargs):
            # Check if ninja is available
            try:
                import ninja
                kwargs.setdefault('use_ninja', True)
                os.environ["USE_NINJA"] = "1"
                print("Ninja build system enabled for faster compilation")
            except ImportError:
                print("ERROR: Ninja build system is required but not found.")
                print("Please install ninja manually using one of the following methods:")
                print("  1. pip install ninja")
                print("  2. conda install ninja")
                print("  3. System package manager (e.g., apt install ninja-build)")
                raise RuntimeError("Ninja is required for compilation but not available. Please install ninja and try again.")
            
            # Set MAX_JOBS if not already set
            if not os.environ.get("MAX_JOBS"):
                try:
                    import psutil
                    max_jobs_cores = max(1, os.cpu_count() // 2)
                    free_memory_gb = psutil.virtual_memory().available / (1024 ** 3)
                    max_jobs_memory = int(free_memory_gb / 9)  # ~9GB per job
                    max_jobs = max(1, min(max_jobs_cores, max_jobs_memory))
                    os.environ["MAX_JOBS"] = str(max_jobs)
                    print(f"Setting MAX_JOBS to {max_jobs} (cores: {max_jobs_cores}, memory: {max_jobs_memory})")
                except ImportError:
                    max_jobs = max(1, os.cpu_count() // 2) if os.cpu_count() else 1
                    os.environ["MAX_JOBS"] = str(max_jobs)
                    print(f"Setting MAX_JOBS to {max_jobs} (fallback)")
            else:
                print(f"Using existing MAX_JOBS setting: {os.environ.get('MAX_JOBS')}")
            
            super().__init__(*args, **kwargs)
    
    return NinjaBuildExtension

def build_cuda_extension():
    """Build CUDA extension if possible"""
    from torch.utils.cpp_extension import CUDAExtension
    
    # Show current build configuration
    show_current_config()
    
    # Detect CUDA architecture
    arch_list = detect_cuda_arch()
    if not arch_list:
        print("ERROR: No valid CUDA architectures detected.")
        print("To build CUDA extensions, either:")
        print("1. Set CPMCU_CUDA_ARCH environment variable (e.g., export CPMCU_CUDA_ARCH=80)")
        print("2. Ensure CUDA devices are available and PyTorch can detect them")
        raise RuntimeError("Cannot determine CUDA architecture for compilation")
    
    # Get compilation configuration
    cxx_args, nvcc_args, link_args, dtype_set = get_compile_config()
    sources, headers = get_sources_and_headers(dtype_set)
    
    # Generate architecture-specific arguments
    gencode_args = []
    arch_defines = []
    for arch in arch_list:
        gencode_args.extend(["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
        arch_defines.append(f"-D_ARCH{arch}")
    
    print(f"Using CUDA architecture compile flags: {arch_list}")
    
    # Add NVCC thread configuration
    nvcc_threads = os.getenv("NVCC_THREADS") or "8"
    final_nvcc_args = nvcc_args + gencode_args + arch_defines + ["-MMD", "-MP", "--threads", nvcc_threads]
    
    # Create extension
    ext_modules = [
        CUDAExtension(
            name='cpmcu.C',
            sources=sources,
            libraries=["cublas", "dl"],
            depends=headers,
            extra_compile_args={
                "cxx": cxx_args,
                "nvcc": final_nvcc_args,
            },
            extra_link_args=link_args,
            include_dirs=[
                f"{this_dir}/src/flash_attn",
                f"{this_dir}/src/flash_attn/src", 
                f"{this_dir}/src/cutlass/include",
                f"{this_dir}/src/",
            ],
        )
    ]
    
    # Create build extension class
    build_ext_class = create_build_extension()
    cmdclass = {'build_ext': build_ext_class} if build_ext_class else {}
    
    return ext_modules, cmdclass

# Main setup execution
ext_modules, cmdclass = build_cuda_extension()

setup(
    packages=find_packages(),
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    zip_safe=False,
) 

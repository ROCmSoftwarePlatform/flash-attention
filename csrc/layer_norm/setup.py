# Adapted from https://github.com/NVIDIA/apex/blob/master/setup.py
import sys
import warnings
import os
import glob
import shutil
from packaging.version import parse, Version

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension, CUDA_HOME, IS_HIP_EXTENSION
from setuptools import setup, find_packages
import subprocess

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))


def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output([cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True)
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def check_cuda_torch_binary_vs_bare_metal(cuda_dir):
    raw_output, bare_metal_version = get_cuda_bare_metal_version(cuda_dir)
    torch_binary_version = parse(torch.version.cuda)

    print("\nCompiling cuda extensions with")
    print(raw_output + "from " + cuda_dir + "/bin\n")

    if (bare_metal_version != torch_binary_version):
        raise RuntimeError(
            "Cuda extensions are being compiled with a version of Cuda that does "
            "not match the version used to compile Pytorch binaries.  "
            "Pytorch binaries were compiled with Cuda {}.\n".format(torch.version.cuda)
            + "In some cases, a minor-version mismatch will not cause later errors:  "
            "https://github.com/NVIDIA/apex/pull/323#discussion_r287021798.  "
            "You can try commenting out this check (at your own risk)."
        )


def raise_if_cuda_home_none(global_option: str) -> None:
    if CUDA_HOME is not None:
        return
    raise RuntimeError(
        f"{global_option} was requested, but nvcc was not found.  Are you sure your environment has nvcc available?  "
        "If you're installing within a container from https://hub.docker.com/r/pytorch/pytorch, "
        "only images whose names contain 'devel' will provide nvcc."
    )


def append_nvcc_threads(nvcc_extra_args):
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version >= Version("11.2"):
        return nvcc_extra_args + ["--threads", "4"]
    return nvcc_extra_args


if not torch.cuda.is_available():
    # https://github.com/NVIDIA/apex/issues/486
    # Extension builds after https://github.com/pytorch/pytorch/pull/23408 attempt to query torch.cuda.get_device_capability(),
    # which will fail if you are compiling in an environment without visible GPUs (e.g. during an nvidia-docker build command).
    print(
        "\nWarning: Torch did not find available GPUs on this system.\n",
        "If your intention is to cross-compile, this is not an error.\n"
        "By default, Apex will cross-compile for Pascal (compute capabilities 6.0, 6.1, 6.2),\n"
        "Volta (compute capability 7.0), Turing (compute capability 7.5),\n"
        "and, if the CUDA version is >= 11.0, Ampere (compute capability 8.0).\n"
        "If you wish to cross-compile for a single specific architecture,\n"
        'export TORCH_CUDA_ARCH_LIST="compute capability" before running setup.py.\n',
    )
    if os.environ.get("TORCH_CUDA_ARCH_LIST", None) is None and CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version >= Version("11.8"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6;9.0"
        elif bare_metal_version >= Version("11.1"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0;8.6"
        elif bare_metal_version == Version("11.0"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5;8.0"
        else:
            os.environ["TORCH_CUDA_ARCH_LIST"] = "6.0;6.1;6.2;7.0;7.5"


print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))
TORCH_MAJOR = int(torch.__version__.split(".")[0])
TORCH_MINOR = int(torch.__version__.split(".")[1])

cmdclass = {}
ext_modules = []

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
def build_for_cuda():
    raise_if_cuda_home_none("flash_attn")
    # Check, if CUDA11 is installed for compute capability 8.0
# Check, if CUDA11 is installed for compute capability 8.0
    cc_flag = []
    _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
    if bare_metal_version < Version("11.0"):
        raise RuntimeError("rotary_emb is only supported on CUDA 11 and above")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    ext_modules.append(
    CUDAExtension(
        name="dropout_layer_norm",
        sources=[
            "ln_api.cpp",
            "ln_fwd_256.cu",
            "ln_bwd_256.cu",
            "ln_fwd_512.cu",
            "ln_bwd_512.cu",
            "ln_fwd_768.cu",
            "ln_bwd_768.cu",
            "ln_fwd_1024.cu",
            "ln_bwd_1024.cu",
            "ln_fwd_1280.cu",
            "ln_bwd_1280.cu",
            "ln_fwd_1536.cu",
            "ln_bwd_1536.cu",
            "ln_fwd_2048.cu",
            "ln_bwd_2048.cu",
            "ln_fwd_2560.cu",
            "ln_bwd_2560.cu",
            "ln_fwd_3072.cu",
            "ln_bwd_3072.cu",
            "ln_fwd_4096.cu",
            "ln_bwd_4096.cu",
            "ln_fwd_5120.cu",
            "ln_bwd_5120.cu",
            "ln_fwd_6144.cu",
            "ln_bwd_6144.cu",
            "ln_fwd_7168.cu",
            "ln_bwd_7168.cu",
            "ln_fwd_8192.cu",
            "ln_bwd_8192.cu",
            "ln_parallel_fwd_256.cu",
            "ln_parallel_bwd_256.cu",
            "ln_parallel_fwd_512.cu",
            "ln_parallel_bwd_512.cu",
            "ln_parallel_fwd_768.cu",
            "ln_parallel_bwd_768.cu",
            "ln_parallel_fwd_1024.cu",
            "ln_parallel_bwd_1024.cu",
            "ln_parallel_fwd_1280.cu",
            "ln_parallel_bwd_1280.cu",
            "ln_parallel_fwd_1536.cu",
            "ln_parallel_bwd_1536.cu",
            "ln_parallel_fwd_2048.cu",
            "ln_parallel_bwd_2048.cu",
            "ln_parallel_fwd_2560.cu",
            "ln_parallel_bwd_2560.cu",
            "ln_parallel_fwd_3072.cu",
            "ln_parallel_bwd_3072.cu",
            "ln_parallel_fwd_4096.cu",
            "ln_parallel_bwd_4096.cu",
            "ln_parallel_fwd_5120.cu",
            "ln_parallel_bwd_5120.cu",
            "ln_parallel_fwd_6144.cu",
            "ln_parallel_bwd_6144.cu",
            "ln_parallel_fwd_7168.cu",
            "ln_parallel_bwd_7168.cu",
            "ln_parallel_fwd_8192.cu",
            "ln_parallel_bwd_8192.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"] + generator_flag,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                ]
                + generator_flag
                + cc_flag
            ),
        },
        include_dirs=[this_dir],
    )
)

def rename_cpp_to_hip(cpp_files):
    for entry in cpp_files:
        shutil.copy(entry, os.path.splitext(entry)[0] + ".hip")
def validate_and_update_archs(archs):
    # List of allowed architectures
    allowed_archs = ["native", "gfx90a", "gfx940", "gfx941", "gfx942"]
    # Validate if each element in archs is in allowed_archs
    assert all(
        arch in allowed_archs for arch in archs
    ), f"One of GPU archs of {archs} is invalid or not supported by Flash-Attention"
def build_for_rocm():
    """build for ROCm platform"""
    archs = os.getenv("GPU_ARCHS", "native").split(";")
    validate_and_update_archs(archs)
    cc_flag = [f"--offload-arch={arch}" for arch in archs]
    if int(os.environ.get("FLASH_ATTENTION_INTERNAL_USE_RTN", 0)):
        print("RTN IS USED")
        cc_flag.append("-DUSE_RTN_BF16_CONVERT")
    else:
        print("RTZ IS USED")
    fa_sources = ["ln_api.cpp","ln_fwd_256.cpp","ln_bwd_256.cpp", "ln_fwd_512.cpp","ln_bwd_512.cpp","ln_fwd_768.cpp",
            "ln_bwd_768.cpp","ln_fwd_1024.cpp","ln_bwd_1024.cpp","ln_fwd_1280.cpp","ln_bwd_1280.cpp","ln_fwd_1536.cpp",
            "ln_bwd_1536.cpp","ln_fwd_2048.cpp","ln_bwd_2048.cpp","ln_fwd_2560.cpp","ln_bwd_2560.cpp","ln_fwd_3072.cpp",
            "ln_bwd_3072.cpp","ln_fwd_4096.cpp","ln_bwd_4096.cpp","ln_fwd_5120.cpp","ln_bwd_5120.cpp","ln_fwd_6144.cpp",
            "ln_bwd_6144.cpp","ln_fwd_7168.cpp","ln_bwd_7168.cpp","ln_fwd_8192.cpp","ln_bwd_8192.cpp","ln_parallel_fwd_256.cpp",
            "ln_parallel_bwd_256.cpp","ln_parallel_fwd_512.cpp","ln_parallel_bwd_512.cpp","ln_parallel_fwd_768.cpp",
            "ln_parallel_bwd_768.cpp","ln_parallel_fwd_1024.cpp","ln_parallel_bwd_1024.cpp","ln_parallel_fwd_1280.cpp",
            "ln_parallel_bwd_1280.cpp", "ln_parallel_fwd_1536.cpp","ln_parallel_bwd_1536.cpp","ln_parallel_fwd_2048.cpp",
            "ln_parallel_bwd_2048.cpp","ln_parallel_fwd_2560.cpp","ln_parallel_bwd_2560.cpp","ln_parallel_fwd_3072.cpp",
            "ln_parallel_bwd_3072.cpp","ln_parallel_fwd_4096.cpp","ln_parallel_bwd_4096.cpp","ln_parallel_fwd_5120.cpp",
            "ln_parallel_bwd_5120.cpp","ln_parallel_fwd_6144.cpp","ln_parallel_bwd_6144.cpp","ln_parallel_fwd_7168.cpp",
            "ln_parallel_bwd_7168.cpp","ln_parallel_fwd_8192.cpp","ln_parallel_bwd_8192.cpp",] #+ glob.glob("src/*.cpp")
    rename_cpp_to_hip(fa_sources)
    ext_modules.append(
        CUDAExtension(
            'dropout_layer_norm', ["ln_api.hip","ln_fwd_256.hip","ln_bwd_256.hip", "ln_fwd_512.hip","ln_bwd_512.hip","ln_fwd_768.hip",
            "ln_bwd_768.hip","ln_fwd_1024.hip","ln_bwd_1024.hip","ln_fwd_1280.hip","ln_bwd_1280.hip","ln_fwd_1536.hip",
            "ln_bwd_1536.hip","ln_fwd_2048.hip","ln_bwd_2048.hip","ln_fwd_2560.hip","ln_bwd_2560.hip","ln_fwd_3072.hip",
            "ln_bwd_3072.hip","ln_fwd_4096.hip","ln_bwd_4096.hip","ln_fwd_5120.hip","ln_bwd_5120.hip","ln_fwd_6144.hip",
            "ln_bwd_6144.hip","ln_fwd_7168.hip","ln_bwd_7168.hip","ln_fwd_8192.hip","ln_bwd_8192.hip","ln_parallel_fwd_256.hip",
            "ln_parallel_bwd_256.hip","ln_parallel_fwd_512.hip","ln_parallel_bwd_512.hip","ln_parallel_fwd_768.hip",
            "ln_parallel_bwd_768.hip","ln_parallel_fwd_1024.hip","ln_parallel_bwd_1024.hip","ln_parallel_fwd_1280.hip",
            "ln_parallel_bwd_1280.hip", "ln_parallel_fwd_1536.hip","ln_parallel_bwd_1536.hip","ln_parallel_fwd_2048.hip",
            "ln_parallel_bwd_2048.hip","ln_parallel_fwd_2560.hip","ln_parallel_bwd_2560.hip","ln_parallel_fwd_3072.hip",
            "ln_parallel_bwd_3072.hip","ln_parallel_fwd_4096.hip","ln_parallel_bwd_4096.hip","ln_parallel_fwd_5120.hip",
            "ln_parallel_bwd_5120.hip","ln_parallel_fwd_6144.hip","ln_parallel_bwd_6144.hip","ln_parallel_fwd_7168.hip",
            "ln_parallel_bwd_7168.hip","ln_parallel_fwd_8192.hip","ln_parallel_bwd_8192.hip",], #+ glob.glob("src/*.cpp")
            extra_compile_args={'cxx': ['-g', '-march=native', '-funroll-loops',"-DNDEBUG"],
                                'nvcc': 
                                    [
                                        "-O3",
                                        "-U__HIP_NO_HALF_OPERATORS__",
                                        "-U__HIP_NO_HALF_CONVERSIONS__",
                                        "-U__HIP_NO_BFLOAT16_OPERATORS__",
                                        "-U__HIP_NO_BFLOAT16_CONVERSIONS__",
                                        "-U__HIP_NO_BFLOAT162_OPERATORS__",
                                        "-U__HIP_NO_BFLOAT162_CONVERSIONS__",
                                        "-U__HIPCC_RTC__",
                                        "-I/opt/rocm/include/hip/nvidia_detail"
                                    ]+
                                    [
                                        '-O3', "-DNDEBUG"
                                    ] + cc_flag
                               }
        )
    )
BUILD_TARGET = os.environ.get("BUILD_TARGET", "auto")
def get_package_version():
    with open(Path(this_dir) / "flash_attn" / "__init__.py", "r") as f:
        version_match = re.search(r"^__version__\s*=\s*(.*)$", f.read(), re.MULTILINE)
    public_version = ast.literal_eval(version_match.group(1))
    local_version = os.environ.get("FLASH_ATTN_LOCAL_VERSION")
    if local_version:
        return f"{public_version}+{local_version}"
    else:
        return str(public_version)
if BUILD_TARGET == "auto":
    if IS_HIP_EXTENSION:
        build_for_rocm()
    else:
        build_for_cuda()
else:
    if BUILD_TARGET == "cuda":
        build_for_cuda()
    elif BUILD_TARGET == "rocm":
        build_for_rocm()
setup(
    name="dropout_layer_norm",
    version="0.1",
    description="Fused dropout + add + layer norm",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension} if ext_modules else {},
)

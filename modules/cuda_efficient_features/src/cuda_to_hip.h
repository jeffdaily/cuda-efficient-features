/*
Copyright 2023 Fixstars Corporation
Copyright 2026 Advanced Micro Devices, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http ://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#ifndef __CUDA_TO_HIP_H__
#define __CUDA_TO_HIP_H__

#if defined(USE_HIP) || defined(__HIP_PLATFORM_AMD__)

#include <hip/hip_runtime.h>

// Override __CV_CUDA_HOST_DEVICE__ for HIP so OpenCV's cuda_types.hpp
// gets the proper device attributes on its PtrStepSz operators.
// This must be defined BEFORE including any OpenCV CUDA headers.
#ifndef __CV_CUDA_HOST_DEVICE__
#define __CV_CUDA_HOST_DEVICE__ __host__ __device__ __forceinline__
#endif

// Runtime API types
#define cudaStream_t              hipStream_t
#define cudaError_t               hipError_t
#define cudaSuccess               hipSuccess
#define cudaGetErrorString        hipGetErrorString
#define cudaGetLastError          hipGetLastError

// Memory operations
#define cudaMalloc                hipMalloc
#define cudaFree                  hipFree
#define cudaMemcpy                hipMemcpy
#define cudaMemcpyAsync           hipMemcpyAsync
#define cudaMemsetAsync           hipMemsetAsync
#define cudaMemcpyToSymbol        hipMemcpyToSymbol
#define cudaMemcpyDeviceToDevice  hipMemcpyDeviceToDevice
#define cudaMemcpyDeviceToHost    hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice    hipMemcpyHostToDevice

// Synchronization
#define cudaStreamSynchronize     hipStreamSynchronize
#define cudaDeviceSynchronize     hipDeviceSynchronize

// Warp intrinsics mask type (HIP requires 64-bit masks)
#define FULL_WARP_MASK            0xffffffffffffffffULL

#else // CUDA

#include <cuda_runtime.h>

// On CUDA, use 32-bit mask
#define FULL_WARP_MASK            0xffffffff

#endif // USE_HIP

#endif // !__CUDA_TO_HIP_H__

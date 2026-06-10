// HIP compatibility shim for cuda_runtime.h
// This allows OpenCV CUDA headers to be included in HIP builds.
#pragma once

#include <hip/hip_runtime.h>

// Define __CUDACC__ so that OpenCV's cuda_types.hpp gets the proper
// __host__ __device__ attributes on its device pointer types.
// This must be defined BEFORE including OpenCV headers.
#ifndef __CUDACC__
#define __CUDACC__ 1
#endif

// Provide CUDA stream/event types for OpenCV headers
// These must be defined BEFORE OpenCV includes cuda_stream_accessor.hpp
using cudaStream_t = hipStream_t;
using cudaEvent_t = hipEvent_t;

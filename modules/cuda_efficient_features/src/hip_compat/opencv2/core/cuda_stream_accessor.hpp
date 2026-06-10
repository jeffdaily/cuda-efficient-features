// HIP-compatible shim for OpenCV cuda_stream_accessor.hpp
// This provides StreamAccessor for HIP builds since OpenCV's version requires CUDA.
#pragma once

#include <hip/hip_runtime.h>

// Use our HIP-native Stream definition
// Forward declaration only -- full definition in cuda.hpp
namespace cv { namespace cuda { class Stream; } }

namespace cv
{
namespace cuda
{

struct StreamAccessor
{
    static hipStream_t getStream(const Stream& stream);
};

} // namespace cuda
} // namespace cv

// Include full Stream definition for inline implementation
#include "cuda.hpp"

inline hipStream_t cv::cuda::StreamAccessor::getStream(const cv::cuda::Stream& stream)
{
    return stream.hipStream();
}

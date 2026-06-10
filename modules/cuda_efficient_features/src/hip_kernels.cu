/*
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

// HIP kernels that replace OpenCV CUDA functionality for ROCm builds.
// These implement: Gaussian blur, bilinear resize, and integral image.

#include "cuda_to_hip.h"

#ifdef USE_HIP

#include <opencv2/core/cuda.hpp>

namespace cv
{
namespace cuda
{
namespace hip
{

static constexpr int BLOCK_DIM = 16;

// Gaussian kernel coefficients for 7x7 kernel with sigma=2
// Generated with: cv::getGaussianKernel(7, 2, CV_32F)
__constant__ float gaussianKernel[7] = {
    0.071303f, 0.131514f, 0.189879f, 0.214607f, 0.189879f, 0.131514f, 0.071303f
};

// Horizontal Gaussian pass
__global__ void gaussianHorizontalKernel(const uchar* src, float* dst,
    int srcStep, int dstStep, int cols, int rows)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    float sum = 0.0f;
    for (int k = -3; k <= 3; k++)
    {
        int sx = x + k;
        // Reflect at borders (BORDER_REFLECT_101)
        if (sx < 0) sx = -sx;
        if (sx >= cols) sx = 2 * cols - sx - 2;

        sum += gaussianKernel[k + 3] * src[y * srcStep + sx];
    }

    dst[y * dstStep + x] = sum;
}

// Vertical Gaussian pass
__global__ void gaussianVerticalKernel(const float* src, uchar* dst,
    int srcStep, int dstStep, int cols, int rows)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= cols || y >= rows)
        return;

    float sum = 0.0f;
    for (int k = -3; k <= 3; k++)
    {
        int sy = y + k;
        // Reflect at borders (BORDER_REFLECT_101)
        if (sy < 0) sy = -sy;
        if (sy >= rows) sy = 2 * rows - sy - 2;

        sum += gaussianKernel[k + 3] * src[sy * srcStep + x];
    }

    // Clamp and convert to uchar
    sum = fminf(fmaxf(sum, 0.0f), 255.0f);
    dst[y * dstStep + x] = static_cast<uchar>(sum + 0.5f);
}

void gaussianBlur7x7(const GpuMat& src, GpuMat& dst, hipStream_t stream)
{
    CV_Assert(src.type() == CV_8UC1);

    dst.create(src.size(), CV_8UC1);

    // Temporary buffer for horizontal pass result
    GpuMat tmp(src.rows, src.cols, CV_32FC1);

    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid((src.cols + block.x - 1) / block.x,
                    (src.rows + block.y - 1) / block.y);

    // Horizontal pass
    gaussianHorizontalKernel<<<grid, block, 0, stream>>>(
        src.data, tmp.ptr<float>(),
        static_cast<int>(src.step), static_cast<int>(tmp.step / sizeof(float)),
        src.cols, src.rows);

    // Vertical pass
    gaussianVerticalKernel<<<grid, block, 0, stream>>>(
        tmp.ptr<float>(), dst.data,
        static_cast<int>(tmp.step / sizeof(float)), static_cast<int>(dst.step),
        src.cols, src.rows);
}

// Bilinear resize kernel
__global__ void resizeBilinearKernel(const uchar* src, uchar* dst,
    int srcCols, int srcRows, int srcStep,
    int dstCols, int dstRows, int dstStep,
    float scaleX, float scaleY)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= dstCols || y >= dstRows)
        return;

    // Map destination to source coordinates
    const float srcX = (x + 0.5f) * scaleX - 0.5f;
    const float srcY = (y + 0.5f) * scaleY - 0.5f;

    const int x0 = static_cast<int>(floorf(srcX));
    const int y0 = static_cast<int>(floorf(srcY));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float dx = srcX - x0;
    const float dy = srcY - y0;

    // Clamp coordinates
    const int sx0 = max(0, min(x0, srcCols - 1));
    const int sx1 = max(0, min(x1, srcCols - 1));
    const int sy0 = max(0, min(y0, srcRows - 1));
    const int sy1 = max(0, min(y1, srcRows - 1));

    // Bilinear interpolation
    const float v00 = src[sy0 * srcStep + sx0];
    const float v01 = src[sy0 * srcStep + sx1];
    const float v10 = src[sy1 * srcStep + sx0];
    const float v11 = src[sy1 * srcStep + sx1];

    const float v0 = v00 + dx * (v01 - v00);
    const float v1 = v10 + dx * (v11 - v10);
    const float v = v0 + dy * (v1 - v0);

    dst[y * dstStep + x] = static_cast<uchar>(v + 0.5f);
}

void resize(const GpuMat& src, GpuMat& dst, Size dstSize, hipStream_t stream)
{
    CV_Assert(src.type() == CV_8UC1);

    dst.create(dstSize, CV_8UC1);

    const float scaleX = static_cast<float>(src.cols) / dstSize.width;
    const float scaleY = static_cast<float>(src.rows) / dstSize.height;

    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid((dstSize.width + block.x - 1) / block.x,
                    (dstSize.height + block.y - 1) / block.y);

    resizeBilinearKernel<<<grid, block, 0, stream>>>(
        src.data, dst.data,
        src.cols, src.rows, static_cast<int>(src.step),
        dstSize.width, dstSize.height, static_cast<int>(dst.step),
        scaleX, scaleY);
}

// Row-wise prefix sum kernel (used for integral image)
__global__ void integralRowKernel(const uchar* src, int* dst,
    int srcStep, int dstStep, int cols, int rows)
{
    const int y = blockIdx.x * blockDim.x + threadIdx.x;
    if (y >= rows)
        return;

    const uchar* srcRow = src + y * srcStep;
    int* dstRow = dst + (y + 1) * dstStep + 1;

    int sum = 0;
    for (int x = 0; x < cols; x++)
    {
        sum += srcRow[x];
        dstRow[x] = sum;
    }
}

// Column-wise prefix sum kernel (to complete integral image)
__global__ void integralColKernel(int* data, int step, int cols, int rows)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= cols)
        return;

    int* col = data + x;

    int sum = 0;
    for (int y = 0; y <= rows; y++)
    {
        sum += col[y * step];
        col[y * step] = sum;
    }
}

void calcIntegralImage(const GpuMat& src, GpuMat& dst, hipStream_t stream)
{
    CV_Assert(src.type() == CV_8UC1);

    const int rows = src.rows;
    const int cols = src.cols;

    dst.create(rows + 1, cols + 1, CV_32SC1);
    dst.setTo(Scalar(0));

    const int stepSrc = static_cast<int>(src.step);
    const int stepDst = static_cast<int>(dst.step / sizeof(int));

    // Row-wise prefix sums
    const int blockSize = 256;
    const int gridRows = (rows + blockSize - 1) / blockSize;
    integralRowKernel<<<gridRows, blockSize, 0, stream>>>(
        src.data, dst.ptr<int>(), stepSrc, stepDst, cols, rows);

    // Column-wise prefix sums
    const int gridCols = (cols + 1 + blockSize - 1) / blockSize;
    integralColKernel<<<gridCols, blockSize, 0, stream>>>(
        dst.ptr<int>(), stepDst, cols + 1, rows);
}

} // namespace hip
} // namespace cuda
} // namespace cv

#endif // USE_HIP

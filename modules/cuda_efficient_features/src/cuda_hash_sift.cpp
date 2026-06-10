/*
Copyright 2023 Fixstars Corporation

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

// Implementation of the article:
//     Iago Suarez, Ghesn Sfeir, Jose M. Buenaposada, and Luis Baumela.
//     Revisiting binary local image description for resource limited devices.
//     IEEE Robotics and Automation Letters, 2021.

#include "cuda_efficient_descriptors.h"

// Include cuda_to_hip.h FIRST to set up the type mappings before OpenCV headers
#include "cuda_to_hip.h"

#ifndef USE_HIP
#include <opencv2/cudaarithm.hpp>
#endif
#include <opencv2/core/cuda_stream_accessor.hpp>
#ifdef USE_HIP
#include <hipblas/hipblas.h>
#else
#include <cublas_v2.h>
#endif

#include "cuda_hash_sift_internal.h"
#include "cuda_efficient_features_internal.h"
#include "device_buffer.h"

namespace cv
{
namespace cuda
{

#ifdef USE_HIP
#define BLAS_CHECK(err) \
do {\
	if (err != HIPBLAS_STATUS_SUCCESS) { \
		printf("[HIPBLAS Error] (code: %d) at %s:%d\n", err, __FILE__, __LINE__); \
	} \
} while (0)

using blasHandle_t = hipblasHandle_t;
using blasOperation_t = hipblasOperation_t;
using blasStream_t = hipStream_t;
constexpr blasOperation_t BLAS_OP_T = HIPBLAS_OP_T;
constexpr blasOperation_t BLAS_OP_N = HIPBLAS_OP_N;

inline hipblasStatus_t blasCreate(blasHandle_t* handle) { return hipblasCreate(handle); }
inline hipblasStatus_t blasDestroy(blasHandle_t handle) { return hipblasDestroy(handle); }
inline hipblasStatus_t blasSetPointerMode(blasHandle_t handle, hipblasPointerMode_t mode)
{ return hipblasSetPointerMode(handle, mode); }
inline hipblasStatus_t blasSetStream(blasHandle_t handle, blasStream_t stream)
{ return hipblasSetStream(handle, stream); }
inline hipblasStatus_t blasSgemm(blasHandle_t handle, blasOperation_t transa, blasOperation_t transb,
	int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
	const float* beta, float* C, int ldc)
{ return hipblasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }
constexpr auto BLAS_POINTER_MODE_HOST = HIPBLAS_POINTER_MODE_HOST;

#else // CUDA
#define BLAS_CHECK(err) \
do {\
	if (err != CUBLAS_STATUS_SUCCESS) { \
		printf("[CUBLAS Error] (code: %d) at %s:%d\n", err, __FILE__, __LINE__); \
	} \
} while (0)

using blasHandle_t = cublasHandle_t;
using blasOperation_t = cublasOperation_t;
using blasStream_t = cudaStream_t;
constexpr blasOperation_t BLAS_OP_T = CUBLAS_OP_T;
constexpr blasOperation_t BLAS_OP_N = CUBLAS_OP_N;

inline cublasStatus_t blasCreate(blasHandle_t* handle) { return cublasCreate_v2(handle); }
inline cublasStatus_t blasDestroy(blasHandle_t handle) { return cublasDestroy_v2(handle); }
inline cublasStatus_t blasSetPointerMode(blasHandle_t handle, cublasPointerMode_t mode)
{ return cublasSetPointerMode_v2(handle, mode); }
inline cublasStatus_t blasSetStream(blasHandle_t handle, blasStream_t stream)
{ return cublasSetStream_v2(handle, stream); }
inline cublasStatus_t blasSgemm(blasHandle_t handle, blasOperation_t transa, blasOperation_t transb,
	int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb,
	const float* beta, float* C, int ldc)
{ return cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc); }
constexpr auto BLAS_POINTER_MODE_HOST = CUBLAS_POINTER_MODE_HOST;
#endif

static void hashSIFTGemm(const GpuMat& src1, const GpuMat& src2, GpuMat& dst, const blasHandle_t& handle)
{
	CV_Assert( src1.type() == CV_32FC1 );
	CV_Assert( src1.cols == src2.cols );

	const float alphaf = 1.0f;
	const float betaf = 0.0f;
	const blasOperation_t transa = BLAS_OP_T;
	const blasOperation_t transb = BLAS_OP_N;

	BLAS_CHECK( blasSgemm(handle, transa, transb, src2.rows, src1.rows, src2.cols,
		&alphaf,
		src2.ptr<float>(), static_cast<int>(src2.step / sizeof(float)),
		src1.ptr<float>(), static_cast<int>(src1.step / sizeof(float)),
		&betaf,
		dst.ptr<float>(), static_cast<int>(dst.step / sizeof(float))) );
}

class MatmulAndSign
{
public:

	MatmulAndSign()
	{
		BLAS_CHECK( blasCreate(&handle_) );
		BLAS_CHECK( blasSetPointerMode(handle_, BLAS_POINTER_MODE_HOST) );
	}

	~MatmulAndSign()
	{
		BLAS_CHECK( blasDestroy(handle_) );
	}

	void operator()(const GpuMat& responses, const GpuMat& bMatrix, GpuMat& descriptors, Stream& stream)
	{
		CV_Assert(responses.rows == descriptors.rows);
		BLAS_CHECK( blasSetStream(handle_, StreamAccessor::getStream(stream)) );

		GpuMat tmp = bufTmp_.createMat(responses.rows, bMatrix.rows, responses.type());
		hashSIFTGemm(responses, bMatrix, tmp, handle_);
		gpu::binarizeDescriptors(tmp, descriptors, StreamAccessor::getStream(stream));
	}

private:

	DeviceBuffer bufTmp_;
	blasHandle_t handle_;
};

class HashSIFTImpl : public HashSIFT
{
public:

	HashSIFTImpl(float croppingScale, int nbits) : croppingScale_(croppingScale)
	{
#include "hash_sift.p512.h"
#include "hash_sift.p256.h"

		if (nbits == SIZE_512_BITS)
			Mat(512, 129, CV_64F, (void*)HASH_SIFT_512_VALS).convertTo(bMatrix_, CV_32F);
		else if (nbits == SIZE_256_BITS)
			Mat(256, 129, CV_64F, (void*)HASH_SIFT_256_VALS).convertTo(bMatrix_, CV_32F);
		else
			CV_Error(Error::StsBadArg, "n_bits should be either SIZE_512_BITS or SIZE_256_BITS");

		nbits_ = bMatrix_.rows;
		d_bMatrix_.upload(bMatrix_);
	}

	void computeHashSIFT(InputArray _image, InputKeyPoints _keypoints, OutputArray _descriptors, Stream& stream)
	{
		if (_image.empty())
			return;

		if (isEmpty(_keypoints))
		{
			// clean output buffer (it may be reused with "allocated" data)
			_descriptors.release();
			return;
		}

		CV_Assert(_image.type() == CV_8U);

		getInputMat(_image, image_, stream);
		getKeypointsMat(_keypoints, keypoints_, stream);
		getOutputMat(_descriptors, descriptors_, keypoints_.rows, descriptorSize(), descriptorType());

		GpuMat responses = bufResponses_.createMat(keypoints_.rows, 129, CV_32F);
		gpu::computePatchSIFTs(image_, keypoints_, responses, croppingScale_, 1./6, 1.6, StreamAccessor::getStream(stream));
		matmulAndSign_(responses, d_bMatrix_, descriptors_, stream);

		if (_descriptors.kind() == _InputArray::KindFlag::MAT)
			descriptors_.download(_descriptors);
	}

	void compute(InputArray _image, KeyPoints& _keypoints, OutputArray _descriptors) override
	{
		computeHashSIFT(_image, _keypoints, _descriptors, Stream::Null());
	}

	void computeAsync(InputArray _image, InputArray _keypoints, OutputArray _descriptors, Stream& stream) override
	{
		computeHashSIFT(_image, _keypoints, _descriptors, stream);
	}

	int descriptorSize() const override { return nbits_ / 8; }
	int descriptorType() const override { return CV_8U; }
	int defaultNorm() const override { return NORM_HAMMING; }

private:

	float croppingScale_;
	Mat bMatrix_;
	int nbits_;

	GpuMat image_, keypoints_, descriptors_, d_bMatrix_;
	DeviceBuffer bufResponses_;
	MatmulAndSign matmulAndSign_;
};

Ptr<HashSIFT> HashSIFT::create(float croppingScale, int nbits)
{
	return makePtr<HashSIFTImpl>(croppingScale, nbits);
}

} // namespace cuda
} // namespace cv

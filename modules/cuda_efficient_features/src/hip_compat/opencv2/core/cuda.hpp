// HIP-compatible GpuMat implementation that replaces OpenCV CUDA operations.
// This provides upload/download/create/setTo/copyTo using HIP runtime.
#pragma once

#include <hip/hip_runtime.h>
#include <opencv2/core.hpp>
#include <memory>
#include <cstdlib>

// Suppress OpenCV's CUDA module usage
#define OPENCV_CORE_CUDA_HPP

namespace cv
{
namespace cuda
{

// Forward declarations
class GpuMat;
class Stream;
class HostMem;

inline int divUp(int a, int b) { return (a + b - 1) / b; }

inline size_t getElemSize(int type)
{
    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);
    size_t elemSize1 = 0;
    switch (depth)
    {
        case CV_8U:  case CV_8S:  elemSize1 = 1; break;
        case CV_16U: case CV_16S: elemSize1 = 2; break;
        case CV_32S: case CV_32F: elemSize1 = 4; break;
        case CV_64F:              elemSize1 = 8; break;
    }
    return elemSize1 * cn;
}

// For device code: PtrStep types (must be defined before GpuMat for conversions)
template <typename T> struct DevPtr
{
    typedef T elem_type;
    typedef int index_type;
    enum { elem_size = sizeof(elem_type) };
    T* data;

    __host__ __device__ DevPtr() : data(0) {}
    __host__ __device__ DevPtr(T* data_) : data(data_) {}
    __host__ __device__ size_t elemSize() const { return elem_size; }
    __host__ __device__ operator T*() { return data; }
    __host__ __device__ operator const T*() const { return data; }
};

template <typename T> struct PtrStep : public DevPtr<T>
{
    size_t step;

    __host__ __device__ PtrStep() : step(0) {}
    __host__ __device__ PtrStep(T* data_, size_t step_) : DevPtr<T>(data_), step(step_) {}

    __host__ __device__ T* ptr(int y = 0) { return (T*)((char*)(this->data) + y * step); }
    __host__ __device__ const T* ptr(int y = 0) const { return (const T*)((const char*)(this->data) + y * step); }
    __host__ __device__ T& operator()(int y, int x) { return ptr(y)[x]; }
    __host__ __device__ const T& operator()(int y, int x) const { return ptr(y)[x]; }
};

template <typename T> struct PtrStepSz : public PtrStep<T>
{
    int cols, rows;

    __host__ __device__ PtrStepSz() : cols(0), rows(0) {}
    __host__ __device__ PtrStepSz(int rows_, int cols_, T* data_, size_t step_)
        : PtrStep<T>(data_, step_), cols(cols_), rows(rows_) {}

    template <typename U>
    explicit PtrStepSz(const PtrStepSz<U>& d) : PtrStep<T>((T*)d.data, d.step), cols(d.cols), rows(d.rows) {}
};

typedef PtrStepSz<unsigned char> PtrStepSzb;
typedef PtrStepSz<unsigned short> PtrStepSzus;
typedef PtrStepSz<float> PtrStepSzf;
typedef PtrStepSz<int> PtrStepSzi;

typedef PtrStep<unsigned char> PtrStepb;
typedef PtrStep<unsigned short> PtrStepus;
typedef PtrStep<float> PtrStepf;
typedef PtrStep<int> PtrStepi;

// Stream wrapper
class Stream
{
public:
    Stream() : stream_(0), owned_(false) {}

    explicit Stream(hipStream_t s) : stream_(s), owned_(false) {}

    static Stream& Null()
    {
        static Stream nullStream;
        return nullStream;
    }

    hipStream_t hipStream() const { return stream_; }

    void waitForCompletion() const
    {
        if (stream_) (void)hipStreamSynchronize(stream_);
    }

private:
    hipStream_t stream_;
    bool owned_;
};

// HostMem wrapper for pinned memory
class HostMem
{
public:
    HostMem() : data_(nullptr), rows_(0), cols_(0), type_(0), step_(0) {}

    ~HostMem() { release(); }

    void create(int rows, int cols, int type)
    {
        release();
        rows_ = rows;
        cols_ = cols;
        type_ = type;
        step_ = cols * getElemSize(type);
        size_t size = rows * step_;
        (void)hipHostMalloc(&data_, size, hipHostMallocDefault);
    }

    void release()
    {
        if (data_) { (void)hipHostFree(data_); data_ = nullptr; }
        rows_ = cols_ = 0;
    }

    Mat createMatHeader() const
    {
        return Mat(rows_, cols_, type_, data_, step_);
    }

    Size size() const { return Size(cols_, rows_); }

    template<typename T> T* ptr(int y = 0) { return (T*)((uchar*)data_ + y * step_); }
    template<typename T> const T* ptr(int y = 0) const { return (const T*)((const uchar*)data_ + y * step_); }

private:
    void* data_;
    int rows_, cols_, type_;
    size_t step_;
};

// GpuMat: HIP-native GPU matrix
class GpuMat
{
public:
    int rows, cols;
    size_t step;
    uchar* data;

    GpuMat() : rows(0), cols(0), step(0), data(nullptr), type_(0), refcount_(nullptr), datastart_(nullptr) {}

    GpuMat(int rows_, int cols_, int type_) : GpuMat()
    {
        create(rows_, cols_, type_);
    }

    GpuMat(int rows_, int cols_, int type_, void* data_, size_t step_ = 0)
        : rows(rows_), cols(cols_), step(step_ ? step_ : cols_ * getElemSize(type_)),
          data(static_cast<uchar*>(data_)), type_(type_), refcount_(nullptr), datastart_(nullptr)
    {
        // External data; no ownership
    }

    GpuMat(Size size_, int type_) : GpuMat(size_.height, size_.width, type_) {}

    GpuMat(const GpuMat& m) : rows(m.rows), cols(m.cols), step(m.step), data(m.data),
                              type_(m.type_), refcount_(m.refcount_), datastart_(m.datastart_)
    {
        if (refcount_) ++(*refcount_);
    }

    GpuMat& operator=(const GpuMat& m)
    {
        if (this != &m)
        {
            release();
            rows = m.rows;
            cols = m.cols;
            step = m.step;
            data = m.data;
            type_ = m.type_;
            refcount_ = m.refcount_;
            datastart_ = m.datastart_;
            if (refcount_) ++(*refcount_);
        }
        return *this;
    }

    ~GpuMat() { release(); }

    void create(int rows_, int cols_, int type)
    {
        if (rows == rows_ && cols == cols_ && type_ == type && data)
            return; // Already allocated with correct size/type

        release();
        rows = rows_;
        cols = cols_;
        type_ = type;

        size_t elemSz = getElemSize(type);
        // For single-column matrices (e.g., keypoint arrays), don't add padding
        // so that array indexing works correctly. For wider matrices, align to 256
        // bytes for texture access performance (PORTING_GUIDE: 256B texture pitch).
        if (cols == 1) {
            step = elemSz;  // No padding for column vectors
        } else {
            step = ((cols * elemSz + 255) / 256) * 256;
            if (step < cols * elemSz) step = cols * elemSz; // Safety
        }

        size_t totalSize = rows * step;
        refcount_ = new int(1);
        (void)hipMalloc(&data, totalSize);
        datastart_ = data;
    }

    void create(Size size_, int type) { create(size_.height, size_.width, type); }

    void release()
    {
        if (refcount_)
        {
            if (--(*refcount_) == 0)
            {
                if (datastart_) (void)hipFree(datastart_);
                delete refcount_;
            }
            refcount_ = nullptr;
        }
        data = nullptr;
        datastart_ = nullptr;
        rows = cols = 0;
        step = 0;
    }

    void upload(InputArray arr, Stream& stream = Stream::Null())
    {
        Mat src = arr.getMat();
        create(src.rows, src.cols, src.type());

        hipStream_t s = stream.hipStream();
        if (src.isContinuous() && step == src.cols * src.elemSize())
        {
            (void)hipMemcpyAsync(data, src.data, src.total() * src.elemSize(), hipMemcpyHostToDevice, s);
        }
        else
        {
            (void)hipMemcpy2DAsync(data, step, src.data, src.step, src.cols * src.elemSize(), src.rows,
                             hipMemcpyHostToDevice, s);
        }
    }

    void download(OutputArray arr, Stream& stream = Stream::Null()) const
    {
        arr.create(rows, cols, type_);
        Mat dst = arr.getMat();

        hipStream_t s = stream.hipStream();
        if (dst.isContinuous() && step == (size_t)dst.cols * dst.elemSize())
        {
            (void)hipMemcpyAsync(dst.data, data, dst.total() * dst.elemSize(), hipMemcpyDeviceToHost, s);
        }
        else
        {
            (void)hipMemcpy2DAsync(dst.data, dst.step, data, step, cols * dst.elemSize(), rows,
                             hipMemcpyDeviceToHost, s);
        }

        if (!s) (void)hipDeviceSynchronize(); // Null stream: sync immediately
    }

    void copyTo(GpuMat& dst, Stream& stream = Stream::Null()) const
    {
        dst.create(rows, cols, type_);
        hipStream_t s = stream.hipStream();

        if (step == dst.step)
        {
            (void)hipMemcpyAsync(dst.data, data, rows * step, hipMemcpyDeviceToDevice, s);
        }
        else
        {
            (void)hipMemcpy2DAsync(dst.data, dst.step, data, step, cols * elemSize(), rows,
                             hipMemcpyDeviceToDevice, s);
        }
    }

    void copyTo(OutputArray arr, Stream& stream = Stream::Null()) const
    {
        if (arr.isGpuMat())
        {
            GpuMat& dst = arr.getGpuMatRef();
            copyTo(dst, stream);
        }
        else
        {
            download(arr, stream);
        }
    }

    void setTo(Scalar s, Stream& stream = Stream::Null())
    {
        hipStream_t hipStream = stream.hipStream();

        // Only handle common case: single-channel or all-same-value
        if (s[0] == 0 && s[1] == 0 && s[2] == 0 && s[3] == 0)
        {
            (void)hipMemset2DAsync(data, step, 0, cols * elemSize(), rows, hipStream);
        }
        else if (elemSize() == 1)
        {
            int val = static_cast<int>(s[0]) & 0xFF;
            (void)hipMemset2DAsync(data, step, val, cols, rows, hipStream);
        }
        else
        {
            // For non-zero multi-byte values, need a kernel. For now, use zero.
            // This suffices for mask creation which uses 0 and 255.
            int val = static_cast<int>(s[0]) & 0xFF;
            if (cols * elemSize() <= step && elemSize() == 1)
            {
                (void)hipMemset2DAsync(data, step, val, cols, rows, hipStream);
            }
        }
    }

    GpuMat operator()(const Rect& roi) const
    {
        GpuMat m;
        m.rows = roi.height;
        m.cols = roi.width;
        m.step = step;
        m.data = data + roi.y * step + roi.x * elemSize();
        m.type_ = type_;
        m.refcount_ = refcount_;
        m.datastart_ = datastart_;
        if (refcount_) ++(*refcount_);
        return m;
    }

    GpuMat rowRange(Range r) const { return (*this)(Rect(0, r.start, cols, r.end - r.start)); }
    GpuMat colRange(Range r) const { return (*this)(Rect(r.start, 0, r.end - r.start, rows)); }

    bool empty() const { return data == nullptr || rows == 0 || cols == 0; }
    Size size() const { return Size(cols, rows); }
    int type() const { return type_; }
    size_t elemSize() const { return getElemSize(type_); }
    int channels() const { return CV_MAT_CN(type_); }
    int depth() const { return CV_MAT_DEPTH(type_); }

    template<typename T> T* ptr(int y = 0) { return (T*)(data + y * step); }
    template<typename T> const T* ptr(int y = 0) const { return (const T*)(data + y * step); }

    // Implicit conversion for kernels
    operator uchar*() { return data; }
    operator const uchar*() const { return data; }

    // Implicit conversion to PtrStep types for kernel arguments
    operator PtrStepb() const { return PtrStepb(const_cast<uchar*>(data), step); }
    operator PtrStepSzb() const { return PtrStepSzb(rows, cols, const_cast<uchar*>(data), step); }
    operator PtrStepf() const { return PtrStepf(reinterpret_cast<float*>(const_cast<uchar*>(data)), step); }
    operator PtrStepSzf() const { return PtrStepSzf(rows, cols, reinterpret_cast<float*>(const_cast<uchar*>(data)), step); }
    operator PtrStepi() const { return PtrStepi(reinterpret_cast<int*>(const_cast<uchar*>(data)), step); }
    operator PtrStepSzi() const { return PtrStepSzi(rows, cols, reinterpret_cast<int*>(const_cast<uchar*>(data)), step); }

private:
    int type_;
    int* refcount_;
    uchar* datastart_;
};

// Helper function to create PtrStep from GpuMat
inline PtrStepb globPtr(const GpuMat& m)
{
    return PtrStepb(const_cast<uchar*>(m.data), m.step);
}

template<typename T>
inline PtrStep<T> globPtr(const GpuMat& m)
{
    return PtrStep<T>(const_cast<T*>(m.ptr<T>()), m.step);
}

} // namespace cuda
} // namespace cv

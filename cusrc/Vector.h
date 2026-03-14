#ifdef _WIN32
#define WINLIB_API __declspec(dllexport)
#else
#define WINLIB_API
#endif

extern "C" {
    WINLIB_API int cuda_run(int, int);
    WINLIB_API void* CudaMalloc(uint32_t);
    WINLIB_API void CudaFree(void*);
    WINLIB_API void CudaMemcpy(void*, void*, uint32_t, uint32_t);
    WINLIB_API void Cuda_Add(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Sub(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Mul(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Div(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Neg(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda_Add_Scalar(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_Sub_Scalar(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_Mul_Scalar(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_Div_Scalar(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_Dsqrt_rn(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda___powf(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda___logf(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda___log2f(uint32_t, uint32_t, double*, double*);
    WINLIB_API void CudaDeviceSynchronize(void);
}

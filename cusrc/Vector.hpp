#ifdef _WIN32
#define WINLIB_API __declspec(dllexport)
#else
#define WINLIB_API
#endif

extern "C" {
    WINLIB_API void Cuda_Get_Slice(uint32_t, uint32_t, size_t, uint32_t*, uint32_t*, uint32_t*, double*, double*);
    WINLIB_API void Cuda_Set_Slice(uint32_t, uint32_t, size_t, uint32_t*, uint32_t*, uint32_t*, double*, double*);
    WINLIB_API void* CudaMalloc(uint32_t);
    WINLIB_API void CudaFree(void*);
    WINLIB_API void CudaMemcpy(void*, void*, uint32_t, uint32_t);
    WINLIB_API void CudaMemcpyShift(void*, void*, uint32_t, uint32_t, uint32_t);
    WINLIB_API void Cuda_Conv2d(uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Add(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Sub(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Mul(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Div(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_Neg(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda_Erff(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda_Ceil(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda_Floor(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda_Round(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda_Log(uint32_t, uint32_t, double*, double*);
    WINLIB_API void Cuda_Pow(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_GE(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_BGE(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_EQ(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_BEQ(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_GT(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_BGT(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_LE(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_BLE(uint32_t, uint32_t, double*, double*, double*);
    WINLIB_API void Cuda_LT(uint32_t, uint32_t, double*, double, double*);
    WINLIB_API void Cuda_BLT(uint32_t, uint32_t, double*, double*, double*);
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

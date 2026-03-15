#include "Vector.h"
#include <stdio.h>
#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <cmath>

__global__ void cuda_add(size_t n, const double* A, const double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

__global__ void cuda_sub(size_t n, const double* A, const double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] - B[i];
    }
}

__global__ void cuda_mul(size_t n, const double* A, const double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B[i];
    }
}

__global__ void cuda_div(size_t n, const double* A, const double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] / B[i];
    }
}

__global__ void cuda_neg(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = -A[i];
    }
}

__global__ void cuda_erff(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = erff(A[i]);
    }
}

__global__ void cuda_ceil(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = std::ceil(A[i]);
    }
}

__global__ void cuda_floor(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = std::floor(A[i]);
    }
}

__global__ void cuda_round(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = roundf(A[i]);
    }
}

__global__ void cuda_log(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = std::log(A[i]);
    }
}

__global__ void cuda_pow(size_t n, const double* A, double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = std::pow(A[i], B);
    }
}

__global__ void cuda_ge(size_t n, const double* A, double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] >= B)? 1.0: 0.0;
    }
}

__global__ void cuda_bge(size_t n, const double* A, double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] >= B[i])? 1.0: 0.0;
    }
}

__global__ void cuda_eq(size_t n, const double* A, double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] == B)? 1.0: 0.0;
    }
}

__global__ void cuda_beq(size_t n, const double* A, double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] == B[i])? 1.0: 0.0;
    }
}

__global__ void cuda_gt(size_t n, const double* A, double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] > B)? 1.0: 0.0;
    }
}

__global__ void cuda_bgt(size_t n, const double* A, double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] > B[i])? 1.0: 0.0;
    }
}

__global__ void cuda_le(size_t n, const double* A, double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] <= B)? 1.0: 0.0;
    }
}

__global__ void cuda_ble(size_t n, const double* A, double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] <= B[i])? 1.0: 0.0;
    }
}

__global__ void cuda_lt(size_t n, const double* A, double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] < B)? 1.0: 0.0;
    }
}

__global__ void cuda_blt(size_t n, const double* A, double* B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = (A[i] < B[i])? 1.0: 0.0;
    }
}

__global__ void cuda_add_scalar(size_t n, const double* A, const double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B;
    }
}

__global__ void cuda_sub_scalar(size_t n, const double* A, const double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] - B;
    }
}

__global__ void cuda_mul_scalar(size_t n, const double* A, const double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] * B;
    }
}

__global__ void cuda_div_scalar(size_t n, const double* A, const double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] / B;
    }
}

__global__ void cuda_dsqrt_rn(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = __dsqrt_rn(A[i]);
    }
}

__global__ void cuda___powf(size_t n, const double* A, double B, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = __powf(A[i], B);
    }
}

__global__ void cuda___logf(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = __logf(A[i]);
    }
}

__global__ void cuda___log2f(size_t n, const double* A, double* C) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        C[i] = __log2f(A[i]);
    }
}



void cuda_get_latest_err(void) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void* CudaMalloc(uint32_t size) {
    cudaError err;
    void* d_arr = NULL;
    err = cudaMalloc((void **)&d_arr, (size_t)size);
    if (err != cudaSuccess)
        printf("Malloc: %s\n", cudaGetErrorString(err));
    return d_arr;
}

void CudaFree(void* d_arr) {
    cudaFree(d_arr);
}

void CudaMemcpy(void* dst_arr, void* src_arr, uint32_t size, uint32_t direction) {
    cudaError_t err;
    cudaMemcpyKind cuda_mem_cpy_kind = cudaMemcpyDefault;
    switch (direction)
    {
    case cudaMemcpyHostToHost:
        cuda_mem_cpy_kind = cudaMemcpyHostToHost;
        break;
    case cudaMemcpyHostToDevice:
        cuda_mem_cpy_kind = cudaMemcpyHostToDevice;
        break;
    case cudaMemcpyDeviceToHost:
        cuda_mem_cpy_kind = cudaMemcpyDeviceToHost;
        break;
    case cudaMemcpyDeviceToDevice:
        cuda_mem_cpy_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        cuda_mem_cpy_kind = cudaMemcpyHostToHost;
        break;
    }
    
    err = cudaMemcpy(dst_arr, src_arr, size, cuda_mem_cpy_kind);
    if (err != cudaSuccess)
        printf("Memcpy: %s\n", cudaGetErrorString(err));
}

void Cuda_Add(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_add<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Sub(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_sub<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Mul(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_mul<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Div(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_div<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Neg(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_neg<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Erff(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_erff<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Ceil(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_ceil<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Floor(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_floor<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Round(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_round<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Log(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_log<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Pow(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_pow<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_GE(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_pow<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_BGE(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_bge<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_EQ(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_eq<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_BEQ(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_beq<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_GT(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_gt<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_BGT(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_bgt<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_LE(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_le<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_BLE(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_ble<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_LT(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_lt<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_BLT(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* B_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_blt<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Add_Scalar(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_add_scalar<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Sub_Scalar(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_sub_scalar<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Mul_Scalar(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_mul_scalar<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Div_Scalar(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_div_scalar<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda_Dsqrt_rn(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_dsqrt_rn<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda___powf(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double B_d, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda___powf<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
    cuda_get_latest_err();
}

void Cuda___logf(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda___logf<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void Cuda___log2f(uint32_t threadsPerBlock, uint32_t num_elements, double* A_d_arr, double* C_d_arr) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda___log2f<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, C_d_arr);
    cuda_get_latest_err();
}

void CudaDeviceSynchronize(void) {
    cudaDeviceSynchronize();
}

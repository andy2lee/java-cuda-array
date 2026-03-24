#include "Vector.hpp"
#include <stdio.h>
#include <iostream>
#include <cstdint>
#include <cuda_runtime.h>
#include <cmath>

__global__ void cuda_conv2d(uint32_t d_row, uint32_t d_col, uint32_t m_row, uint32_t m_col, double* d_arr, double* m_arr, double* res_arr) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int row_cnt = (d_row-m_row + 1), col_cnt = (d_col-m_col + 1);

    if ((i < row_cnt) && (j < col_cnt)) {
        double res = 0.0;
        for (int m = 0; m < m_row; m++) {
            for (int n = 0; n < m_col; n++) {
                res+=d_arr[(i * d_col + j) + (m * d_col + n)] * m_arr[m * m_col + n];
                // res += d_arr[(i + m) * d_col + (j + n)] * m_arr[m * m_col + n];
            }
        }
        res_arr[i * col_cnt + j] = res;
    }
}

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

__global__ void cuda_matmul(
    const double* A_arr, 
    const double* B_arr,
    double* C_arr, 
    const uint32_t a_row_len, 
    const uint32_t b_col_len, 
    const uint32_t ab_mid_len
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if ((i < a_row_len) && (j < b_col_len)) {
        double res = 0;
        for (int k = 0; k < ab_mid_len; k++) {
            res+=(A_arr[i*ab_mid_len+k] * B_arr[j+k*b_col_len]);
        }
        C_arr[i*b_col_len+j] = res;
    }
}

__global__ void cuda_get_slice(
    size_t n, 
    size_t shape_len, 
    uint32_t* ndcu_slice_start, 
    uint32_t* re_stride, 
    uint32_t* stride,  
    double* res_cu_na_arr, 
    double* cu_na_arr
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        uint32_t i_rem = i;
        uint32_t index = 0;
        
        for (uint32_t j = 0; j < shape_len; j++) {
            uint32_t i_div = i_rem / re_stride[j];
            i_rem = i_rem % re_stride[j];
            index+=((ndcu_slice_start[j] + i_div) * stride[j]);
        }
        res_cu_na_arr[i] = cu_na_arr[index];
    }
}

__global__ void cuda_set_slice(
    size_t n, 
    size_t shape_len, 
    uint32_t* ndcu_slice_start, 
    uint32_t* re_stride, 
    uint32_t* stride,  
    double* res_cu_na_arr,
    double* cu_na_arr
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) {
        uint32_t i_rem = i;
        uint32_t index = 0;
        
        for (uint32_t j = 0; j < shape_len; j++) {
            uint32_t i_div = i_rem / re_stride[j];
            i_rem = i_rem % re_stride[j];
            index+=((ndcu_slice_start[j] + i_div) * stride[j]);
        }
        res_cu_na_arr[index] = cu_na_arr[i];
    }
}

void cuda_get_latest_err(void) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

void Cuda_Matmul(uint32_t threadsPerBlock, uint32_t num_elements, double* A_arr, double* B_arr, double* C_arr, uint32_t a_row_len, uint32_t b_col_len, uint32_t ab_mid_len, uint32_t A_arr_shift_p, uint32_t B_arr_shift_p, uint32_t C_arr_shift_p) {
    int row_cnt = a_row_len;
    int col_cnt = b_col_len;

    dim3 block(threadsPerBlock, threadsPerBlock);

    dim3 grid(
        (col_cnt + block.x - 1) / block.x,
        (row_cnt + block.y - 1) / block.y
    );

    cuda_matmul<<<grid, block>>>((A_arr+A_arr_shift_p), (B_arr+B_arr_shift_p), (C_arr+C_arr_shift_p), a_row_len, b_col_len, ab_mid_len);
    cuda_get_latest_err();
}

void Cuda_Get_Slice(
    uint32_t threadsPerBlock, uint32_t num_elements, 
    size_t shape_len,
    uint32_t* ndcu_slice_start, 
    uint32_t* re_stride, 
    uint32_t* stride,  
    double* res_cu_na_arr, 
    double* cu_na_arr
) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_get_slice<<<blocksPerGrid, threadsPerBlock>>>(num_elements, shape_len, ndcu_slice_start, re_stride, stride, res_cu_na_arr, cu_na_arr);
    cuda_get_latest_err();
}

void Cuda_Set_Slice(
    uint32_t threadsPerBlock, uint32_t num_elements, 
    size_t shape_len,
    uint32_t* ndcu_slice_start, 
    uint32_t* re_stride, 
    uint32_t* stride,  
    double* res_cu_na_arr, 
    double* cu_na_arr
) {
    size_t blocksPerGrid   = (num_elements + threadsPerBlock - 1) / threadsPerBlock;
    cuda_set_slice<<<blocksPerGrid, threadsPerBlock>>>(num_elements, shape_len, ndcu_slice_start, re_stride, stride, res_cu_na_arr, cu_na_arr);
    cuda_get_latest_err();
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

void CudaMemcpyShift(void* dst_arr, void* src_arr, uint32_t size, uint32_t shift, uint32_t direction) {
    cudaError_t err;
    cudaMemcpyKind cuda_mem_cpy_kind = cudaMemcpyDefault;
    char* curr_dst_arr = (char*)dst_arr;
    char* curr_src_arr = (char*)src_arr;

    switch (direction)
    {
    case cudaMemcpyHostToHost:
        cuda_mem_cpy_kind = cudaMemcpyHostToHost;
        break;
    case cudaMemcpyHostToDevice:
        cuda_mem_cpy_kind = cudaMemcpyHostToDevice;
        curr_dst_arr = curr_dst_arr + shift;
        break;
    case cudaMemcpyDeviceToHost:
        cuda_mem_cpy_kind = cudaMemcpyDeviceToHost;
        curr_src_arr = curr_src_arr + shift;
        break;
    case cudaMemcpyDeviceToDevice:
        cuda_mem_cpy_kind = cudaMemcpyDeviceToDevice;
        break;
    default:
        cuda_mem_cpy_kind = cudaMemcpyHostToHost;
        break;
    }

    err = cudaMemcpy(curr_dst_arr, curr_src_arr, size, cuda_mem_cpy_kind);

    if (err != cudaSuccess)
        printf("Memcpy: %s\n", cudaGetErrorString(err));
}

void Cuda_Conv2d(uint32_t threadsPerBlock, uint32_t d_row, uint32_t d_col, uint32_t m_row, uint32_t m_col, double* d_arr, double* m_arr, double* res_arr) {
    int row_cnt = d_row - m_row + 1;
    int col_cnt = d_col - m_col + 1;

    dim3 block(threadsPerBlock, threadsPerBlock);

    dim3 grid(
        (col_cnt + block.x - 1) / block.x,
        (row_cnt + block.y - 1) / block.y
    );

    cuda_conv2d<<<grid, block>>>(d_row, d_col, m_row, m_col, d_arr, m_arr, res_arr);
    cuda_get_latest_err();
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
    cuda_ge<<<blocksPerGrid, threadsPerBlock>>>(num_elements, A_d_arr, B_d, C_d_arr);
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

// __global__ void cuda_ndcu_reshape(uint32_t re_elem, uint32_t* re_shapes, uint32_t* re_strides) {
//     int i = blockDim.x * blockIdx.x + threadIdx.x;
//     if (i < n) {
//         C[i] = (A[i] >= B)? 1.0: 0.0;
//     }
// }

// void Cuda_NDcu_Reshape(uint32_t threadsPerBlock, uint32_t re_elem, uint32_t* re_shapes, uint32_t* re_strides) {
//     double* cu_arr =  (double*)CudaMalloc(re_elem);
//     size_t blocksPerGrid   = (re_elem + threadsPerBlock - 1) / threadsPerBlock;
//     cuda_ndcu_reshape<<<blocksPerGrid, threadsPerBlock>>>(re_elem, re_shapes, re_strides);

//     cuda_get_latest_err();
// }

void CudaDeviceSynchronize(void) {
    cudaDeviceSynchronize();
}

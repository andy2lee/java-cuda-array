import ctypes
import os
import numpy as np

dll_path = os.path.join(os.path.dirname(__file__), "Vector.dll")
vec = ctypes.CDLL(dll_path)

# void* CudaMalloc(int size)
vec.CudaMalloc.argtypes = [ctypes.c_int]
vec.CudaMalloc.restype = ctypes.c_void_p

# void CudaMemcpy(void* dst, void* src, int size, int kind)
vec.CudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
vec.CudaMemcpy.restype = None

# void Cuda_Add(int threadsPerBlock, int numElements, void* a, void* b, void* c)
vec.Cuda_Add.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
vec.Cuda_Add.restype = None

# void Cuda___powf(int threadsPerBlock, int numElements, void* a, double power, void* out)
vec.Cuda___powf.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_double, ctypes.c_void_p]
vec.Cuda___powf.restype = None

# void CudaDeviceSynchronize()
vec.CudaDeviceSynchronize.argtypes = []
vec.CudaDeviceSynchronize.restype = None

# void CudaFree(void* ptr)
vec.CudaFree.argtypes = [ctypes.c_void_p]
vec.CudaFree.restype = None

cudaMemcpyKind = {
    "cudaMemcpyHostToHost": 0,
    "cudaMemcpyHostToDevice": 1,
    "cudaMemcpyDeviceToHost": 2,
    "cudaMemcpyDeviceToDevice": 3,
    "cudaMemcpyDefault": 4,
}

num_elements = 1000
threadsPerBlock = 256

a_arr = np.zeros(num_elements, dtype=np.float64)
b_arr = np.zeros(num_elements, dtype=np.float64)

a_arr[:5] = 10.0
b_arr[:5] = 10.0

byte_size_arr = a_arr.nbytes

cu_a_arr = vec.CudaMalloc(byte_size_arr)
cu_b_arr = vec.CudaMalloc(byte_size_arr)
cu_c_arr = vec.CudaMalloc(byte_size_arr)

a_ptr = a_arr.ctypes.data_as(ctypes.c_void_p)
b_ptr = b_arr.ctypes.data_as(ctypes.c_void_p)

vec.CudaMemcpy(cu_a_arr, a_ptr, byte_size_arr, cudaMemcpyKind["cudaMemcpyHostToDevice"])
vec.CudaMemcpy(cu_b_arr, b_ptr, byte_size_arr, cudaMemcpyKind["cudaMemcpyHostToDevice"])

vec.Cuda_Add(threadsPerBlock, num_elements, cu_a_arr, cu_b_arr, cu_c_arr)
vec.Cuda___powf(threadsPerBlock, num_elements, cu_c_arr, 2.0, cu_c_arr)
vec.CudaDeviceSynchronize()

c_arr = np.zeros(num_elements, dtype=np.float64)
c_ptr = c_arr.ctypes.data_as(ctypes.c_void_p)

vec.CudaMemcpy(c_ptr, cu_c_arr, byte_size_arr, cudaMemcpyKind["cudaMemcpyDeviceToHost"])

print(c_arr[:5])

vec.CudaFree(cu_a_arr)
vec.CudaFree(cu_b_arr)
vec.CudaFree(cu_c_arr)
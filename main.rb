require 'ffi'

module Vector
  extend FFI::Library

  dll_path = File.join(__dir__, 'Vector.dll')
  ffi_lib dll_path

  # Attach function
  attach_function :CudaMalloc, [:int], :pointer
  attach_function :CudaMemcpy, [:pointer, :pointer, :int, :int], :void
  attach_function :Cuda_Add,   [:int, :int, :pointer, :pointer, :pointer], :void
  attach_function :Cuda___powf,[:int, :int, :pointer, :double, :pointer], :void
  attach_function :CudaDeviceSynchronize, [], :void
  attach_function :CudaFree,   [:pointer], :void
  attach_function :Cuda_Conv2d, [:int, :int, :int, :int, :int, :int, :pointer, :pointer, :pointer], :void
end

cudaMemcpyKind = {
  cudaMemcpyHostToHost: 0,
  cudaMemcpyHostToDevice: 1,
  cudaMemcpyDeviceToHost: 2,
  cudaMemcpyDeviceToDevice: 3,
  cudaMemcpyDefault: 4
}

num_elements, threadsPerBlock = 1000000, 256

a_arr = Array.new num_elements, 0
b_arr = Array.new num_elements, 0

for i in 0..5-1 do
  a_arr[i] = 10
  b_arr[i] = 10
end

a_arr_ptr = FFI::MemoryPointer.new :double, a_arr.size
b_arr_ptr = FFI::MemoryPointer.new :double, b_arr.size
c_arr_ptr = FFI::MemoryPointer.new :double, a_arr.size
a_arr_ptr.write_array_of_double a_arr
b_arr_ptr.write_array_of_double b_arr

byte_size_arr = num_elements * 8
cu_a_arr = Vector.CudaMalloc byte_size_arr
cu_b_arr = Vector.CudaMalloc byte_size_arr
cu_c_arr = Vector.CudaMalloc byte_size_arr

Vector.CudaMemcpy cu_a_arr, a_arr_ptr, byte_size_arr, cudaMemcpyKind[:cudaMemcpyHostToDevice]
Vector.CudaMemcpy cu_b_arr, b_arr_ptr, byte_size_arr, cudaMemcpyKind[:cudaMemcpyHostToDevice]

Vector.Cuda_Add threadsPerBlock, num_elements, cu_a_arr, cu_b_arr, cu_c_arr
Vector.Cuda___powf threadsPerBlock, num_elements, cu_c_arr, 2.0, cu_c_arr
Vector.CudaDeviceSynchronize

Vector.CudaMemcpy c_arr_ptr, cu_c_arr, byte_size_arr, cudaMemcpyKind[:cudaMemcpyDeviceToHost]

c_arr = c_arr_ptr.read_array_of_double a_arr.size

for i in 0..5-1 do
  puts c_arr[i]
end

Vector.CudaFree cu_a_arr
Vector.CudaFree cu_b_arr
Vector.CudaFree cu_c_arr

data_arr = [
     1,  2,  3,  4,  5,  6,  7,  8,  9, 
    10, 11, 12, 13, 14, 15, 16, 17, 18, 
    19, 20, 21, 22, 23, 24, 25, 26, 27,
    28, 29, 30, 31, 32, 33, 34, 35, 36 ]
data_arr = data_arr.map { |i| i.to_f }
mask_arr = [
     3,  4,  5,
     6,  7,  8,
     9, 10, 11 ]
mask_arr = mask_arr.map { |i| i.to_f }

data_row, data_col = 4, 9
mask_row, mask_col = 3, 3

data_arr_ptr   = FFI::MemoryPointer.new :double, data_arr.size
mask_arr_ptr   = FFI::MemoryPointer.new :double, mask_arr.size
result_arr_ptr = FFI::MemoryPointer.new :double, ((data_row-mask_row+1)*(data_col-mask_col+1))

data_arr_ptr.write_array_of_double data_arr
mask_arr_ptr.write_array_of_double mask_arr

cu_d_a_arr   = Vector.CudaMalloc data_arr.size*8
cu_m_b_arr   = Vector.CudaMalloc mask_arr.size*8
cu_res_c_arr = Vector.CudaMalloc ((data_row-mask_row+1)*(data_col-mask_col+1)*8)

Vector.CudaMemcpy cu_d_a_arr, data_arr_ptr, data_arr.size*8, cudaMemcpyKind[:cudaMemcpyHostToDevice]
Vector.CudaMemcpy cu_m_b_arr, mask_arr_ptr, mask_arr.size*8, cudaMemcpyKind[:cudaMemcpyHostToDevice]

Vector.Cuda_Conv2d threadsPerBlock, num_elements, data_row, data_col, mask_row, mask_col, cu_d_a_arr, cu_m_b_arr, cu_res_c_arr
Vector.CudaDeviceSynchronize

Vector.CudaMemcpy result_arr_ptr, cu_res_c_arr, ((data_row-mask_row+1)*(data_col-mask_col+1)*8), cudaMemcpyKind[:cudaMemcpyDeviceToHost]
result_arr = result_arr_ptr.read_array_of_double ((data_row-mask_row+1)*(data_col-mask_col+1))

result_arr.each do |item|
  puts item
end

Vector.CudaFree cu_d_a_arr
Vector.CudaFree cu_m_b_arr
Vector.CudaFree cu_res_c_arr

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
end

cudaMemcpyKind = {
  cudaMemcpyHostToHost: 0,
  cudaMemcpyHostToDevice: 1,
  cudaMemcpyDeviceToHost: 2,
  cudaMemcpyDeviceToDevice: 3,
  cudaMemcpyDefault: 4
}

num_elements, threadsPerBlock = 1000, 256

a_arr = Array.new num_elements, 0
b_arr = Array.new num_elements, 0

for i in 0..5-1 do
  a_arr[i] = 10.0
  b_arr[i] = 10.0
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

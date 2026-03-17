import CudaLib.CudaMemObj;
import static java.lang.foreign.ValueLayout.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;

import static CudaLib.CudaNumLib.*;
import static CudaLib.CudaNumLib.cudaMemcpyKind.*;

public class Main {
    public static void main(String[] args) throws Throwable {
        int num_elements = 1000, threadsPerBlock = 256;
        
        CudaMemObj a_arr = hostMalloc(JAVA_DOUBLE, num_elements);
        CudaMemObj b_arr = hostMalloc(JAVA_DOUBLE, num_elements);
        CudaMemObj c_arr = hostMalloc(JAVA_DOUBLE, num_elements);

        for (int i = 0; i < 5; i++) {
            a_arr.get_ptr().setAtIndex(JAVA_DOUBLE, i, 10.0);
            b_arr.get_ptr().setAtIndex(JAVA_DOUBLE, i, 10.0);
        }

        int byte_size_arr = (int)a_arr.get_ptr().byteSize();
        CudaMemObj cu_a_arr = cudaMalloc(byte_size_arr);
        CudaMemObj cu_b_arr = cudaMalloc(byte_size_arr);
        CudaMemObj cu_c_arr = cudaMalloc(byte_size_arr);

        cudaMemcpy(cu_a_arr, a_arr, byte_size_arr, cudaMemcpyHostToDevice);
        cudaMemcpy(cu_b_arr, b_arr, byte_size_arr, cudaMemcpyHostToDevice);
        cuda_add(threadsPerBlock, num_elements, cu_a_arr, cu_b_arr, cu_c_arr);
        cuda__powf(threadsPerBlock, num_elements, cu_c_arr, 2.0, cu_c_arr);
        cudaDeviceSynchronize();
        cudaMemcpy(c_arr, cu_c_arr, byte_size_arr, cudaMemcpyDeviceToHost);

        for (int i = 0; i < 5; i++) {
            IO.println(c_arr.get_ptr().getAtIndex(JAVA_DOUBLE, i));
        }

        cudaFree(cu_a_arr);
        cudaFree(cu_b_arr);
        cudaFree(cu_c_arr);

        // Conv2d
        double[] data_arr = { 1,  2,  3,  4,  5,  6,  7,  8,  9, 
                             10, 11, 12, 13, 14, 15, 16, 17, 18, 
                             19, 20, 21, 22, 23, 24, 25, 26, 27,
                             28, 29, 30, 31, 32, 33, 34, 35, 36 };
        double[] mask_arr = { 3,  4,  5,
                              6,  7,  8,
                              9, 10, 11 };

        int data_row = 4, data_col = 9;
        int mask_row = 3, mask_col = 3;

        CudaMemObj data_arr_ptr = hostMalloc(JAVA_DOUBLE, data_arr.length);
        CudaMemObj mask_arr_ptr = hostMalloc(JAVA_DOUBLE, mask_arr.length);
        CudaMemObj result_arr_ptr = hostMalloc(JAVA_DOUBLE, (data_row-mask_row+1)*(data_col-mask_col+1));
        
        for (int i = 0; i < data_arr.length; i++) {
            data_arr_ptr.get_ptr().setAtIndex(JAVA_DOUBLE, i, data_arr[i]);
        }
        for (int i = 0; i < mask_arr.length; i++) {
            mask_arr_ptr.get_ptr().setAtIndex(JAVA_DOUBLE, i, mask_arr[i]);
        }

        CudaMemObj cu_d_a_arr = cudaMalloc((int)data_arr_ptr.get_ptr().byteSize());
        CudaMemObj cu_m_b_arr = cudaMalloc((int)mask_arr_ptr.get_ptr().byteSize());
        CudaMemObj cu_res_c_arr = cudaMalloc((int)result_arr_ptr.get_ptr().byteSize());

        cudaMemcpy(cu_d_a_arr, data_arr_ptr, (int)data_arr_ptr.get_ptr().byteSize(), cudaMemcpyHostToDevice);
        cudaMemcpy(cu_m_b_arr, mask_arr_ptr, (int)mask_arr_ptr.get_ptr().byteSize(), cudaMemcpyHostToDevice);
        cuda_conv2d(16, data_row, data_col, mask_row, mask_col, cu_d_a_arr, cu_m_b_arr, cu_res_c_arr);
        cudaDeviceSynchronize();

        IO.println((int)result_arr_ptr.get_ptr().byteSize());
        IO.println(((data_row-mask_row+1)*(data_col-mask_col+1))*8);
        cudaMemcpy(result_arr_ptr, cu_res_c_arr, (int)result_arr_ptr.get_ptr().byteSize(), cudaMemcpyDeviceToHost);

        for (int i = 0; i < (data_row-mask_row+1)*(data_col-mask_col+1); i++) {
            IO.println(result_arr_ptr.get_ptr().getAtIndex(JAVA_DOUBLE, i));
        }
    }
}

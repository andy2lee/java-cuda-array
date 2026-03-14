import CudaLib.CudaMemObj;
import static java.lang.foreign.ValueLayout.*;
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
    }
}

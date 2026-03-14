package CudaLib;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import static java.lang.foreign.ValueLayout.*;

public class CudaNumLib {
    private static Linker linker = Linker.nativeLinker();
    private static Arena arena = Arena.ofConfined();
    private static SymbolLookup lookup = SymbolLookup.libraryLookup("Vector", arena);
    
    private static MethodHandle __cudaMalloc;
    private static MethodHandle __cudaFree;
    private static MethodHandle __cudaMemcpy;
    private static MethodHandle __cudaDeviceSynchronize;
    private static MethodHandle __cuda_add, __cuda_sub, __cuda_mul, __cuda_div, __cuda_neg;
    private static MethodHandle __cuda_add_scalar, __cuda_sub_scalar, __cuda_mul_scalar, __cuda_div_scalar;
    private static MethodHandle __cuda_dsqrt_rn, __cuda__powf, __cuda__logf, __cuda__log2f; 

    public static enum cudaMemcpyKind {
        cudaMemcpyHostToHost      ,
        cudaMemcpyHostToDevice    ,
        cudaMemcpyDeviceToHost    ,
        cudaMemcpyDeviceToDevice  ,
        cudaMemcpyDefault          
    }

    static {
        try {
            __cudaMalloc = linker.downcallHandle(
                lookup.find("CudaMalloc").get(),
                    FunctionDescriptor.of(ADDRESS, JAVA_INT)
            );
            __cudaFree = linker.downcallHandle(
                lookup.find("CudaFree").get(),
                    FunctionDescriptor.ofVoid(ADDRESS)
                );
            __cudaMemcpy = linker.downcallHandle(
                lookup.find("CudaMemcpy").get(),
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT)
                );
            __cudaDeviceSynchronize = linker.downcallHandle(
                lookup.find("CudaDeviceSynchronize").get(),
                    FunctionDescriptor.ofVoid()
                );
            __cuda_add = linker.downcallHandle(
                lookup.find("Cuda_Add").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                );
            __cuda_sub = linker.downcallHandle(
                lookup.find("Cuda_Sub").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                );
            __cuda_mul = linker.downcallHandle(
                lookup.find("Cuda_Mul").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                );
            __cuda_div = linker.downcallHandle(
                lookup.find("Cuda_Div").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                );
            __cuda_neg = linker.downcallHandle(
                lookup.find("Cuda_Neg").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                );
            __cuda_add_scalar = linker.downcallHandle(
                lookup.find("Cuda_Add_Scalar").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                );
            __cuda_sub_scalar = linker.downcallHandle(
                lookup.find("Cuda_Sub_Scalar").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                );
            __cuda_mul_scalar = linker.downcallHandle(
                lookup.find("Cuda_Mul_Scalar").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                );
            __cuda_div_scalar = linker.downcallHandle(
                lookup.find("Cuda_Div_Scalar").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                );

            __cuda_dsqrt_rn = linker.downcallHandle(
                lookup.find("Cuda_Dsqrt_rn").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                );

            __cuda__powf = linker.downcallHandle(
                lookup.find("Cuda___powf").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                );
            
            __cuda__logf = linker.downcallHandle(
                lookup.find("Cuda___logf").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                );

            __cuda__log2f = linker.downcallHandle(
                lookup.find("Cuda___log2f").get(),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                );
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static CudaMemObj hostMalloc(MemoryLayout layout_type, int num_elements) {
        try {
            return new CudaMemObj((MemorySegment)arena.allocate(layout_type, num_elements));
        } catch (Throwable e) {
            e.printStackTrace();
        }
        return null;
    }

    public static CudaMemObj cudaMalloc(int size) {
        try {
            return new CudaMemObj((MemorySegment)__cudaMalloc.invoke(size));
        } catch (Throwable e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void cudaFree(CudaMemObj cuda_mem_obj) {
        try {
            __cudaFree.invoke(cuda_mem_obj.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cudaMemcpy(CudaMemObj d_obj_ptr, CudaMemObj h_obj_ptr, int size, cudaMemcpyKind direction) {
        try {
            __cudaMemcpy.invoke(d_obj_ptr.get_ptr(), h_obj_ptr.get_ptr(), size, direction.ordinal());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cudaDeviceSynchronize() {
        try {
            __cudaDeviceSynchronize.invoke();
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_add(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_add.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_sub(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_sub.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_mul(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_mul.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_div(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_div.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_neg(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_neg.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_add_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_add_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_sub_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_sub_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_mul_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_mul_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_div_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_div_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_dsqrt_rn(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda_dsqrt_rn.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }


    public static void cuda__powf(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda__powf.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda__logf(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda__logf.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda__log2f(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            __cuda__log2f.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }
}

package CudaLib;

import java.lang.foreign.*;
import java.lang.invoke.MethodHandle;
import java.util.HashMap;
import java.util.Map;

import static java.lang.foreign.ValueLayout.*;

public class CudaNumLib {
    static Linker linker = Linker.nativeLinker();
    static Arena arena = Arena.ofConfined();
    static SymbolLookup lookup = SymbolLookup.libraryLookup("Vector", arena);
    static Map<String, MethodHandle> method_handle_map = new HashMap<>();
    
    public static final int CU_DOUBLE = 8;
    public static final int threadsPerBlock = 256;

    public static enum cudaMemcpyKind {
        cudaMemcpyHostToHost      ,
        cudaMemcpyHostToDevice    ,
        cudaMemcpyDeviceToHost    ,
        cudaMemcpyDeviceToDevice  ,
        cudaMemcpyDefault          
    }

    public static void cuda_matmul(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr, int a_row_len, int b_col_len, int ab_mid_len, int A_arr_shift_p, int B_arr_shift_p, int C_arr_shift_p) {
        try {
            var __cuda_matmul = method_handle_map.computeIfAbsent("Cuda_Matmul", k -> linker.downcallHandle(
                lookup.find("Cuda_Matmul")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")), 
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT
                    )
                )
            );
            __cuda_matmul.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr(), a_row_len, b_col_len, ab_mid_len, A_arr_shift_p, B_arr_shift_p, C_arr_shift_p);
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_get_slice(int threadsPerBlock, int num_elements, int shape_len, CudaMemObj ndcu_slice_start, CudaMemObj re_stride, CudaMemObj stride, CudaMemObj res_cu_na_arr, CudaMemObj cu_na_arr) {
        try {
            var __cuda_get_slice = method_handle_map.computeIfAbsent("Cuda_Get_Slice", k -> linker.downcallHandle(
                lookup.find("Cuda_Get_Slice")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")), 
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_get_slice.invoke(threadsPerBlock, num_elements, shape_len, ndcu_slice_start.get_ptr(), re_stride.get_ptr(), stride.get_ptr(), res_cu_na_arr.get_ptr(), cu_na_arr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_set_slice(int threadsPerBlock, int num_elements, int shape_len, CudaMemObj ndcu_slice_start, CudaMemObj re_stride, CudaMemObj stride, CudaMemObj res_cu_na_arr, CudaMemObj cu_na_arr) {
        try { 
            var __cuda_set_slice = method_handle_map.computeIfAbsent("Cuda_Set_Slice", k -> linker.downcallHandle(
                lookup.find("Cuda_Set_Slice")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")), 
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_set_slice.invoke(threadsPerBlock, num_elements, shape_len, ndcu_slice_start.get_ptr(), re_stride.get_ptr(), stride.get_ptr(), res_cu_na_arr.get_ptr(), cu_na_arr.get_ptr());
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
            var __cudaMalloc = method_handle_map.computeIfAbsent("CudaMalloc", k -> linker.downcallHandle(
                lookup.find("CudaMalloc")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.of(ADDRESS, JAVA_INT)
                )
            );
            return new CudaMemObj((MemorySegment)__cudaMalloc.invoke(size));
        } catch (Throwable e) {
            e.printStackTrace();
        }
        return null;
    }

    public static void cudaFree(CudaMemObj cuda_mem_obj) {
        try {
            var __cudaFree = method_handle_map.computeIfAbsent("CudaFree", k -> linker.downcallHandle(
                lookup.find("CudaFree")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(ADDRESS)
                )
            );
            __cudaFree.invoke(cuda_mem_obj.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cudaMemcpy(MemorySegment d_obj_ptr, CudaMemObj h_obj_ptr, int size, cudaMemcpyKind direction) {
        try {
            var __cudaMemcpy = method_handle_map.computeIfAbsent("CudaMemcpy", k -> linker.downcallHandle(
                lookup.find("CudaMemcpy")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT)
                )
            );
            __cudaMemcpy.invoke(d_obj_ptr, h_obj_ptr.get_ptr(), size, direction.ordinal());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cudaMemcpy(CudaMemObj d_obj_ptr, MemorySegment h_obj_ptr, int size, cudaMemcpyKind direction) {
        try {
            var __cudaMemcpy = method_handle_map.computeIfAbsent("CudaMemcpy", k -> linker.downcallHandle(
                lookup.find("CudaMemcpy")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT)
                )
            );
            __cudaMemcpy.invoke(d_obj_ptr.get_ptr(), h_obj_ptr, size, direction.ordinal());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cudaMemcpy(MemorySegment d_obj_ptr, CudaMemObj h_obj_ptr, int size, int shift, cudaMemcpyKind direction) {
        try {
            var __cudaMemcpy_shift = method_handle_map.computeIfAbsent("CudaMemcpyShift", k -> linker.downcallHandle(
                lookup.find("CudaMemcpyShift")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT)
                )
            );
            __cudaMemcpy_shift.invoke(d_obj_ptr, h_obj_ptr.get_ptr(), size, shift, direction.ordinal());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cudaMemcpy(CudaMemObj d_obj_ptr, MemorySegment h_obj_ptr, int size, int shift, cudaMemcpyKind direction) {
        try {
            var __cudaMemcpy_shift = method_handle_map.computeIfAbsent("CudaMemcpyShift", k -> linker.downcallHandle(
                lookup.find("CudaMemcpyShift")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(ADDRESS, ADDRESS, JAVA_INT, JAVA_INT, JAVA_INT)
                )
            );
            __cudaMemcpy_shift.invoke(d_obj_ptr.get_ptr(), h_obj_ptr, size, shift, direction.ordinal());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cudaDeviceSynchronize() {
        try {
            var __cudaDeviceSynchronize = method_handle_map.computeIfAbsent("CudaDeviceSynchronize", k -> linker.downcallHandle(
                lookup.find("CudaDeviceSynchronize")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid()
                )
            );
            __cudaDeviceSynchronize.invoke();
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_add(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_add = method_handle_map.computeIfAbsent("Cuda_Add", k -> linker.downcallHandle(
                lookup.find("Cuda_Add")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_add.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_sub(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_sub = method_handle_map.computeIfAbsent("Cuda_Sub", k -> linker.downcallHandle(
                lookup.find("Cuda_Sub")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_sub.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_mul(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_mul = method_handle_map.computeIfAbsent("Cuda_Mul", k -> linker.downcallHandle(
                lookup.find("Cuda_Mul")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_mul.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_div(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_div = method_handle_map.computeIfAbsent("Cuda_Div", k -> linker.downcallHandle(
                lookup.find("Cuda_Div")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_div.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_neg(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_neg = method_handle_map.computeIfAbsent("Cuda_Neg", k -> linker.downcallHandle(
                lookup.find("Cuda_Neg")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_neg.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_erff(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_erff = method_handle_map.computeIfAbsent("Cuda_Erff", k -> linker.downcallHandle(
                lookup.find("Cuda_Erff")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_erff.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_ceil(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_ceil = method_handle_map.computeIfAbsent("Cuda_Ceil", k -> linker.downcallHandle(
                lookup.find("Cuda_Ceil")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_ceil.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_floor(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_floor = method_handle_map.computeIfAbsent("Cuda_Floor", k -> linker.downcallHandle(
                lookup.find("Cuda_Floor")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_floor.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_round(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_round = method_handle_map.computeIfAbsent("Cuda_Round", k -> linker.downcallHandle(
                lookup.find("Cuda_Round")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_round.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_log(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_log = method_handle_map.computeIfAbsent("Cuda_Log", k -> linker.downcallHandle(
                lookup.find("Cuda_Log")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_log.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_pow(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_pow = method_handle_map.computeIfAbsent("Cuda_Pow", k -> linker.downcallHandle(
                lookup.find("Cuda_Pow")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_pow.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_ge(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_ge = method_handle_map.computeIfAbsent("Cuda_GE", k -> linker.downcallHandle(
                lookup.find("Cuda_GE")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_ge.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_bge(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_bge = method_handle_map.computeIfAbsent("Cuda_BGE", k -> linker.downcallHandle(
                lookup.find("Cuda_BGE")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_bge.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_eq(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_eq = method_handle_map.computeIfAbsent("Cuda_EQ", k -> linker.downcallHandle(
                lookup.find("Cuda_EQ")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_eq.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_beq(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_beq = method_handle_map.computeIfAbsent("Cuda_BEQ", k -> linker.downcallHandle(
                lookup.find("Cuda_BEQ")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_beq.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_gt(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_gt = method_handle_map.computeIfAbsent("Cuda_GT", k -> linker.downcallHandle(
                lookup.find("Cuda_GT")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_gt.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_bgt(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_bgt = method_handle_map.computeIfAbsent("Cuda_BGT", k -> linker.downcallHandle(
                lookup.find("Cuda_BGT")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_bgt.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_le(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_le = method_handle_map.computeIfAbsent("Cuda_LE", k -> linker.downcallHandle(
                lookup.find("Cuda_LE")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_le.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_ble(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_ble = method_handle_map.computeIfAbsent("Cuda_BLE", k -> linker.downcallHandle(
                lookup.find("Cuda_BLE")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_ble.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_lt(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_lt = method_handle_map.computeIfAbsent("Cuda_LT", k -> linker.downcallHandle(
                lookup.find("Cuda_LT")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_lt.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_blt(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_blt = method_handle_map.computeIfAbsent("Cuda_BLT", k -> linker.downcallHandle(
                lookup.find("Cuda_BLT")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_blt.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_add_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_add_scalar = method_handle_map.computeIfAbsent("Cuda_Add_Scalar", k -> linker.downcallHandle(
                lookup.find("Cuda_Add_Scalar")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_add_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_sub_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_sub_scalar = method_handle_map.computeIfAbsent("Cuda_Sub_Scalar", k -> linker.downcallHandle(
                lookup.find("Cuda_Sub_Scalar")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_sub_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_mul_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_mul_scalar = method_handle_map.computeIfAbsent("Cuda_Mul_Scalar", k -> linker.downcallHandle(
                lookup.find("Cuda_Mul_Scalar")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_mul_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_div_scalar(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_div_scalar = method_handle_map.computeIfAbsent("Cuda_Div_Scalar", k -> linker.downcallHandle(
                lookup.find("Cuda_Div_Scalar")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda_div_scalar.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_dsqrt_rn(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_dsqrt_rn = method_handle_map.computeIfAbsent("Cuda_Dsqrt_rn", k -> linker.downcallHandle(
                lookup.find("Cuda_Dsqrt_rn")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_dsqrt_rn.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda__powf(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, double d_b, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda__powf = method_handle_map.computeIfAbsent("Cuda___powf", k -> linker.downcallHandle(
                lookup.find("Cuda___powf")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        JAVA_DOUBLE,
                        ADDRESS
                    )
                )
            );
            __cuda__powf.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_b, d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda__logf(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda__logf = method_handle_map.computeIfAbsent("Cuda___logf", k -> linker.downcallHandle(
                lookup.find("Cuda___logf")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda__logf.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda__log2f(int threadsPerBlock, int num_elements, CudaMemObj d_a_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda__log2f = method_handle_map.computeIfAbsent("Cuda___log2f", k -> linker.downcallHandle(
                lookup.find("Cuda___log2f")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda__log2f.invoke(threadsPerBlock, num_elements, d_a_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }

    public static void cuda_conv2d(int threadsPerBlock, int d_row, int d_col, int m_row, int m_col, CudaMemObj d_a_obj_ptr, CudaMemObj d_b_obj_ptr, CudaMemObj d_c_obj_ptr) {
        try {
            var __cuda_conv2d = method_handle_map.computeIfAbsent("Cuda_Conv2d", k -> linker.downcallHandle(
                lookup.find("Cuda_Conv2d")
                      .orElseThrow(() -> new RuntimeException("Symbol not found")),
                    FunctionDescriptor.ofVoid(
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        JAVA_INT,
                        ADDRESS,
                        ADDRESS,
                        ADDRESS
                    )
                )
            );
            __cuda_conv2d.invoke(threadsPerBlock, d_row, d_col, m_row, m_col, d_a_obj_ptr.get_ptr(), d_b_obj_ptr.get_ptr(), d_c_obj_ptr.get_ptr());
        } catch (Throwable e) {
            e.printStackTrace();
        }
    }
}

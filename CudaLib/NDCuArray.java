package CudaLib;


import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.AutoCloseable;

import static CudaLib.CudaNumLib.*;
import static CudaLib.CudaNumLib.cudaMemcpyKind.*;
import static java.lang.foreign.ValueLayout.*;

public class NDCuArray implements AutoCloseable {
    public int      arr_size;
    public int[]    shape;
    public int[]    stride;
    CudaMemObj      cu_na_arr;
    Arena           arena = Arena.ofConfined();

    public NDCuArray(double[] _arr) {
        arr_size = _arr.length;
        shape    = new int[] { arr_size };
        stride   = new int[] { 1 };

        // GPU copy
        try (Arena tmp_arena = Arena.ofConfined()) {
            MemorySegment na_arr_mem_seg = tmp_arena.allocate(JAVA_DOUBLE, arr_size);
            for (int i = 0; i < arr_size; i++) {
                na_arr_mem_seg.setAtIndex(JAVA_DOUBLE, i, _arr[i]);
            }
            int na_arr_b_size = (int)na_arr_mem_seg.byteSize();
            CudaMemObj _cu_na_arr = CudaNumLib.cudaMalloc(na_arr_b_size);
            CudaNumLib.cudaMemcpy(_cu_na_arr, na_arr_mem_seg, na_arr_b_size, cudaMemcpyHostToDevice);
            cu_na_arr = _cu_na_arr;
        }
    }

    public NDCuArray(CudaMemObj _cu_na_arr, int _arr_size, int[] _shape, int[] _stride) {
        arr_size = _arr_size;
        shape    = _shape.clone();
        stride   = _stride.clone();
        cu_na_arr = _cu_na_arr;
    }

    @Override
    public void close() {
        cudaFree(cu_na_arr);
    }

    //a0 a1 a2    a3       ...
    //0  a0 a0*a1 a0*a1*a2 ...

    public int reshape(int... shape_sizes) {
        // check reshape size is legal or not.
        int reshape_cnt = 1;
        for (int item: shape_sizes) {
            reshape_cnt*=item;
        }
        if (reshape_cnt != arr_size) {
            IO.println("Error reshape size.");
            return -1;
        }
        shape = shape_sizes;

        // calculate stride array settings. col - row - depth
        /*
        int[] new_stride = new int[shape.length];
        int accu_val = 1;
        for (int i = 0; i < new_stride.length; i++) {
            new_stride[i] = accu_val;
            accu_val*=shape[i];
        }
        stride = new_stride;
        */

        // calculate stride array settings. depth - row - col (C/Cpp, numpy like)
        int[] new_stride = new int[shape.length];
        int accu_val = 1;
        for (int i = (new_stride.length-1); i >= 0; i--) {
            new_stride[i] = accu_val;
            accu_val*=shape[i];
        }
        stride = new_stride;

        return 0;
    }

    public NDCuArray get(NDCuSlice... ndcu_slice) {
        int[] re_shape = new int[shape.length];
        int[] re_stride = new int[shape.length];

        if (ndcu_slice.length != shape.length) {
            IO.println("Error get size.");
            return null;
        }
        int re_elem = 1, dim_len = 0;
        for (int i = 0; i < shape.length; i++) {
            dim_len = ndcu_slice[i].end - ndcu_slice[i].start;
            re_elem*=dim_len;
            re_shape[i] = dim_len;
        }

        int accu_val = 1;
        for (int i = (re_stride.length-1); i >= 0; i--) {
            re_stride[i] = accu_val;
            accu_val*=re_shape[i];
        }

        double[] re_res = new double[re_elem];
        for (int i = 0; i < re_elem; i++) {
            int i_rem = i;
            int[] query_shapes = new int[shape.length];
            for (int j = 0; j < query_shapes.length; j++) {
                query_shapes[j] = ndcu_slice[j].start;
            }

            for (int j = 0; j < re_stride.length; j++) {
                int i_div = i_rem / re_stride[j];
                i_rem = i_rem % re_stride[j];
                query_shapes[j]+=i_div;
            }

            int index = 0;
            for (int k = 0; k < stride.length; k++) {
                index+=query_shapes[k] * stride[k];
            }

            MemorySegment na_arr_mem_seg = arena.allocate(JAVA_DOUBLE, 1); // get only 1 element
            int na_arr_b_size = (int)na_arr_mem_seg.byteSize();
            cudaMemcpy(na_arr_mem_seg, cu_na_arr, na_arr_b_size, 8*index, cudaMemcpyDeviceToHost);
            double res = na_arr_mem_seg.getAtIndex(JAVA_DOUBLE, 0);
            IO.println(res);
            re_res[i] = res;
        }

        // GPU copy
        CudaMemObj _cu_na_arr;
        try (Arena tmp_arena = Arena.ofConfined()) {
            MemorySegment na_arr_mem_seg = tmp_arena.allocate(JAVA_DOUBLE, re_elem);
            for (int i = 0; i < re_elem; i++) {
                na_arr_mem_seg.setAtIndex(JAVA_DOUBLE, i, re_res[i]);
            }
            int na_arr_b_size = (int)na_arr_mem_seg.byteSize();
            _cu_na_arr = CudaNumLib.cudaMalloc(na_arr_b_size);
            CudaNumLib.cudaMemcpy(_cu_na_arr, na_arr_mem_seg, na_arr_b_size, cudaMemcpyHostToDevice);
        }
        return new NDCuArray(_cu_na_arr, re_elem, re_shape, re_stride);
    }

    public double get(int... indexes) {
        if (indexes.length != shape.length) {
            IO.println("Error get size.");
            return 0.0;
        }
        for (int i = 0; i < shape.length; i++) {
            if ( (indexes[i] < 0) || (indexes[i] >= shape[i]) ) {
                IO.println("Error access array index.");
                return 0.0; 
            }
        }

        MemorySegment na_arr_mem_seg = arena.allocate(JAVA_DOUBLE, 1); // get only 1 element
        int na_arr_b_size = (int)na_arr_mem_seg.byteSize();
        cudaMemcpy(na_arr_mem_seg, cu_na_arr, na_arr_b_size, 8*cal_idx_loc(indexes), cudaMemcpyDeviceToHost);
        double res = na_arr_mem_seg.getAtIndex(JAVA_DOUBLE, 0);

        return res;
    }

    public void set(double value, int... indexes) {
        if (indexes.length != shape.length) {
            IO.println("Error get size.");
            return;
        }
        for (int i = 0; i < shape.length; i++) {
            if ( (indexes[i] < 0) || (indexes[i] >= shape[i]) ) {
                IO.println("Error access array index.");
                return; 
            }
        }

        MemorySegment na_arr_mem_seg = arena.allocate(JAVA_DOUBLE, 1); // set only 1 element
        na_arr_mem_seg.setAtIndex(JAVA_DOUBLE, 0, value);
        int na_arr_b_size = (int)na_arr_mem_seg.byteSize();
        cudaMemcpy(cu_na_arr, na_arr_mem_seg, na_arr_b_size, 8*cal_idx_loc(indexes), cudaMemcpyHostToDevice);
    }

    // Ops
    public NDCuArray add(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * 8);
        cuda_add(256, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray add(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * 8);
        cuda_add(256, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    public NDCuArray sub(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * 8);
        cuda_sub(256, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray sub(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * 8);
        cuda_sub(256, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    public NDCuArray mul(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * 8);
        cuda_mul(256, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray mul(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * 8);
        cuda_mul(256, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    public NDCuArray div(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * 8);
        cuda_div(256, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray div(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * 8);
        cuda_div(256, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    private int cal_idx_loc(int... indexes) {
        int index = 0;
        for (int i = 0; i < shape.length; i++) {
            index+=indexes[i] * stride[i];
        }

        return index;
    }
}

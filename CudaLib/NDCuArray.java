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
            CudaMemObj _cu_na_arr = cudaMalloc(na_arr_b_size);
            cudaMemcpy(_cu_na_arr, na_arr_mem_seg, na_arr_b_size, cudaMemcpyHostToDevice);
            cu_na_arr = _cu_na_arr;
        }
    }

    public NDCuArray(int... _shape) {
        arr_size = 1;
        for (int i = 0; i < _shape.length; i++) {
            arr_size*=_shape[i];
        }
        shape = _shape.clone();

        // calculate stride array settings. depth - row - col (C/Cpp, numpy like)
        int[] _stride = new int[shape.length];
        int accu_val = 1;
        for (int i = (_stride.length-1); i >= 0; i--) {
            _stride[i] = accu_val;
            accu_val*=shape[i];
        }
        stride = _stride;
        cu_na_arr = cudaMalloc(arr_size*CU_DOUBLE);
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

    public void print() {
        try (Arena tmp_arena = Arena.ofConfined()) {
            MemorySegment na_arr_mem_seg = tmp_arena.allocate(JAVA_DOUBLE, arr_size); // get only 1 element
            int na_arr_size = (int)na_arr_mem_seg.byteSize();
            cudaMemcpy(na_arr_mem_seg, cu_na_arr, na_arr_size, cudaMemcpyDeviceToHost);
            System.out.printf("[ ");
            for (int i = 0; i < arr_size; i++) {
                System.out.printf("%f ", na_arr_mem_seg.getAtIndex(JAVA_DOUBLE, i));
            }
            System.out.printf("]\n");
        }
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

    public void set(NDCuArray b_arr, NDCuSlice... slices) {
        int[] re_shape = new int[shape.length];
        int re_elem = 1;
        
        if (slices.length != shape.length) {
            IO.println("Error get size.");
            return;
        }
        
        for (int i = 0; i < slices.length; i++) {
            if (slices[i].start > slices[i].end) {
                IO.println("Error access array index when start is larget than end index.");
                return;
            }
            if ((slices[i].start < 0) || (slices[i].start > shape[i])) {
                IO.println("Error access array index.");
                return;
            }
            if ((slices[i].end < 0) || (slices[i].end > shape[i])) {
                IO.println("Error access array index.");
                return;
            }
        }

        try (Arena tmp_arena = Arena.ofConfined()) {
            MemorySegment stride_mem_seg = tmp_arena.allocate(JAVA_INT, shape.length);
            MemorySegment re_stride_mem_seg = tmp_arena.allocate(JAVA_INT, shape.length);
            MemorySegment slice_start_mem_seg = tmp_arena.allocate(JAVA_INT, shape.length);
            
            // set each slice len and their slice start index
            int dim_len = 0;
            for (int i = 0; i < shape.length; i++) {
                dim_len = slices[i].end - slices[i].start;
                re_elem*=dim_len;
                re_shape[i] = dim_len;
                slice_start_mem_seg.setAtIndex(JAVA_INT, i, slices[i].start);
            }

            // set re_stride mem and original stride mem
            int accu_val = 1;
            for (int i = (shape.length-1); i >= 0; i--) {
                re_stride_mem_seg.setAtIndex(JAVA_INT, i, accu_val);
                accu_val*=re_shape[i];
            }

            for (int i = 0; i < stride.length; i++) {
                stride_mem_seg.setAtIndex(JAVA_INT, i, stride[i]);
            }

            // GPU allocate mem
            CudaMemObj cu_nd_slice_start = cudaMalloc((int)slice_start_mem_seg.byteSize());
            cudaMemcpy(cu_nd_slice_start, slice_start_mem_seg, (int)slice_start_mem_seg.byteSize(), cudaMemcpyHostToDevice);
            CudaMemObj cu_re_stride = cudaMalloc((int)re_stride_mem_seg.byteSize());
            cudaMemcpy(cu_re_stride, re_stride_mem_seg, (int)re_stride_mem_seg.byteSize(), cudaMemcpyHostToDevice);
            CudaMemObj cu_stride = cudaMalloc((int)stride_mem_seg.byteSize());
            cudaMemcpy(cu_stride, stride_mem_seg, (int)stride_mem_seg.byteSize(), cudaMemcpyHostToDevice);

            cuda_set_slice(
                threadsPerBlock,
                re_elem, 
                shape.length, 
                cu_nd_slice_start,
                cu_re_stride,
                cu_stride,
                cu_na_arr,
                b_arr.cu_na_arr
            );
            cudaDeviceSynchronize();
        }
    }

    public NDCuArray get(NDCuSlice... slices) {
        int[] re_shape = new int[shape.length];
        int[] re_stride;
        int re_elem = 1;
        CudaMemObj _cu_na_arr;
        
        if (slices.length != shape.length) {
            IO.println("Error get size.");
            return null;
        }
        
        for (int i = 0; i < slices.length; i++) {
            if (slices[i].start > slices[i].end) {
                IO.println("Error access array index when start is larget than end index.");
                return null;
            }
            if ((slices[i].start < 0) || (slices[i].start > shape[i])) {
                IO.println("Error access array index.");
                return null;
            }
            if ((slices[i].end < 0) || (slices[i].end > shape[i])) {
                IO.println("Error access array index.");
                return null;
            }
        }

        try (Arena tmp_arena = Arena.ofConfined()) {
            MemorySegment stride_mem_seg = tmp_arena.allocate(JAVA_INT, shape.length);
            MemorySegment re_stride_mem_seg = tmp_arena.allocate(JAVA_INT, shape.length);
            MemorySegment slice_start_mem_seg = tmp_arena.allocate(JAVA_INT, shape.length);
            
            // set each slice len and their slice start index
            int dim_len = 0;
            for (int i = 0; i < shape.length; i++) {
                dim_len = slices[i].end - slices[i].start;
                re_elem*=dim_len;
                re_shape[i] = dim_len;
                slice_start_mem_seg.setAtIndex(JAVA_INT, i, slices[i].start);
            }

            // set re_stride mem and original stride mem
            int accu_val = 1;
            for (int i = (shape.length-1); i >= 0; i--) {
                re_stride_mem_seg.setAtIndex(JAVA_INT, i, accu_val);
                accu_val*=re_shape[i];
            }

            for (int i = 0; i < stride.length; i++) {
                stride_mem_seg.setAtIndex(JAVA_INT, i, stride[i]);
            }

            // GPU allocate mem
            CudaMemObj res_cu_na_arr = cudaMalloc(re_elem * CU_DOUBLE);
            CudaMemObj cu_nd_slice_start = cudaMalloc((int)slice_start_mem_seg.byteSize());
            cudaMemcpy(cu_nd_slice_start, slice_start_mem_seg, (int)slice_start_mem_seg.byteSize(), cudaMemcpyHostToDevice);
            CudaMemObj cu_re_stride = cudaMalloc((int)re_stride_mem_seg.byteSize());
            cudaMemcpy(cu_re_stride, re_stride_mem_seg, (int)re_stride_mem_seg.byteSize(), cudaMemcpyHostToDevice);
            CudaMemObj cu_stride = cudaMalloc((int)stride_mem_seg.byteSize());
            cudaMemcpy(cu_stride, stride_mem_seg, (int)stride_mem_seg.byteSize(), cudaMemcpyHostToDevice);

            cuda_get_slice(
                threadsPerBlock,
                re_elem, 
                shape.length, 
                cu_nd_slice_start,
                cu_re_stride,
                cu_stride,
                res_cu_na_arr,
                cu_na_arr
            );
            cudaDeviceSynchronize();
            
            re_stride = re_stride_mem_seg.toArray(JAVA_INT);
            _cu_na_arr = res_cu_na_arr;
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
        cudaMemcpy(na_arr_mem_seg, cu_na_arr, na_arr_b_size, CU_DOUBLE*cal_idx_loc(indexes), cudaMemcpyDeviceToHost);
        double res = na_arr_mem_seg.getAtIndex(JAVA_DOUBLE, 0);

        return res;
    }

    public void set(double num, int... indexes) {
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
        na_arr_mem_seg.setAtIndex(JAVA_DOUBLE, 0, num);
        int na_arr_b_size = (int)na_arr_mem_seg.byteSize();
        cudaMemcpy(cu_na_arr, na_arr_mem_seg, na_arr_b_size, CU_DOUBLE*cal_idx_loc(indexes), cudaMemcpyHostToDevice);
    }

    // Ops
    public NDCuArray add(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * CU_DOUBLE);
        cuda_add(threadsPerBlock, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray add(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * CU_DOUBLE);
        cuda_add(threadsPerBlock, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    public NDCuArray sub(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * CU_DOUBLE);
        cuda_sub(threadsPerBlock, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray sub(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * CU_DOUBLE);
        cuda_sub(threadsPerBlock, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    public NDCuArray mul(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * CU_DOUBLE);
        cuda_mul(threadsPerBlock, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray mul(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * CU_DOUBLE);
        cuda_mul(threadsPerBlock, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    public NDCuArray div(NDCuArray b_arr) {
        if (arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(arr_size * CU_DOUBLE);
        cuda_div(threadsPerBlock, arr_size, cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, arr_size, shape, stride);
    }

    public NDCuArray div(NDCuArray a_arr, NDCuArray b_arr) {
        if (a_arr.arr_size != b_arr.arr_size)
            return null;

        CudaMemObj _cu_na_arr = cudaMalloc(a_arr.arr_size * CU_DOUBLE);
        cuda_div(threadsPerBlock, a_arr.arr_size, a_arr.cu_na_arr, b_arr.cu_na_arr, _cu_na_arr);
        cudaDeviceSynchronize();
        return new NDCuArray(_cu_na_arr, a_arr.arr_size, a_arr.shape, a_arr.stride);
    }

    int cal_idx_loc(int... indexes) {
        int index = 0;
        for (int i = 0; i < shape.length; i++) {
            index+=indexes[i] * stride[i];
        }

        return index;
    }
}

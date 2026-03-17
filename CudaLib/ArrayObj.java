public class ArrayObj {
    public int      arr_size;
    public double[] arr;
    public int[]    shape;
    public int[]    stride;

    public ArrayObj(double[] _arr) {
        arr      = _arr;
        arr_size = _arr.length;
        shape    = new int[] { arr_size };
        stride   = new int[] { 1 };
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

        // calculate stride array settings.
        int[] new_stride = new int[shape.length];
        int accu_val = 1;
        for (int i = 0; i < new_stride.length; i++) {
            new_stride[i] = accu_val;
            accu_val*=shape[i];
        }
        stride = new_stride;

        return 0;
    }

    public double get(int... indexes) {
        if (indexes.length != shape.length) {
            IO.println("Error get size.");
            return 0.0;
        }
        int index = 0;
        for (int i = 0; i < shape.length; i++) {
            index+=indexes[i] * stride[i];
        }

        return arr[index];
    }

    public void set(int... indexes) {
        if (indexes.length != (shape.length+1)) {
            IO.println("Error get size.");
            return;
        }

        arr[cal_idx_loc(indexes)] = indexes[indexes.length-1];
    }

    private int cal_idx_loc(int... indexes) {
        int index = 0;
        for (int i = 0; i < shape.length; i++) {
            index+=indexes[i] * stride[i];
        }

        return index;
    }
}

import CudaLib.CudaMemObj;
import CudaLib.NDCuArray;
import CudaLib.NDCuSlice;

import static java.lang.foreign.ValueLayout.*;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.util.Arrays;

import static CudaLib.CudaNumLib.*;
import static CudaLib.CudaNumLib.cudaMemcpyKind.*;
import CudaLib.NDCuArray;
import static CudaLib.NDCuSlice.Sliceof;

public class Main {
    public static void main(String[] args) throws Throwable {
        // Array test.

        NDCuArray arr = new NDCuArray(new double[]{ 1,  2,  3,  4,  5,  6,  7,  8,  9, 
                                                 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                                                 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                                 28, 29, 30, 31, 32, 33, 34, 35, 36 });
                                                 
        IO.println(arr.arr_size);
        arr.reshape(2, 2, 9);
        double get_val = arr.get(1, 1, 3); // depth - row - col
        IO.println(get_val);
        arr.set(1234.0, 1, 1, 3);
        double get_val_02 = arr.get(1, 1, 3);
        IO.println(get_val_02);

        NDCuArray arr02 = new NDCuArray(new double[]{ 1,  2,  3,  4,  5,  6,  7,  8,  9, 
                                                 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                                                 19, 20, 21, 22, 23, 24, 25, 26, 27,
                                                 28, 29, 30, 31, 32, 33, 34, 35, 36 });
        
        NDCuArray arr03 = arr.add(arr02);
        IO.println(arr03.get(1, 1, 3));

        NDCuArray arr04 = new NDCuArray(new double[]{ 1,  2,  3,  4,  5,  6,  7,  8,  9, 
                                                 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                                                 19, 20, 21, 22, 23, 24 });
        arr04.reshape(2, 3, 4);
        // 1,  2,  3,  4  //  13, 14, 15, 16
        // 5,  [6,  7,  8  //  17, [18, 19, 20
        // 9,  10, 11,] 12  // 21, 22, 23,] 24

        NDCuArray arr06 = arr04.get(Sliceof(0, 2), Sliceof(1, 3), Sliceof(1, 3));
        IO.println(Arrays.toString(arr06.shape));
        IO.println(Arrays.toString(arr04.shape));

        IO.println(arr06.get(0, 0, 0));
        IO.println(arr06.get(0, 0, 1));
        IO.println(arr06.get(0, 1, 0));
        IO.println(arr06.get(0, 1, 1));
        IO.println(arr06.get(1, 0, 0));
        IO.println(arr06.get(1, 0, 1));
        IO.println(arr06.get(1, 1, 0));
        IO.println(arr06.get(1, 1, 1));

        arr06.set(100.0, 0, 0, 0);
        arr06.set(200.0, 0, 0, 1);
        arr06.set(300.0,0, 1, 0);
        arr06.set(400.0,0, 1, 1);
        arr06.set(500.0,1, 0, 0);
        arr06.set(600.0,1, 0, 1);
        arr06.set(700.0,1, 1, 0);
        arr06.set(800.0,1, 1, 1);

        IO.println(arr06.get(0, 0, 0));
        IO.println(arr06.get(0, 0, 1));
        IO.println(arr06.get(0, 1, 0));
        IO.println(arr06.get(0, 1, 1));
        IO.println(arr06.get(1, 0, 0));
        IO.println(arr06.get(1, 0, 1));
        IO.println(arr06.get(1, 1, 0));
        IO.println(arr06.get(1, 1, 1));
        
        arr04.set(arr06, Sliceof(0, 2), Sliceof(1, 3), Sliceof(2, 4));
        IO.println(arr04.get(0, 1, 2));
        IO.println(arr04.get(0, 1, 3));
        IO.println(arr04.get(0, 2, 2));
        IO.println(arr04.get(0, 2, 3));
        IO.println(arr04.get(1, 1, 2));
        IO.println(arr04.get(1, 1, 3));
        IO.println(arr04.get(1, 2, 2));
        IO.println(arr04.get(1, 2, 3));
        arr04.print();

        NDCuArray cu_arr_zeros = new NDCuArray(2, 3, 4);
        cu_arr_zeros.print();
    }
}

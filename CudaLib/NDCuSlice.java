package CudaLib;

public class NDCuSlice {
    public int start, end;
    public NDCuSlice(int _start, int _end) {
        start = _start;
        end   = _end;
    }
    public static NDCuSlice Sliceof(int start, int end) {
        return new NDCuSlice(start, end);
    }
}

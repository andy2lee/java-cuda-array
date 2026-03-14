package CudaLib;

import java.lang.foreign.MemorySegment;

public class CudaMemObj {
    private MemorySegment ptr;
    
    public CudaMemObj(MemorySegment _ptr) {
        ptr = _ptr;
    }

    public void set_ptr(MemorySegment _ptr) {
        ptr = _ptr;
    }

    public MemorySegment get_ptr() {
        return ptr;
    }
}

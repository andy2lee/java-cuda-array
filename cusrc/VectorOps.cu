extern "C"
__global__ void vec_set (size_t n, double *result, double  value)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = value;
    }
}


//=== Vector arithmetic ======================================================

extern "C"
__global__ void vec_add (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] + y[id];
    }
}


extern "C"
__global__ void vec_sub (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] - y[id];
    }
}


extern "C"
__global__ void vec_mul (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] * y[id];
    }
}


extern "C"
__global__ void vec_div (size_t n, double *result, double  *x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] / y[id];
    }
}

extern "C"
__global__ void vec_negate (size_t n, double *result, double  *x)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = -x[id];
    }
}




//=== Vector-and-scalar arithmetic ===========================================

extern "C"
__global__ void vec_addScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] + y;
    }
}


extern "C"
__global__ void vec_subScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] - y;
    }
}


extern "C"
__global__ void vec_mulScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] * y;
    }
}


extern "C"
__global__ void vec_divScalar (size_t n, double *result, double  *x, double  y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x[id] / y;
    }
}




extern "C"
__global__ void vec_scalarAdd (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x + y[id];
    }
}


extern "C"
__global__ void vec_scalarSub (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x - y[id];
    }
}


extern "C"
__global__ void vec_scalarMul (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x * y[id];
    }
}


extern "C"
__global__ void vec_scalarDiv (size_t n, double *result, double  x, double  *y)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < n)
    {
        result[id] = x / y[id];
    }
}

extern "C" {
    __declspec(dllexport) int cudaMalloc(void**, int);
    __declspec(dllexport) int cudaMemcpy(float*, float*, int, int); // para4: cudaMemcpyHostToDevice
    __declspec(dllexport) int cudaFree(float*);
}

void cudaMalloc_func() {
    
}

void cudaMemcpy_func() {
    
}

void cudaFree_func(void* ptr) {
    cudaFree(ptr);
}

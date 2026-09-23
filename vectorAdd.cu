#include <iostream>
#include <chrono>
#include <cuda_runtime.h>

__global__ void vectoradd(const float *A, const float *B, float *C, const int numsElements) {
    int index = blockDim.x + blockIdx.x + threadIdx.x;
    if(index < numsElements) {
        C[index] = A[index] + B[index];
    }
}

int main() {
    int numsElements = 50000;
    const size_t size = numsElements * sizeof(float);

    auto h_A = new float[numsElements];
    auto h_B = new float[numsElements];
    auto h_C = new float[numsElements];

    float *d_A, *d_B, *d_C;

    for(int i = 0;i<numsElements;i++) {
        h_A[i] = static_cast<float>(rand()/RAND_MAX);
        h_B[i] = static_cast<float>(rand()/RAND_MAX);
    }
    
    cudaMalloc(reinterpret_cast<void **>(&d_A), size);
    cudaMalloc(reinterpret_cast<void **>(&d_B), size);
    cudaMalloc(reinterpret_cast<void **>(&d_C), size);

    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (numsElements + threadsPerBlock - 1) / threadsPerBlock;

    auto time_start = std::chrono::high_resolution_clock::now();
    vectoradd<<<blocksPerGrid, threadsPerBlock>>>(d_A,d_B,d_C,numsElements);
    cudaDeviceSynchronize();
    auto time_end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_start);
    std::cout << "耗时(微秒)" << duration.count() << std::endl;

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
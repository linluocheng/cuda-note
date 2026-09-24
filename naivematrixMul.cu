#include <iostream>
#include <cuda_runtime.h>
#include <iomanip>

__global__ void naivematrixMul(float *A, float *B, float *C, int hA, int wB, int K) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    float sum = 0.0f;
    if(i<hA && j<wB) {
        for(int p=0;p<K;++p) {
            sum += A[i*K + p] * B[p*wB + j];
        }
        C[i * wB + j] = sum;
    }
}

int main() {
    // 单个block最多32*32=1024个线程
    int blockSize = 32;
    int hA = 320, wB = 320, K = 320;

    int sizeA = hA * K;
    int sizeB = K * wB;
    int sizeC = hA * wB;

    float *h_A,*h_B,*h_C;
    float *d_A,*d_B,*d_C;

    cudaMallocHost(&h_A, sizeA * sizeof(float));
    cudaMallocHost(&h_B, sizeB * sizeof(float));
    cudaMallocHost(&h_C, sizeC * sizeof(float));

    cudaMalloc(&d_A, sizeA * sizeof(float));
    cudaMalloc(&d_B, sizeB * sizeof(float));
    cudaMalloc(&d_C, sizeC * sizeof(float));

    //赋值
    for(int i = 0;i<sizeA;i++) h_A[i] = 1.0f;
    for(int i = 0;i<sizeB;i++) h_B[i] = 0.0f;

    cudaMemcpy(d_A, h_A, sizeof(float) * sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * sizeB, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(blockSize, blockSize);
    dim3 blocksPerGrid((wB + threadsPerBlock.x - 1) / threadsPerBlock.x, (hA + threadsPerBlock.y - 1) / threadsPerBlock.y);

    //GPU 时间计数
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    naivematrixMul<<<threadsPerBlock,blocksPerGrid>>>(d_A, d_B, d_C, hA, wB, K);
    cudaEventRecord(stop,0);

    //等待gpu事件全部结束
    cudaEventSynchronize(stop);

    cudaMemcpy(h_C, d_C, sizeof(float) * sizeC, cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms,start,stop);

    std::cout << "naivematrixMul(ms)：" << std::fixed << std::setprecision(3) << ms << std::endl;

    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
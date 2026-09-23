// device_query.cu
#include <stdio.h>
#include <cuda_runtime.h> // CUDA运行时头文件

int main() {
    // 1. 获取设备数量
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    
    if (deviceCount == 0) {
        printf("No CUDA-capable device found.\n");
        return -1;
    }

    // 2. 遍历并打印每个设备的信息
    for (int dev = 0; dev < deviceCount; ++dev) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\n=== Device %d: \"%s\" ===\n", dev, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total Global Memory: %.2f GB\n", 
               deviceProp.totalGlobalMem / (1024.0*1024.0*1024.0));
        printf("  Multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Warp Size: %d\n", deviceProp.warpSize);
    }

    return 0;
}
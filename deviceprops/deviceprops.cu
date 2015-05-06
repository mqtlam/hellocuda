#include <cuda.h>
#include <iostream>

using namespace std;

// macros
#define printCudaProperty(PROPERTY) { \
	cout << "\t" #PROPERTY "=" << dev_prop.PROPERTY << endl; \
}

// print cuda device properties
void printCudaDeviceProperties()
{
	int dev_count;
	cudaGetDeviceCount(&dev_count);

	cout << "Number of devices=" << dev_count << endl << endl;

	cudaDeviceProp dev_prop;
	for (int i = 0; i < dev_count; i++)
	{
		cout << "Device " << i << ":" << endl;

		cudaGetDeviceProperties(&dev_prop, i);
		printCudaProperty(name);
		printCudaProperty(totalGlobalMem);
		printCudaProperty(sharedMemPerBlock);
		printCudaProperty(regsPerBlock);
		printCudaProperty(warpSize);
		printCudaProperty(memPitch);
		printCudaProperty(maxThreadsPerBlock);
		printCudaProperty(maxThreadsDim[0]);
		printCudaProperty(maxThreadsDim[1]);
		printCudaProperty(maxThreadsDim[2]);
		printCudaProperty(maxGridSize[0]);
		printCudaProperty(maxGridSize[1]);
		printCudaProperty(maxGridSize[2]);
		printCudaProperty(clockRate);
		printCudaProperty(totalConstMem);
		printCudaProperty(major);
		printCudaProperty(minor);
		printCudaProperty(deviceOverlap);
		printCudaProperty(multiProcessorCount);
		printCudaProperty(kernelExecTimeoutEnabled);
		printCudaProperty(integrated);
		printCudaProperty(canMapHostMemory);
		printCudaProperty(computeMode);
		printCudaProperty(concurrentKernels);
		printCudaProperty(ECCEnabled);
		printCudaProperty(pciBusID);
		printCudaProperty(pciDeviceID);
		printCudaProperty(pciDomainID);
		printCudaProperty(tccDriver);
		printCudaProperty(asyncEngineCount);
		printCudaProperty(unifiedAddressing);
		printCudaProperty(memoryClockRate);
		printCudaProperty(memoryBusWidth);
		printCudaProperty(l2CacheSize);
		printCudaProperty(maxThreadsPerMultiProcessor);
	}
}

// This program prints out the CUDA device properties.
int main(int argc, char* argv[])
{
	printCudaDeviceProperties();
	return 0;
}
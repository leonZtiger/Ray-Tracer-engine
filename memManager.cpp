#include "memManager.h"
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )
void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line) {
	if (result) {
		cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
			file << ":" << line << " '" << func << "' \n";
		// Make sure we call CUDA Device Reset before exiting
		cudaDeviceReset();
		exit(99);
	}
}
	void* memManager::operator new (size_t len) {
		void* ptr;
		checkCudaErrors(cudaMallocManaged(&ptr, len));
		checkCudaErrors(cudaDeviceSynchronize());
		return ptr;
	}
	void memManager::operator delete(void* ptr) {

		checkCudaErrors(cudaDeviceSynchronize());
		cudaFree(ptr);
	}



#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>

using namespace std;

void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);
class memManager
{
public:
	void* operator new(size_t len);
	void operator delete(void* ptr);

};



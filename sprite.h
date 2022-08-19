
#pragma once

#include<string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include"memManager.h"

class buffer;

class buffer :public memManager {

public:

	float* data;
	int size;

	buffer(float *pixels,int length);

};


class sprite;

class sprite : public memManager{

public:

	sprite(std::string file);

	__host__
		int getBytes();
	
	__device__
		int getWidth();
	__device__
		int getHeight();
	
	buffer *rBuff;
	buffer *gBuff;
	buffer *bBuff;

	int width;
	int height;

  
};



#include"sprite.h"
#include <opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<iostream>
#include<string>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "memManager.h"
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )



	buffer::buffer(float *pixels, int length) {
		size = length * sizeof(float);
		checkCudaErrors(cudaMallocManaged((void**)&data,size));
		memcpy(data,pixels, size);
	//	data = new int[length];
		
	//	for (int i = 0; i < length; i++)
	//		data[i] = pixels[i];
	//	cudaMalloc((void**)&data, size);

	//	cudaMemcpy(data,pixels, size,cudaMemcpyHostToDevice);

	}


	sprite::sprite(std::string file) {
	
		cv::Mat image = cv::imread(file,cv::IMREAD_COLOR);
       
		 width = image.cols;
		 height = image.rows;

		 float *r = new float[width * height];
		 float*g = new float[width * height];
		 float* b = new float[width * height];

		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
				r[y * width + x] = (float)pixel[2]/255;
				g[y * width  + x] = (float)pixel[1] / 255;
				b[y * width + x ] = (float)pixel[0] / 255;
			}
		 }    
		rBuff = new buffer(r, width * height);
		gBuff = new buffer(g, width * height);
		bBuff = new buffer(b, width * height);
	}
	__host__ 
		int sprite::getBytes() {
		return sizeof(float) * width * height * 3;
	}
	
	__device__
		int sprite::getWidth() {
		return this->width - 1;
	}
	__device__
		int sprite::getHeight() {
		return this->height - 1;
	}
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include "img.h"
#include <cmath>

using namespace std;

#define NX 32
#define NY 32

static void HandleError(cudaError_t err,
                        const char *file,
                        int line)
{
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err),
            file, line);
        exit(EXIT_FAILURE);
    }
}
#define HANDLE_ERROR( error ) (HandleError( error, __FILE__, __LINE__ ))


__global__ void canny_filter(unsigned int* srcimg,unsigned int* newimg,int width,int height)
{
	unsigned int K[]={2,4,5,4,2,
					   4,9,12,9,4,
					   5,12,15,12,5,
					   4,9,12,9,4,
					   2,4,5,4,2};
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	int j=threadIdx.y+blockDim.y*blockIdx.y;
	int rez=0;
	for(int F=0; F<25;F++){
		int I = i + F%5-2;
		int J = j + F/5-2;
		I<0||I>=height||J<0||J>=width?rez+=0: rez += srcimg[I*width+J]*K[24-F];
	}
	newimg[i*width+j]=rez/159;
}

__global__ void sobel_filter(unsigned int* srcimg,unsigned int* newimg,int width,int height)
{
	const int Gx[]={-1,0,1,
			  -2,0,2,
			  -1,0,1};
	const int Gy[]={-1,-2,-1,
	           0, 0, 0,
			   1, 2, 1};
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	int j=threadIdx.y+blockDim.y*blockIdx.y;
	int rez1=0,rez2=0;
	for(int F=0; F<9;F++){
			int I = i + F%3-1;
			int J = j + F/3-1;
			I<0||I>=height||J<0||J>=width?rez1+=0: rez1 += srcimg[I*width+J]*Gx[8-F];
			I<0||I>=height||J<0||J>=width?rez2+=0: rez2 += srcimg[I*width+J]*Gy[8-F];
	}

	newimg[i*width+j]= __fsqrt_ru(rez1*rez1+rez2*rez2);

}

__global__ void border_filter(unsigned int* srcimg,unsigned int* newimg,int width,int height, unsigned int lower, unsigned int upper)
{
	int i=threadIdx.x+blockDim.x*blockIdx.x;
	int j=threadIdx.y+blockDim.y*blockIdx.y;
	srcimg[i*width+j]<upper&&srcimg[i*width+j]>lower?newimg[i*width+j]=255:newimg[i*width+j]=0;
}


void get_borders(char *source_img, char *border_img, size_t width,size_t height, unsigned int lower, unsigned int upper){
	img source_image(source_img, width, height);
	unsigned int *CPU_source_img = source_image.to_massiv();
	unsigned int *GPU_source_img = source_image.get_copy_in_GPU(), *GPU_filter_img = NULL;
	unsigned int *CPU_filter_img = NULL;
	CPU_filter_img = new unsigned int[source_image.get_size()];
	HANDLE_ERROR( cudaMalloc(&GPU_filter_img,source_image.get_size()) );
	dim3 threads(NX,NY,1);
	dim3 blocks(height%NX==0?height/NX:height/NX+1,width%NY==0?width/NY:width/NY+1);
	canny_filter<<<blocks,threads>>>(GPU_source_img, GPU_filter_img, width, height);
	swap(GPU_source_img, GPU_filter_img);
	sobel_filter<<<blocks,threads>>>(GPU_source_img, GPU_filter_img, width, height);
	swap(GPU_source_img, GPU_filter_img);
	border_filter<<<blocks,threads>>>(GPU_source_img, GPU_filter_img, width, height, lower, upper);
	HANDLE_ERROR( cudaMemcpy(CPU_filter_img,GPU_filter_img,source_image.get_size(),cudaMemcpyDeviceToHost));
	img filter_image(width, height, CPU_filter_img);
	filter_image.save(border_img);
	HANDLE_ERROR( cudaFree(GPU_source_img) );
	HANDLE_ERROR( cudaFree(GPU_filter_img) );
	delete[] CPU_filter_img;
}

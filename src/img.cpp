#include <vector>
#include "img.h"
#include <fstream>
#include <istream>
#include <iterator>
#include <iostream>
#include <cstddef>
#include <cuda.h>
#include <cuda_runtime.h>


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

img::img(const char *filename, std::size_t width, std::size_t height) {
	this->width = width;
	this->height = height;
	GPU_img = NULL;
	std::ifstream is(filename, std::ios::in);
	std::copy(std::istream_iterator<unsigned int>(is), std::istream_iterator<unsigned int>(), std::back_inserter<std::vector<unsigned int> >(massiv));
	is.close();
}
img::img(std::size_t width, std::size_t height,unsigned int *img_mass){
	GPU_img = NULL;
	this->width = width;
	this->height = height;
	massiv.assign(img_mass, img_mass+width*height);
}
img::~img() {
	HANDLE_ERROR( cudaFree(GPU_img) );
}

int img::operator() (int x, int y){
	if (x<width && y<height && x>=0 && y>=0){
		return massiv[y*height+x];
	}
	else return 0;
	return 0;
}

void img::save(const char *filename){
	std::fstream is(filename, std::ios::out);
	for(int i=0; i < massiv.size(); i++){
		is<<massiv[i]<<' ';
		if((i+1)%width==0) is<<'\n';
	}
	is.close();
}
void img::copy_to_GPU(){
	HANDLE_ERROR( cudaFree(GPU_img) );
	HANDLE_ERROR( cudaMalloc(&GPU_img,get_size()) );
	HANDLE_ERROR( cudaMemcpy(GPU_img,to_massiv(),get_size(),cudaMemcpyHostToDevice));
}

unsigned int* img::to_massiv(){
	return &massiv[0];
}

unsigned int* img::get_copy_in_GPU(){
	unsigned int * GPU_copy = NULL;
	HANDLE_ERROR( cudaMalloc(&GPU_copy,get_size()) );
	HANDLE_ERROR( cudaMemcpy(GPU_copy,to_massiv(),get_size(),cudaMemcpyHostToDevice));
	return GPU_copy;
}



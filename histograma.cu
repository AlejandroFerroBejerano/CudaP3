#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <image.h>  
#define Tamaño IMAGE_WIDTH*IMAGE_HEIGTH;
#define RangoColores 256
#define Nbloques 1
#define NThreads 256

__global__ void histograma_kernel(unsigned char *buffer, long size, unsigned int *hist){
	
		/*Buffer de histograma temporal en memoria compartida*/	
	__shared__ unsigned int temp[256];
	temp[threadIdx.x]=0;
	__syncthreads();

	int posicion = threadIdx.x + blockIdx.x * blockDim.x;
	int desplazamiento = blockDim.x * gridDim.x;

	while(posicion < size){
		/*Bloquea la variable de memoria compartida para que no escriban en la misma */		
		atomicAdd(&temp[buffer[posicion]], 1);
		posicion += desplazamiento;
	}
	
	/*Esperamos a que todos lo hilos hayan terminado */
	__syncthreads();
	/*Copiamos nuestro histograma en memoria compartida*/
	atomicAdd( &(hist[threadIdx.x]), temp[treadIdx.x]);
}

int main(void){
	unsigned char *image =(unsigned char*)image;
	unsigned int histograma[RangoColores];
	unsigned char *dev_image;
	unsigned int *dev_histograma;

	long = tamaño IMAGE_WIDTH*IMAGE_HEIGTH;

	cudaMalloc((void**) &dev_image, tamaño);
	cudaMalloc((void**) &dev_histograma, RangoColores * sizeof(long));

	cudaMemcpy(dev_image, image, Tamaño, cudaMemcpyHostToDevice);
	histograma_kernel<<<Nbloques,NThreads>>>(dev_image,tamaño,dev_histograma);
	cudaMemcpy(histograma, &dev_histograma, RangoColores * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i=0; i<tamaño; i++){
		printf("%d\t"histograma[i]);	
	}
}

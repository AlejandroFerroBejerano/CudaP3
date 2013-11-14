#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "image.h"

#define RangoColores 256
#define Nbloques 1
#define NThreads 256

__global__ void histograma_kernel(unsigned char *buffer, long size, unsigned int *hist){
	
		/*Buffer de histograma temporal en memoria compartida*/	
	__shared__ unsigned int temp[RangoColores];
	temp[threadIdx.x]=0;
	__syncthreads();

	int posicion = threadIdx.x + blockIdx.x * blockDim.x;
	int desplazamiento = blockDim.x * gridDim.x;

	while(posicion < size){
		/*Bloquea la variable de memoria compartida para que no escriban en la misma */		
		atomicAdd(&temp[buffer[posicion]], 1);
		posicion +=desplazamiento;
	}
	
	/*Esperamos a que todos lo hilos hayan terminado */
	__syncthreads();
	/*Copiamos nuestro histograma en memoria compartida*/
	atomicAdd( &(hist[threadIdx.x]), temp[threadIdx.x]);
}

int main(void){
	/*Cargamos la imagen*/
	unsigned char *img =(unsigned char*)image;

	/*Declaramos el array histograma y los punteros a la imagen y al histograma en memoria*/
	unsigned int histograma[RangoColores];
	unsigned char *dev_image;
	unsigned int *dev_histograma;

	/*Calculamos la longitud máxima en memoria que ocupa la imagen*/
	long size = IMAGE_WIDTH * IMAGE_HEIGHT;
	
	/*Reservamos memoria e inicializamos a 0 
	todo el rango donde se almacenara el histograma*/
	cudaMalloc((void**) &dev_image, size);
	cudaMalloc((void**) &dev_histograma, RangoColores * sizeof(int));
	cudaMemset( dev_histograma, 0,RangoColores * sizeof( int ) );

	cudaMemcpy(dev_image, img, size, cudaMemcpyHostToDevice);
	histograma_kernel<<<Nbloques,NThreads>>>(dev_image,size,dev_histograma);
	cudaMemcpy(histograma, dev_histograma, RangoColores * sizeof(int), cudaMemcpyDeviceToHost);
	
	/*Comprobamos uqe el cálculo se ha hecho correctamente haciendo
	la operación inversa con la CPU*/
	for (int i=0; i<size; i++) histograma[img[i]]--;
	for(int i=0; i< RangoColores; i++){
		if (histograma[i] !=0) printf("Error en %d  Valor %d\n ", i, histograma[i]);
	}
	
	cudaFree(dev_image);
	cudaFree(dev_histograma);
	
	return 0;
}

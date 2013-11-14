#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "image.h"

#define RangoColores 256
#define Nbloques 16
#define NThreads 256

__global__ void histograma_kernel(unsigned char *buffer, long size, unsigned int *hist){
	
		/*Buffer de histograma temporal en memoria compartida*/	
	__shared__ unsigned int temp[NThreads * RangoColores];
	for (int i=0; i< 64; i++){
		temp[threadIdx.x + RangoColores * i]=0;		
	}
	__syncthreads();

	int posicion = threadIdx.x + blockIdx.x * blockDim.x;
	int desplazamiento = blockDim.x * gridDim.x;
	int i=0;
	for(int pos = posicion; pos < size; pos += desplazamiento){
		atomicAdd(&temp[buffer[pos]+ 4096*i], 1);
		i++;
	}

			
	/*Esperamos a que todos lo hilos hayan terminado */
	__syncthreads();
	for (int i=0; i < 512; i++){
		atomicAdd( &(hist[threadIdx.x + 4096 * i]), temp[threadIdx.x + RangoColores * i]);	
	}
		__syncthreads();
}


int main(void){
	/*Cargamos la imagen*/
	unsigned char *img =(unsigned char*)image;

	/*Declaramos el array histograma, los punteros a la imagen y al histograma en memoria*/
	unsigned int parcial_histograma[RangoColores * Nbloques], histograma[RangoColores];
	unsigned char *dev_image;
	unsigned int *dev_parcialHistograma;

	/*Calculamos la longitud máxima en memoria que ocupa la imagen*/
	long size = IMAGE_WIDTH * IMAGE_HEIGHT;
	
	/*Reservamos memoria e inicializamos a 0 
	todo el rango donde se almacenara el histograma*/
	cudaMalloc((void**) &dev_image, size);
	cudaMalloc((void**) &dev_parcialHistograma, RangoColores * Nbloques * sizeof(int));
	cudaMemset( dev_parcialHistograma, 0,RangoColores * Nbloques * sizeof( int ) );

	cudaMemcpy(dev_image, img, size, cudaMemcpyHostToDevice);
		histograma_kernel<<<Nbloques,NThreads>>>(dev_image,size,dev_parcialHistograma);
	cudaMemcpy(parcial_histograma, dev_parcialHistograma, RangoColores * Nbloques * sizeof(int), cudaMemcpyDeviceToHost);
	
	/*Calculamos las sumas parciales*/
	for (int i=0; i < RangoColores; i++){
		for (int j=0; j< Nbloques; j++){
			histograma[i] += parcial_histograma[RangoColores * j + i];
		}
	}

		/*Comprobamos uqe el cálculo se ha hecho correctamente haciendo
	la operación inversa con la CPU*/
	for (int i=0; i<size; i++) histograma[img[i]]--;
	for(int i=0; i< RangoColores; i++){
		if (histograma[i] !=0) printf("Error en %d  Valor %d\n ", i, histograma[i]);
	}

	
	cudaFree(dev_image);
	cudaFree(dev_parcialHistograma);
	
	return 0;
}

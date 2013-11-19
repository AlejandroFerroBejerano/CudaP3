/*autor: Alejandro Ferro Bejerano*/

#include <stdio.h>
#include "image.h"

#define TRUE 1
#define FALSE 0
#define SIZE IMAGE_WIDTH * IMAGE_HEIGHT
#define RangoColores 256
#define Nbloques 1 
#define NThreads 256

__global__ void histograma_kernel(unsigned char *img, unsigned int *hist){
	
	/*Inicializamos histograma temporal en memoria compartida a 0*/	
	__shared__ unsigned int temp[RangoColores];
	temp[threadIdx.x]=0;
	__syncthreads();

	int posicion = threadIdx.x + blockIdx.x * blockDim.x;
	int desplazamiento = blockDim.x * gridDim.x;

	while(posicion < SIZE){
		/*Bloquea la variable de memoria compartida para que no escriban en la misma */		
		atomicAdd(&temp[img[posicion]], 1);
		posicion +=desplazamiento;
	}
	
	/*Esperamos a que todos lo hilos hayan terminado */
	__syncthreads();
	/*Copiamos de nuestra memoria compartida a nuestro histograma*/
	atomicAdd( &(hist[threadIdx.x]), temp[threadIdx.x]);
}

int main(void){

	int hist_correcto = FALSE;
	/*Cargamos la imagen*/
	unsigned char *img =(unsigned char*)image;

	/*Declaramos el array histograma y los punteros a la imagen y al histograma en memoria*/
	unsigned int histograma[RangoColores];
	unsigned char *dev_image;
	unsigned int *dev_histograma;


	/*Reservamos memoria e inicializamos a 0 
	todo el rango donde se almacenara el histograma*/
	cudaMalloc((void**) &dev_image, SIZE);
	cudaMalloc((void**) &dev_histograma, RangoColores * sizeof(int));
	cudaMemset( dev_histograma, 0,RangoColores * sizeof( int ) );

	cudaMemcpy(dev_image, img, SIZE, cudaMemcpyHostToDevice);
	histograma_kernel<<<Nbloques,NThreads>>>(dev_image,dev_histograma);
	cudaMemcpy(histograma, dev_histograma, RangoColores * sizeof(int), cudaMemcpyDeviceToHost);
	
		/*Comprobamos uqe el cálculo se ha hecho correctamente haciendo
	la operación inversa con la CPU*/
	for (int i=0; i<SIZE; i++) histograma[img[i]]--;
	for(int i=0; i< RangoColores; i++){
		if (histograma[i] !=0){
			printf("\nError: El cálculo del histograma, no corresponde con el generado por la CPU\n\n");
			hist_correcto = FALSE;
			exit(-1);
		}else{
			hist_correcto = TRUE;
		}
	}
	if(hist_correcto == TRUE) printf("Histograma generado correctamente, ;-)\n\n");

	
	cudaFree(dev_image);
	cudaFree(dev_histograma);
	
	return 0;
}

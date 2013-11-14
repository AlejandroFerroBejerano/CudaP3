
all: histograma histograma_multiblocks

histograma: histograma.cu
	    nvcc -arch=sm_20 $ -o $@ $^

histograma_multiblocks: histograma_multiblocks.cu
	    		nvcc -arch=sm_20 $ -o $@ $^

clean:
	$(RM) *~ *.o histograma

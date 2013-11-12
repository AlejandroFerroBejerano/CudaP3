
all: histograma

histograma: histograma.cu
	    nvcc -arch=sm_20 $ -o $@ $^

clean:
	$(RM) *~ *.o histograma

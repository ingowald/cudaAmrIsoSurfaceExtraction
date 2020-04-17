all: iso

iso: iso.cu
	nvcc -O3 $< -o $@


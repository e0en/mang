
all: libcudamat.so libcudamat_conv.so

libcudamat.so: cudamat.cu cudamat_kernels.cu cudamat.cuh cudamat_kernels.cuh
	nvcc -O3 \
		-v \
		--compiler-options '-fPIC' -o libcudamat.so \
		--shared cudamat.cu cudamat_kernels.cu -lcublas -lcudart

libcudamat_conv.so: cudamat_conv.cu cudamat_conv_kernels.cu cudamat_conv.cuh cudamat.cuh cudamat_conv_kernels.cuh
	nvcc -O3 \
		-v \
		--compiler-options '-fPIC' -o libcudamat_conv.so \
		--shared cudamat_conv.cu cudamat_conv_kernels.cu -lcublas -lcudart
	
clean:
	rm -rf *.so

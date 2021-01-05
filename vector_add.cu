#include<stdio.h>

__global__ void parallel_vector_add(int *d_a, int *d_b, int *d_c, int *d_n){
	int i = (blockIdx.x*blockDim.x)+threadIdx.x ;
	printf("I am thread #%d\n", i) ;
	if(i < *d_n){
		printf("T am about to compute c[%d].\n", i) ;
		d_c[i] = d_a[i] + d_b[i] ;
	}
	else{
		printf("I am doing nothing.\n") ;
	}
}

int main(){


	// daclare input and output in host
	int n ;
	scanf("%d", &n) ;
	int h_a[n] ;
	int h_b[n] ;
	int h_c[n] ;


	for(int i=0; i<n; i++){
		h_a[i] = i ;
		h_b[i] = n-i ;
	}


	// copy data from host to device
	int *d_a ;
	int *d_b ;
	int *d_c ;
	int *d_n ;
	cudaMalloc((void **) &d_a, n*sizeof(int));
	cudaMalloc((void **) &d_b, n*sizeof(int));
	cudaMalloc((void **) &d_c, n*sizeof(int));
	cudaMalloc((void **) &d_n, sizeof(int));

	// timing
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaMemcpy(d_a, &h_a, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &h_b, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_n, &n, sizeof(int), cudaMemcpyHostToDevice);

	// kernel launch
	int numBlock = n/1024 ;
	if(n%1024){
		numBlock++ ;
	}
	cudaEventRecord(start);
	parallel_vector_add<<<numBlock, 1024>>>(d_a, d_b, d_c, d_n) ;
	/*
	parallel_vector_add<<<n, 1>>>(d_a, d_b, d_c, d_n) ;
	
	^
	|
	|
	use 1 thread per block 
	test speed for n = 100000
	if we use 1024 thread per block it use time about 3021 ms
	but if we use 1 thread per block it use time about 416 ms 
	 */
	cudaEventRecord(stop);
	
	// copy data from device back to host and free
	cudaMemcpy(&h_c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);


	for(int i = 0; i<n; i++){
		printf("%d ", h_c[i]);
	}

	printf("\ntime used = %f\n", milliseconds);
}

#include <stdio.h>
#include <math.h>

#define N 10000

__global__ void inclusive_scan(int *d_in){

    __shared__ int temp_in[N] ;
    int i = threadIdx.x ;
    temp_in[i] = d_in[i];
    __syncthreads() ;

    for (unsigned int s = 1; s <= N-1; s <<= 1){
        if ((i >= s) && (i < N)){
            int a = temp_in[i] ;
            int b = temp_in[i-s] ;
            __syncthreads() ;
            int c = a+b ;
            temp_in[i] = c ;
        }
        __syncthreads();
    }
    d_in[i] = temp_in[i] ;
}

__global__ void exclusive_scan(int *d_in){

    __shared__ int temp_in[N] ;
    int i = threadIdx.x ;
    temp_in[i] = d_in[i];
    __syncthreads() ;

    int s = 1 ;
    for (; s <= N-1; s <<= 1){
        int i = 2*s*(threadIdx.x+1)-1 ;
        if ((i-s >= 0) && (i < N)){
            int a = temp_in[i] ;
            int b = temp_in[i-s] ;
            __syncthreads() ;
            temp_in[i] = a+b ;
        }
        __syncthreads() ;
    }
    d_in[i] = temp_in[i] ;


    if (threadIdx.x == 0 ){
        d_in[N-1] = 0 ;
        temp_in[N-1] = 0 ;
    }
    for (s = s/2; s >= 1; s >>= 1){
        int i = 2*s*(threadIdx.x+1)-1 ;
        if((i-s >= 0) && (i < N)){
            int r = temp_in[i] ;
            int l = temp_in[i-s] ;
            __syncthreads() ;
            temp_in[i] = l+r ;
            temp_in[i-s] = r ;
            __syncthreads() ;
        }
        __syncthreads() ;
    }
    
    d_in[i] = temp_in[i];

}

int main(){

    int h_in[N];
	int h_out[N];

	for(int i=0; i < N; i++)
		h_in[i] = 1;

	int *d_in;
	int *d_out;

	cudaMalloc((void**) &d_in, N*sizeof(int));
	cudaMalloc((void**) &d_out, N*sizeof(int));
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;

	
    int numblock = N/1024 ;
    if (N%1024){
        numblock++ ;
    }
	//Implementing kernel call
    cudaEventRecord(start) ;
    exclusive_scan<<<numblock, 1024>>>(d_in) ;
    cudaEventRecord(stop) ;

    cudaEventSynchronize(stop) ;
	float millisec = 0 ;
	cudaEventElapsedTime(&millisec, start, stop) ;


	cudaMemcpy(&h_out, d_in, N*sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);

	for(int i=0; i<N; i++)
		printf("out[%d] =  %d\n", i, h_out[i]); 


    printf("Time used: %f\n", millisec) ;
	return -1;
}
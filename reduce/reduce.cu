#include <stdio.h>
#include <math.h>

#define BLOCK_SIZE 1024
#define N 2048



__global__ void interleaved_reduce(int *d_in, int *d_out){
	int i = (blockIdx.x*blockDim.x)+threadIdx.x ;
	int M = ceilf(N/2.0f) ;
	__shared__ int shareMem[N] ;
	shareMem[i] = d_in[i] ;
	__syncthreads() ;
	for(int s=1; s<=N; s=s*2){
		if(i<M && i){
			//printf("s = %d, thread = %d\n", s, i) ;
			shareMem[(2*s)*i] = d_in[(2*s)*i] + d_in[(2*s)*i+s] ;
		}
		__syncthreads() ;
		M = ceilf(M/2.0f) ;
	}
	if(i == 0){
		d_out[0] = shareMem[0] ;
	}
}

__global__ void contiguous_reduce(int *d_in, int *d_out){
	int i = (blockIdx.x*blockDim.x)+threadIdx.x ;
	int M = ceilf(N/2.0f) ;
	__shared__ int shareMem[N] ;
	shareMem[i] = d_in[i] ;
	__syncthreads() ;
	for(int s=M; s>0; s=s/2){
		if(i<M){
			//printf("s = %d, thread = %d\n", s, i) ;
			shareMem[i] = shareMem[i] + shareMem[i+s] ;
		}
		__syncthreads() ;
		M = ceilf(M/2.0f) ;
	}
	if(i == 0){
		d_out[0] = shareMem[0] ;
	}	
}

__global__ void setShareMem(int *d_in){
	int i = (blockIdx.x*blockDim.x)+threadIdx.x ;
	//shareMem[i] = d_in[i] ;
}


int main(){
	int h_in[N] ;
	int h_out ;

	for(int i=0; i<N; i++){
		h_in[i] = 1 ;
		//shareMem[i] = 1 ;
	}

	int *d_in, *d_out ;

	cudaMalloc((void**) &d_in, N*sizeof(int)) ;
	cudaMalloc((void**) &d_out, N*sizeof(int)) ;
	cudaMemcpy(d_in, &h_in, N*sizeof(int), cudaMemcpyHostToDevice) ;

	cudaEvent_t start, stop ;
	cudaEventCreate(&start) ;
	cudaEventCreate(&stop) ;

	int numBlock = N/(BLOCK_SIZE*2) ;
	if(N%(BLOCK_SIZE*2)){
		numBlock++ ;
	}
	printf("numBlock=%d\n", numBlock) ;

	//setShareMem<<<numBlock, BLOCK_SIZE>>>(d_in) ;

	cudaEventRecord(start) ;
	interleaved_reduce<<<1, BLOCK_SIZE>>>(d_in, d_out) ;
	//contiguous_reduce<<<numBlock, BLOCK_SIZE>>>(d_in, d_out) ;
	cudaEventRecord(stop) ;

	cudaEventSynchronize(stop) ;
	float millisec = 0 ;
	cudaEventElapsedTime(&millisec, start, stop) ;

	cudaMemcpy(&h_out, d_out, sizeof(int), cudaMemcpyDeviceToHost) ;

	cudaFree(d_in) ;
	cudaFree(d_out) ;

	printf("Output: %d\n", h_out) ;
	printf("Time used: %f\n", millisec) ;
}

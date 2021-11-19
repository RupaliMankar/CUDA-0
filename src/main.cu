#include <iostream>
#include <fstream>
#include <string>
#include "cuda_runtime.h"
#include <errorh.h>

#define N 10

__global__ void kernelInc(float* B, float *A){
int i;
i = blockIdx.x * blockDim.x + threadIdx.x;
	B[i] = A[i]+1.0;
}

__global__ void kernelDec(float* C, float *A){
int i;
i = blockIdx.x * blockDim.x + threadIdx.x;
	C[i] = A[i]-1.0;
}
int main(int argc, char* argv[]){
	int gpuN;													//stores the number of CUDA compatible devices
	float* A = (float*)malloc(N*sizeof(float));
	float* B = (float*)malloc(N*sizeof(float));
	float* C = (float*) malloc(N*sizeof(float));
	//code to take input A from user
//see if user has given any file name otherwise give default file name
	HANDLE_ERROR(cudaGetDeviceCount(&gpuN));							//get the number of devices with compute capability < 1.0
		
	if(gpuN>=2){
		std::cout<<"There are "<<gpuN<<"GPU devices fast execution using two GPUs is possible"<<std::endl;
	
	int sizeA = N * sizeof(float);
    //Setting up device 1
	HANDLE_ERROR(cudaSetDevice(0));
	std::cout<<"executing code on gpu device 1"<<std::endl;
	
	float *g_A0, *g_B;//declare gpu pointers

	HANDLE_ERROR(cudaMalloc(&g_A0, sizeA)); //allocate device 1 memory
	HANDLE_ERROR(cudaMemcpy(g_A0, A, sizeA, cudaMemcpyHostToDevice)); //copy from host to device
	HANDLE_ERROR(cudaMalloc(&g_B, sizeA)); //allocate device 1 memory
	kernelInc<<<(N/2), 2>>>(g_B, g_A0);//launch the kernel on device 1
	HANDLE_ERROR(cudaMemcpy(B, g_B, sizeA, cudaMemcpyDeviceToHost));//copy results from kernel to host
	cudaFree(g_A0); //cuda free memory
	cudaFree(g_B);

	//setting up device 2
	HANDLE_ERROR(cudaSetDevice(1));
	float *g_A1, *g_C;//
	std::cout<<"executing code on gpu device 2"<<std::endl;
	HANDLE_ERROR(cudaMalloc(&g_A1, sizeA));//alloate memory on device 2 shared memory
	HANDLE_ERROR(cudaMemcpy(g_A1, A, sizeA, cudaMemcpyHostToDevice));//copy from host tp device 2 memory
	HANDLE_ERROR(cudaMalloc(&g_C, sizeA));//allocate device memory
	kernelDec<<<(N/2), 2>>>(g_C, g_A1);//launch kernel on device 2
	HANDLE_ERROR(cudaMemcpy(C, g_C, sizeA, cudaMemcpyDeviceToHost));//
	//Free cuda memory
	cudaFree(g_A1),
	cudaFree(g_C);
	}else if(gpuN==1){
		std::cout<<"There is only "<<gpuN<<"GPU device"<<std::endl;
	}
	std::cout<<"incremented array is :"<<B<<std::endl;
	std::cout<<"decremented array is :"<<C<<std::endl;
	free(A);
	free(B);
	free(C) ;
	
}
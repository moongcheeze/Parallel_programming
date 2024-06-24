#include <iostream>
#include <math.h>
#include "mkClockMeasure.h"
#include "mkCuda.h"

#define TILE_WIDTH 16
#define COARSE_FACTOR 4

using namespace std;

const int MAX_ITER = 10;
const int WIDTH = 1000; //matrix 넓이
const float MAX_NUM=100.0; //matrix 값 제한

//input matrix M,N
float inputM[WIDTH*WIDTH];
float inputN[WIDTH*WIDTH];
// output matrix P (cpu / gpu)
float cpuAns[WIDTH*WIDTH];
float gpu1Ans[WIDTH*WIDTH];
float gpu2Ans[WIDTH*WIDTH];
float gpu3Ans[WIDTH*WIDTH];

//matrix 값 생성 함수
void generateRandomValues(float *array, float max, const int size){
	for(int i = 0; i < size; i++){
		array[i] = float(rand())/float(RAND_MAX) * max;
	}
}

//cpu 코드
void cpuCode(float* M, float* N, float* P, int width){
    for (int m = 0; m < width; m++) {
        for (int n = 0; n < width; n++) {
			float Pvalue = 0;
            for (int k = 0; k < width; k++) {
            	Pvalue += M[m * width + k] * N[k * width + n];
            }
			P[m * width  + n] = Pvalue;
        }
    }
}

//gpu 코드 (data locality x)
__global__
void gpuCode3(float* M, float* N, float* P, int width){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    if ((row < width) && (col < width)){
        float Pvalue = 0;
        for (int k = 0; k < width; ++k) {
            Pvalue += M[row*width + k] * N[k*width + col];
        }
        P[row*width + col] = Pvalue;
    }
}


__global__
void gpuCode5(float* M, float* N, float* P, int width){

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by*TILE_WIDTH + ty;
    int Col = bx*TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph=0; ph < ceil(width/(float)TILE_WIDTH); ++ph){
        if((Row < width) && (ph*TILE_WIDTH + tx) < width){
            Mds[ty][tx] = M[Row * width + ph*TILE_WIDTH + tx];
        }
        else{
            Mds[ty][tx] = 0.0f;
        }
        if((ph*TILE_WIDTH + ty) < width && Col < width){
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + Col];
        }
        else{
            Nds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }
    if((Row < width) && (Col < width)){
        P[Row*width + Col] = Pvalue;
    }
}

__global__
void gpuCode6(float* M, float* N, float* P, int width){

    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by*TILE_WIDTH + ty;
    int colStart = bx * TILE_WIDTH * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    for(int c = 0; c < COARSE_FACTOR; ++c){
        Pvalue[c] = 0.0f;
    }   
    for(int ph = 0; ph <  ceil(width/(float)TILE_WIDTH); ++ph){
        if((row < width) && (ph*TILE_WIDTH + tx) < width){
            Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];
        }
        else{
            Mds[ty][tx] = 0.0f;
        }
        
        
        for (int c = 0; c < COARSE_FACTOR; ++c){
            int col = colStart + c*TILE_WIDTH;

            if((ph*TILE_WIDTH + ty) < width && col < width){
                Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + col];
            }
            else{
            Nds[ty][tx] = 0.0f;
            }
           
            
            __syncthreads();
            

            for(int k = 0; k < TILE_WIDTH; ++k){
                Pvalue[c] += Mds[ty][k] * Nds[k][tx];
            }
            __syncthreads();
        }
    }
    for (int c = 0; c < COARSE_FACTOR; ++c){
        int col = colStart + c*TILE_WIDTH;
        if((row < width) && (col < width)){
            P[row*width + col] = Pvalue[c];
        }
    }
    

}


//matrix multiplication 결과 검사 함수
void checkAnswer(float *h_a, float *d_a, const int size){
	bool isSame = true;
	for(int i = 0; i < size*size; i++){
		if(fabsf(h_a[i] - d_a[i]) > 1){
			cout<<"-\tERROR: IDX - "<< i << " (" << h_a[i] << " != " << d_a[i] << " )" << endl;
			isSame = false;
		}
	}
	if(isSame)
		printf("All values are same\n");
	else
		printf("Some values are not same\n");
}


// 메인 함수
int main(){
    //create random value of matrix M,N
    generateRandomValues(inputM, MAX_NUM, WIDTH*WIDTH);
	generateRandomValues(inputN, MAX_NUM, WIDTH*WIDTH);


    //cudaMalloc
    float *d_M, *d_N, *d_P1, *d_P2, *d_P3;
    int arraySize = WIDTH * WIDTH * sizeof(float);
    cudaError_t err = cudaMalloc((void **) &d_M, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_N, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_P1, arraySize);
	checkCudaError(err);    
    err = cudaMalloc((void **) &d_P2, arraySize);
	checkCudaError(err);    
    err = cudaMalloc((void **) &d_P3, arraySize);
	checkCudaError(err);    


    //cudaMemcpy
    err = cudaMemcpy(d_M, inputM, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(d_N, inputN, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);


    //dim3 define
    const int tSize = 16;
	dim3 blockSize(tSize, tSize, 1);
	dim3 gridSize(ceil((float)WIDTH/tSize), ceil((float)WIDTH/tSize), 1);

    //clock define
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU");
	mkClockMeasure *ckGpu1 = new mkClockMeasure("GPU");
    mkClockMeasure *ckGpu2 = new mkClockMeasure("GPU");
    mkClockMeasure *ckGpu3 = new mkClockMeasure("GPU");

	ckCpu->clockReset();
	ckGpu1->clockReset();
    ckGpu2->clockReset();
    ckGpu3->clockReset();


    //Kernel 실행 
    for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuCode(inputM, inputN, cpuAns, WIDTH);
		ckCpu->clockPause();
		
		ckGpu1->clockResume();
		gpuCode3<<<gridSize, blockSize>>>(d_M, d_N, d_P1, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu1->clockPause();
		checkCudaError(err);

        ckGpu2->clockResume();
		gpuCode5<<<gridSize, blockSize>>>(d_M, d_N, d_P2, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu2->clockPause();
		checkCudaError(err);

        ckGpu3->clockResume();
		gpuCode6<<<gridSize, blockSize>>>(d_M, d_N, d_P3, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu3->clockPause();
		checkCudaError(err);

	}

    // Memcpy
	err = cudaMemcpy(gpu1Ans, d_P1, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpu2Ans, d_P2, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

    err = cudaMemcpy(gpu3Ans, d_P3, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

    //cudaFree
	cudaFree(d_M);
	cudaFree(d_N);
	cudaFree(d_P1);
    cudaFree(d_P2);
    cudaFree(d_P3);

    //checking Ansewer
	checkAnswer(cpuAns, gpu1Ans, WIDTH);
    checkAnswer(cpuAns, gpu2Ans, WIDTH);
    checkAnswer(cpuAns, gpu3Ans, WIDTH);

    printf("CPU ");
	ckCpu->clockPrint();
    printf("GPU03 ");
	ckGpu1->clockPrint();
    printf("GPU05 ");
    ckGpu2->clockPrint();
    printf("GPU06 ");
    ckGpu3->clockPrint();

	return 0;
}
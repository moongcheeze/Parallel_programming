#include <iostream>
#include <math.h>
#include <stdlib.h> //srand, rand를 사용하기 위한 헤더파일
#include <time.h> //time 사용하기 위한 헤더파일 
#include "mkClockMeasure.h"
#include "mkCuda.h"


using namespace std;



const int MAX_ITER = 10;
const int MAX_NUM = 10; // use to create matrix random value
const int FILTER_RADIUS = 2; // filter radius
const int WIDTH = 3000; //matrix size

const int tSize = 32;
const int IN_TILE_DIM = 32;
const int OUT_TILE_DIM =((IN_TILE_DIM)-2*(FILTER_RADIUS));


int InputN[WIDTH*WIDTH];
int Filter[(2*FILTER_RADIUS + 1)*(2*FILTER_RADIUS + 1)];

int cpuAns[WIDTH*WIDTH];
int gpu1Ans[WIDTH*WIDTH];
int gpu2Ans[WIDTH*WIDTH];
int gpu3Ans[WIDTH*WIDTH];
int gpu4Ans[WIDTH*WIDTH];

//define filter using constant memory
__constant__ int F_c[(2*FILTER_RADIUS + 1)*(2*FILTER_RADIUS + 1)];


//matrix 값 생성 함수
void generateRandomValues(int *array, int max, const int size){
    srand(time(NULL)); // 난수 초기화
	for(int i = 0; i < size; i++){
		array[i] = rand() % max + 1;
	}
}

//matrix multiplication 결과 검사 함수
void checkAnswer(int *h_a, int *d_a, const int size){
	bool isSame = true;
	for(int i = 0; i < size; i++){
		if(h_a[i] != d_a[i]){
			cout<<"-\tERROR: IDX - "<< i << " (" << h_a[i] << " != " << d_a[i] << " )" << endl;
			isSame = false;
		}
	}
	if(isSame)
		printf("All values are same\n");
	else
		printf("Some values are not same\n");
}


//cpu 
void convolution_cpu(int *N, int *F, int *P, int r, int width){
    for(int outRow = 0; outRow < width; outRow++){
        for(int outCol = 0; outCol < width; outCol++){
            int Pvalue = 0;
            for(int fRow = 0; fRow < 2*r + 1; fRow++){
                for(int fCol = 0; fCol < 2*r + 1; fCol++){
                    int inRow = outRow - r + fRow;
                    int inCol = outCol - r + fCol;
                    if(inRow >= 0 && inRow < width && inCol >= 0 && inCol < width){
                        Pvalue += F[fRow * (2*r + 1) + fCol] * N[inRow*width + inCol];
                    }
                }
            }
            P[outRow*width + outCol] = Pvalue; 
        }
    }
}


//basic algorithm of 2D convolution
__global__
void convolution_ver1(int *N, int *F, int *P,int r,int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    
    if (col < width && row < width) { //thread boundary condition
        int Pvalue = 0;
        for (int i = 0; i < (2*r+1); i++) {
            for (int j = 0; j < (2*r+1); j++) {
                int inputRow = row - r + i;
                int inputCol = col - r + j;
                if (inputRow >= 0 && inputRow < width && inputCol >= 0 && inputCol < width) {
                    Pvalue += N[inputRow * width + inputCol] * F[i * (2*r+1) + j];               
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}


//algorithm of 2D convolution using constant memory F_c
__global__ void convolution_ver2(int *N, int *P, int r, int width){
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;


    if (col < width && row < width) { //thread boundary condition
        int Pvalue = 0;
        for (int i = 0; i < (2*r+1); i++) {
            for (int j = 0; j < (2*r+1); j++) {
                int inputRow = row - r + i;
                int inputCol = col - r + j;
                if (inputRow >= 0 && inputRow < width && inputCol >= 0 && inputCol < width) {
                    Pvalue += N[inputRow * width + inputCol] * F_c[i * (2*r+1) + j];               
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}


__global__ void convolution_ver3(int *N, int *P, int r, int width)
{
 
    int Pvalue = 0;

    __shared__ int N_s[IN_TILE_DIM][IN_TILE_DIM];

    int n_row = blockIdx.y * OUT_TILE_DIM + threadIdx.y;
    int n_col = blockIdx.x * OUT_TILE_DIM + threadIdx.x;

    int m_row = n_row - r;
    int m_col = n_col - r;
 
    if(m_row >= 0 && m_row < width && m_col >= 0 && m_col < width)
    {
        N_s[threadIdx.y][threadIdx.x] = N[m_row * width + m_col];
    }
    else
    {
        N_s[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    if(threadIdx.y < OUT_TILE_DIM && threadIdx.x < OUT_TILE_DIM && n_row < width && n_col < width)
    {
        for(int i = 0; i < (2*r + 1); ++i)
        {
            for(int j = 0; j < (2*r + 1); ++j)
            {
                Pvalue += F_c[i * (2*r + 1) + j] * N_s[threadIdx.y + i][threadIdx.x + j];
            }
        }
        P[n_row * width + n_col] = Pvalue;
    }
}



__global__ void convolution_ver4(int *N, int *P, int r, int width){

    int col = blockIdx.x *  blockDim.x  + threadIdx.x;
    int row = blockIdx.y *  blockDim.y  + threadIdx.y;

    __shared__ int N_s[tSize][tSize]; //tSize = blockDim.x

    if(row < width &&col < width){
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    }else{
        N_s[threadIdx.y][threadIdx.x] = 0;
    }

    __syncthreads();

    if(row < width && col < width){
        int Pvalue = 0;
        for (int fRow = 0; fRow < 2*r + 1 ; fRow++){
            for(int fCol = 0; fCol < 2*r + 1; fCol++){

                int tile_x = threadIdx.x - r + fCol;
                int tile_y = threadIdx.y - r + fRow;

                
                if( tile_x >= 0 && tile_x <tSize && tile_y >= 0 && tile_y < tSize){ //tile 범위 비교
                    Pvalue += F_c[fRow * (2 * r + 1) + fCol]*N_s[tile_y][tile_x];
                }
                else{
                    if( (row-r+fRow) >= 0 && row-r+fRow < width && (col-r+fCol) >=0 && col-r+fCol <width){ //matrix 범위 비교
                        Pvalue += F_c[fRow * (2 * r + 1) + fCol]*N[(row-r+fRow)*width + col-r+fCol];
                    }
                }
            }
        }
        P[row*width+col] = Pvalue;
    }
}



int main(){
    //create random value of input matrix and filter matrix
    generateRandomValues(InputN, MAX_NUM, WIDTH*WIDTH);
    generateRandomValues(Filter, MAX_NUM, (2*FILTER_RADIUS + 1)*(2*FILTER_RADIUS + 1));

    //cudaMalloc
    int *d_N, *d_F, *d_P1, *d_P2, *d_P3, *d_P4;
    int arraySize = WIDTH * WIDTH * sizeof(int);
    int filterSize = (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(int);
    cudaError_t err = cudaMalloc((void **) &d_N, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_F, filterSize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_P1, arraySize);
	checkCudaError(err);    
    err = cudaMalloc((void **) &d_P2, arraySize);
	checkCudaError(err);    
    err = cudaMalloc((void **) &d_P3, arraySize);
	checkCudaError(err);    
    err = cudaMalloc((void **) &d_P4, arraySize);
	checkCudaError(err);    
    

    //cudaMemcpy (host -> device)
    err = cudaMemcpy(d_N, InputN, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(d_F, Filter, filterSize, cudaMemcpyHostToDevice);
	checkCudaError(err);

    //cudaMemcpy constant memory 
    err = cudaMemcpyToSymbol(F_c, Filter, (2*FILTER_RADIUS+1) * (2*FILTER_RADIUS+1) * sizeof(int));
    checkCudaError(err);


    //dim3 define
	dim3 blockSize(tSize, tSize, 1); //block: (tSize x tSize) threads
	dim3 gridSize(ceil((float)WIDTH/tSize), ceil((float)WIDTH/tSize), 1);
    dim3 gridSizeTiled(ceil((float)WIDTH/OUT_TILE_DIM), ceil((float)WIDTH/OUT_TILE_DIM), 1);

    //clock define
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU");
	mkClockMeasure *ckGpu1 = new mkClockMeasure("GPU");
    mkClockMeasure *ckGpu2 = new mkClockMeasure("GPU");
    mkClockMeasure *ckGpu3 = new mkClockMeasure("GPU");
    mkClockMeasure *ckGpu4 = new mkClockMeasure("GPU");

    //clock reset
	ckCpu->clockReset();
	ckGpu1->clockReset();
    ckGpu2->clockReset();
    ckGpu3->clockReset();
    ckGpu4->clockReset();


    //Kernel 실행 
    for(int i = 0; i < MAX_ITER; i++){

        ckCpu->clockResume();
		convolution_cpu(InputN, Filter, cpuAns, FILTER_RADIUS, WIDTH);
		ckCpu->clockPause();

		ckGpu1->clockResume();
		convolution_ver1<<<gridSize, blockSize>>>(d_N, d_F, d_P1, FILTER_RADIUS, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu1->clockPause();
		checkCudaError(err);

        ckGpu2->clockResume();
		convolution_ver2<<<gridSize, blockSize>>>(d_N, d_P2, FILTER_RADIUS, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu2->clockPause();
		checkCudaError(err);

        ckGpu3->clockResume();
		convolution_ver3<<<gridSizeTiled, blockSize>>>(d_N, d_P3, FILTER_RADIUS, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu3->clockPause();
		checkCudaError(err);    

        ckGpu4->clockResume();
		convolution_ver3<<<gridSize, blockSize>>>(d_N, d_P4, FILTER_RADIUS, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu4->clockPause();
		checkCudaError(err);    
    }

    //cudaMemcpy (device -> host) 
	err = cudaMemcpy(gpu1Ans, d_P1, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpu2Ans, d_P2, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

	err = cudaMemcpy(gpu3Ans, d_P3, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

    err = cudaMemcpy(gpu4Ans, d_P4, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

    //cudaFree
	cudaFree(d_N);
	cudaFree(d_F);
	cudaFree(d_P1);
    cudaFree(d_P2);
    cudaFree(d_P3);
    cudaFree(d_P4);

    //checking Ansewer
    checkAnswer(cpuAns, gpu1Ans, WIDTH*WIDTH);
    checkAnswer(cpuAns, gpu2Ans, WIDTH*WIDTH);
    checkAnswer(cpuAns, gpu3Ans, WIDTH*WIDTH);
    checkAnswer(cpuAns, gpu4Ans, WIDTH*WIDTH);
    

    printf("CPU ");
	ckCpu->clockPrint();
    printf("GPU01 ");
	ckGpu1->clockPrint();
    printf("GPU02 ");
	ckGpu2->clockPrint();
    printf("GPU03 ");
	ckGpu3->clockPrint();
    printf("GPU04 ");
    ckGpu4->clockPrint();


	return 0;
}
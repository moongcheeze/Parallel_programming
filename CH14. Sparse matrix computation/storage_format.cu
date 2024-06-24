#include <iostream>
#include <math.h>
#include "mkClockMeasure.h"
#include "mkCuda.h"

using namespace std;

const int MAX_ITER = 1;
const int WIDTH = 2048; //matrix width
const int MAX_NUM=100; //matrix maximum value
const float SPARSITY = 0.01; //sparse matrix의 sparsity value (0~1)

//input matrix (sparse x dense)
int sparseMatrix_h[WIDTH*WIDTH];
int denseMatrix_h[WIDTH];


//sparse matrix storage format
//COO storage format
struct COOMatrix{
    int rowIdx[WIDTH*WIDTH] = {0,};
    int colIdx[WIDTH*WIDTH] = {0,};
    int value[WIDTH*WIDTH] = {0,};
    int count = 0;
}cooMatrix;

//CSR stroage format
int rowPtrs[WIDTH+1] = {0,};

//ELL stroage format
int nnzPerRow[WIDTH] = {0,};
int maxNNzPerRow = 0;


//output matrix
int cpuAns[WIDTH];
int gpuAns[WIDTH];
int gpuCOOAns[WIDTH];
int gpuCSRAns[WIDTH];
int gpuELLAns[WIDTH];

//generate random sparse matrix 
void generateSparseMatrix(int *array, int max, const int size, float sparsity) {
    int totalElements = size;
    int nonZeroElements = totalElements * sparsity;

    //모든 matrix element 0으로 초기화
    for (int i = 0; i < totalElements; i++) {
        array[i] = 0;
    }
    
    //sparsity에 따라 matrix value 설정
    for (int i = 0; i < nonZeroElements; i++) {
        int index = rand() % size;
        if(array[index] == 0){ //index 중복 체크
            array[index] = rand() % max + 1;
        }
        else{
            i--;
        }
    }
}

//generate random dense matrix
void generateDenseMatrix(int *array, int max, const int size){
    srand(time(NULL)); // 난수 초기화
	for(int i = 0; i < size; i++){
		array[i] = rand() % max + 1;
	}
}


void generateCOO(int *array, int width){
    int index = 0;
    for(int i=0; i<width; i++){
        for(int j=0; j<width; j++){
            if(array[i*width + j] != 0){
                cooMatrix.rowIdx[index] = i;
                cooMatrix.colIdx[index] = j;
                cooMatrix.value[index] = array[i*width + j];
                index ++;
            }
        }
    }
    cooMatrix.count = index;
}


void generateCSR(int* array, int width, int *row_ptr, int count)
{
    int idx = 0;
    for (int i = 0; i < width; i++) {
        row_ptr[i] = idx;
        for (int j = 0; j < width; j++) {
            if (array[i*width + j] > 0) {
                idx++;
            }
        }
    }
    row_ptr[width] = idx;
    row_ptr[width+1] = count;
}


//ELL storage formating
void generateNNzPerRow(int *array, int width){ 
    for(int i=0; i<width; i++){//row
        nnzPerRow[i]=0;
        for(int j=0; j<width; j++){//col
            if(array[i*width + j] != 0){
                nnzPerRow[i] += 1;
            }
        }
    }
}
int generateMaxNNzPerRow(int *nnzPerRow, int width){
    int max = 0;
    for(int i = 0; i < width; i++){
        if(nnzPerRow[i] > max ){ 
            max = nnzPerRow[i];
        }
    }
    return max;
}




void paddingELL(int *paddingColIdx, int *paddingValue, int *nnzPerRow, int *colIdx, int *value, int maxNNZ, int width){ // used to coo format
    int index = 0; // coo element 접근 index
    for(int row = 0; row < width; row++){
        for(int element = 0; element < maxNNZ; element++){
            if(element < nnzPerRow[row]){
                paddingColIdx[row*maxNNZ + element] = colIdx[index];
                paddingValue[row*maxNNZ + element] = value[index];
                index ++;
            }
            else{ // 0을 패딩
                paddingColIdx[row*maxNNZ + element] = 0;
                paddingValue[row*maxNNZ + element] = 0;
            }
        }
    }
}

void transpositionELL(int *inputColIdx, int *inputValue, int *transposeColIdx, int *transposeValue, int maxNNZ, int width){
    for(int row = 0; row < width; row++){
        for(int element = 0; element < maxNNZ; element++){
            transposeColIdx[element*width + row] = inputColIdx[row*maxNNZ + element];
            transposeValue[element*width + row] = inputValue[row*maxNNZ + element];
        }
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


//cpu 코드
void cpuCode(int* M, int* N, int* P, int width){
    for(int row = 0; row<width; row++){
        int Pvalue = 0;
        for(int col = 0; col < width; col++){
            Pvalue += M[row*width + col] * N[col];
        }
        P[row] = Pvalue;
    }
}



//gpu 코드 (storage format x)
__global__
void gpuCode(int* M, int* N, int* P, int width){
    int row = blockIdx.x*blockDim.x + threadIdx.x;

    if (row < width){ //boundary check
        int Pvalue = 0;
        for (int k = 0; k < width; ++k) {
            Pvalue += M[row*width + k] * N[k];
        }
        P[row] = Pvalue;
    }
}


//SpMV/COO
__global__ 
void gpuCode2(int* rowIdx, int* colIdx, int* value, int* count, int* x, int* y, int width){

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < *count){
        int row = rowIdx[i];
        int column = colIdx[i];
        atomicAdd(&y[row], value[i] * x[column]);
    }

}




//SpMV/CSR
__global__
void gpuCode3(int width, int* colIdx, int* value, int* row_ptr, int* x, int* y){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if(row < width){
        int result = 0;
        for (int i = row_ptr[row]; i < row_ptr[row+1]; ++i){
            result += value[i] * x[colIdx[i]];
        }
        y[row] = result;
    }
}


//SpMV/ELL
__global__ 
void gpuCode4(int width, int* colIdx, int* value, int* maxNNZ, int* x, int* y) {
    int row = blockIdx.x * blockDim.x + threadIdx.x; 
    if (row < width) {
        int result = 0;
        for (int i = 0; i < *maxNNZ; i++) {
            result += value[row+i*width] * x[colIdx[row+i*width]];
        }
        y[row] = result;
    }
}
    

 
int main(){

    //create input matrix
    generateSparseMatrix(sparseMatrix_h, MAX_NUM, WIDTH*WIDTH, SPARSITY);
	generateDenseMatrix(denseMatrix_h, MAX_NUM, WIDTH);
    generateCOO(sparseMatrix_h, WIDTH);
    generateCSR(sparseMatrix_h, WIDTH, rowPtrs, cooMatrix.count);

    rowPtrs[WIDTH+1] = cooMatrix.count;

    //ELL
    generateNNzPerRow(sparseMatrix_h, WIDTH);
    maxNNzPerRow = generateMaxNNzPerRow(nnzPerRow,WIDTH);

    int *paddingColIdx = (int*)calloc(WIDTH * maxNNzPerRow,sizeof(int));
    int *paddingValue = (int*)calloc(WIDTH * maxNNzPerRow,sizeof(int));
    paddingELL(paddingColIdx, paddingValue, nnzPerRow, cooMatrix.colIdx, cooMatrix.value, maxNNzPerRow, WIDTH);

    int *ellColIdx = (int*)calloc(WIDTH * maxNNzPerRow,sizeof(int));
    int *ellValue = (int*)calloc(WIDTH * maxNNzPerRow,sizeof(int));
    transpositionELL(paddingColIdx, paddingValue, ellColIdx, ellValue, maxNNzPerRow, WIDTH);


/*
    for(int i = 0 ; i < WIDTH; i++){
        printf("nnzPerRow[%d]: %d  ", i,nnzPerRow[i]);
    }
    printf("\n");
    printf("maxNNzPerRow: %d\n", maxNNzPerRow);

    for(int i = 0 ; i < WIDTH; i++){
        printf("row[%d]: ", i);
        for(int j = 0; j < maxNNzPerRow; j++){
            printf("%d ", paddingColIdx[i*maxNNzPerRow + j]);
        }
        printf("/ ");
        for(int j = 0; j < maxNNzPerRow; j++){
            printf("%d ", paddingValue[i*maxNNzPerRow + j]);
        }
        printf("\n");
    }

    for(int i = 0 ; i < WIDTH*maxNNzPerRow; i++){
        printf("%d ", ellColIdx[i]);
    }

   
    for(int i = 0; i<cooMatrix.count; i++){
        printf("row[%d],col[%d]: %d", cooMatrix.rowIdx[i],cooMatrix.colIdx[i],cooMatrix.value[i]);
        printf("\n");
    }

    printf("%d\n", cooMatrix.count);
    printf("%d\n", maxNNzPerRow);
*/




    //cudaMalloc
    int *d_sparseMatrix, *d_denseMatrix;
    int *d_outputMatrix, *d_outputMatrixCOO, *d_outputMatrixCSR, *d_outputMatrixELL;
    int *d_cooRowIdx, *d_cooColIdx, *d_cooValue, *d_cooCount;
    int *d_rowPtrs;
    int *d_maxNNzPerRow, *d_ellColIdx, *d_ellValue;

    int matrixSize = WIDTH * WIDTH * sizeof(int);
    int arraySize = WIDTH * sizeof(int);
    int ellarraySize = WIDTH * maxNNzPerRow * sizeof(int);


    cudaError_t err = cudaMalloc((void **) &d_sparseMatrix, matrixSize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_denseMatrix, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_outputMatrix, arraySize);
	checkCudaError(err);    
	err = cudaMalloc((void **) &d_outputMatrixCOO, arraySize);
	checkCudaError(err);    
    err = cudaMalloc((void **) &d_outputMatrixCSR, arraySize);
	checkCudaError(err); 
    err = cudaMalloc((void **) &d_outputMatrixELL, arraySize);
	checkCudaError(err); 
    err = cudaMalloc((void **) &d_cooRowIdx, matrixSize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_cooColIdx, matrixSize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_cooValue, matrixSize);
	checkCudaError(err);   
	err = cudaMalloc((void **) &d_cooCount, sizeof(int));
	checkCudaError(err); 
    err = cudaMalloc((void **) &d_rowPtrs, (WIDTH+1) * sizeof(int));
	checkCudaError(err);  
    err = cudaMalloc((void **) &d_maxNNzPerRow, sizeof(int));
	checkCudaError(err);   
    err = cudaMalloc((void **) &d_ellColIdx, ellarraySize);
	checkCudaError(err);    
    err = cudaMalloc((void **) &d_ellValue, ellarraySize);
	checkCudaError(err);
    

    //cudaMemcpy
    err = cudaMemcpy(d_sparseMatrix, sparseMatrix_h, matrixSize, cudaMemcpyHostToDevice);
	checkCudaError(err);
	err = cudaMemcpy(d_denseMatrix, denseMatrix_h, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);
    err = cudaMemcpy(d_cooRowIdx, cooMatrix.rowIdx, matrixSize, cudaMemcpyHostToDevice);
	checkCudaError(err);
    err = cudaMemcpy(d_cooColIdx, cooMatrix.colIdx, matrixSize, cudaMemcpyHostToDevice);
	checkCudaError(err);
    err = cudaMemcpy(d_cooValue, cooMatrix.value, matrixSize, cudaMemcpyHostToDevice);
	checkCudaError(err);
    err = cudaMemcpy(d_cooCount, &cooMatrix.count, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaError(err);
    err = cudaMemcpy(d_rowPtrs, rowPtrs, (WIDTH+1) * sizeof(int), cudaMemcpyHostToDevice);
	checkCudaError(err);    
    err = cudaMemcpy(d_maxNNzPerRow, &maxNNzPerRow, sizeof(int), cudaMemcpyHostToDevice);
	checkCudaError(err);
    err = cudaMemcpy(d_ellColIdx, ellColIdx, ellarraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);
    err = cudaMemcpy(d_ellValue, ellValue, ellarraySize, cudaMemcpyHostToDevice);
	checkCudaError(err); 


    //dim3 define
    const int tSize = 256;
	dim3 blockSize(tSize, 1, 1);
	dim3 gridSize(ceil((float)WIDTH/tSize), 1, 1);

    const int tSize_coo = 256;
	dim3 blockSize_coo(tSize_coo, 1, 1);
	dim3 gridSize_coo(ceil((float)cooMatrix.count/tSize), 1, 1);

    const int tSize_csr = 256;
	dim3 blockSize_csr(tSize_csr, 1, 1);
	dim3 gridSize_csr(ceil((float)WIDTH/tSize), 1, 1);

    const int tSize_ell = 256;
	dim3 blockSize_ell(tSize_ell, 1, 1);
	dim3 gridSize_ell(ceil((float)WIDTH/tSize), 1, 1);

    //clock define
	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	mkClockMeasure *ckGpu = new mkClockMeasure("GPU CODE");
    mkClockMeasure *ckGpu2 = new mkClockMeasure("GPU COO format CODE");
    mkClockMeasure *ckGpu3 = new mkClockMeasure("GPU CSR format CODE");
    mkClockMeasure *ckGpu4 = new mkClockMeasure("GPU ELL format CODE");

	ckCpu->clockReset();
	ckGpu->clockReset();
    ckGpu2->clockReset();
    ckGpu3->clockReset();
    ckGpu4->clockReset();

    //Kernel 실행 
    for(int i = 0; i < MAX_ITER; i++){
		ckCpu->clockResume();
		cpuCode(sparseMatrix_h, denseMatrix_h, cpuAns, WIDTH);
		ckCpu->clockPause();
		
		ckGpu->clockResume();
		gpuCode<<<gridSize, blockSize>>>(d_sparseMatrix, d_denseMatrix, d_outputMatrix, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);

        ckGpu2->clockResume();
		gpuCode2<<<gridSize_coo, blockSize_coo>>>(d_cooRowIdx, d_cooColIdx, d_cooValue, d_cooCount, d_denseMatrix, d_outputMatrixCOO, WIDTH);
		err=cudaDeviceSynchronize();
		ckGpu2->clockPause();
		checkCudaError(err);

        ckGpu3->clockResume();
		gpuCode3<<<gridSize_csr, blockSize_csr>>>(WIDTH, d_cooColIdx, d_cooValue, d_rowPtrs, d_denseMatrix, d_outputMatrixCSR);
		err=cudaDeviceSynchronize();
		ckGpu3->clockPause();
		checkCudaError(err);

        
        ckGpu4->clockResume();
		gpuCode4<<<gridSize_ell, blockSize_ell>>>(WIDTH, d_ellColIdx, d_ellValue, d_maxNNzPerRow, d_denseMatrix, d_outputMatrixELL);
		err=cudaDeviceSynchronize();
		ckGpu4->clockPause();
		checkCudaError(err);
        
	}

    // Memcpy
	err = cudaMemcpy(gpuAns, d_outputMatrix, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);
    err = cudaMemcpy(gpuCOOAns, d_outputMatrixCOO, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);
    err = cudaMemcpy(gpuCSRAns, d_outputMatrixCSR, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);
    err = cudaMemcpy(gpuELLAns, d_outputMatrixELL, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);

    //cudaFree
	cudaFree(d_sparseMatrix);
	cudaFree(d_denseMatrix);
	cudaFree(d_outputMatrix);
    cudaFree(d_outputMatrixCOO);
    cudaFree(d_outputMatrixCSR);
    cudaFree(d_outputMatrixELL);
    cudaFree(d_cooRowIdx);
	cudaFree(d_cooColIdx);
	cudaFree(d_cooValue);
    cudaFree(d_cooCount);
    cudaFree(rowPtrs);
    cudaFree(d_maxNNzPerRow);
    cudaFree(d_ellColIdx);
    cudaFree(d_ellValue);

    free(paddingColIdx);
    free(paddingValue);
    free(ellColIdx);
    free(ellValue);

    //checking Ansewer
	checkAnswer(cpuAns, gpuAns, WIDTH);
	checkAnswer(cpuAns, gpuCOOAns, WIDTH);
    checkAnswer(cpuAns, gpuCSRAns, WIDTH);
    checkAnswer(cpuAns, gpuELLAns, WIDTH);

	ckCpu->clockPrint();
	ckGpu->clockPrint();
	ckGpu2->clockPrint();
    ckGpu3->clockPrint();
    ckGpu4->clockPrint();


    return 0;
}
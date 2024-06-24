/*
<<To Do>>
1. frontier
2. frontier - shared mem
*/

#include <iostream>
#include <math.h>
#include "mkClockMeasure.h"
#include "mkCuda.h"

using namespace std;

const int MAX_ITER = 1;
const int WIDTH = 1000; //The number of vertices & Width of adjacency matrix 
const int MAX = 10000;
const int local_frontier_capacity = 256;

int adjacencyMatrix[WIDTH*WIDTH] = {0,};


//BFS result Array
int cpuResult[WIDTH];
int gpuResultE[WIDTH];
int gpuResultPush[WIDTH];
int gpuResultPull[WIDTH];
int gpuResultFrontier[WIDTH];
int gpuResultPrivatization[WIDTH];

//sparse matrix storage format
//COO storage format
struct COOMatrix{
    int rowIdx[WIDTH*WIDTH] = {0,};
    int colIdx[WIDTH*WIDTH] = {0,};
    int count = 0;
}cooMatrix;

//CSR stroage format
int rowPtrs[WIDTH+1] = {0,};

//CSC storage format
int dstPtrs[WIDTH+1] = {0,};
int src[WIDTH*WIDTH] = {0,};


typedef struct Queue
{int front, rear;
int data[MAX];
}Queue;



//Create the random direction graph in the adjacency form
void createGraph(int width, int* matrix){

    int vertexCheck[width]; // connected vertex = 1, unconnected vertex = 0
    for(int i = 0; i < width; i++){
        vertexCheck[i] = 0;
    }

    int flag = 0; //모든 vertex가 연결되었는지 확인해주는 flag
    int srcVertex = 0;
    int dstVertex = 0;
    
    while(flag == 0){ //모든 vertex가 연결되기 전까지 loop
        flag = 1;
        for(int f = 0; f < width; f++){ //flag setting 
            if(vertexCheck[f] == 0){
                flag = 0;
                break;
            }
        }
        
        do{ 
            dstVertex = rand() % width;
        }while(srcVertex == dstVertex);
        
        matrix[srcVertex*WIDTH + dstVertex] = 1;
        vertexCheck[srcVertex] = 1;
        srcVertex = dstVertex;        
    }
}


//COO format 
void generateCOO(int *array, int width){
    int index = 0;
    for(int i=0; i<width; i++){
        for(int j=0; j<width; j++){
            if(array[i*width + j] != 0){
                cooMatrix.rowIdx[index] = i;
                cooMatrix.colIdx[index] = j;
                index ++;
            }
        }
    }
    cooMatrix.count = index;
}

//CSR format
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

//CSC format
void generateCSC(int* array, int width, int *dst_ptr, int *src_arr, int count)
{
    int idx = 0;
    for (int i = 0; i < width; i++) {
        dst_ptr[i] = idx;
        for (int j = 0; j < width; j++) {
            if (array[width*j + i] > 0) {
                src_arr[idx] = j;
                idx++;
            }
        }
    }
    dst_ptr[width] = idx;
    dst_ptr[width+1] = count;
}



void cpuBfsCode(int* level, int width, int* matrix) {
    int root = 0;
    int here, there;     
    Queue q;    
    q.front = -1;    
    q.rear = -1;     
    q.data[++q.rear] = root; // push     
    
    //queue 꺼내기    
    while(q.front < q.rear) {        
        here = q.data[++q.front]; // pop               

        //인접행렬 구하기        
        for(int i = 0; i < width; i++) {            
            if(matrix[here*width + i] == 1) {               
                there = i;                 
                if(level[there] == MAX) {                    
                    level[there] = level[here] + 1;                    
                    q.data[++q.rear] = there; // push
                }            
            }        
        }    
    }
} 



//Edge-centric BFS kernel (Using COO storage format)
__global__
void edgeCentricKernel(int* rowIdx, int* colIdx, int* count, int* level, int* newVertexVisted, int* currLevel){
    int edge = blockIdx.x * blockDim.x + threadIdx.x; //Edge에 thread를 부여
    if(edge < *count){
        int vertex = rowIdx[edge];
        if(level[vertex] == *currLevel - 1){
            int neighbor = colIdx[edge];
            if(level[neighbor] == MAX){
                level[neighbor] = *currLevel;
                *newVertexVisted = 1;
            }
        }
    }
}


//Vertex-centric Push BFS Kernel (Using CSR storage format)
__global__
void vertexCentricPushKernel(int* rowPtr, int* colIdx, int* level, int* newVertexVisted, int* currLevel){
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if(vertex < WIDTH){
        if(level[vertex] == *currLevel - 1){
            for(int edge = rowPtr[vertex]; edge <rowPtr[vertex+1]; edge++){
                int neighbor = colIdx[edge];
                if(level[neighbor] == MAX){
                    level[neighbor] = *currLevel;
                    *newVertexVisted = 1;
                }
            }
        }
    }
}

//Vertex-centric Pull BFS Kernel (Using CSC storage format)
__global__
void vertexCentricPullKernel(int* dstPtr, int* srcIdx, int* level, int* newVertexVisted, int* currLevel){
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;

    if(vertex < WIDTH){
        if(level[vertex] == MAX){
            for(int edge = dstPtr[vertex]; edge < dstPtr[vertex+1]; edge++){
                int neighbor = srcIdx[edge];
                if(level[neighbor] == *currLevel - 1){
                    level[vertex] = *currLevel;
                    *newVertexVisted = 1;
                    break;
                }
            }
        }
    }
}

//Vertex-centric Push BFS Kernel (Using CSR storage format) with frontiers
__global__
void frontierKernel(int* rowPtr, int* colIdx, int* level, int* prevFrontier, int* currFrontier, 
                    int* numPrevFrontier, int* numCurrFrontier, int* currLevel){
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < *numPrevFrontier){
        int vertex = prevFrontier[i];
        for(int edge = rowPtr[vertex]; edge < rowPtr[vertex+1]; edge++){
            int neighbor = colIdx[edge];
            if(atomicCAS(&level[neighbor], MAX, *currLevel) == MAX){
                int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                currFrontier[currFrontierIdx] = neighbor;
            }
        }
    }
}

//Vertex-centric Push BFS Kernel (Using CSR storage format) with privatization of frontiers
__global__
void privatizationKernel(int* rowPtr, int* colIdx, int* level, int* prevFrontier, int* currFrontier, 
                    int* numPrevFrontier, int* numCurrFrontier, int* currLevel){
    
    //Initialize privatized frontier 
    __shared__ int currFrontier_s[local_frontier_capacity];
    __shared__ int numCurrFrontier_s;

    if(threadIdx.x == 0){
        numCurrFrontier_s = 0;
    }
    __syncthreads();


    //Perform BFS
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < *numPrevFrontier){
        int vertex = prevFrontier[i];
        for(int edge = rowPtr[vertex]; edge < rowPtr[vertex+1]; edge++){
            int neighbor = colIdx[edge];
            if(atomicCAS(&level[neighbor], MAX, *currLevel) == MAX){
                int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1);
                if(currFrontierIdx_s < local_frontier_capacity){
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                }else{
                    numCurrFrontier_s = local_frontier_capacity;
                    int currFrontierIdx = atomicAdd(numCurrFrontier, 1);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }

    __syncthreads();

    //Allocate in global frontier
    __shared__ int currFrontierStartIdx;
    if(threadIdx.x == 0){
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
    }
    __syncthreads();

    //Commit to global frontier
    for(int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s; currFrontierIdx_s += blockDim.x){
        int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}



void printGraph(int width, int* matrix){
    printf("\n");
    for(int i = 0; i < width; i++){
        for(int j = 0; j < width; j++){
            printf("%d ", matrix[i*width + j]);
        }
        printf("\n");
    }
}

//결과 검사 함수
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







int main(){

createGraph(WIDTH, adjacencyMatrix);
generateCOO(adjacencyMatrix, WIDTH);
generateCSR(adjacencyMatrix, WIDTH, rowPtrs, cooMatrix.count);
generateCSC(adjacencyMatrix, WIDTH, dstPtrs, src, cooMatrix.count);



int newVertexVisted = 1;
int currLevel = 1;

//forntier variable & array
int prevFrontier[WIDTH] = {0,};
int currFrontier[WIDTH] = {0,};
int numPrevFrontier = 1;
int numCurrFrontier = 0;


//Initialize to result arr & Select root vertex
for(int i=0; i<WIDTH; i++){
    cpuResult[i] = MAX; //Sets the value of an unvisited vertex to the MAX value
}
cpuResult[0] = 0; //The value of Root vertex should be zero (start level == 0)

for(int i=0; i<WIDTH; i++){
    gpuResultE[i] = MAX;
}
gpuResultE[0] = 0;

for(int i=0; i<WIDTH; i++){
    gpuResultPush[i] = MAX;
}
gpuResultPush[0] = 0;

for(int i=0; i<WIDTH; i++){
    gpuResultPull[i] = MAX;
}
gpuResultPull[0] = 0;

for(int i=0; i<WIDTH; i++){
    gpuResultFrontier[i] = MAX;
}
gpuResultFrontier[0] = 0;

for(int i=0; i<WIDTH; i++){
    gpuResultPrivatization[i] = MAX;
}
gpuResultPrivatization[0] = 0;





//cudaMalloc
int *d_adjacencyMatrix;
int *d_outputEdgeCentric, *d_outputVertexCentricPush, *d_outputVertexCentricPull, *d_outputFrontier, *d_outputPrivatization;
int *d_cooRowIdx, *d_cooColIdx, *d_cooCount;
int *d_rowPtrs;
int *d_dstPtrs, *d_src;
int *d_currLevel;
int *d_newVertexVisted;
int *d_prevFrontier, *d_currFrontier;
int *d_numPrevFrontier, *d_numCurrFrontier;

int matrixSize = WIDTH * WIDTH * sizeof(int);
int ptrSize = (WIDTH+1) * sizeof(int);
int arraySize = WIDTH * sizeof(int);
    
cudaError_t err = cudaMalloc((void **) &d_adjacencyMatrix, matrixSize);
checkCudaError(err);
err = cudaMalloc((void **) &d_outputEdgeCentric, arraySize);
checkCudaError(err);
err = cudaMalloc((void **) &d_outputVertexCentricPush, arraySize);
checkCudaError(err);
err = cudaMalloc((void **) &d_outputVertexCentricPull, arraySize);
checkCudaError(err);
err = cudaMalloc((void **) &d_outputFrontier, arraySize);
checkCudaError(err);
err = cudaMalloc((void **) &d_outputPrivatization, arraySize);
checkCudaError(err);
err = cudaMalloc((void **) &d_cooRowIdx, matrixSize);
checkCudaError(err);
err = cudaMalloc((void **) &d_cooColIdx, matrixSize);
checkCudaError(err);
err = cudaMalloc((void **) &d_cooCount, sizeof(int));
checkCudaError(err);   
err = cudaMalloc((void **) &d_rowPtrs, ptrSize);
checkCudaError(err);  
err = cudaMalloc((void **) &d_dstPtrs, ptrSize);
checkCudaError(err);  
err = cudaMalloc((void **) &d_src, matrixSize);
checkCudaError(err);
err = cudaMalloc((void **) &d_currLevel, sizeof(int));
checkCudaError(err);   
err = cudaMalloc((void **) &d_newVertexVisted, sizeof(int));
checkCudaError(err);   
err = cudaMalloc((void **) &d_prevFrontier, arraySize);
checkCudaError(err);   
err = cudaMalloc((void **) &d_currFrontier, arraySize);
checkCudaError(err);   
err = cudaMalloc((void **) &d_numPrevFrontier, sizeof(int));
checkCudaError(err);   
err = cudaMalloc((void **) &d_numCurrFrontier, sizeof(int));
checkCudaError(err);   

//cudaMemcpy
err = cudaMemcpy(d_adjacencyMatrix, adjacencyMatrix, matrixSize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_outputEdgeCentric, gpuResultE, arraySize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_outputVertexCentricPush, gpuResultPush, arraySize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_outputVertexCentricPull, gpuResultPull, arraySize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_outputFrontier, gpuResultFrontier, arraySize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_outputPrivatization, gpuResultPrivatization, arraySize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_cooRowIdx, cooMatrix.rowIdx, matrixSize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_cooColIdx, cooMatrix.colIdx, matrixSize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_cooCount, &cooMatrix.count, sizeof(int), cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_rowPtrs, rowPtrs, ptrSize, cudaMemcpyHostToDevice);	
checkCudaError(err); 
err = cudaMemcpy(d_dstPtrs, dstPtrs, ptrSize, cudaMemcpyHostToDevice);	
checkCudaError(err); 
err = cudaMemcpy(d_src, src, matrixSize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_newVertexVisted, &newVertexVisted, sizeof(int), cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_prevFrontier, prevFrontier, arraySize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_currFrontier, currFrontier, arraySize, cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_numPrevFrontier, &numPrevFrontier, sizeof(int), cudaMemcpyHostToDevice);
checkCudaError(err);
err = cudaMemcpy(d_numCurrFrontier, &numCurrFrontier, sizeof(int), cudaMemcpyHostToDevice);
checkCudaError(err);


const int tSize = 256;
dim3 blockSize(tSize, 1, 1);
dim3 gridSize(ceil((float)cooMatrix.count/tSize), 1, 1);
dim3 gridSizeV(ceil((float) WIDTH/tSize), 1, 1);

mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
mkClockMeasure *ckGpuEdegCentric = new mkClockMeasure("GPU Edge Centric CODE");
mkClockMeasure *ckGpuVertexCentricPush = new mkClockMeasure("GPU Vertex Centric Push CODE");
mkClockMeasure *ckGpuVertexCentricPull = new mkClockMeasure("GPU Vertex Centric Pull CODE");
mkClockMeasure *ckGpuFrontier = new mkClockMeasure("GPU Frontier CODE");
mkClockMeasure *ckGpuPrivatization = new mkClockMeasure("GPU Privatization CODE");

ckCpu->clockReset();
ckGpuEdegCentric->clockReset();
ckGpuVertexCentricPush->clockReset();
ckGpuVertexCentricPull->clockReset();
ckGpuFrontier->clockReset();
ckGpuPrivatization->clockReset();

//Kernel 실행 
for(int i = 0; i < MAX_ITER; i++){
    ckCpu->clockResume();
	cpuBfsCode(cpuResult, WIDTH, adjacencyMatrix);
	ckCpu->clockPause();

    //GPU: edge-centric kernel

    while(newVertexVisted == 1){
        newVertexVisted = 0;
        err = cudaMemcpy(d_newVertexVisted, &newVertexVisted, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);
	    ckGpuEdegCentric->clockResume();
        edgeCentricKernel<<<gridSize, blockSize>>>(d_cooRowIdx, d_cooColIdx, d_cooCount, d_outputEdgeCentric, d_newVertexVisted, d_currLevel);
	    err=cudaDeviceSynchronize();
        checkCudaError(err); 
        ckGpuEdegCentric->clockPause();
        currLevel += 1;
        err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        err = cudaMemcpy(&newVertexVisted, d_newVertexVisted, sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaError(err);
    }



    //GPU: vertex-centric push kernel
    currLevel = 1;
    newVertexVisted = 1;
    err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err);
    
    while(newVertexVisted == 1){
        newVertexVisted = 0;
        err = cudaMemcpy(d_newVertexVisted, &newVertexVisted, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        ckGpuVertexCentricPush->clockResume();
        vertexCentricPushKernel<<<gridSizeV, blockSize>>>(d_rowPtrs, d_cooColIdx, d_outputVertexCentricPush, d_newVertexVisted, d_currLevel);
	    err=cudaDeviceSynchronize();
        checkCudaError(err); 
        ckGpuVertexCentricPush->clockPause();

        currLevel += 1;
        err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        err = cudaMemcpy(&newVertexVisted, d_newVertexVisted, sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaError(err);
    }
	
        

    //GPU: vertex-centric pull kernel
    currLevel = 1;
    newVertexVisted = 1;
    err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err);
    
    while(newVertexVisted == 1){
        newVertexVisted = 0;
        err = cudaMemcpy(d_newVertexVisted, &newVertexVisted, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        ckGpuVertexCentricPull->clockResume();
        vertexCentricPullKernel<<<gridSizeV, blockSize>>>(d_dstPtrs, d_src, d_outputVertexCentricPull, d_newVertexVisted, d_currLevel);
	    err=cudaDeviceSynchronize();
        checkCudaError(err);
        ckGpuVertexCentricPull->clockPause();
        
        currLevel += 1;
        err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        err = cudaMemcpy(&newVertexVisted, d_newVertexVisted, sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaError(err);
    }
	
           

    //GPU: frontier kernel
    currLevel = 1;
    err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err);
    while(numPrevFrontier > 0){
        printf("%d ", numPrevFrontier);
        dim3 gridSizeF(ceil((float) numPrevFrontier/tSize), 1, 1);
        ckGpuFrontier->clockResume();
        frontierKernel<<<gridSizeF, blockSize>>>(d_rowPtrs, d_cooColIdx, d_outputFrontier, d_prevFrontier, d_currFrontier, d_numPrevFrontier, d_numCurrFrontier, d_currLevel);
	    err=cudaDeviceSynchronize();
        checkCudaError(err);  
        ckGpuFrontier->clockPause();

        currLevel += 1;
        err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        err = cudaMemcpy(&numPrevFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaError(err);
        err = cudaMemcpy(d_prevFrontier, d_currFrontier, numPrevFrontier * sizeof(int), cudaMemcpyDeviceToDevice);
        checkCudaError(err);
        err = cudaMemcpy(d_numPrevFrontier, &numPrevFrontier, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        numCurrFrontier = 0;

        err = cudaMemcpy(d_numCurrFrontier, &numCurrFrontier, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);
    }

    //GPU: privatization with frontier kernel
    currLevel = 1;
    err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err);

    numPrevFrontier = 1;
    err = cudaMemcpy(d_numPrevFrontier, &numPrevFrontier, sizeof(int), cudaMemcpyHostToDevice);
    checkCudaError(err);
    prevFrontier[0] = 0;
    err = cudaMemcpy(d_prevFrontier, &prevFrontier, arraySize, cudaMemcpyHostToDevice);
    checkCudaError(err);
    while(numPrevFrontier > 0){

        dim3 gridSizeF(ceil((float) numPrevFrontier/tSize), 1, 1);
        ckGpuPrivatization->clockResume();
        privatizationKernel<<<gridSizeF, blockSize>>>(d_rowPtrs, d_cooColIdx, d_outputPrivatization, d_prevFrontier, d_currFrontier, d_numPrevFrontier, d_numCurrFrontier, d_currLevel);
	    err=cudaDeviceSynchronize();
        checkCudaError(err);  
        ckGpuPrivatization->clockPause();

        currLevel += 1;
        err = cudaMemcpy(d_currLevel, &currLevel, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        err = cudaMemcpy(&numPrevFrontier, d_numCurrFrontier, sizeof(int), cudaMemcpyDeviceToHost);
        checkCudaError(err);
        err = cudaMemcpy(d_prevFrontier, d_currFrontier, numPrevFrontier * sizeof(int), cudaMemcpyDeviceToDevice);
        checkCudaError(err);
        err = cudaMemcpy(d_numPrevFrontier, &numPrevFrontier, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);

        numCurrFrontier = 0;

        err = cudaMemcpy(d_numCurrFrontier, &numCurrFrontier, sizeof(int), cudaMemcpyHostToDevice);
        checkCudaError(err);
    }
}


// Result Arr Memcpy (device to host)
err = cudaMemcpy(gpuResultE, d_outputEdgeCentric, arraySize, cudaMemcpyDeviceToHost);
checkCudaError(err);
err = cudaMemcpy(gpuResultPush, d_outputVertexCentricPush, arraySize, cudaMemcpyDeviceToHost);
checkCudaError(err);
err = cudaMemcpy(gpuResultPull, d_outputVertexCentricPull, arraySize, cudaMemcpyDeviceToHost);
checkCudaError(err);
err = cudaMemcpy(gpuResultFrontier, d_outputFrontier, arraySize, cudaMemcpyDeviceToHost);
checkCudaError(err);
err = cudaMemcpy(gpuResultPrivatization, d_outputPrivatization, arraySize, cudaMemcpyDeviceToHost);
checkCudaError(err);


cudaFree(adjacencyMatrix);
cudaFree(d_outputEdgeCentric);
cudaFree(d_outputVertexCentricPush);
cudaFree(d_outputVertexCentricPull);
cudaFree(d_outputFrontier);
cudaFree(d_outputPrivatization);
cudaFree(d_cooRowIdx);
cudaFree(d_cooColIdx);
cudaFree(d_cooCount);
cudaFree(d_rowPtrs);
cudaFree(d_dstPtrs);
cudaFree(src);
cudaFree(d_currLevel);
cudaFree(d_newVertexVisted);



checkAnswer(cpuResult, gpuResultE, WIDTH);
checkAnswer(cpuResult, gpuResultPush, WIDTH);
checkAnswer(cpuResult, gpuResultPull, WIDTH);
checkAnswer(cpuResult, gpuResultFrontier, WIDTH);
checkAnswer(cpuResult, gpuResultPrivatization, WIDTH);

ckCpu->clockPrint();
ckGpuVertexCentricPush->clockPrint();
ckGpuVertexCentricPull->clockPrint();
ckGpuEdegCentric->clockPrint();
ckGpuFrontier->clockPrint();
ckGpuPrivatization->clockPrint();

/*
printGraph(WIDTH, adjacencyMatrix);
printf("\n\n");
for(int i = 0; i < WIDTH; i++){
    printf("%d ",gpuResultE[i]);
}*/

printf("\n%d\n", WIDTH);
printf("%d\n", cooMatrix.count);

return 0;
}

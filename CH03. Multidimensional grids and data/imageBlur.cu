/*
 * =====================================================================================
 *
 *       Filename:  imageBlur.cu
 *
 *    Description:  Ch03 Samples
 *
 *        Version:  1.0
 *        Created:  01/14/2024
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Yi, Ji Yeong, dlwldud1151@naver.com
 *   Organization:  EWHA Womans Unversity
 *
 * =====================================================================================
 */

#include<iostream>
#include "mkPpm.h"
#include "mkCuda.h"
#include "mkClockMeasure.h"

using namespace std;

const int MAX_ITER = 1000;
const int BLUR_SIZE = 4;



void cpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	for(int i = 0; i < w * h * 3; i+=3){ // 배열 돌면서
		int row = (i/3) / w;
		int col = (i/3) % w;  
        int pixRedVal = 0;
		int pixBlueVal = 0;
	    int pixGreenVal = 0;
        int pixels = 0;
		for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
			for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol){
				int curRow = row + blurRow;
                int curCol = col + blurCol;
                if(curRow >= 0 && curRow < h && curCol >= 0 && curCol <w){
                    pixRedVal += inArray[(curRow*w + curCol) * 3];
					pixBlueVal += inArray[(curRow*w + curCol) * 3 + 1];
					pixGreenVal += inArray[(curRow*w + curCol) * 3 + 2];
                    ++pixels;
                }
			}
		}
        outArray[i] = (unsigned char) (pixRedVal/pixels);
		outArray[i + 1] = (unsigned char) (pixBlueVal/pixels);
		outArray[i + 2] = (unsigned char) (pixGreenVal/pixels);
	}
}

__global__
void gpuCode(unsigned char *outArray, const unsigned char *inArray, const int w, const int h){
	int col = threadIdx.x + blockIdx.x * blockDim.x;
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int offset = (row * w + col) * 3;

	if(col < w && row < h){
        int pixRedVal = 0;
		int pixBlueVal = 0;
	    int pixGreenVal = 0;
        int pixels = 0;

        for(int blurRow = -BLUR_SIZE; blurRow < BLUR_SIZE + 1; ++blurRow){
            for (int blurCol = -BLUR_SIZE; blurCol < BLUR_SIZE + 1; ++blurCol){
                int curRow = row + blurRow;
                int curCol = col + blurCol;
                if(curRow >= 0 && curRow < h && curCol >= 0 && curCol <w){
                    pixRedVal += inArray[(curRow*w + curCol) * 3];
					pixBlueVal += inArray[(curRow*w + curCol) * 3 + 1];
					pixGreenVal += inArray[(curRow*w + curCol) * 3 + 2];
                    ++pixels;
                }
            }
        }

        outArray[offset] = (unsigned char) (pixRedVal/pixels);
		outArray[offset + 1] = (unsigned char) (pixBlueVal/pixels);
		outArray[offset + 2] = (unsigned char) (pixGreenVal/pixels);
	}
}

int main(){
	int w, h;
	unsigned char *h_imageArray;
	unsigned char *h_outImageArray;
	unsigned char *d_imageArray;
	unsigned char *d_outImageArray;
	unsigned char *h_outImageArray2;

	ppmLoad("./data/1024.ppm", &h_imageArray, &w, &h);

	size_t arraySize = sizeof(unsigned char) * h * w * 3;

	h_outImageArray = (unsigned char*)malloc(arraySize);
	h_outImageArray2 = (unsigned char*)malloc(arraySize);

	cudaError_t err = cudaMalloc((void **) &d_imageArray, arraySize);
	checkCudaError(err);
	err = cudaMalloc((void **) &d_outImageArray, arraySize);
	checkCudaError(err);

	err = cudaMemcpy(d_imageArray, h_imageArray, arraySize, cudaMemcpyHostToDevice);
	checkCudaError(err);

	const int tSize = 16;
	dim3 blockSize(tSize, tSize, 1);
	dim3 gridSize(ceil((float)w/tSize), ceil((float)h/tSize), 1);

	mkClockMeasure *ckCpu = new mkClockMeasure("CPU CODE");
	mkClockMeasure *ckGpu = new mkClockMeasure("GPU CODE");

	ckCpu->clockReset();
	ckGpu->clockReset();


	for(int i = 0; i < MAX_ITER; i++){
		
		ckCpu->clockResume();
		cpuCode(h_outImageArray, h_imageArray, w, h);
		ckCpu->clockPause();

		ckGpu->clockResume();
		gpuCode<<<gridSize, blockSize>>>(d_outImageArray, d_imageArray, w, h);
		err=cudaDeviceSynchronize();
		ckGpu->clockPause();
		checkCudaError(err);

	}
	ckCpu->clockPrint();
	ckGpu->clockPrint();

	err = cudaMemcpy(h_outImageArray2, d_outImageArray, arraySize, cudaMemcpyDeviceToHost);
	checkCudaError(err);
			
	ppmSave("blur9_1024_cpu.ppm", h_outImageArray, w, h);
	ppmSave("blur9_1024_gpu.ppm", h_outImageArray2, w, h);
	return 0;
}



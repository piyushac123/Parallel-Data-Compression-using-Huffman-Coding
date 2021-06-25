#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <limits.h>
#include <math.h> 
#include <cuda.h>
#include <algorithm>

#define BLOCK_SIZE 1024
__device__ unsigned int counter, counter_2;

__constant__ const unsigned int INTMAX = 2147483647;

__global__ void CalculateFrequency(unsigned char * device_inputFileData , unsigned int * device_frequency, unsigned int inputFileLength)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x ;
    if(id < inputFileLength){
        atomicAdd(& device_frequency[device_inputFileData[id]] , 1);
    }
}

int main(int argc, char ** argv){
    unsigned int distinctCharacterCount, inputFileLength;
    unsigned int frequency[256];
    unsigned char * inputFileData, bitSequenceLength = 0, bitSequence[255];
    unsigned int * compressedDataOffset, cpuTimeUsed;
    long unsigned int  memOffset;
    clock_t start, end;

    FILE * inputFile, * compressedFile;

    // check the arguments
    if(argc != 3){
        printf("Arguments should be input file and output file");
        return -1;
    }

    // read input file, get length and data
    inputFile = fopen(argv[1], "rb");
    fseek(inputFile, 0, SEEK_END);
    inputFileLength = ftell(inputFile);
    printf("Input File length : %d\n", inputFileLength);
    fseek(inputFile, 0, SEEK_SET);
    inputFileData = (unsigned char *) malloc(inputFileLength * sizeof(unsigned char));
    fread(inputFileData, sizeof(unsigned char), inputFileLength, inputFile);
    fclose(inputFile);

    // starting the clock, tick tick
    start = clock();

    // find frequency of each symbols
    for(int i = 0; i < 256; i++)
        frequency[i] = 0;

    unsigned int *device_frequency;
    cudaMalloc(& device_frequency, 256*sizeof(unsigned int));
    cudaMemcpy(device_frequency, frequency, 256*sizeof(unsigned int), cudaMemcpyHostToDevice);

    unsigned char * device_inputFileData;
    cudaMalloc(& device_inputFileData, inputFileLength*sizeof(unsigned char));
    cudaMemcpy(device_inputFileData, inputFileData, inputFileLength*sizeof(unsigned char), cudaMemcpyHostToDevice);

    int NumBlocks;
    if( inputFileLength > 1024){
        NumBlocks = ceil( (float)inputFileLength / BLOCK_SIZE );
    }
    else{
        NumBlocks = 1;
    }

    printf("Num of blocks %d\n",NumBlocks);

    CalculateFrequency<<< NumBlocks, BLOCK_SIZE >>>(device_inputFileData, device_frequency, inputFileLength);
    cudaMemcpy(frequency, device_frequency, 256*sizeof(unsigned int), cudaMemcpyDeviceToHost);

    cudaFree(device_inputFileData);
    cudaFree(device_frequency);

    // end the clock, tick tick
    end = clock();

    cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
    printf("\n\nTime taken :: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);

    free(inputFileData);

    return 0;
}

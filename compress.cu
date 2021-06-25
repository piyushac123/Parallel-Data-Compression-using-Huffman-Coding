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

// structure for dictionary
struct huffmanDictionary{
    unsigned char bitSequence[256][191];
    unsigned char bitSequenceLength[256];
};

// structure for node
struct huffmanNode{
    unsigned char letter;
    unsigned int frequency;
    struct huffmanNode * left, * right;
};

struct huffmanNode * huffmanTreeNode_head;
struct huffmanDictionary huffmanDict;
struct huffmanNode huffmanTreeNode[512];
unsigned char bitSequenceConstMemory[256][255];

__global__ void CalculateFrequency(unsigned char * device_inputFileData , unsigned int * device_frequency, unsigned int inputFileLength)
{
    unsigned int id = blockIdx.x * blockDim.x + threadIdx.x ;
    if(id < inputFileLength){
        atomicAdd(& device_frequency[device_inputFileData[id]] , 1);
    }
}

__device__ int findIndex(unsigned int *freq, unsigned int size,unsigned int search){
    for(int i=0;i<size;i++){
        if(freq[i] == search){
            return i;
        }
    }
    return -1;
}
__global__ void findLeastFrequent(unsigned int *freq, unsigned int *min, int size, unsigned int threads, unsigned int* count, unsigned int *index){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    counter_2 = 0;
    __syncthreads();
    int ind;
    
    if(id<threads){
        
        while(1){
            min[counter_2] = INTMAX;
            
            atomicMin(&min[counter_2], freq[id]);
            // Need global barrier
            __syncthreads();
            
            ind = findIndex(freq, threads, min[counter_2]);
            index[counter_2] = ind;
            // Need global barrier
            __syncthreads();
            freq[ind] = INTMAX;
            
            if(id == 0) atomicInc(&counter_2, size);
            // Need global barrier
            __syncthreads();

            min[counter_2] = INTMAX;
            
            atomicMin(&min[counter_2], freq[id]);
            // Need global barrier
            __syncthreads();
            
            ind = findIndex(freq, threads, min[counter_2]);
            index[counter_2] = ind;
            // Need global barrier
            __syncthreads();
            freq[ind] = min[counter_2] + min[counter_2-1];
            
            if(id == 0) atomicInc(&counter_2, size);
            // Need global barrier
            __syncthreads();
            

            if(min[counter_2] == INTMAX || min[counter_2-1] == INTMAX){
                count[0] = counter_2;
                break;
            }
            
        }
    }
}

__global__ void searchSimilarIndex(unsigned int *index, unsigned int *resultIndex, unsigned int *cnt, int threads){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    
    __syncthreads();
    counter = 0;
    if(id != threads){
        if(index[id] == index[threads]){
            int temp = atomicInc(&counter, threads+1);
            resultIndex[temp] = id;
        }
        __syncthreads();
        cnt[0] = counter;
    }
}

void buildHuffmanTree(int count,unsigned char *uniqueChar, unsigned int *frequency,int newIndex, int childIndex){
    if(count == 0){
        
        huffmanTreeNode[newIndex].frequency = frequency[childIndex];
        huffmanTreeNode[newIndex].letter = uniqueChar[childIndex];
        huffmanTreeNode[newIndex].left = NULL;
        huffmanTreeNode[newIndex].right = NULL;
    }
    else{
        
        huffmanTreeNode[newIndex].frequency = huffmanTreeNode[childIndex].frequency + huffmanTreeNode[childIndex + 1].frequency;
        huffmanTreeNode[newIndex].left = & huffmanTreeNode[childIndex];
        huffmanTreeNode[newIndex].right = & huffmanTreeNode[childIndex + 1];
        huffmanTreeNode_head = & (huffmanTreeNode[newIndex]);
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

    // initialize the nodes
    distinctCharacterCount = 0;
    for(int i = 0; i < 256; i++){
        if(frequency[i] > 0){
            distinctCharacterCount ++;
        }
    }

    int unique = 0;
    unsigned char *uniqueChar, *duniqueChar;
    uniqueChar = (unsigned char *)malloc(256*sizeof(unsigned char));
    cudaMalloc(&duniqueChar, 256*sizeof(unsigned char));
    for(int i = 0; i<256; i++){
        if(frequency[i] > 0){
            uniqueChar[unique++] = i;
            printf("%d ",frequency[i]);
        }
    }
    printf("\n");
    cudaMemcpy(duniqueChar, uniqueChar, 256*sizeof(unsigned char), cudaMemcpyHostToDevice);

    // *** FIND MINIMUM 2 FREQUENCY FOR ADDING NEW NODE ***
    unsigned int *tempFreq, *tempDFreq;
    unsigned int *min, *dmin;
    unsigned int *cntMin, *dcntMin;
    unsigned int *indMin, *dindMin;
    int ctr;

    tempFreq = (unsigned int *)malloc(unique*sizeof(unsigned int));
    min = (unsigned int *)malloc(inputFileLength*sizeof(unsigned int));
    cntMin = (unsigned int *)malloc(sizeof(unsigned int));
    indMin = (unsigned int *)malloc(inputFileLength*sizeof(unsigned int));
    ctr = 0;
    for(unsigned int i=0;i<256;i++){
        if(frequency[i]!=0){
            tempFreq[ctr++] = frequency[i];
        }
    }
    // for(unsigned int i=0;i<unique;i++) printf("%d:%c ",tempFreq[i],uniqueChar[i]);
    // printf("\n");
    cudaMalloc(&tempDFreq, unique*sizeof(unsigned int));
    cudaMalloc(&dmin, inputFileLength*sizeof(unsigned int));
    cudaMalloc(&dindMin, inputFileLength*sizeof(unsigned int));
    cudaMalloc(&dcntMin, sizeof(unsigned int));
    cudaMemcpy(tempDFreq, tempFreq, unique*sizeof(unsigned int), cudaMemcpyHostToDevice);

    float num = (float)(unique)/(float)BLOCK_SIZE;
    
    int mod = BLOCK_SIZE;
    if(unique < BLOCK_SIZE) mod = unique%BLOCK_SIZE;
    
    int n = ceil(num);
    printf("%d %d\n",n,mod);
    findLeastFrequent<<<n, mod>>>(tempDFreq, dmin, inputFileLength, unique, dcntMin, dindMin);
    
    cudaDeviceSynchronize();

    cudaMemcpy(min, dmin, inputFileLength*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(indMin, dindMin, inputFileLength*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cntMin, dcntMin, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    // printf("count : %d\n",cntMin[0]);
    // for(unsigned int i=0;i<cntMin[0];i++){
    //     printf("%d:%d:%d ",i,indMin[i],min[i]);
    // } 
    // printf("\n");
    // printf("Min:\n");
    // for(unsigned int i=0;i<cntMin[0];i++) printf("%d ",min[i]);
    // printf("\nIndMin:\n");
    // for(unsigned int i=0;i<cntMin[0];i++) printf("%d ",indMin[i]);

    // Get all children
    unsigned int *resultIndex, *dresultIndex;
    unsigned int *cnt, *dcnt;
    resultIndex = (unsigned int *)malloc(cntMin[0]*sizeof(unsigned int));
    cudaMalloc(&dresultIndex, cntMin[0]*sizeof(unsigned int));
    cnt = (unsigned int *)malloc(sizeof(unsigned int));
    cudaMalloc(&dcnt, sizeof(unsigned int));

    int indexChild;
    for(int i=0;i<cntMin[0]-1;i++){
        num = (float)(i+1)/(float)BLOCK_SIZE;
        mod = BLOCK_SIZE;
        if(i+1 < BLOCK_SIZE) mod = (i+1)%BLOCK_SIZE;
        n = ceil(num);
        
        searchSimilarIndex<<<n, mod>>>(dindMin, dresultIndex, dcnt, i);
        cudaDeviceSynchronize();

        cudaMemcpy(resultIndex, dresultIndex, cntMin[0]*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cnt, dcnt, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        
        if(cnt[0] == 0) indexChild = indMin[i];
        else indexChild = *std::max_element(resultIndex, resultIndex + cnt[0])-1;
        buildHuffmanTree(cnt[0], uniqueChar, tempFreq, i, indexChild);
    }
    // for(int j=0;j<cntMin[0]-1;j++){
    //         printf("Index %d:Frequency %u",j,huffmanTreeNode[j].frequency);
    //         if(huffmanTreeNode[j].letter != '\0') printf(":Letter %c\n",huffmanTreeNode[j].letter);
    //         if(huffmanTreeNode[j].left != NULL) printf(":Left %u:Right %u\n",(huffmanTreeNode[j].left)->frequency,(huffmanTreeNode[j].right)->frequency);
    //     }

    if(distinctCharacterCount == 1){
        huffmanTreeNode_head = & huffmanTreeNode[0];
    }

    // end the clock, tick tick
    end = clock();

    cpuTimeUsed = ((end - start)) * 1000 / CLOCKS_PER_SEC;
    printf("\n\nTime taken :: %d:%d s\n", cpuTimeUsed / 1000, cpuTimeUsed % 1000);

    free(inputFileData);

    return 0;
}

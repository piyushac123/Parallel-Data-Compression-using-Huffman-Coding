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
//__device__ unsigned int flag;
__constant__ const unsigned int INTMAX = 2147483647;

// struct for dictionary
struct huffmanDictionary{
    unsigned char bitSequence[255];
    unsigned char bitSequenceLength;
};

// struct for node
struct huffmanNode{
    unsigned char letter;  // char to store
    unsigned int frequency;  // frequency of the char
    struct huffmanNode * left;  // left sub tree
    struct huffmanNode * right;  // right sub tree
};

struct huffmanDictionary huffmanDictionary[256];
struct huffmanNode * huffmanNode_head;
struct huffmanNode huffmanTreeNode[512];

#define DEBUG 1

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
    //printf("%d:%d ",size,threads);
    if(id<threads){
        //printf("\n:%d:\n",counter_2);
        while(1){
            min[counter_2] = INTMAX;
            //printf("%d ",min[counter_2]);
            atomicMin(&min[counter_2], freq[id]);
            // Need global barrier
            __syncthreads();
            //printf("1:%d:%d ",counter_2,min[counter_2]);
            //printf("\n");
            ind = findIndex(freq, threads, min[counter_2]);
            index[counter_2] = ind;
            // Need global barrier
            __syncthreads();
            freq[ind] = INTMAX;
            //printf("%d:%d ",ind,freq[ind]);
            //printf("\n");
            if(id == 0) atomicInc(&counter_2, size);
            // Need global barrier
            __syncthreads();

            min[counter_2] = INTMAX;
            //printf("%d:%d ",counter_2,min[counter_2]);
            atomicMin(&min[counter_2], freq[id]);
            // Need global barrier
            __syncthreads();
            //printf("2:%d:%d ",counter_2,min[counter_2]);
            //printf("\n");
            ind = findIndex(freq, threads, min[counter_2]);
            index[counter_2] = ind;
            // Need global barrier
            __syncthreads();
            freq[ind] = min[counter_2] + min[counter_2-1];
            //printf("%d:%d ",ind,freq[ind]);
            //printf("\n");
            if(id == 0) atomicInc(&counter_2, size);
            // Need global barrier
            __syncthreads();
            //for(int i=0;i<counter_2;i++) printf("%d:%d ",i,min[i]);

            if(min[counter_2] == INTMAX || min[counter_2-1] == INTMAX){
                count[0] = counter_2;
                break;
            }
            //break;
        }
    }
}

__global__ void searchSimilarIndex(unsigned int *index, unsigned int *resultIndex, unsigned int *cnt, int threads){
    int id = blockIdx.x*blockDim.x + threadIdx.x;
    //printf("%d ",threads);
    __syncthreads();
    counter = 0;
    if(id != threads){
        if(index[id] == index[threads]){
            //printf("%d ",id);
            int temp = atomicInc(&counter, threads+1);
            resultIndex[temp] = id;
        }
        __syncthreads();
        cnt[0] = counter;
    }
}

/*
* Sorting the nodes based on the frequency
* The man frequency is represented by the distinct char count
*/
void sortHuffmanTreeNodes(int a, int distinctCharacterCount, int combinedHuffmanNodes){
    for(int i = combinedHuffmanNodes; i < distinctCharacterCount - 1 + a; i++){
        for(int j = combinedHuffmanNodes; j < distinctCharacterCount - 1 + a; j++){

            // perform swapping
            if(huffmanTreeNode[j].frequency > huffmanTreeNode[j + 1].frequency){
                struct huffmanNode tempNode = huffmanTreeNode[j];
                huffmanTreeNode[j] = huffmanTreeNode[j + 1];
                huffmanTreeNode[j + 1] = tempNode;
            }

        }
    }
}

/*
* Build the tree from the sorted results
* The tree here is the min heap
*/
void buildHuffmanTree(int count,unsigned char *uniqueChar, unsigned int *frequency,int newIndex, int childIndex){
    if(count == 0){
        //printf("1:%d:%c\n",frequency[childIndex],uniqueChar[childIndex]);
        huffmanTreeNode[newIndex].frequency = frequency[childIndex];
        huffmanTreeNode[newIndex].letter = uniqueChar[childIndex];
        huffmanTreeNode[newIndex].left = NULL;
        huffmanTreeNode[newIndex].right = NULL;
    }
    else{
        //printf("2:%d:%d:%d\n",childIndex,childIndex+1,huffmanTreeNode[childIndex].frequency + huffmanTreeNode[childIndex + 1].frequency);
        huffmanTreeNode[newIndex].frequency = huffmanTreeNode[childIndex].frequency + huffmanTreeNode[childIndex + 1].frequency;
        huffmanTreeNode[newIndex].left = & huffmanTreeNode[childIndex];
        huffmanTreeNode[newIndex].right = & huffmanTreeNode[childIndex + 1];
        huffmanNode_head = & (huffmanTreeNode[newIndex]);
    }
}
/*void buildHuffmanTreeNodes(int i, int distinctCharacterCount, int combinedHuffmanNodes){
    huffmanTreeNode[distinctCharacterCount + i].frequency = huffmanTreeNode[combinedHuffmanNodes].frequency + huffmanTreeNode[combinedHuffmanNodes + 1].frequency;
    huffmanTreeNode[distinctCharacterCount + i].left = & huffmanTreeNode[combinedHuffmanNodes];
    huffmanTreeNode[distinctCharacterCount + i].right = & huffmanTreeNode[combinedHuffmanNodes + 1];
    huffmanNode_head = & (huffmanTreeNode[distinctCharacterCount + i]);
}*/

/*
* Build the dictionary for the huffman tree
* It will store the bit sequence and their respective lengths
*/
void buildHuffmanDictionary(struct huffmanNode * root, unsigned char * bitSequence, unsigned char bitSequenceLength){
    if(root -> left){
        bitSequence[bitSequenceLength] = 0;
        buildHuffmanDictionary(root -> left, bitSequence, bitSequenceLength + 1);
    }

    if(root -> right){
        bitSequence[bitSequenceLength] = 1;
        buildHuffmanDictionary(root -> right, bitSequence, bitSequenceLength + 1);
    }

    // copy the bit sequence and the length to the dictionary
    if(root -> right == NULL && root -> left == NULL){
        huffmanDictionary[root -> letter].bitSequenceLength = bitSequenceLength;
        memcpy(huffmanDictionary[root -> letter].bitSequence, bitSequence, bitSequenceLength * sizeof(unsigned char));
    }
}


int main(int argc, char ** argv){
//int main(){
    clock_t start, end;
    unsigned int cpuTime;
    unsigned int compressedFileLength, outputFileLengthCounter, outputFileLength, extra;
    unsigned int mergedHuffmanNodes, distinctCharacterCount;
    unsigned int frequency[256];

    unsigned char currentInputBit, currentInputByte, bitSequenceLength = 0, bitSequence[255];
    unsigned char * compressedData, * outputData;
    struct huffmanNode * huffmanNode_current;

    FILE *compressedFile, *outputFile;

    // read input file get length, frequency and data
    compressedFile = fopen(argv[1], "r");
    //compressedFile = fopen("xyz.txt", "r");
    fread(& outputFileLength, sizeof(unsigned int), 1, compressedFile);
    //no. of extra bits added, calculate here
    fread(& extra, sizeof(unsigned int), 1, compressedFile);
    fread(frequency, 256 * sizeof(unsigned int), 1, compressedFile);

    // find length of the compressed file
    fseek(compressedFile, 0, SEEK_END);
    compressedFileLength = ftell(compressedFile) - 1032;
    fseek(compressedFile, 1032, SEEK_SET);

    // allocate the required memory, read the file
    compressedData = (unsigned char *)malloc((compressedFileLength) * sizeof(unsigned char));
    fread(compressedData, sizeof(unsigned char), compressedFileLength, compressedFile);

    // start the clock, tick tick
    start = clock();

    for(int i=0;i<256;i++) printf("%c ",frequency[i]);
    printf("\n");
    // initialize the huffman tree
    distinctCharacterCount = 0;
    for(int i = 0; i < 256; i++){
        if(frequency[i] > 0){
            /*huffmanTreeNode[distinctCharacterCount].frequency = frequency[i];
            huffmanTreeNode[distinctCharacterCount].letter = i;
            huffmanTreeNode[distinctCharacterCount].left = NULL;
            huffmanTreeNode[distinctCharacterCount].right = NULL;*/
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
    //tempFreq = (unsigned int *)malloc(5*sizeof(unsigned int));
    //tempFreq[0] = 2;tempFreq[1] = 3;tempFreq[2] = 4;tempFreq[3] = 5;tempFreq[4] = 6;
    min = (unsigned int *)malloc(outputFileLength*sizeof(unsigned int));
    cntMin = (unsigned int *)malloc(sizeof(unsigned int));
    indMin = (unsigned int *)malloc(outputFileLength*sizeof(unsigned int));
    ctr = 0;
    for(unsigned int i=0;i<256;i++){
        if(frequency[i]!=0){
            tempFreq[ctr++] = frequency[i];
        }
    }
    for(unsigned int i=0;i<unique;i++) printf("%d:%c ",tempFreq[i],uniqueChar[i]);
    printf("\n");
    //printf("%d\n",unique);
    cudaMalloc(&tempDFreq, unique*sizeof(unsigned int));
    //cudaMalloc(&tempDFreq, 5*sizeof(unsigned int));
    cudaMalloc(&dmin, outputFileLength*sizeof(unsigned int));
    cudaMalloc(&dindMin, outputFileLength*sizeof(unsigned int));
    cudaMalloc(&dcntMin, sizeof(unsigned int));
    cudaMemcpy(tempDFreq, tempFreq, unique*sizeof(unsigned int), cudaMemcpyHostToDevice);

    float num = (float)(unique)/(float)BLOCK_SIZE;
    //num = (float)(5)/(float)BLOCK_SIZE;
    int mod = BLOCK_SIZE;
    if(unique < BLOCK_SIZE) mod = unique%BLOCK_SIZE;
    //if(5 < BLOCKSIZE) mod = 5%BLOCK_SIZE;
    int n = ceil(num);
    printf("%d %d\n",n,mod);
    findLeastFrequent<<<n, mod>>>(tempDFreq, dmin, outputFileLength, unique, dcntMin, dindMin);
    //findLeastFrequent<<<n, mod>>>(tempDFreq, dmin, outputFileLength, 5, dcntMin);
    cudaDeviceSynchronize();

    cudaMemcpy(min, dmin, outputFileLength*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(indMin, dindMin, outputFileLength*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(cntMin, dcntMin, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("count : %d\n",cntMin[0]);
    for(unsigned int i=0;i<cntMin[0];i++){
        printf("%d:%d:%d ",i,indMin[i],min[i]);
    } 
    printf("\n");
    printf("Min:\n");
    for(unsigned int i=0;i<cntMin[0];i++) printf("%d ",min[i]);
    printf("\nIndMin:\n");
    for(unsigned int i=0;i<cntMin[0];i++) printf("%d ",indMin[i]);
    
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
        //printf("\n\n%d %d\n",n,mod);
        searchSimilarIndex<<<n, mod>>>(dindMin, dresultIndex, dcnt, i);
        cudaDeviceSynchronize();

        cudaMemcpy(resultIndex, dresultIndex, cntMin[0]*sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(cnt, dcnt, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        /*printf("count : %d\n",cnt[0]);
        for(unsigned int j=0;j<cnt[0];j++){
            printf("%d:%d ",indMin[resultIndex[j]],resultIndex[j]);
        } 
        printf("\n");
        printf("%d:%d\n",indMin[*std::max_element(resultIndex, resultIndex + cnt[0])],*std::max_element(resultIndex, resultIndex + cnt[0]));*/

        if(cnt[0] == 0) indexChild = indMin[i];
        else indexChild = *std::max_element(resultIndex, resultIndex + cnt[0])-1;
        buildHuffmanTree(cnt[0], uniqueChar, tempFreq, i, indexChild);

    }
    for(int j=0;j<cntMin[0]-1;j++){
            printf("Index %d:Frequency %u",j,huffmanTreeNode[j].frequency);
            if(huffmanTreeNode[j].letter != '\0') printf(":Letter %c\n",huffmanTreeNode[j].letter);
            if(huffmanTreeNode[j].left != NULL) printf(":Left %u:Right %u\n",(huffmanTreeNode[j].left)->frequency,(huffmanTreeNode[j].right)->frequency);
        }

    // build the tree
    /*for(int i = 0; i < distinctCharacterCount - 1; i++){
        mergedHuffmanNodes = 2 * i;
        sortHuffmanTreeNodes(i, distinctCharacterCount, mergedHuffmanNodes);
        buildHuffmanTreeNodes(i, distinctCharacterCount, mergedHuffmanNodes);
    }*/

    // build the huffman dictionary with the bit sequence and its length
    buildHuffmanDictionary(huffmanNode_head, bitSequence, bitSequenceLength);

    // write data to the file
    outputData = (unsigned char *)malloc(outputFileLength * sizeof(unsigned char));
    huffmanNode_current = huffmanNode_head;
    outputFileLengthCounter = 0;
    for(int i = 0; i < compressedFileLength-extra; i++){

            // value is 0 then left sub tree
            if(compressedData[i] == 0){
                huffmanNode_current = huffmanNode_current -> left;
                if(huffmanNode_current -> left == NULL){
                    outputData[outputFileLengthCounter] = huffmanNode_current -> letter;
                    huffmanNode_current = huffmanNode_head;
                    outputFileLengthCounter ++;
                }
            }
            // value is 1 the right sub tree
            else {
                huffmanNode_current = huffmanNode_current -> right;
                if(huffmanNode_current -> right == NULL){
                    outputData[outputFileLengthCounter] = huffmanNode_current -> letter;
                    huffmanNode_current = huffmanNode_head;
                    outputFileLengthCounter ++;
                }
            }
    }

    // end the clock, tick tick
    end = clock();

    // write the data to the output file
    outputFile = fopen(argv[2], "wb");
    //outputFile = fopen("abc_.txt", "wb");
    fwrite(outputData, sizeof(unsigned char), outputFileLength, outputFile);
    fclose(outputFile);

    // printing debug info if debug is on
    if(DEBUG){
        printf("\nCompressed file length :: %d", compressedFileLength);
        printf("\nOutput file length counter :: %d", outputFileLengthCounter);
        printf("\nOutput file length :: %d", outputFileLength);
        printf("\nMerged Huffman Nodes :: %d", mergedHuffmanNodes);
        printf("\nDistinct character count :: %d", distinctCharacterCount);
    }

    cpuTime = (end - start) * 1000 / CLOCKS_PER_SEC;
    printf("\nTime taken: %d:%d s\n", cpuTime / 1000, cpuTime % 1000);

	// clean up
	free(outputData);
	free(compressedData);

	return 0;
}

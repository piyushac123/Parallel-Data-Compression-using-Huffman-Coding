#include<stdio.h>
#include<cuda.h>

void perm_func(char *perm_str,char *str, int i, int N){
    int ind1 = 0;
    // while(str[i] != '\0') printf("%c ",str[i++]);
    // printf("\n");
    for(int j=i;j<N;j++){
        perm_str[i*N+ind1] = str[j];
        ind1++;
    }
    int ind2 = 0;
    for(int j=ind1;j<N;j++){
        perm_str[i*N+j] = str[ind2];
        ind2++;
    }
}

int main(){
    //  Step 1
    char str[] = {'t','h','i','s',' ','i','s',' ','t','h','e','\0'};
    // str = (char *)malloc(11*sizeof(char));
    int N = 11;
    int i=0;
    while(str[i] != '\0') printf("%c ",str[i++]);
    printf("\n");

    //Therefore, only the original string needs to be stored, while
    //the rotations are represented by pointers or indices into a
    //memory buffer.
    //  Step 2
    char *perm_str = (char *)malloc(N*N*sizeof(char));
    for(int i=0;i<N;i++){
        perm_func(perm_str,str, i, N);
    }
    for(int i=0;i<N;i++){
        for(int j=0;j<N;j++){
            printf("%c ",perm_str[i*N+j]);
        }
        printf("\n");
    }

    //  Step 3

    return 0;
}
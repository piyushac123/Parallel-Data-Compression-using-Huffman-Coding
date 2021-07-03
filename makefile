all:
	nvcc -dc compress.cu 
	nvcc *.o -o ../bin/compress
	nvcc -dc decompress.cu 
	nvcc *.o -o ../bin/decompress
	gcc -o ../bin/GenerateInput GenerateInput.c
  rm -rf *.o

clean:
	rm -f ../bin/compress
  	rm -f ../bin/decompress
 	rm -f ../bin/GenerateInput

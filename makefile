all:
	nvcc -dc compress.cu 
	nvcc *.o -o ../bin/compress
  gcc -o ../bin/decompress decompress.c
	gcc -o ../bin/GenerateInput GenerateInput.c
  rm -rf *.o

clean:
	rm -f ../bin/compress
  rm -f ../bin/decompress
  rm -f ../bin/GenerateInput
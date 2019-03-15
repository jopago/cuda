CC=nvcc
OPENCV = `pkg-config opencv --cflags --libs`

all:
	$(CC) haar.cu -o haar $(OPENCV)
	
clean:
	rm haar
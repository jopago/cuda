CC=nvcc
OPENCV = `pkg-config opencv --cflags --libs`

all:
	$(CC) conv2d.cu -o conv2d -I../ $(OPENCV)

clean:
	rm conv2d
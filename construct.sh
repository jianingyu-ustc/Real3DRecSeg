rm Matrix.o
rm Frame.o
rm glad.o
rm Viewer.o
rm Kernel.o
rm Voxel.o
rm main.o
rm main

g++ -c Matrix.cc
g++ -c Frame.cc -fopenmp
g++ -c glad.c 
g++ -c Viewer.cc
nvcc -c Kernel.cu -Xcompiler -fopenmp
g++ -c Voxel.cc
nvcc -c main.cc  
g++ -o main main.o Kernel.o -lcudart -L/usr/local/cuda/lib64 Voxel.o Viewer.o Frame.o Matrix.o glad.o -lglfw3 -lGL -lopencv_core -lm -lXrandr -lXi -lX11 -lXxf86vm -fopenmp -lpthread -ldl -lXinerama -lXcursor `pkg-config opencv --cflags --libs`

./main $1

rm Matrix.o
rm Frame.o
rm glad.o
rm Viewer.o
rm Kernel.o
rm Voxel.o
rm main.o
rm main

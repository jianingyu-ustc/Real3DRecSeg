#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <chrono>
#include <filesystem>

float* data1;
float* data2;
float* data3;
float* data4;
float* data5;

void ReadGpuBlock(std::string block_name, int block_size, int len){
    std::string file_name = "GpuBlock/" + block_name + ".txt";
    FILE *pFile = fopen(file_name.c_str(), "r");
    if(pFile == nullptr) {
        printf("fail\n");
        return;
    }
    fread(data1, sizeof(float), len, pFile);
    fread(data2, sizeof(float), len, pFile);
    fread(data3, sizeof(float), len, pFile);
    fread(data4, sizeof(float), len, pFile);
    fread(data5, sizeof(float), len, pFile);

    fclose(pFile);
    return;
}

void WriteGpuBlock(std::string block_name, int block_size, int len){
    // std::string directory_name("GpuBlock/");
    // std::filesystem::create_directory(directory_name);

    std::string file_name = "GpuBlock/" + block_name + ".txt";
    FILE *pFile = fopen(file_name.c_str(), "wt");

    fwrite(data1, sizeof(float), len, pFile);
    fwrite(data2, sizeof(float), len, pFile);
    fwrite(data3, sizeof(float), len, pFile);
    fwrite(data4, sizeof(float), len, pFile);
    fwrite(data5, sizeof(float), len, pFile);
    
    fclose(pFile);
    return;
}

int main()
{
    int num = 108000000;

    data1 = (float*)malloc(sizeof(float) * num);
    data2 = (float*)malloc(sizeof(float) * num);
    data3 = (float*)malloc(sizeof(float) * num);
    data4 = (float*)malloc(sizeof(float) * num);
    data5 = (float*)malloc(sizeof(float) * num);

    float val = 0;
    for(int i=0; i < num; ++i){
        data1[i] = ++val;
        data2[i] = ++val;
        data3[i] = ++val;
        data4[i] = ++val;
        data5[i] = ++val;
    }

    WriteGpuBlock("+0+0+0", 4, num);
    free(data1); data1 = nullptr;
    free(data2); data2 = nullptr;
    free(data3); data3 = nullptr;
    free(data4); data4 = nullptr;
    free(data5); data5 = nullptr;

    data1 = (float*)malloc(sizeof(float) * num);
    data2 = (float*)malloc(sizeof(float) * num);
    data3 = (float*)malloc(sizeof(float) * num);
    data4 = (float*)malloc(sizeof(float) * num);
    data5 = (float*)malloc(sizeof(float) * num);
    ReadGpuBlock("+0+0+0", 4, num);

    return 0;
}

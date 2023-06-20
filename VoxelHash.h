#ifndef JIAMERA_VOXELHASH_H_
#define JIAMERA_VOXELHASH_H_



#include <iostream>
#include "Matrix.h"

struct VoxelBlock {
    int x_index;   // 该块在体素空间中的序号
    int y_index;
    int z_index;
    int block_index;
    int grid_index[8 * 8 * 8];  // 该块包含的网格的序号
    VoxelBlock(int x, int y, int z, int index) 
        : x_index(x), y_index(y), z_index(z), block_index(index) {
        //hash_value = get_hash(x, y, z);
        for (int i = 0; i < 8 * 8 * 8; ++i) 
            grid_index[i] = 512 * block_index + i;
    }
};

struct HashBucket {     // hash_table 由 Buckets 组成，
    VoxelBlock* entrance[4];    // 指向空间内的 4 个块

    HashBucket* next_Bucket;
    HashBucket() {
        for (int i = 0; i < 3; ++i) entrance[i] = nullptr;
        next_Bucket = nullptr;
    }

};

class VoxelHashInterface
{
public:
    VoxelHashInterface() {
        std::cout << "Creating VoxelHashInterface Class -----------------------------------------\n";
    }

    VoxelHashInterface(int x_num, int y_num, int z_num)
        : block_number_x_(x_num), block_number_y_(y_num), 
          block_number_z_(z_num) {

        std::cout << "Setting VoxelHashInterface Class -----------------------------------------\n";
        // 构造体素块
        block_volume_ = block_number_x_ * block_number_y_ * block_number_z_;

        block_list_ = new Matrix<VoxelBlock*>(block_number_x_, block_number_y_, 
                                              block_number_z_, NULL);

        size_t block_index = 0;
        for (size_t block_index_z = 0; block_index_z < block_number_z_; ++block_index_z)
            for (size_t block_index_y = 0; block_index_y < block_number_y_; ++block_index_y)
                for (size_t block_index_x = 0; block_index_x < block_number_x_; ++block_index_x) {

                    block_list_->grid_[block_index] = 
                        new VoxelBlock(block_index_x, block_index_y, 
                                       block_index_z, block_index);
                    ++block_index;
                }

        // 哈希桶实例化，一个哈希值构造一个桶，并按序连接起来，构造环形链表
        hash_table_size_ = compute_block_hash_table_size(block_volume_, 0.75);
        HashBucket* previous_bucket = nullptr;
        HashBucket* first_bucket;

        for (size_t bucket_index = 0; bucket_index < hash_table_size_; ++bucket_index) {
            HashBucket* current_bucket = new HashBucket();
            block_hash_table_[bucket_index] = current_bucket;
            if (bucket_index == 0) first_bucket = current_bucket;
            else previous_bucket->next_Bucket = current_bucket;
            previous_bucket = current_bucket;
        }
        previous_bucket->next_Bucket = first_bucket;

    }




protected:
    int block_number_x_;    // x 轴上 block 个数
    int block_number_y_;
    int block_number_z_;
    int block_volume_;
    Matrix<VoxelBlock*>* block_list_;   // 将体素空间分割成块，每个块为一个 VoxelBlock 结构体，包含 8^3 个网格

    std::unordered_map<int, HashBucket*> block_hash_table_;    // 根据哈希值找到对应的 Block
    int hash_table_size_;    // 哈希表长度

public:
    int ComputeHash(int block_index_x, int block_index_y, int block_index_z, 
                    int n) {  // 根据 block 坐标计算哈希值
        /* 冲突的块
         序号    哈希值
        176188 1901056644
        176193 121702485
        177637 1557463991
        177642 719051110
    */

        long long p1 = 73856093;
        long long p2 = 19349669;
        long long p3 = 83492791;

        long long hash_value = (block_index_x * p1) ^ (block_index_y * p2) ^ 
                               (block_index_z * p3) % n;
        return hash_value;
    }

    // 求哈希表长度 n，使得表长为素数且块数量 / 3n 小于装填因子
    int compute_block_hash_table_size(const int block_size, float factor) {
        int m = block_size / (3 * factor);
        int i, j;
        int res;
        for (i = m + 1; i < 2 * m; i++)
            for (j = 2; j < (i / j + 1); j++) {
                res = i / j;
                if ((res - j) <= 1 && i % j != 0) return i;
                if (i % j == 0) break;
                else continue;
            }
        return 0;
    }

    VoxelBlock* LinearFindBlock(int block_index_x, int block_index_y, 
                                int block_index_z) {
        int idx = 0;
        VoxelBlock* target_block = block_list_->grid_[idx];

        while (idx < block_volume_ && 
               (target_block->x_index != block_index_x ||
                target_block->y_index != block_index_y || 
                target_block->z_index != block_index_z)) {
            ++idx;
            target_block = block_list_->grid_[idx];
        }

        if (idx >= block_volume_) {
            printf("Error: block (%d, %d, %d) does not exist.\n", 
                block_index_x, block_index_y, block_index_z);
            return nullptr;
        }
        return target_block;
    }

    // 根据 block 的坐标计算出哈希值，并找到对应的目标桶
    // 调用 find_block_in_bucket 递归查找目标 block
    // 若没有找到目标 block 则直接线性查找，并调用 insert_block_in_bucket 递归插入
    // 返回目标 block
    VoxelBlock* HashFindBlock(int block_index_x, int block_index_y, 
                              int block_index_z) {
        int current_hash = ComputeHash(block_index_x, block_index_y, 
                                       block_index_z, hash_table_size_);
        HashBucket* currnt_bucket = block_hash_table_[current_hash];

        VoxelBlock* target_block = FindBlockInBucket(currnt_bucket, 
                                                     block_index_x, block_index_y,
                                                     block_index_z);
        if (!target_block) {
            target_block = LinearFindBlock(block_index_x, block_index_y, 
                                                     block_index_z);
            InsertBlockInBucket(currnt_bucket, target_block);
        }
        return target_block;
    }

    // 遍历当前桶的 entrance，查找目标 block
    // 若找到目标 block 则直接返回
    // 若桶满且未找到则递归查找下一个桶
    // 若桶未满且未找到则返回 nullptr
    VoxelBlock* FindBlockInBucket(HashBucket* bucket, int block_index_x, 
                                  int block_index_y, int block_index_z) {
        int entrance_index = 0;
        VoxelBlock* current_block = bucket->entrance[entrance_index];
        while (current_block && entrance_index < 2 && 
               (current_block->x_index != block_index_x ||
                current_block->y_index != block_index_y || 
                current_block->z_index != block_index_z)) {
            current_block = bucket->entrance[++entrance_index];
        }

        if (current_block && current_block->x_index == block_index_x &&
            current_block->y_index == block_index_y && 
            current_block->z_index == block_index_z) {
            return current_block;
        }
        else if (entrance_index == 2) {
            return FindBlockInBucket(bucket->next_Bucket, block_index_x, 
                                     block_index_y, block_index_z);
        }
        else {
            return nullptr;
        }
    }

    // 遍历当前桶的 entrance，查找第一个空 block 并插入
    // 若桶满则递归地在下一个桶中插入
    bool InsertBlockInBucket(HashBucket* bucket, VoxelBlock* block);

    bool HashDeleteBlock(int block_idx_x, int block_idx_y,
                         int block_idx_z);











};

#endif
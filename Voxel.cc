#include "Voxel.h"

namespace Jiamera {

    Voxel::Voxel() {
        // std::cout << "Creating Voxel Class\n";

        grid_num_x_ = k_space_x / k_grid_size_;
        grid_num_y_ = k_space_y / k_grid_size_;
        grid_num_z_ = k_space_z / k_grid_size_;
        grid_num_ = grid_num_y_ * grid_num_y_ * grid_num_z_;

        truncation_margin_ = k_grid_size_ * 5;
    }

    /*
    x
    原点 偏左 0.7 0.5 0.4 合适 0.3 偏右
    尺寸 偏小 10 合适 15 偏大

    y
    原点 偏上 0.7 合适 0.71 0.72 0.725 0.75 0.8 偏下
    尺寸 偏小 合适 8 偏大

    z (左视图)
    原点 偏左 0.75 0.5 0.3 合适 0.2 偏右
    尺寸 偏小 8 合适 10 偏大
    */


    Voxel::Voxel(float x, float y, float z, float size) :
        k_space_x(x), k_space_y(y), k_space_z(z), k_grid_size_(size),
        // k_origin_x_(0), k_origin_y_(0), k_origin_z_(0) {
        k_origin_x_(-x*0.35), k_origin_y_(-y * 0.705), k_origin_z_(-z*0.25) {

        // std::cout << "Setting Voxel Class\n";

        grid_num_x_ = k_space_x / k_grid_size_;
        grid_num_y_ = k_space_y / k_grid_size_;
        grid_num_z_ = k_space_z / k_grid_size_;

        grid_num_ = grid_num_x_ * grid_num_y_ * grid_num_z_;

        truncation_margin_ = k_grid_size_ * 3;

        data_init();

        gl_data_init();

        //VoxelHashInterface(ceil(grid_num_x_ / 8.0), ceil(grid_num_y_ / 8.0), ceil(grid_num_z_ / 8.0));

        gl_data_update();   // 向 OpenGL 渲染数组传输点云，本次传输界面为空

    }

    void Voxel::data_init() {
        //data_ = (float*)malloc(grid_num_ * attrib_num_ * sizeof(float));
        //memset(data_, 0, grid_num_ * attrib_num_ * sizeof(float));

        color_b_ = (float*)malloc(grid_num_ * sizeof(float));
        memset(color_b_, 0, grid_num_ * sizeof(float));

        color_g_ = (float*)malloc(grid_num_ * sizeof(float));
        memset(color_g_, 0, grid_num_ * sizeof(float));

        color_r_ = (float*)malloc(grid_num_ * sizeof(float));
        memset(color_r_, 0, grid_num_ * sizeof(float));

        instance_ = (int*)malloc(grid_num_ * sizeof(int));
        memset(instance_, 0, grid_num_ * sizeof(int));

        label_ = (int*)malloc(grid_num_ * sizeof(int));
        std::fill(label_, label_ + grid_num_, -1);
        //memset(label_, 0, grid_num_ * sizeof(int));

        tsdf_ = (float*)malloc(grid_num_ * sizeof(float));
        memset(tsdf_, 0, grid_num_ * sizeof(float));

        bgr_weight_ = (float*)malloc(grid_num_ * sizeof(float));
        memset(bgr_weight_, 0, grid_num_ * sizeof(float));

        label_weight_ = (float*)malloc(grid_num_ * sizeof(float));
        memset(label_weight_, 0, grid_num_ * sizeof(float));

        instance_weight_ = (float*)malloc(grid_num_ * sizeof(float));
        memset(instance_weight_, 0, grid_num_ * sizeof(float));

        int grid_index = 0;
        for (int j = 0; j < grid_num_z_; ++j) {
            for (int k = 0; k < grid_num_y_; ++k) {
                for (int i = 0; i < grid_num_x_; ++i) {

                    // 初始化三个 weight = 0
                    // r g b tsdf 第0帧不会用到

                    // 初始化 x y z
                    //data_[grid_index * attrib_num_ + x_offset_] = k_origin_x_ + (float)i * k_grid_size_;
                    //data_[grid_index * attrib_num_ + y_offset_] = k_origin_y_ + (float)k * k_grid_size_;
                    //data_[grid_index * attrib_num_ + z_offset_] = k_origin_z_ + (float)j * k_grid_size_;

                    // 初始化 label
                    //data_[grid_index * attrib_num_ + label_offset_] = -1.0f;

                    //bool flag1 = (i == 0 || i == grid_num_x_ - 1);
                    //bool flag2 = (j == 0 || j == grid_num_z_ - 1);
                    //bool flag3 = (k == 0 || k == grid_num_y_ - 1);
                    //if ((!flag1 && flag2 && flag3) || (!flag1 && flag2 && flag3) || (!flag1 && flag2 && flag3)) {

                    // 显示边框
                    {
                        //bool flag1 = (i == 0);
                        //bool flag2 = (j == 0);
                        //bool flag3 = (k == 0);

                        //bool flag4 = (i == grid_num_x_ - 1);
                        //bool flag5 = (j == grid_num_z_ - 1);
                        //bool flag6 = (k == grid_num_y_ - 1);

                        //if (flag1 && flag2 && !flag3 && !flag4 && !flag5 && !flag6 ||   // 12
                        //    flag1 && !flag2 && flag3 && !flag4 && !flag5 && !flag6 ||   // 13

                        //    flag1 && !flag2 && !flag3 && flag4 && !flag5 && !flag6 ||   // 14

                        //    flag1 && !flag2 && !flag3 && !flag4 && flag5 && !flag6 ||   // 15
                        //    flag1 && !flag2 && !flag3 && !flag4 && !flag5 && flag6 ||   // 16

                        //    !flag1 && flag2 && flag3 && !flag4 && !flag5 && !flag6 ||   // 23
                        //    !flag1 && flag2 && !flag3 && flag4 && !flag5 && !flag6 ||   // 24
                        //    !flag1 && flag2 && !flag3 && !flag4 && !flag5 && flag6 ||   // 26

                        //    !flag1 && !flag2 && flag3 && flag4 && !flag5 && !flag6 ||   // 34
                        //    !flag1 && !flag2 && flag3 && !flag4 && flag5 && !flag6 ||   // 35

                        //    !flag1 && !flag2 && !flag3 && flag4 && flag5 && !flag6 ||   // 45
                        //    !flag1 && !flag2 && !flag3 && flag4 && !flag5 && flag6 ||   // 46

                        //    !flag1 && !flag2 && !flag3 && !flag4 && flag5 && flag6    // 56
                        //    ) {
                        //    data_[grid_index * attrib_num_ + rgb_weight_offset] = 1.0f;

                        //}
                    }

                    ++grid_index;
                }
            }
        }
        // std::cout << grid_index << " points\n";
    }

    void Voxel::SaveCloud(const std::string &scene_id, float* pose) {
        std::string file_name = "../pred/" + scene_id + ".ply";
        // std::cout << "Saving surface point cloud." << std::endl;

        FILE* fp = fopen(file_name.c_str(), "w");
        fprintf(fp, "ply\n");
        fprintf(fp, "format binary_little_endian 1.0\n");
        fprintf(fp, "element vertex %d\n", this->gl_point_num_);
        fprintf(fp, "property float x\n");
        fprintf(fp, "property float y\n");
        fprintf(fp, "property float z\n");
        fprintf(fp, "property uchar color_r\n");
        fprintf(fp, "property uchar color_g\n");
        fprintf(fp, "property uchar color_b\n");
        fprintf(fp, "property uchar label_r\n");
        fprintf(fp, "property uchar label_g\n");
        fprintf(fp, "property uchar label_b\n");
        fprintf(fp, "property int instance\n");
        fprintf(fp, "property int label\n");

        fprintf(fp, "element face 7\n");
        fprintf(fp, "end_header\n");

        // FILE* txt_writer = fopen("CloudPoint.txt", "w"); 

        int index = 0;
        int idx = 0;
        for (int k = 0; k < grid_num_z_; ++k) {
            for (int j = 0; j < grid_num_y_; ++j) {
                for (int i = 0; i < grid_num_x_; ++i) {
                    if (std::abs(tsdf_[index]) < 0.2f && bgr_weight_[index] > 0.0f) {

                        float temp_x = k_origin_x_ + (float)i * k_grid_size_;
                        float temp_y = k_origin_y_ + (float)j * k_grid_size_;
                        float temp_z = k_origin_z_ + (float)k * k_grid_size_;

                        // 与 Scannet GT 模型重合
                        float x = temp_x*pose[0]+temp_y*pose[1]+temp_z*pose[2]+pose[3];
                        float y = temp_x*pose[4]+temp_y*pose[5]+temp_z*pose[6]+pose[7];
                        float z = temp_x*pose[8]+temp_y*pose[9]+temp_z*pose[10]+pose[11];

                        uchar color_r = (uchar)(int)(255 * color_r_[index]);
                        uchar color_g = (uchar)(int)(255 * color_g_[index]);
                        uchar color_b = (uchar)(int)(255 * color_b_[index]);

                        int ins = instance_[index];
                        int label = label_[index];     // -1 -> 65535

                        float label_r, label_g, label_b;
                        if (label == 0) { label_r = 174; label_g = 199; label_b = 232; }
                        else if (label == 1) { label_r = 152; label_g = 223; label_b = 138; }
                        else if (label == 2) { label_r = 31; label_g = 119; label_b = 180; }
                        else if (label == 3) { label_r = 255; label_g = 187; label_b = 120; }
                        else if (label == 4) { label_r = 188; label_g = 189; label_b = 34; }
                        else if (label == 5) { label_r = 140; label_g = 86; label_b = 75; }
                        else if (label == 6) { label_r = 255; label_g = 152; label_b = 150; }
                        else if (label == 7) { label_r = 214; label_g = 39; label_b = 40; }
                        else if (label == 8) { label_r = 197; label_g = 176; label_b = 213; }
                        else if (label == 9) { label_r = 148; label_g = 103; label_b = 189; }
                        else if (label == 10) { label_r = 196; label_g = 156; label_b = 148; }
                        else if (label == 11) { label_r = 23; label_g = 190; label_b = 207; }
                        else if (label == 12) { label_r = 178; label_g = 76; label_b = 76; }
                        else if (label == 13) { label_r = 247; label_g = 182; label_b = 210; }
                        else if (label == 14) { label_r = 66; label_g = 188; label_b = 102; }
                        else if (label == 15) { label_r = 219; label_g = 219; label_b = 141; }
                        else if (label == 16) { label_r = 140; label_g = 57; label_b = 197; }
                        else if (label == 17) { label_r = 202; label_g = 185; label_b = 52; }
                        else if (label == 18) { label_r = 51; label_g = 176; label_b = 203; }
                        else if (label == 19) { label_r = 200; label_g = 54; label_b = 131; }
                        else if (label == 20) { label_r = 92; label_g = 193; label_b = 61; }
                        else if (label == 21) { label_r = 78; label_g = 71; label_b = 183; }
                        else if (label == 22) { label_r = 172; label_g = 114; label_b = 82; }
                        else if (label == 23) { label_r = 255; label_g = 127; label_b = 14; }
                        else if (label == 24) { label_r = 91; label_g = 163; label_b = 138; }
                        else if (label == 25) { label_r = 153; label_g = 98; label_b = 156; }
                        else if (label == 26) { label_r = 140; label_g = 153; label_b = 101; }
                        else if (label == 27) { label_r = 158; label_g = 218; label_b = 229; }
                        else if (label == 28) { label_r = 100; label_g = 125; label_b = 154; }
                        else if (label == 29) { label_r = 178; label_g = 127; label_b = 135; }
                        else if (label == 30) { label_r = 120; label_g = 185; label_b = 128; }
                        else if (label == 31) { label_r = 146; label_g = 111; label_b = 194; }
                        else if (label == 32) { label_r = 44; label_g = 160; label_b = 44; }
                        else if (label == 33) { label_r = 112; label_g = 128; label_b = 144; }
                        else if (label == 34) { label_r = 96; label_g = 207; label_b = 209; }
                        else if (label == 35) { label_r = 227; label_g = 119; label_b = 194; }
                        else if (label == 36) { label_r = 213; label_g = 92; label_b = 176; }
                        else if (label == 37) { label_r = 94; label_g = 106; label_b = 211; }
                        else if (label == 38) { label_r = 82; label_g = 84; label_b = 163; }
                        else if (label == 39) { label_r = 100; label_g = 85; label_b = 144; }
                        else { label_r = 0.0f; label_g = 0.0f; label_b = 0.0f; }

                        uchar uc_label_r = (uchar)label_r;
                        uchar uc_label_g = (uchar)label_g;
                        uchar uc_label_b = (uchar)label_b;

                        
                        fwrite(&x, sizeof(float), 1, fp);
                        fwrite(&y, sizeof(float), 1, fp);
                        fwrite(&z, sizeof(float), 1, fp);

                        fwrite(&color_r, sizeof(uchar), 1, fp);
                        fwrite(&color_g, sizeof(uchar), 1, fp);
                        fwrite(&color_b, sizeof(uchar), 1, fp);

                        fwrite(&uc_label_r, sizeof(uchar), 1, fp);
                        fwrite(&uc_label_g, sizeof(uchar), 1, fp);
                        fwrite(&uc_label_b, sizeof(uchar), 1, fp);

                        fwrite(&ins, sizeof(int), 1, fp);
                        fwrite(&label, sizeof(int), 1, fp);
                    }
                    ++index;
                }
            }
        }
        fclose(fp);
        // fclose(txt_writer);
    }

 /*
    void Voxel::SaveParamaters() {
        std::string file_name = "tsdf.bin";
        std::cout << "Saving TSDF voxel grid values to disk." << std::endl;

        std::ofstream outFile(file_name, std::ios::binary | std::ios::out);

        float voxel_grid_dim_xf = (float)grid_num_x_;
        float voxel_grid_dim_yf = (float)grid_num_y_;
        float voxel_grid_dim_zf = (float)grid_num_z_;

        outFile.write((char*)&voxel_grid_dim_xf, sizeof(float));
        outFile.write((char*)&voxel_grid_dim_yf, sizeof(float));
        outFile.write((char*)&voxel_grid_dim_zf, sizeof(float));
        outFile.write((char*)&k_origin_x_, sizeof(float));
        outFile.write((char*)&k_origin_y_, sizeof(float));
        outFile.write((char*)&k_origin_z_, sizeof(float));
        outFile.write((char*)&k_grid_size_, sizeof(float));
        outFile.write((char*)&truncation_margin_, sizeof(float));

        for (int i = 0; i < grid_num_; ++i)
            outFile.write((char*)&data_[long long(i * attrib_num_) + tsdf_offset_],
                          sizeof(float));
        outFile.close();
    }
    */


}
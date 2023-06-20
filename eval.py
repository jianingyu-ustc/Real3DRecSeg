# -*- coding: utf-8 -*-
import math
import os, sys, argparse
import inspect
import numpy as np
import util_3d
import csv

def IoU():
    row = [scene_id]
    # 每个label统计一次IoU
    for i in range(len(CLASS_LABELS)):
        label = VALID_CLASS_IDS[i]
        if m1[label] + m2[label] - n[label] != 0:
            print CLASS_LABELS[i], m1[label], m2[label], n[label], round(n[label] / float(m1[label] + m2[label] - n[label]), 5)
            row.append(round(n[label] / float(m1[label] + m2[label] - n[label]), 5))
        else:
            print CLASS_LABELS[i], m1[label], m2[label], n[label], "nan"
            row.append("nan")
            

    with open("eval_5_10.csv","a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(row)
        
def PQ():
    return

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
sys.path.insert(0, '/home/jia/ScanNet/Benchmark')

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']
VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39])

'''
预测模型、gt模型的xyz的范围(米)
pred:
  x 1.53 ~ 7.29
  y -0.26 ~ 7.92
  z  -0.07 ~ 2.62
gt:
  x 1.61 ~ 7.45
  y  -0.03 ~ 8.21
  z  -0.01 ~ 2.60
'''

scene_id = sys.argv[1]
# 读取预测点云模型, 每个点的xyz均乘以20，变成以5cm为单位
pred_grid = util_3d.read_mesh_vertices('/home/jia/pred_5cm/' + scene_id + '.ply', label=True)
# index = np.lexsort((pred_grid[:, 2], pred_grid[:, 1], pred_grid[:, 0]))
# pred_grid = pred_grid[index]

# 创建点云查找表，形式为：pred_table[x][y][z]=label，xyz的单位：5cm，四舍五入找到最近邻的整数点
pred_table = np.zeros((1000,1000,1000),dtype=int)
for v in pred_grid:
    pred_table[v[0],v[1],v[2]] = v[3]
    
    
# 读取gt ply文件, 和预测模型的读取方式相同, 并且两个模型已经确保重合
gt_grid = util_3d.read_mesh_vertices('/home/jia/scannet/' + scene_id + '/' + scene_id + '_vh_clean_2.labels.ply', label=True)  # (237360, 3) line 0: [2.5091114  0.4083811  0.14877559] x y z
# index = np.lexsort((gt_grid[:,2], gt_grid[:,1], gt_grid[:,0]))
# gt_grid = gt_grid[index]


# 计算IoU =============================================================================================
m1 = np.zeros((100), dtype=int)
m2 = np.zeros((100), dtype=int)
n = np.zeros((100), dtype=int)

# 遍历gt模型中的每个点(xyz-label)，查找该点在pred_table中对应的点
for v in gt_grid:
    pred_label = pred_table[v[0], v[1], v[2]] + 1
    gt_label = v[3]
    flag1 = False
    flag2 = False
    if(pred_label in VALID_CLASS_IDS):
        flag1 = True
        m1[pred_label] = m1[pred_label] +1
    if(gt_label in VALID_CLASS_IDS):
        flag2 = True
        m2[gt_label] = m2[gt_label] +1
    if(flag1 and flag2 and pred_label == gt_label):
        n[pred_label] = n[pred_label] + 1

IoU()
# if(sys.argv[2] == "iou"):
#     IoU()
# elif(sys.argv[2] == "pq"):
#     PQ()

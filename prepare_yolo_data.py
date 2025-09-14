#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HOD数据集转换为YOLOv5格式
"""

import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
import random

def create_yolo_dataset():
    # 设置路径
    base_dir = '/root/HOD_reproduction'
    dataset_dir = os.path.join(base_dir, 'HOD-Benchmark-Dataset/dataset')
    yolo_dir = os.path.join(base_dir, 'yolo_data')
    
    # 读取metadata
    metadata_path = os.path.join(dataset_dir, 'metadata.csv')
    df = pd.read_csv(metadata_path)
    
    print(f"总样本数: {len(df)}")
    print(f"类别分布:")
    print(df['Category'].value_counts())
    
    # 类别映射
    class_names = ['alcohol', 'blood', 'cigarette', 'gun', 'insulting_gesture', 'knife']
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    
    # 为了快速演示，我们只使用部分数据
    # 每个类别最多使用200个样本
    sampled_data = []
    for category in class_names:
        category_data = df[df['Category'] == category]
        if len(category_data) > 200:
            category_data = category_data.sample(n=200, random_state=42)
        sampled_data.append(category_data)
    
    df_sampled = pd.concat(sampled_data, ignore_index=True)
    print(f"\n采样后样本数: {len(df_sampled)}")
    print(f"采样后类别分布:")
    print(df_sampled['Category'].value_counts())
    
    # 分割数据集 (70% train, 20% val, 10% test)
    train_df, temp_df = train_test_split(df_sampled, test_size=0.3, random_state=42, stratify=df_sampled['Category'])
    val_df, test_df = train_test_split(temp_df, test_size=0.33, random_state=42, stratify=temp_df['Category'])
    
    print(f"\n数据分割:")
    print(f"训练集: {len(train_df)}")
    print(f"验证集: {len(val_df)}")
    print(f"测试集: {len(test_df)}")
    
    # 处理每个数据集
    for split_name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        print(f"\n处理 {split_name} 数据集...")
        
        images_dir = os.path.join(yolo_dir, split_name, 'images')
        labels_dir = os.path.join(yolo_dir, split_name, 'labels')
        
        for idx, row in split_df.iterrows():
            # 复制图片
            src_img = os.path.join(dataset_dir, 'all/jpg', row['Image Name'])
            dst_img = os.path.join(images_dir, row['Image Name'])
            
            if os.path.exists(src_img):
                shutil.copy2(src_img, dst_img)
                
                # 复制并转换标注文件
                src_label = os.path.join(dataset_dir, 'all/txt', row['Annotation Name (YOLOv5)'])
                dst_label = os.path.join(labels_dir, row['Annotation Name (YOLOv5)'])
                
                if os.path.exists(src_label):
                    # 读取原始标注并转换类别ID
                    with open(src_label, 'r') as f:
                        lines = f.readlines()
                    
                    with open(dst_label, 'w') as f:
                        for line in lines:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                # 将类别名转换为ID
                                category = row['Category']
                                class_id = class_to_id[category]
                                # 写入新格式: class_id x_center y_center width height
                                f.write(f"{class_id} {' '.join(parts[1:])}\n")
    
    # 创建数据集配置文件
    yaml_content = f"""# HOD Dataset Configuration for YOLOv5

# Dataset paths
path: {yolo_dir}  # dataset root dir
train: train/images  # train images (relative to 'path')
val: val/images  # val images (relative to 'path')
test: test/images  # test images (relative to 'path')

# Classes
nc: {len(class_names)}  # number of classes
names: {class_names}  # class names
"""
    
    yaml_path = os.path.join(yolo_dir, 'hod_dataset.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n数据集配置文件已创建: {yaml_path}")
    print("YOLOv5数据集准备完成!")

if __name__ == '__main__':
    create_yolo_dataset()
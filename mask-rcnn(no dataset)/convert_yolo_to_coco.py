import os
import json
import random
import cv2
import numpy as np

# ================= 配置区域 =================
IMG_DIR = 'images'       # 原始图片文件夹 (混合 jpg, png, tif)
LBL_DIR = 'labels'       # 标签文件夹
OUT_IMG_DIR = 'images_jpg' # 【新生成】统一后的 JPG 图片文件夹
OUT_TRAIN_JSON = 'train.json'
OUT_VAL_JSON = 'val.json'
TRAIN_RATIO = 0.8
CLASS_NAME = 'tree'
# 支持的后缀
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.tif', '.tiff')
# ===========================================

def yolo_to_bbox(x_c, y_c, w, h, img_w, img_h):
    x_center_px = float(x_c) * img_w
    y_center_px = float(y_c) * img_h
    w_px = float(w) * img_w
    h_px = float(h) * img_h
    
    x_min = x_center_px - (w_px / 2)
    y_min = y_center_px - (h_px / 2)
    
    return [max(0, x_min), max(0, y_min), w_px, h_px]

def create_pseudo_mask(bbox, img_h, img_w):
    x, y, w, h = bbox
    # 创建矩形伪掩码
    segmentation = [[
        x, y, 
        x + w, y, 
        x + w, y + h, 
        x, y + h
    ]]
    return segmentation, int(w * h)

def convert():
    # 1. 创建输出目录
    if not os.path.exists(OUT_IMG_DIR):
        os.makedirs(OUT_IMG_DIR)
        print(f"创建目录: {OUT_IMG_DIR}")

    # 2. 获取所有有效图片文件
    all_files = os.listdir(IMG_DIR)
    img_files = [f for f in all_files if f.lower().endswith(VALID_EXTENSIONS)]
    
    print(f"发现 {len(img_files)} 张图片 (混合格式)")

    # 3. 预处理图片：统一转为 JPG (解决通道问题和格式混乱)
    # 建立映射表：原始文件名 -> 新文件名 (94.png -> 94.jpg)
    file_map = {} 
    
    for fname in img_files:
        base_name = os.path.splitext(fname)[0] # 去掉后缀，如 "94"
        new_fname = f"{base_name}.jpg"
        new_path = os.path.join(OUT_IMG_DIR, new_fname)
        
        # 如果已经转换过，跳过
        if os.path.exists(new_path):
            file_map[fname] = new_fname
            continue
            
        # 读取图片 (自动处理 jpg, png, tif)
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) # IMREAD_COLOR 强制转为 3 通道 BGR
        
        if img is None:
            print(f"⚠️ 警告：无法读取图片 {fname}，跳过。")
            continue
            
        # 保存为 JPG
        cv2.imwrite(new_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        file_map[fname] = new_fname

    print(f"已统一转换 {len(file_map)} 张图片到 {OUT_IMG_DIR}")

    # 4. 打乱并划分数据集 (基于原始文件名列表)
    original_names = list(file_map.keys())
    random.shuffle(original_names)
    
    split_idx = int(len(original_names) * TRAIN_RATIO)
    train_orig = original_names[:split_idx]
    val_orig = original_names[split_idx:]
    
    print(f"训练集: {len(train_orig)}, 验证集: {len(val_orig)}")

    def process_dataset(orig_list, output_json):
        coco_data = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": CLASS_NAME}]
        }
        
        img_id_counter = 0
        ann_id_counter = 0
        
        for orig_name in orig_list:
            new_name = file_map[orig_name] # 获取新的 .jpg 文件名
            base_name = os.path.splitext(orig_name)[0] # 获取基础名用于找 txt
            
            img_path = os.path.join(OUT_IMG_DIR, new_name)
            lbl_path = os.path.join(LBL_DIR, f"{base_name}.txt")
            
            # 读取统一后的图片以获取尺寸
            img = cv2.imread(img_path)
            if img is None: continue
            h, w, _ = img.shape
            
            # 添加图像信息 (使用新文件名)
            coco_data["images"].append({
                "id": img_id_counter,
                "file_name": new_name, 
                "width": w,
                "height": h
            })
            
            # 读取标签
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    lines = f.readlines()
                
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 5: continue
                    
                    cls_id = int(parts[0])
                    if cls_id != 0: continue 
                    
                    bbox = yolo_to_bbox(parts[1], parts[2], parts[3], parts[4], w, h)
                    seg, area = create_pseudo_mask(bbox, h, w)
                    
                    coco_data["annotations"].append({
                        "id": ann_id_counter,
                        "image_id": img_id_counter,
                        "category_id": 1,
                        "bbox": bbox,
                        "segmentation": seg,
                        "area": area,
                        "iscrowd": 0
                    })
                    ann_id_counter += 1
            
            img_id_counter += 1
            
        with open(output_json, 'w') as f:
            json.dump(coco_data, f)
        print(f"已生成 {output_json}")

    process_dataset(train_orig, OUT_TRAIN_JSON)
    process_dataset(val_orig, OUT_VAL_JSON)
    print("\n✅ 完成！请使用 'images_jpg' 文件夹作为你的图片输入目录。")

if __name__ == '__main__':
    convert()
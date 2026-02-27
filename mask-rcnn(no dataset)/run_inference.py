# run_inference.py (适配 MMDetection 3.x)
from mmdet.apis import init_detector, inference_detector
import cv2
import os
import torch

def main():
    # 1. 配置文件路径
    config_file = 'configs/mask_rcnn_config.py'
    
    # 2. 模型权重路径，这一行是需要根据你的训练结果进行修改的，不一定是epoch_6.pth
    checkpoint_file = 'work_dirs/tree_mask_rcnn/epoch_6.pth' 
    
    if not os.path.exists(checkpoint_file):
        print(f"❌ 错误：找不到权重文件 {checkpoint_file}")
        return

    # 3. 要预测的图片路径
    img_path = 'images_jpg/94.jpg' 
    
    if not os.path.exists(img_path):
        print(f"❌ 错误：找不到图片 {img_path}")
        return

    print("🤖 正在加载模型...")
    # 初始化检测器
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    print(f"🔍 正在预测图片：{img_path} ...")
    # 进行推理
    result = inference_detector(model, img_path)

    # 🚨 关键修改：3.x 版本需要从 result.pred_instances 获取数据
    pred_instances = result.pred_instances
    
    # 将数据从 GPU 移到 CPU 并转为 numpy
    bboxes = pred_instances.bboxes.cpu().numpy()
    scores = pred_instances.scores.cpu().numpy()
    masks = pred_instances.masks.cpu().numpy()

    # 4. 可视化结果
    img = cv2.imread(img_path)
    if img is None:
        print("❌ 无法读取图片")
        return
    
    count = 0
    for i in range(len(bboxes)):
        score = scores[i]
        
        # 只显示置信度大于 0.5 的结果
        if score > 0.5:
            bbox = bboxes[i]
            mask = masks[i]
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # 画边界框 (绿色, 厚度2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 画掩码 (红色半透明)
            # 确保 mask 是布尔值或 0/1
            overlay = img.copy()
            # mask 形状通常是 (H, W)，值为 0 或 1
            overlay[mask > 0.5] = [0, 0, 255] 
            cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            
            # 写标签
            label = f'Tree {score:.2f}'
            cv2.putText(img, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            count += 1

    # 5. 保存结果
    output_path = 'prediction_result.jpg'
    cv2.imwrite(output_path, img)
    
    print(f"✨ 预测完成！检测到 {count} 棵树 (置信度>0.5)。")
    print(f"📷 结果已保存至：{output_path}")
    print("💡 请在当前文件夹查看 prediction_result.jpg")

if __name__ == '__main__':
    main()
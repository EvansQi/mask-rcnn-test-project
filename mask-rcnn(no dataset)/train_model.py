# train_model.py (终极调试版)

import sys
import os

print(f"✅ [1] Python 路径: {sys.executable}")

try:
    from mmengine.config import Config
    from mmengine.runner import Runner
    print("✅ [2] 库导入成功 (mmengine, Runner)")
except Exception as e:
    print(f"❌ [2] 库导入失败: {e}")
    sys.exit(1)

def main():
    try:
        print("🚀 [3] 开始加载配置...")
        
        # 尝试加载配置
        cfg_path = 'configs/mask_rcnn_config.py'
        if not os.path.exists(cfg_path):
            print(f"❌ [3] 配置文件不存在: {cfg_path}")
            return
            
        cfg = Config.fromfile(cfg_path)
        print("✅ [4] 配置文件加载成功!")
        
        # 设置工作目录
        cfg.work_dir = './work_dirs/tree_mask_rcnn'
        os.makedirs(cfg.work_dir, exist_ok=True)
        print(f"📂 [5] 工作目录已准备: {cfg.work_dir}")
        
        cfg.gpu_ids = range(1)
        
        print("⏳ [6] 正在构建 Runner (这可能需要几秒钟)...")
        runner = Runner.from_cfg(cfg)
        print("✅ [7] Runner 构建成功!")
        
        print("🔥 [8] 开始训练 (runner.train())...")
        runner.train()
        
        print("✅ [9] 训练完成!")
        
    except Exception as e:
        print(f"\n💥💥 发生严重错误 💥💥")
        print(f"错误类型: {type(e).__name__}")
        print(f"错误信息: {e}")
        import traceback
        traceback.print_exc() # 打印详细堆栈信息

if __name__ == '__main__':
    print("✅ 开始执行脚本...")
    main()
    print("🏁 脚本执行结束。")
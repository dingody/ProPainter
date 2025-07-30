#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProPainter Colab初始化脚本
专为Google Colab环境优化的快速初始化脚本
"""

def init_propainter_colab():
    """
    Google Colab专用的ProPainter初始化函数
    一键完成所有环境配置和模型下载
    """
    
    print("🚀 ProPainter Colab 环境初始化")
    print("=" * 60)
    
    # 1. 检查环境
    print("1️⃣ 检查运行环境...")
    
    import torch
    import os
    import sys
    from pathlib import Path
    
    # 检查CUDA
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        print("⚠️  未检测到GPU，将使用CPU运行（速度较慢）")
    
    # 检查当前目录
    current_dir = Path.cwd()
    print(f"📁 当前目录: {current_dir}")
    
    # 2. 安装依赖
    print(f"\n2️⃣ 检查Python依赖...")
    
    required_packages = [
        'torch', 'torchvision', 'opencv-python', 'numpy', 
        'scipy', 'pillow', 'tqdm', 'requests', 'rapidocr'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            if package == 'opencv-python':
                import cv2
            elif package == 'pillow':
                import PIL
            elif package == 'rapidocr':
                from rapidocr import RapidOCR
            else:
                __import__(package)
            print(f"✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} - 需要安装")
    
    if missing_packages:
        print(f"\n📦 安装缺失的包: {', '.join(missing_packages)}")
        import subprocess
        
        # 特殊处理某些包的安装
        install_commands = []
        for pkg in missing_packages:
            if pkg == 'opencv-python':
                install_commands.append('pip install opencv-python-headless')
            elif pkg == 'rapidocr':
                install_commands.append('pip install rapidocr')
            else:
                install_commands.append(f'pip install {pkg}')
        
        for cmd in install_commands:
            print(f"执行: {cmd}")
            result = subprocess.run(cmd.split(), capture_output=True, text=True)
            if result.returncode != 0:
                print(f"❌ 安装失败: {cmd}")
                print(result.stderr)
            else:
                print(f"✅ 安装成功")
    
    # 3. 下载预训练模型
    print(f"\n3️⃣ 下载预训练模型...")
    
    # 使用我们的模型下载脚本
    try:
        from init_models import init_models
        success = init_models(skip_i3d=True)  # 在Colab中跳过I3D模型以节省空间
        if not success:
            print("❌ 模型下载失败")
            return False
    except Exception as e:
        print(f"❌ 模型下载出错: {e}")
        # 备用方案：手动下载核心模型
        print("🔄 尝试备用下载方案...")
        if not download_core_models():
            return False
    
    # 4. 创建示例脚本
    print(f"\n4️⃣ 创建快速测试脚本...")
    create_colab_examples()
    
    # 5. 验证安装
    print(f"\n5️⃣ 验证安装...")
    if verify_installation():
        print(f"\n🎉 ProPainter Colab环境初始化完成！")
        print_usage_guide()
        return True
    else:
        print(f"\n❌ 安装验证失败")
        return False

def download_core_models():
    """备用模型下载方案"""
    import requests
    from tqdm import tqdm
    import os
    from pathlib import Path
    
    # 创建weights目录
    weights_dir = Path('weights')
    weights_dir.mkdir(exist_ok=True)
    
    # 核心模型（跳过I3D以节省空间和时间）
    core_models = {
        'ProPainter.pth': 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth',
        'recurrent_flow_completion.pth': 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth',
        'raft-things.pth': 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth',
    }
    
    for model_name, url in core_models.items():
        model_path = weights_dir / model_name
        
        if model_path.exists():
            print(f"✅ 已存在: {model_name}")
            continue
        
        print(f"📥 下载: {model_name}")
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f, tqdm(
                desc=model_name,
                total=total_size,
                unit='B',
                unit_scale=True
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            print(f"✅ 完成: {model_name}")
            
        except Exception as e:
            print(f"❌ 下载失败 {model_name}: {e}")
            return False
    
    return True

def create_colab_examples():
    """创建Colab示例脚本"""
    
    # 创建快速测试脚本
    quick_test_script = '''# ProPainter 快速测试脚本

import os
import sys
from pathlib import Path

# 添加项目路径
sys.path.append('.')

def quick_test():
    """快速测试ProPainter是否正常工作"""
    
    print("🧪 ProPainter 快速测试")
    print("-" * 40)
    
    # 检查模型文件
    weights_dir = Path('weights')
    required_models = ['ProPainter.pth', 'recurrent_flow_completion.pth', 'raft-things.pth']
    
    for model in required_models:
        model_path = weights_dir / model
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024*1024)
            print(f"✅ {model} ({size_mb:.1f}MB)")
        else:
            print(f"❌ {model} - 文件缺失")
            return False
    
    # 测试OCR功能
    try:
        from rapidocr import RapidOCR
        ocr = RapidOCR()
        print("✅ OCR引擎初始化成功")
    except Exception as e:
        print(f"❌ OCR引擎初始化失败: {e}")
        return False
    
    # 测试PyTorch CUDA
    import torch
    if torch.cuda.is_available():
        print(f"✅ CUDA可用: {torch.version.cuda}")
        print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  CUDA不可用，将使用CPU")
    
    print("\\n🎉 所有测试通过！ProPainter已准备就绪")
    return True

if __name__ == "__main__":
    quick_test()
'''
    
    with open('quick_test.py', 'w', encoding='utf-8') as f:
        f.write(quick_test_script)
    
    # 创建使用示例
    usage_example = '''# ProPainter 使用示例

# 1. 生成OCR掩码（去除文字/字幕/水印）
!python generate_ocr_mask.py -i /path/to/your/video.mp4 -o ocr_masks --confidence 0.6

# 2. 使用ProPainter进行视频修复
!python inference_propainter.py -i /path/to/your/video.mp4 -m ocr_masks/video_name_mask --fp16

# 3. 处理图像序列
!python generate_ocr_mask.py -i /path/to/image/folder -o masks
!python inference_propainter.py -i /path/to/image/folder -m masks/folder_name_mask

# 4. 高级选项
!python inference_propainter.py \\
    -i input_video.mp4 \\
    -m mask_folder \\
    --width 1280 \\
    --height 720 \\
    --fp16 \\
    --subvideo_length 60 \\
    --save_frames
'''
    
    with open('usage_examples.py', 'w', encoding='utf-8') as f:
        f.write(usage_example)
    
    print("✅ 已创建示例脚本:")
    print("   - quick_test.py: 快速验证安装")
    print("   - usage_examples.py: 使用示例")

def verify_installation():
    """验证安装是否成功"""
    
    try:
        # 检查核心依赖
        import torch
        import torchvision
        import cv2
        import numpy as np
        from rapidocr import RapidOCR
        
        # 检查模型文件
        from pathlib import Path
        weights_dir = Path('weights')
        required_models = ['ProPainter.pth', 'recurrent_flow_completion.pth', 'raft-things.pth']
        
        for model in required_models:
            if not (weights_dir / model).exists():
                print(f"❌ 缺失模型: {model}")
                return False
        
        # 简单的功能测试
        from model.misc import get_device
        device = get_device()
        print(f"✅ 设备检测: {device}")
        
        # OCR测试
        ocr = RapidOCR()
        print("✅ OCR引擎: 正常")
        
        return True
        
    except Exception as e:
        print(f"❌ 验证失败: {e}")
        return False

def print_usage_guide():
    """打印使用指南"""
    
    guide = """
🎯 使用指南:

1️⃣ OCR掩码生成（去除文字/字幕/水印）:
   python generate_ocr_mask.py -i your_video.mp4 -o masks

2️⃣ 视频修复（推荐设置）:
   python inference_propainter.py -i your_video.mp4 -m masks/video_mask --fp16

3️⃣ 内存优化设置（适用于大视频）:
   python inference_propainter.py -i video.mp4 -m masks --fp16 --subvideo_length 40

4️⃣ 快速测试:
   python quick_test.py

📝 更多示例请查看: usage_examples.py

💡 提示:
   - 使用 --fp16 可以减少GPU内存占用
   - 调整 --subvideo_length 控制内存使用
   - OCR置信度建议设置为 0.5-0.8
   - 大视频建议先分段处理
    """
    
    print(guide)

# 如果是直接运行此脚本
if __name__ == "__main__":
    success = init_propainter_colab()
    if not success:
        exit(1)
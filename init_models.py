#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProPainter模型下载初始化脚本
自动下载所有必需的预训练模型到weights文件夹
适用于Google Colab环境的快速初始化
"""

import os
import hashlib
import requests
from pathlib import Path
from tqdm import tqdm
import sys

# 添加项目根目录到path
sys.path.append('.')
from utils.download_util import load_file_from_url, sizeof_fmt

# 模型配置信息
MODELS_CONFIG = {
    'ProPainter.pth': {
        'url': 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/ProPainter.pth',
        'size_mb': 280.5,  # 大约文件大小
        'description': 'ProPainter主模型 - 视频修复核心模型'
    },
    'recurrent_flow_completion.pth': {
        'url': 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/recurrent_flow_completion.pth',
        'size_mb': 55.6,
        'description': 'Flow补全模型 - 光流场修复'
    },
    'raft-things.pth': {
        'url': 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/raft-things.pth',
        'size_mb': 47.2,
        'description': 'RAFT模型 - 光流估计'
    },
    'i3d_rgb_imagenet.pt': {
        'url': 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/i3d_rgb_imagenet.pt',
        'size_mb': 50.1,
        'description': 'I3D模型 - VFID指标评估 (可选)'
    }
}

def get_file_size(file_path):
    """获取文件大小（MB）"""
    if os.path.exists(file_path):
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    return 0

def check_file_integrity(file_path, expected_size_mb, tolerance_mb=5.0):
    """检查文件完整性"""
    if not os.path.exists(file_path):
        return False
    
    actual_size_mb = get_file_size(file_path)
    size_diff = abs(actual_size_mb - expected_size_mb)
    
    if size_diff <= tolerance_mb:
        return True
    else:
        print(f"⚠️  文件大小异常: {file_path}")
        print(f"   预期: {expected_size_mb:.1f}MB, 实际: {actual_size_mb:.1f}MB")
        return False

def download_with_progress(url, file_path, description=""):
    """带进度条的下载函数"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        print(f"📥 下载: {description}")
        print(f"   URL: {url}")
        print(f"   大小: {sizeof_fmt(total_size)}")
        
        with open(file_path, 'wb') as file, tqdm(
            desc=os.path.basename(file_path),
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    pbar.update(len(chunk))
        
        print(f"✅ 下载完成: {file_path}")
        return True
        
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        if os.path.exists(file_path):
            os.remove(file_path)
        return False

def init_models(weights_dir='weights', force_download=False, skip_i3d=False):
    """
    初始化ProPainter模型
    
    Args:
        weights_dir (str): 模型保存目录
        force_download (bool): 强制重新下载
        skip_i3d (bool): 跳过I3D模型下载（用于VFID评估，非必需）
    """
    
    print("🚀 ProPainter模型初始化")
    print("=" * 60)
    
    # 创建weights目录
    weights_path = Path(weights_dir)
    weights_path.mkdir(exist_ok=True)
    
    print(f"📁 模型目录: {weights_path.absolute()}")
    
    # 统计信息
    total_models = len(MODELS_CONFIG)
    if skip_i3d:
        total_models -= 1
    
    downloaded_models = 0
    skipped_models = 0
    failed_models = 0
    total_size_mb = 0
    
    print(f"📊 需要下载 {total_models} 个模型文件")
    print("-" * 60)
    
    for model_name, config in MODELS_CONFIG.items():
        # 跳过I3D模型（如果指定）
        if skip_i3d and model_name == 'i3d_rgb_imagenet.pt':
            print(f"⏭️  跳过: {model_name} (VFID评估模型)")
            continue
            
        model_path = weights_path / model_name
        url = config['url']
        expected_size = config['size_mb']
        description = config['description']
        
        print(f"\n🔍 检查: {model_name}")
        print(f"   描述: {description}")
        
        # 检查文件是否存在且完整
        if not force_download and check_file_integrity(model_path, expected_size):
            actual_size = get_file_size(model_path)
            print(f"✅ 已存在: {model_path} ({actual_size:.1f}MB)")
            skipped_models += 1
            continue
        
        # 下载文件
        print(f"🔄 开始下载: {model_name}")
        
        if download_with_progress(url, model_path, description):
            # 验证下载的文件
            if check_file_integrity(model_path, expected_size):
                downloaded_models += 1
                total_size_mb += get_file_size(model_path)
                print(f"✅ 验证通过: {model_name}")
            else:
                failed_models += 1
                print(f"❌ 文件验证失败: {model_name}")
        else:
            failed_models += 1
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 下载汇总")
    print("-" * 60)
    print(f"✅ 成功下载: {downloaded_models} 个文件")
    print(f"⏭️  已跳过: {skipped_models} 个文件")
    print(f"❌ 下载失败: {failed_models} 个文件")
    print(f"💾 总下载量: {total_size_mb:.1f}MB")
    
    if failed_models > 0:
        print(f"\n⚠️  有 {failed_models} 个模型下载失败！")
        print("请检查网络连接或手动下载：")
        print("https://github.com/sczhou/ProPainter/releases/tag/v0.1.0")
        return False
    
    print(f"\n🎉 所有模型初始化完成！")
    print(f"📁 模型位置: {weights_path.absolute()}")
    
    # 列出所有模型文件
    print(f"\n📋 模型文件列表:")
    for model_file in sorted(weights_path.glob('*.p*')):
        size_mb = get_file_size(model_file)
        print(f"   - {model_file.name} ({size_mb:.1f}MB)")
    
    return True

def create_readme():
    """创建README文件"""
    readme_content = """# ProPainter 预训练模型

本目录包含 ProPainter 所需的预训练模型：

## 模型文件说明

- **ProPainter.pth** (280MB): ProPainter主模型，用于视频修复
- **recurrent_flow_completion.pth** (56MB): 递归光流补全模型
- **raft-things.pth** (47MB): RAFT光流估计模型
- **i3d_rgb_imagenet.pt** (50MB): I3D模型，用于VFID指标评估（可选）

## 自动下载

使用初始化脚本自动下载所有模型：

```bash
python init_models.py
```

## 手动下载

如果自动下载失败，可以从以下地址手动下载：
https://github.com/sczhou/ProPainter/releases/tag/v0.1.0

## 注意事项

- 所有模型文件总大小约 433MB
- 建议在良好的网络环境下进行下载
- I3D模型仅用于评估，推理时非必需
"""
    
    readme_path = Path('weights') / 'README.md'
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"📝 已创建: {readme_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ProPainter模型初始化脚本")
    parser.add_argument('--weights-dir', type=str, default='weights',
                      help='模型保存目录 (默认: weights)')
    parser.add_argument('--force', action='store_true',
                      help='强制重新下载所有模型')
    parser.add_argument('--skip-i3d', action='store_true',
                      help='跳过I3D模型下载（仅用于评估）')
    parser.add_argument('--create-readme', action='store_true',
                      help='创建README文件')
    
    args = parser.parse_args()
    
    # 初始化模型
    success = init_models(
        weights_dir=args.weights_dir,
        force_download=args.force,
        skip_i3d=args.skip_i3d
    )
    
    # 创建README
    if args.create_readme:
        create_readme()
    
    if success:
        print("\n🚀 ProPainter已准备就绪！")
        print("现在可以运行推理脚本：")
        print("python inference_propainter.py -i input_video.mp4 -m mask_folder")
    else:
        print("\n❌ 初始化失败，请检查网络连接后重试")
        sys.exit(1)

if __name__ == "__main__":
    main()
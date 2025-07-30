#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProPainter 快速修复脚本
针对Colab显存不足问题的一键解决方案
"""

def quick_fix_memory_issue():
    """快速修复显存问题"""
    
    print("🚨 ProPainter 显存不足快速修复")
    print("=" * 50)
    
    import os
    import torch
    import gc
    
    print("1️⃣ 设置环境变量...")
    # 设置PyTorch内存管理
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    print("2️⃣ 清理GPU内存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        
        # 显示内存状态
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        free_memory = total_memory - allocated_memory
        
        print(f"   GPU总内存: {total_memory:.2f}GB")
        print(f"   已用内存: {allocated_memory:.2f}GB")
        print(f"   可用内存: {free_memory:.2f}GB")
    
    print("\n3️⃣ 为720p视频推荐的命令:")
    
    commands = [
        {
            "name": "超保守模式 (推荐)",
            "cmd": "!python inference_propainter.py -i /content/input_fixed.mp4 -m /content/input_fixed_mask --fp16 --width 640 --height 360 --subvideo_length 8 --neighbor_length 5"
        },
        {
            "name": "低内存模式",  
            "cmd": "!python inference_propainter.py -i /content/input_fixed.mp4 -m /content/input_fixed_mask --fp16 --width 854 --height 480 --subvideo_length 12 --neighbor_length 6"
        },
        {
            "name": "中等模式 (如果上面成功)",
            "cmd": "!python inference_propainter.py -i /content/input_fixed.mp4 -m /content/input_fixed_mask --fp16 --width 960 --height 540 --subvideo_length 15 --neighbor_length 8"
        }
    ]
    
    for i, cmd_info in enumerate(commands, 1):
        print(f"\n📋 {cmd_info['name']}:")
        print(f"   {cmd_info['cmd']}")
    
    print(f"\n💡 参数说明:")
    print(f"   --width/--height: 降低分辨率减少内存使用")
    print(f"   --subvideo_length: 减少每次处理的帧数")
    print(f"   --neighbor_length: 减少邻近帧数量")
    print(f"   --fp16: 使用半精度浮点数")
    
    print(f"\n🔧 如果还是内存不足，尝试:")
    print(f"   1. 重启Colab Runtime清理内存")
    print(f"   2. 使用更小的 --subvideo_length (如 6 或 4)")
    print(f"   3. 进一步降低分辨率 (如 480x270)")
    print(f"   4. 使用分段处理模式")

def create_batch_commands():
    """创建批处理命令文件"""
    
    commands_script = '''#!/bin/bash
# ProPainter 批处理命令

echo "🚀 ProPainter 批处理开始..."

# 设置环境变量
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1

# 清理内存的Python脚本
python -c "
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print('✅ 内存已清理')
"

echo "📊 尝试不同的内存优化设置..."

# 尝试1: 超保守模式
echo "🎯 尝试1: 超保守模式 (640x360)"
python inference_propainter.py \\
    -i /content/input_fixed.mp4 \\
    -m /content/input_fixed_mask \\
    --fp16 \\
    --width 640 \\
    --height 360 \\
    --subvideo_length 8 \\
    --neighbor_length 5 \\
    --ref_stride 15

if [ $? -eq 0 ]; then
    echo "✅ 超保守模式成功完成！"
    exit 0
fi

echo "⚠️  超保守模式失败，尝试更小设置..."

# 尝试2: 极限保守模式
echo "🎯 尝试2: 极限保守模式 (480x270)"
python inference_propainter.py \\
    -i /content/input_fixed.mp4 \\
    -m /content/input_fixed_mask \\
    --fp16 \\
    --width 480 \\
    --height 270 \\
    --subvideo_length 6 \\
    --neighbor_length 4 \\
    --ref_stride 20

if [ $? -eq 0 ]; then
    echo "✅ 极限保守模式成功完成！"
    exit 0
fi

echo "❌ 所有尝试都失败了，请考虑："
echo "   1. 重启Colab Runtime"
echo "   2. 使用CPU模式（很慢）"
echo "   3. 分段处理视频"
'''
    
    with open('run_propainter_optimized.sh', 'w') as f:
        f.write(commands_script)
    
    print("✅ 已创建批处理脚本: run_propainter_optimized.sh")
    print("运行: !bash run_propainter_optimized.sh")

def create_segment_processor():
    """创建分段处理脚本"""
    
    segment_script = '''#!/usr/bin/env python3
# 视频分段处理脚本

import os
import subprocess
import cv2
from pathlib import Path

def split_and_process(input_video, mask_folder, segment_duration=20):
    """分段处理视频"""
    
    print(f"✂️  分段处理: {input_video}")
    
    # 获取视频信息
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = frame_count / fps
    cap.release()
    
    print(f"📊 视频时长: {total_duration:.1f}秒")
    print(f"🔢 分段长度: {segment_duration}秒")
    
    num_segments = int(total_duration / segment_duration) + 1
    print(f"📋 总共 {num_segments} 段")
    
    # 创建输出目录
    output_dir = Path("segments")
    output_dir.mkdir(exist_ok=True)
    
    video_name = Path(input_video).stem
    
    for i in range(num_segments):
        start_time = i * segment_duration
        end_time = min((i + 1) * segment_duration, total_duration)
        
        print(f"\\n🎬 处理第 {i+1}/{num_segments} 段...")
        
        # 分割视频
        segment_name = f"{video_name}_seg_{i:03d}.mp4"
        segment_path = output_dir / segment_name
        
        cmd_split = [
            "ffmpeg", "-y",
            "-i", input_video,
            "-ss", str(start_time),
            "-t", str(end_time - start_time),
            "-c", "copy",
            str(segment_path)
        ]
        
        subprocess.run(cmd_split, check=True, capture_output=True)
        
        # 处理这一段
        cmd_process = [
            "python", "inference_propainter.py",
            "-i", str(segment_path),
            "-m", mask_folder,
            "--fp16",
            "--width", "640",
            "--height", "360",
            "--subvideo_length", "10"
        ]
        
        try:
            subprocess.run(cmd_process, check=True)
            print(f"✅ 第 {i+1} 段完成")
        except subprocess.CalledProcessError:
            print(f"❌ 第 {i+1} 段失败")
    
    print("\\n🎉 分段处理完成！")
    print("📁 结果保存在 results/ 目录中")

if __name__ == "__main__":
    split_and_process("/content/input_fixed.mp4", "/content/input_fixed_mask")
'''
    
    with open('segment_processor.py', 'w') as f:
        f.write(segment_script)
    
    print("✅ 已创建分段处理脚本: segment_processor.py")
    print("运行: !python segment_processor.py")

if __name__ == "__main__":
    quick_fix_memory_issue()
    print("\n" + "="*50)
    create_batch_commands()
    print("\n" + "="*50)
    create_segment_processor()
    
    print("\n🎯 推荐解决方案（按优先级）:")
    print("1️⃣ 先运行上面推荐的超保守模式命令")
    print("2️⃣ 如果失败，使用: !bash run_propainter_optimized.sh")
    print("3️⃣ 如果还失败，使用: !python segment_processor.py")
    print("4️⃣ 最后选择：重启Colab Runtime清理内存")
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProPainter 显存优化脚本
针对Google Colab环境的显存不足问题提供解决方案
"""

import torch
import gc
import os
import sys
from pathlib import Path

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        print("🧹 GPU内存已清理")

def get_gpu_memory_info():
    """获取GPU内存信息"""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated_memory = torch.cuda.memory_allocated(0) / 1024**3
        cached_memory = torch.cuda.memory_reserved(0) / 1024**3
        free_memory = total_memory - cached_memory
        
        print(f"🖥️  GPU内存状态:")
        print(f"   总容量: {total_memory:.2f}GB")
        print(f"   已分配: {allocated_memory:.2f}GB")
        print(f"   缓存: {cached_memory:.2f}GB")
        print(f"   可用: {free_memory:.2f}GB")
        
        return {
            'total': total_memory,
            'allocated': allocated_memory,
            'cached': cached_memory,
            'free': free_memory
        }
    return None

def optimize_pytorch_memory():
    """优化PyTorch内存设置"""
    # 设置内存管理策略
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 设置其他优化选项
    torch.backends.cudnn.benchmark = False  # 禁用benchmark以节省内存
    torch.backends.cudnn.deterministic = True
    
    print("⚙️  PyTorch内存优化设置已应用")

def recommend_settings(video_path):
    """根据视频分析推荐设置"""
    import cv2
    
    print(f"🔍 分析视频: {video_path}")
    
    try:
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        print(f"📊 视频信息:")
        print(f"   分辨率: {width}x{height}")
        print(f"   帧率: {fps:.2f}fps")
        print(f"   总帧数: {frame_count}")
        print(f"   时长: {duration:.2f}秒")
        
        # 获取GPU内存信息
        gpu_info = get_gpu_memory_info()
        
        if gpu_info:
            free_memory = gpu_info['free']
            
            # 根据分辨率和可用内存推荐设置
            if width >= 1920:  # 1080p+
                if free_memory < 8:
                    recommended_width = 960
                    recommended_height = 540
                    subvideo_length = 10
                    print("🎯 推荐设置 (超低内存模式):")
                elif free_memory < 12:
                    recommended_width = 1280
                    recommended_height = 720
                    subvideo_length = 15
                    print("🎯 推荐设置 (低内存模式):")
                else:
                    recommended_width = width
                    recommended_height = height
                    subvideo_length = 25
                    print("🎯 推荐设置 (标准模式):")
            
            elif width >= 1280:  # 720p
                if free_memory < 6:
                    recommended_width = 854
                    recommended_height = 480
                    subvideo_length = 15
                    print("🎯 推荐设置 (超低内存模式):")
                elif free_memory < 10:
                    recommended_width = 960
                    recommended_height = 540
                    subvideo_length = 20
                    print("🎯 推荐设置 (低内存模式):")
                else:
                    recommended_width = width
                    recommended_height = height
                    subvideo_length = 30
                    print("🎯 推荐设置 (标准模式):")
            
            else:  # 480p及以下
                recommended_width = width
                recommended_height = height
                subvideo_length = 40
                print("🎯 推荐设置 (小分辨率):")
            
            print(f"   --width {recommended_width}")
            print(f"   --height {recommended_height}")
            print(f"   --subvideo_length {subvideo_length}")
            print(f"   --fp16")
            
            return {
                'width': recommended_width,
                'height': recommended_height,
                'subvideo_length': subvideo_length
            }
    
    except Exception as e:
        print(f"❌ 视频分析失败: {e}")
        return None

def run_optimized_inference(video_path, mask_path, custom_settings=None):
    """运行优化的推理命令"""
    
    # 清理内存
    clear_gpu_memory()
    optimize_pytorch_memory()
    
    # 获取推荐设置
    if custom_settings:
        settings = custom_settings
    else:
        settings = recommend_settings(video_path)
        if not settings:
            # 默认保守设置
            settings = {
                'width': 640,
                'height': 360,
                'subvideo_length': 10
            }
    
    # 构建命令
    cmd = [
        'python', 'inference_propainter.py',
        '-i', str(video_path),
        '-m', str(mask_path),
        '--fp16',
        '--width', str(settings['width']),
        '--height', str(settings['height']),
        '--subvideo_length', str(settings['subvideo_length']),
        '--neighbor_length', '5',  # 减少邻近帧数量
        '--ref_stride', '15'  # 增加参考帧间隔
    ]
    
    print(f"\n🚀 执行优化命令:")
    cmd_str = ' '.join(cmd)
    print(f"   {cmd_str}")
    
    # 执行命令
    import subprocess
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print("✅ 推理完成！")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 推理失败: {e}")
        return False

def progressive_processing(video_path, mask_path):
    """渐进式处理：从最保守设置开始，逐步提高质量"""
    
    print("🔄 渐进式处理模式")
    print("从最保守设置开始，逐步提高质量直到成功")
    
    # 定义不同的质量级别
    quality_levels = [
        {
            'name': '超低质量模式 (节省显存)',
            'width': 480,
            'height': 270,
            'subvideo_length': 8
        },
        {
            'name': '低质量模式',
            'width': 640,
            'height': 360,
            'subvideo_length': 12
        },
        {
            'name': '中等质量模式',
            'width': 854,
            'height': 480,
            'subvideo_length': 16
        },
        {
            'name': '高质量模式',
            'width': 1280,
            'height': 720,
            'subvideo_length': 20
        }
    ]
    
    for i, settings in enumerate(quality_levels):
        print(f"\n🎯 尝试 {settings['name']}")
        print(f"   分辨率: {settings['width']}x{settings['height']}")
        print(f"   子视频长度: {settings['subvideo_length']}")
        
        # 清理内存
        clear_gpu_memory()
        
        # 尝试运行
        success = run_optimized_inference(video_path, mask_path, settings)
        
        if success:
            print(f"✅ 成功完成！使用设置: {settings['name']}")
            return True
        else:
            print(f"❌ {settings['name']} 失败，尝试更保守设置...")
            if i == len(quality_levels) - 1:
                print("😵 所有设置都失败了，请检查输入文件或尝试更小的视频")
                return False
    
    return False

def split_video_processing(video_path, mask_path, segment_duration=30):
    """分段处理：将长视频分成小段分别处理"""
    
    print(f"✂️  分段处理模式 (每段 {segment_duration} 秒)")
    
    import cv2
    import subprocess
    from pathlib import Path
    
    try:
        # 获取视频信息
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps
        cap.release()
        
        print(f"📊 视频总时长: {duration:.2f}秒")
        
        # 计算分段数量
        num_segments = int(duration / segment_duration) + 1
        print(f"🔢 将分为 {num_segments} 段处理")
        
        video_stem = Path(video_path).stem
        output_dir = Path('temp_segments')
        output_dir.mkdir(exist_ok=True)
        
        processed_segments = []
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min((i + 1) * segment_duration, duration)
            
            print(f"\n🎬 处理第 {i+1}/{num_segments} 段 ({start_time:.1f}s - {end_time:.1f}s)")
            
            # 分割视频段
            segment_path = output_dir / f"{video_stem}_segment_{i:03d}.mp4"
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-ss', str(start_time),
                '-t', str(end_time - start_time),
                '-c', 'copy',
                str(segment_path)
            ]
            
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            
            # 处理这一段
            segment_mask_path = f"{mask_path}_segment_{i:03d}"
            success = run_optimized_inference(segment_path, segment_mask_path, {
                'width': 854,
                'height': 480,
                'subvideo_length': 20
            })
            
            if success:
                processed_segments.append(i)
                print(f"✅ 第 {i+1} 段处理完成")
            else:
                print(f"❌ 第 {i+1} 段处理失败")
        
        print(f"\n📊 处理结果: {len(processed_segments)}/{num_segments} 段成功")
        
        if len(processed_segments) == num_segments:
            print("🎉 所有段都处理成功！")
            print("💡 提示: 可以使用视频编辑软件将结果段合并")
            return True
        else:
            print("⚠️  部分段处理失败")
            return False
    
    except Exception as e:
        print(f"❌ 分段处理失败: {e}")
        return False

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="ProPainter 显存优化处理")
    parser.add_argument('-i', '--input', type=str, required=True, help='输入视频路径')
    parser.add_argument('-m', '--mask', type=str, required=True, help='掩码文件夹路径')
    parser.add_argument('--mode', type=str, choices=['auto', 'progressive', 'split'], 
                       default='auto', help='处理模式')
    parser.add_argument('--segment-duration', type=int, default=30,
                       help='分段模式的每段时长（秒）')
    
    args = parser.parse_args()
    
    print("💾 ProPainter 显存优化处理")
    print("=" * 50)
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"❌ 输入文件不存在: {args.input}")
        return
    
    if not os.path.exists(args.mask):
        print(f"❌ 掩码文件夹不存在: {args.mask}")
        return
    
    # 初始内存状态
    get_gpu_memory_info()
    
    if args.mode == 'auto':
        print("\n🎯 自动优化模式")
        success = run_optimized_inference(args.input, args.mask)
    elif args.mode == 'progressive':
        print("\n🔄 渐进式处理模式")
        success = progressive_processing(args.input, args.mask)
    elif args.mode == 'split':
        print("\n✂️  分段处理模式")
        success = split_video_processing(args.input, args.mask, args.segment_duration)
    
    if success:
        print("\n🎉 处理完成！")
    else:
        print("\n❌ 处理失败，请尝试其他模式或减小视频分辨率")

if __name__ == "__main__":
    main()
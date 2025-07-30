#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ProPainter 显存问题快速解决方案
专为你的720p视频显存不足问题设计
"""

import os
import torch
import gc
import subprocess
import sys

def immediate_fix():
    """立即解决显存问题的方案"""
    
    print("🚨 显存不足紧急修复方案")
    print("=" * 60)
    
    # 1. 设置环境变量
    print("1️⃣ 设置PyTorch内存管理...")
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # 2. 强制清理内存
    print("2️⃣ 强制清理GPU内存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        gc.collect()
        
        # 显示清理后的内存状态
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3  
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        free = total - reserved
        
        print(f"   总内存: {total:.2f}GB")
        print(f"   已分配: {allocated:.2f}GB")
        print(f"   已保留: {reserved:.2f}GB") 
        print(f"   可用: {free:.2f}GB")
        
        if free < 3.0:
            print("⚠️  可用内存不足3GB，建议重启Runtime")
    
    # 3. 提供具体的解决命令
    print("\n3️⃣ 针对你的问题的具体解决方案:")
    print("-" * 40)
    
    solutions = [
        {
            "level": "🟢 方案1: 超低内存模式 (强烈推荐)",
            "resolution": "480x270", 
            "memory": "约2GB",
            "cmd": "python inference_propainter.py -i /content/input_fixed.mp4 -m /content/input_fixed_mask --fp16 --width 480 --height 270 --subvideo_length 6 --neighbor_length 4 --ref_stride 20"
        },
        {
            "level": "🟡 方案2: 低内存模式",
            "resolution": "640x360",
            "memory": "约3GB", 
            "cmd": "python inference_propainter.py -i /content/input_fixed.mp4 -m /content/input_fixed_mask --fp16 --width 640 --height 360 --subvideo_length 8 --neighbor_length 5 --ref_stride 15"
        },
        {
            "level": "🟠 方案3: 中等内存模式",
            "resolution": "854x480",
            "memory": "约4GB",
            "cmd": "python inference_propainter.py -i /content/input_fixed.mp4 -m /content/input_fixed_mask --fp16 --width 854 --height 480 --subvideo_length 10 --neighbor_length 6"
        }
    ]
    
    for solution in solutions:
        print(f"\n{solution['level']}")
        print(f"   分辨率: {solution['resolution']}")
        print(f"   预估内存: {solution['memory']}")
        print(f"   命令: {solution['cmd']}")
    
    return solutions

def create_emergency_script():
    """创建紧急处理脚本"""
    
    script_content = '''#!/usr/bin/env python3
import os
import torch
import gc
import subprocess

# 设置环境变量
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 清理内存
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

print("🧹 内存已清理")

# 尝试不同的设置，从最保守开始
configs = [
    {
        "name": "超保守模式",
        "args": ["--width", "480", "--height", "270", "--subvideo_length", "6", "--neighbor_length", "4"]
    },
    {
        "name": "保守模式", 
        "args": ["--width", "640", "--height", "360", "--subvideo_length", "8", "--neighbor_length", "5"]
    },
    {
        "name": "中等模式",
        "args": ["--width", "854", "--height", "480", "--subvideo_length", "10", "--neighbor_length", "6"]
    }
]

for config in configs:
    print(f"\\n🎯 尝试 {config['name']}...")
    
    cmd = [
        "python", "inference_propainter.py",
        "-i", "/content/input_fixed.mp4",
        "-m", "/content/input_fixed_mask", 
        "--fp16"
    ] + config["args"]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✅ {config['name']} 成功完成！")
        break
    except subprocess.CalledProcessError as e:
        print(f"❌ {config['name']} 失败，尝试下一个...")
        # 清理内存后继续
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        continue
else:
    print("❌ 所有模式都失败了，请考虑重启Runtime或分段处理")
'''
    
    with open('emergency_fix.py', 'w') as f:
        f.write(script_content)
    
    print("✅ 已创建紧急修复脚本: emergency_fix.py")

def restart_runtime_guide():
    """重启Runtime指南"""
    
    print("\n🔄 如果上述方案都失败，请按以下步骤重启Runtime:")
    print("-" * 50)
    print("1. 在Colab菜单栏点击 '代码执行程序' → '重新启动会话'")
    print("2. 或者使用快捷键: Ctrl+M .")
    print("3. 重启后重新运行初始化脚本")
    print("4. 然后使用方案1的超低内存模式")
    
    print("\n💡 重启后的推荐流程:")
    print("# 重新初始化")
    print("!python init_colab.py")
    print()
    print("# 使用超保守设置")
    print("!python inference_propainter.py -i /content/input_fixed.mp4 -m /content/input_fixed_mask --fp16 --width 480 --height 270 --subvideo_length 6 --neighbor_length 4")

def main():
    """主执行函数"""
    print("🚑 ProPainter 显存问题急救包")
    print("=" * 60)
    
    # 立即执行修复
    solutions = immediate_fix()
    
    # 创建紧急脚本
    print("\n4️⃣ 创建自动处理脚本...")
    create_emergency_script()
    
    # 提供重启指南
    restart_runtime_guide()
    
    print("\n" + "=" * 60)
    print("🎯 立即行动方案:")
    print("1️⃣ 复制并运行方案1的命令（超低内存模式）")
    print("2️⃣ 如果失败，运行: !python emergency_fix.py")
    print("3️⃣ 如果还失败，重启Runtime后重试")
    print("4️⃣ 最后选择：考虑将视频分段处理")
    
    print(f"\n🔥 最推荐的命令（直接复制运行）:")
    print("!" + solutions[0]["cmd"])

if __name__ == "__main__":
    main()
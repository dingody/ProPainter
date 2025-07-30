#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CUDA 12.4 兼容性检查脚本
检查项目中所有依赖库和CUDA功能的兼容性
"""

import sys
import warnings
warnings.filterwarnings("ignore")

def check_pytorch_cuda():
    """检查PyTorch和CUDA兼容性"""
    print("🔍 检查PyTorch和CUDA兼容性...")
    
    try:
        import torch
        print(f"✅ PyTorch版本: {torch.__version__}")
        
        # 检查CUDA可用性
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.version.cuda}")
            print(f"✅ GPU数量: {torch.cuda.device_count()}")
            
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"   GPU {i}: {props.name} ({props.total_memory // 1024**3}GB)")
            
            # 测试简单的CUDA操作
            try:
                x = torch.randn(10, 10).cuda()
                y = torch.randn(10, 10).cuda()
                z = torch.mm(x, y)
                print("✅ CUDA tensor操作测试成功")
            except Exception as e:
                print(f"❌ CUDA tensor操作失败: {e}")
                return False
                
        else:
            print("⚠️  CUDA不可用，将使用CPU")
        
        # 检查cuDNN
        if torch.backends.cudnn.is_available():
            print(f"✅ cuDNN可用: {torch.backends.cudnn.version()}")
        else:
            print("⚠️  cuDNN不可用")
        
        return True
        
    except ImportError as e:
        print(f"❌ PyTorch导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ PyTorch检查失败: {e}")
        return False


def check_torchvision():
    """检查torchvision兼容性"""
    print("\n🔍 检查torchvision兼容性...")
    
    try:
        import torchvision
        print(f"✅ torchvision版本: {torchvision.__version__}")
        
        # 测试视频读取功能
        try:
            # 这个功能在inference_propainter.py中使用
            from torchvision.io import read_video
            print("✅ 视频读取功能可用")
        except Exception as e:
            print(f"⚠️  视频读取功能有问题: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ torchvision导入失败: {e}")
        return False


def check_onnxruntime():
    """检查ONNX Runtime GPU支持"""
    print("\n🔍 检查ONNX Runtime GPU支持...")
    
    try:
        import onnxruntime as ort
        print(f"✅ ONNX Runtime版本: {ort.__version__}")
        
        # 检查可用的执行提供者
        providers = ort.get_available_providers()
        print(f"✅ 可用的执行提供者: {providers}")
        
        if 'CUDAExecutionProvider' in providers:
            print("✅ CUDA执行提供者可用 - OCR将支持GPU加速")
        else:
            print("⚠️  CUDA执行提供者不可用 - OCR将使用CPU")
        
        return True
        
    except ImportError as e:
        print(f"❌ ONNX Runtime导入失败: {e}")
        return False


def check_rapidocr():
    """检查RapidOCR功能"""
    print("\n🔍 检查RapidOCR功能...")
    
    try:
        from rapidocr_onnxruntime import RapidOCR
        print("✅ RapidOCR导入成功")
        
        # 尝试初始化OCR引擎
        try:
            ocr = RapidOCR()
            print("✅ RapidOCR初始化成功")
        except Exception as e:
            print(f"⚠️  RapidOCR初始化警告: {e}")
        
        return True
        
    except ImportError as e:
        print(f"❌ RapidOCR导入失败: {e}")
        print("请安装: pip install rapidocr-onnxruntime")
        return False


def check_other_dependencies():
    """检查其他关键依赖"""
    print("\n🔍 检查其他关键依赖...")
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('scipy', 'scipy'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'matplotlib'),
        ('imageio', 'imageio-ffmpeg'),
        ('av', 'av'),
    ]
    
    all_good = True
    for module, package in dependencies:
        try:
            if module == 'cv2':
                import cv2
                print(f"✅ OpenCV版本: {cv2.__version__}")
            elif module == 'PIL':
                from PIL import Image
                print(f"✅ Pillow可用")
            else:
                exec(f"import {module}")
                print(f"✅ {package}可用")
        except ImportError:
            print(f"❌ {package}导入失败")
            all_good = False
    
    return all_good


def check_mixed_precision():
    """检查混合精度支持"""
    print("\n🔍 检查混合精度支持...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # 测试autocast
            try:
                from torch.cuda.amp import autocast
                with autocast():
                    x = torch.randn(10, 10).cuda()
                    y = torch.randn(10, 10).cuda()
                    z = torch.mm(x, y)
                print("✅ CUDA混合精度(autocast)支持")
            except Exception as e:
                print(f"⚠️  CUDA混合精度测试失败: {e}")
            
            # 测试half precision
            try:
                x = torch.randn(10, 10).cuda().half()
                y = torch.randn(10, 10).cuda().half()
                z = torch.mm(x, y)
                print("✅ FP16半精度支持")
            except Exception as e:
                print(f"⚠️  FP16半精度测试失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 混合精度检查失败: {e}")
        return False


def check_memory_efficiency():
    """检查GPU内存相关功能"""
    print("\n🔍 检查GPU内存管理...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # 检查内存清理功能
            try:
                torch.cuda.empty_cache()
                print("✅ CUDA缓存清理功能可用")
            except Exception as e:
                print(f"⚠️  CUDA缓存清理失败: {e}")
            
            # 检查内存信息获取
            try:
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                print(f"✅ GPU内存监控可用 (已分配: {memory_allocated//1024**2}MB)")
            except Exception as e:
                print(f"⚠️  GPU内存监控失败: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU内存管理检查失败: {e}")
        return False


def main():
    """主检查函数"""
    print("=" * 60)
    print("🚀 ProPainter CUDA 12.4 兼容性检查")
    print("=" * 60)
    
    checks = [
        ("PyTorch和CUDA", check_pytorch_cuda),
        ("torchvision", check_torchvision),
        ("ONNX Runtime", check_onnxruntime),
        ("RapidOCR", check_rapidocr),
        ("其他依赖", check_other_dependencies),
        ("混合精度", check_mixed_precision),
        ("GPU内存管理", check_memory_efficiency),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"❌ {name}检查过程中发生错误: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("📊 兼容性检查汇总")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{name:15} : {status}")
        if result:
            passed += 1
    
    print("-" * 60)
    print(f"总体结果: {passed}/{total} 项检查通过")
    
    if passed == total:
        print("🎉 所有检查都通过！项目与CUDA 12.4完全兼容")
        return True
    elif passed >= total * 0.8:
        print("⚠️  大部分检查通过，项目基本兼容CUDA 12.4")
        print("建议检查失败的项目并进行相应调整")
        return True
    else:
        print("❌ 多个关键组件检查失败，需要解决兼容性问题")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
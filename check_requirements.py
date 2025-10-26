#!/usr/bin/env python3
"""
检查并安装必要的依赖包
"""

import subprocess
import sys

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✅ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ 安装 {package} 失败")
        return False

def check_and_install():
    """检查并安装必要的包"""
    required_packages = [
        "tqdm",
        "numpy",
        "torch",
        "tensorboard"
    ]
    
    print("🔍 检查依赖包...")
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"⚠️  {package} 未安装，正在安装...")
            install_package(package)

if __name__ == "__main__":
    check_and_install()
    print("\n🎉 依赖检查完成!")

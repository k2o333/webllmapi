import sys
import os
from pathlib import Path

print("=== 环境测试 ===")
print(f"Python路径: {sys.executable}")
print(f"工作目录: {os.getcwd()}")
print(f"文件存在: {Path('config.yaml').exists()}")
print("=== 测试完成 ===")
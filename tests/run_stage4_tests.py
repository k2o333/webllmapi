import os
import sys
import time
import pytest
import asyncio
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent))

# 导入清理模块
from cleanup import run_cleanup

def run_tests():
    """运行阶段4测试并记录结果"""
    # 记录开始时间
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("="*80)
    print(f"开始阶段4测试 - {start_datetime}")
    print("="*80)
    
    # 首先清理环境
    print("\n[1/3] 清理测试环境...")
    run_cleanup()
    
    # 运行测试
    print("\n[2/3] 运行阶段4测试...")
    test_file = Path(__file__).parent / "test_stage4.py"
    result = pytest.main(["-v", str(test_file)])
    
    # 计算测试时间
    end_time = time.time()
    duration = end_time - start_time
    end_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 准备测试结果摘要
    print("\n[3/3] 生成测试报告...")
    
    # 获取测试结果文件路径
    results_file = Path(__file__).parent / "test_results.txt"
    
    # 添加本次测试的结果到测试结果文件
    with open(results_file, "a") as f:
        f.write("\n\n")
        f.write("="*80 + "\n")
        f.write(f"测试执行摘要 - {end_datetime}\n")
        f.write("="*80 + "\n")
        f.write(f"开始时间: {start_datetime}\n")
        f.write(f"结束时间: {end_datetime}\n")
        f.write(f"测试持续时间: {duration:.2f} 秒\n")
        f.write(f"测试结果代码: {result}\n")
        f.write(f"测试状态: {'成功' if result == 0 else '失败'}\n")
        f.write("="*80 + "\n")
    
    # 打印测试摘要
    print("\n" + "="*80)
    print(f"测试执行摘要")
    print("="*80)
    print(f"开始时间: {start_datetime}")
    print(f"结束时间: {end_datetime}")
    print(f"测试持续时间: {duration:.2f} 秒")
    print(f"测试状态: {'成功' if result == 0 else '失败'}")
    print("="*80)
    print(f"\n测试结果已添加到: {results_file}")
    
    return result

if __name__ == "__main__":
    sys.exit(run_tests())
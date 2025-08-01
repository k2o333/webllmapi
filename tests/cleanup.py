import os
import shutil
import psutil
import asyncio
from pathlib import Path

async def cleanup_test_environment():
    """清理测试环境，删除所有测试相关的临时文件和进程"""
    print("开始清理测试环境...")
    
    # 获取项目根目录
    root_dir = Path(__file__).parent.parent
    
    # 清理 __pycache__ 目录
    def clean_pycache(directory):
        for root, dirs, files in os.walk(directory):
            # 删除 __pycache__ 目录
            if '__pycache__' in dirs:
                pycache_path = os.path.join(root, '__pycache__')
                print(f"删除 {pycache_path}")
                shutil.rmtree(pycache_path, ignore_errors=True)
            # 删除 .pyc 文件
            for file in files:
                if file.endswith('.pyc'):
                    file_path = os.path.join(root, file)
                    print(f"删除 {file_path}")
                    os.remove(file_path)
    
    # 清理测试缓存目录
    cache_dirs = [
        '.pytest_cache',
        '.mypy_cache',
        '.coverage'
    ]
    
    for cache_dir in cache_dirs:
        cache_path = root_dir / cache_dir
        if cache_path.exists():
            print(f"删除缓存目录: {cache_path}")
            shutil.rmtree(cache_path, ignore_errors=True)
    
    # 清理 Python 缓存文件
    clean_pycache(root_dir)
    
    # 清理日志文件
    logs_dir = root_dir / 'logs'
    if logs_dir.exists():
        for log_file in logs_dir.glob('*.log'):
            print(f"删除日志文件: {log_file}")
            log_file.unlink()
    
    # 终止可能残留的浏览器进程（根据需求只关闭Firefox）
    browser_processes = ['firefox', 'playwright']
    for proc in psutil.process_iter(['name']):
        try:
            for browser in browser_processes:
                if browser in proc.info['name'].lower():
                    print(f"终止进程: {proc.info['name']} (PID: {proc.pid})")
                    proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    
    print("测试环境清理完成")

def run_cleanup():
    """运行清理程序"""
    asyncio.run(cleanup_test_environment())

if __name__ == "__main__":
    run_cleanup()
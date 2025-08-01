import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

def test_config_loading() -> bool:
    """测试配置加载"""
    try:
        print("\n=== 开始配置测试 ===")
        
        # 检查配置文件是否存在
        config_path = Path("config.yaml")
        if not config_path.exists():
            print("错误: 找不到配置文件 config.yaml")
            return False
        
        print("✓ 配置文件存在")
        
        # 尝试导入配置管理器
        try:
            print("正在导入配置管理器...")
            from config_manager import get_config, get_llm_site_config, get_enabled_llm_sites
            print("✓ 成功导入配置管理器")
        except ImportError as e:
            print(f"错误: 无法导入配置管理器 - {str(e)}")
            return False
        
        # 尝试加载配置
        try:
            print("正在加载配置...")
            config = get_config()
            print("✓ 配置加载成功")
            
            # 测试全局设置
            print("\n全局设置:")
            print(f"  超时时间: {config.global_settings.timeout}秒")
            print(f"  最大重试次数: {config.global_settings.max_retries}")
            
            # 测试 API 设置
            print("\nAPI设置:")
            print(f"  OpenAI模型: {config.api_settings.openai.model}")
            print(f"  温度: {config.api_settings.openai.temperature}")
            
            # 测试代理设置
            print("\n代理设置:")
            print(f"  代理启用状态: {config.proxy_settings.enabled}")
            
            # 测试 LLM 站点
            enabled_sites = get_enabled_llm_sites()
            print(f"\n已启用的LLM站点 ({len(enabled_sites)}):")
            
            for site in enabled_sites:
                print("\n站点信息:")
                print(f"  ID: {site.id}")
                print(f"  名称: {site.name}")
                print(f"  URL: {site.url}")
                print(f"  实例池大小: {site.pool_size}")
                print(f"  每实例最大请求数: {site.max_requests_per_instance}")
                print(f"  每实例最大内存(MB): {site.max_memory_per_instance_mb}")
                
                # 测试选择器
                print("\n  选择器配置:")
                print(f"    输入区域: {site.selectors.get('input_area')}")
                print(f"    提交按钮: {site.selectors.get('submit_button')}")
                
                # 测试响应处理
                print("\n  响应处理配置:")
                print(f"    处理类型: {site.response_handling.type}")
                print(f"    提取选择器: {site.response_handling.extraction_selector_key}")
                
                # 测试健康检查
                print("\n  健康检查配置:")
                print(f"    启用状态: {site.health_check.enabled}")
                print(f"    检查间隔: {site.health_check.interval_seconds}秒")
            
        except Exception as e:
            print(f"错误: 配置加载失败 - {str(e)}")
            import traceback
            print(traceback.format_exc())
            return False
        
        print("\n=== 配置测试完成 ===")
        return True
        
    except Exception as e:
        print(f"错误: 测试过程中发生异常 - {str(e)}")
        import traceback
        print(traceback.format_exc())
        return False

if __name__ == "__main__":
    try:
        print("开始执行测试...")
        success = test_config_loading()
        if success:
            print("\n✅ 配置测试通过")
            sys.exit(0)
        else:
            print("\n❌ 配置测试失败")
            sys.exit(1)
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        print(traceback.format_exc())
        sys.exit(1)
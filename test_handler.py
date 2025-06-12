import asyncio
import logging
from browser_handler import LLMWebsiteAutomator
from config_manager import get_config, get_llm_site_config
from utils import setup_logging

# 直接在这里配置日志，用于本次测试
setup_logging(log_level="DEBUG")
logger = logging.getLogger("wrapper_api")

async def main_test():
    """
    一个独立的测试函数，用于验证 LLMWebsiteAutomator 的核心功能。
    """
    logger.info("--- 开始独立测试 browser_handler ---")
    
    try:
        # 1. 加载配置
        logger.info("正在加载配置...")
        # 注意：这里直接用 get_config() 可能会因为没有 lifespan 而出问题
        # 我们手动调用 load_config 来模拟
        from config_manager import load_config
        config = load_config()
        logger.info("配置加载成功。")

        # 2. 获取启用的站点配置
        enabled_sites = [s for s in config.llm_sites if s.enabled]
        if not enabled_sites:
            logger.error("在 config.yaml 中没有找到启用的站点！测试无法继续。")
            return
        
        site_config = enabled_sites[0]
        logger.info(f"将要测试的站点: {site_config.id}")

        # 3. 创建并初始化 Automator 实例
        logger.info(f"正在为 {site_config.id} 创建 LLMWebsiteAutomator 实例...")
        automator = LLMWebsiteAutomator(site_config)
        
        logger.info("正在调用 automator.initialize()...")
        await automator.initialize()
        logger.info("automator.initialize() 调用成功！Playwright 实例已启动。")

        # 4. (可选) 进行一次简单的健康检查
        is_healthy = await automator.is_healthy()
        logger.info(f"实例健康检查结果: {'健康' if is_healthy else '不健康'}")
        
        # 5. (可选) 进行一次模拟请求
        # logger.info("正在发送一个测试消息...")
        # response = await automator.send_message("你好")
        # logger.info(f"收到响应: {response[:100]}...")

    except Exception as e:
        logger.critical("独立测试过程中发生严重错误！", exc_info=True)
    finally:
        if 'automator' in locals() and automator:
            logger.info("正在清理 automator...")
            await automator.cleanup()
            logger.info("清理完成。")
    
    logger.info("--- 独立测试结束 ---")

if __name__ == "__main__":
    asyncio.run(main_test())
# 文件: browser_utils.py

import logging
import shutil
from pathlib import Path
from typing import List

from playwright.async_api import Page
from models import LLMSiteConfig

logger = logging.getLogger("wrapper_api.browser_utils")


async def robust_click(page: Page, selector: str, timeout: int = 10000):
    """
    一个健壮的点击辅助函数，它会按顺序尝试多种方法来点击一个元素。
    (此函数从 browser_handler.py 移至此处)
    """
    log_prefix = f"[robust_click for '{selector[:60]}...']"
    
    try:
        await page.wait_for_selector(selector, state="attached", timeout=timeout)
    except Exception as find_exc:
        logger.error(f"{log_prefix} 无法在DOM中定位到元素，所有点击尝试中止。错误: {find_exc}")
        error_snapshot_dir = Path("error_snapshots")
        error_snapshot_dir.mkdir(exist_ok=True)
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        screenshot_path = error_snapshot_dir / f"robust_click_fail_{timestamp}.png"
        try:
            await page.screenshot(path=screenshot_path)
            logger.info(f"Saved error screenshot for robust_click failure to {screenshot_path}")
        except Exception as se:
            logger.error(f"Failed to save error screenshot: {se}")
        raise find_exc

    # 策略 1: Playwright 标准点击
    try:
        logger.debug(f"{log_prefix} 尝试策略 1: 标准 page.click()")
        await page.click(selector, timeout=timeout)
        logger.info(f"{log_prefix} 标准 click 成功。")
        return
    except Exception as e:
        logger.warning(f"{log_prefix} 策略 1 (标准 click) 失败: {e}")

    # 获取元素句柄用于后续策略
    element_handle = await page.locator(selector).element_handle()
    if not element_handle:
        raise RuntimeError(f"无法为选择器 '{selector}' 获取元素句柄")

    # 策略 2: JavaScript evaluation click
    try:
        logger.debug(f"{log_prefix} 尝试策略 2: JavaScript evaluation click")
        await element_handle.evaluate("(element) => { if (element && typeof element.click === 'function') element.click(); }")
        logger.info(f"{log_prefix} JavaScript evaluation click 成功。")
        return
    except Exception as e:
        logger.warning(f"{log_prefix} 策略 2 (JS evaluation) 失败: {e}")

    # 策略 3: Dispatch Event 'click'
    try:
        logger.debug(f"{log_prefix} 尝试策略 3: dispatch_event('click')")
        await element_handle.dispatch_event('click')
        logger.info(f"{log_prefix} dispatch_event('click') 成功。")
        return
    except Exception as e:
        logger.warning(f"{log_prefix} 策略 3 (dispatch_event) 失败: {e}")

    final_error_msg = f"{log_prefix} 所有点击策略均失败。"
    logger.error(final_error_msg)
    raise RuntimeError(final_error_msg)


def cleanup_profile_templates(base_profile_dir: Path, configured_sites: List[LLMSiteConfig]):
    """
    在应用关闭时清理profiles目录。
    只保留在config.yaml中定义的站点的模板目录，删除所有其他的。
    """
    if not base_profile_dir.exists():
        logger.info(f"基础 Profile 目录 '{base_profile_dir}' 不存在，无需清理。")
        return

    logger.info(f"开始清理 Profile 模板目录: '{base_profile_dir}'")

    # 1. 获取所有在 config.yaml 中配置的 profile 目录名
    configured_profile_names = set()
    for site in configured_sites:
        # 将相对路径转换为纯目录名
        profile_path = Path(site.firefox_profile_dir)
        configured_profile_names.add(profile_path.name)
    
    logger.info(f"将保留以下已配置的 Profile 模板: {configured_profile_names}")

    # 2. 遍历磁盘上实际存在的目录
    for item in base_profile_dir.iterdir():
        if item.is_dir():
            if item.name not in configured_profile_names:
                logger.warning(f"发现未配置的 Profile 目录 '{item.name}'，正在删除...")
                try:
                    shutil.rmtree(item)
                    logger.info(f"成功删除目录: '{item}'")
                except Exception as e:
                    logger.error(f"删除目录 '{item}' 时失败: {e}")
    
    logger.info("Profile 模板目录清理完成。")
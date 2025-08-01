import asyncio
import time
from pathlib import Path
from playwright.async_api import async_playwright, Page, BrowserContext, ElementHandle
import logging
from collections import Counter, defaultdict

# --- 配置区 ---
TARGET_URL = "https://www.wenxiaobai.com/"  # 修改为你的 LLM 网页 URL
MONITOR_DURATION_SECONDS = 100  # 手动提交后，监控页面变化的时长
POLL_INTERVAL_SECONDS = 0.2   # 检查 MutationObserver 记录的频率
MIN_TEXT_CHANGE_LENGTH = 3    # 文本节点内容变化被认为是“有效”的最小长度
MAX_XPATH_DEPTH = 10          # 向上追溯父节点以寻找共同容器的最大深度
MIN_OCCURRENCES_FOR_CANDIDATE = 3 # 一个 XPath 需要至少变化这么多次才被认为是候选容器

LOG_FILE_PATH = "logs/dynamic_stream_detector.log"

# --- 日志设置 ---
detector_logger = logging.getLogger("DynamicStreamDetector")
detector_logger.setLevel(logging.INFO) # INFO 或 DEBUG
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
detector_logger.addHandler(ch)

Path("logs").mkdir(exist_ok=True)
fh = logging.FileHandler(LOG_FILE_PATH, mode='w', encoding='utf-8')
fh.setFormatter(formatter)
detector_logger.addHandler(fh)

ENABLE_PLAYWRIGHT_TRACING = True
TRACE_DIR = Path("traces/dynamic_stream_detection")

# --- JavaScript Code to Inject ---
JS_MUTATION_OBSERVER_SCRIPT = """
async (options) => {
    if (window.llmStreamChanges) { // Clear previous if any
        window.llmStreamChanges.observer.disconnect();
        window.llmStreamChanges.records = [];
    }

    window.llmStreamChanges = {
        records: [],
        observer: null,
        options: options || {} // minTextChangeLength
    };

    const getElementXPath = (element) => {
        if (!element || !element.parentNode) return null; // Element might be detached
        if (element.id !== '') return `id("${element.id}")`;
        if (element === document.body) return '/html/body'; // More stable than just 'body'

        let ix = 0;
        const siblings = element.parentNode.childNodes;
        for (let i = 0; i < siblings.length; i++) {
            const sibling = siblings[i];
            if (sibling === element) {
                const parentPath = getElementXPath(element.parentNode);
                if (!parentPath) return null; // If parent has no path, this one won't either
                return parentPath + '/' + element.tagName.toLowerCase() + '[' + (ix + 1) + ']';
            }
            if (sibling.nodeType === 1 && sibling.tagName === element.tagName) {
                ix++;
            }
        }
        return null; // Should not happen if element is in DOM
    };

    const observer = new MutationObserver((mutationsList) => {
        for (const mutation of mutationsList) {
            let targetElement = null;
            let changeType = mutation.type;
            let textSample = "";

            if (mutation.type === 'childList') {
                if (mutation.addedNodes.length > 0) {
                    targetElement = mutation.addedNodes[0].parentElement || mutation.target; // Prefer parent if adding nodes
                    // Try to get text from the added node or its children
                    const addedNode = mutation.addedNodes[0];
                    if (addedNode.textContent && addedNode.textContent.trim().length > 0) {
                        textSample = addedNode.textContent.trim().substring(0, 50);
                    } else if (addedNode.nodeType === Node.ELEMENT_NODE && addedNode.innerText) {
                         textSample = addedNode.innerText.trim().substring(0,50);
                    }
                } else if (mutation.removedNodes.length > 0) {
                    // Less likely for streaming output, but good to note
                    targetElement = mutation.target;
                }
            } else if (mutation.type === 'characterData') {
                targetElement = mutation.target.parentElement; // The text node's parent
                if (mutation.target.textContent &&
                    mutation.target.textContent.trim().length >= (window.llmStreamChanges.options.minTextChangeLength || 1)) {
                    textSample = mutation.target.textContent.trim().substring(0, 50);
                } else {
                    continue; // Ignore small character data changes
                }
            }

            if (targetElement) {
                const xpath = getElementXPath(targetElement);
                if (xpath) { // Only record if xpath could be generated
                    window.llmStreamChanges.records.push({
                        xpath: xpath,
                        type: changeType,
                        timestamp: Date.now(),
                        textSample: textSample.replace(/\\n/g, ' '),
                        targetTagName: targetElement.tagName.toLowerCase(),
                        // attributes: Array.from(targetElement.attributes).map(attr => ({name: attr.name, value: attr.value}))
                    });
                }
            }
        }
    });

    observer.observe(document.documentElement, {
        childList: true,
        subtree: true,
        characterData: true,
        characterDataOldValue: false // Don't need old value for this
    });
    window.llmStreamChanges.observer = observer;
    return true; // Signal observer started
}
"""

JS_GET_CHANGES_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.records) {
        const records = window.llmStreamChanges.records;
        window.llmStreamChanges.records = []; // Clear after fetching
        return records;
    }
    return [];
}
"""

JS_STOP_OBSERVER_SCRIPT = """
() => {
    if (window.llmStreamChanges && window.llmStreamChanges.observer) {
        window.llmStreamChanges.observer.disconnect();
        delete window.llmStreamChanges;
        return true;
    }
    return false;
}
"""

async def analyze_changes(all_changes: list):
    detector_logger.info(f"\n--- 分析 {len(all_changes)} 条 DOM 变化记录 ---")
    if not all_changes:
        detector_logger.info("没有检测到 DOM 变化。")
        return

    xpath_counts = Counter()
    xpath_texts = defaultdict(list)
    xpath_types = defaultdict(Counter)
    xpath_parents = defaultdict(Counter)

    for change in all_changes:
        xpath = change['xpath']
        xpath_counts[xpath] += 1
        if change['textSample']:
            xpath_texts[xpath].append(change['textSample'])
        xpath_types[xpath][change['type']] += 1

        # Analyze parent XPaths
        current_xpath_parts = xpath.split('/')
        for i in range(1, min(len(current_xpath_parts) -1, MAX_XPATH_DEPTH + 1)): # -1 to avoid full document path
            parent_xpath = "/".join(current_xpath_parts[:-i])
            if parent_xpath: # Ensure not empty
                 xpath_parents[parent_xpath][xpath] +=1 # Count how many changed children a parent has


    detector_logger.info("\n--- XPath 变化频率统计 (按变化次数排序) ---")
    # Sort by count, then by XPath string length (shorter preferred for same count)
    sorted_xpaths = sorted(xpath_counts.items(), key=lambda item: (item[1], -len(item[0])), reverse=True)

    candidate_containers = []

    for xpath, count in sorted_xpaths:
        if count < MIN_OCCURRENCES_FOR_CANDIDATE: # Filter out infrequent changes
            continue

        types_str = ", ".join([f"{t}:{c}" for t,c in xpath_types[xpath].items()])
        detector_logger.info(f"XPath: {xpath}")
        detector_logger.info(f"  发生变化次数: {count}")
        detector_logger.info(f"  变化类型: {types_str}")
        text_samples = xpath_texts.get(xpath, [])
        if text_samples:
            unique_samples = list(set(text_samples)) # Show unique samples
            detector_logger.info(f"  关联文本片段 (最多显示3个不同样本): {unique_samples[:3]}")
        candidate_containers.append(xpath) # Add to raw candidates

    detector_logger.info(f"\n--- 初步候选容器 XPath (出现次数 >= {MIN_OCCURRENCES_FOR_CANDIDATE}): ---")
    if candidate_containers:
        for i, xpath_cand in enumerate(candidate_containers[:10]): # Show top 10
            detector_logger.info(f"  {i+1}. {xpath_cand} (变化次数: {xpath_counts[xpath_cand]})")
    else:
        detector_logger.info("  未找到足够频繁变化的 XPath 作为初步候选。")


    detector_logger.info("\n--- 潜在父容器分析 (基于子元素变化频率) ---")
    # Sort parent candidates by the number of distinct children that changed under them,
    # and then by the total number of changes under them.
    sorted_parent_candidates = sorted(
        xpath_parents.items(),
        key=lambda item: (len(item[1]), sum(item[1].values())), # (num_distinct_changed_children, total_child_changes)
        reverse=True
    )

    final_suggested_containers = []
    if sorted_parent_candidates:
        detector_logger.info("最有可能是流容器的父 XPath (按其下不同变化子元素的数量和总变化量排序):")
        for parent_xpath, children_counts in sorted_parent_candidates[:5]: # Top 5 parent candidates
            num_distinct_children = len(children_counts)
            total_child_changes = sum(children_counts.values())
            detector_logger.info(f"  父 XPath: {parent_xpath}")
            detector_logger.info(f"    其下有 {num_distinct_children} 个不同子路径发生变化, 总计 {total_child_changes} 次子变化。")
            if num_distinct_children > 1 or total_child_changes >= MIN_OCCURRENCES_FOR_CANDIDATE * 1.5 : # Heuristic
                final_suggested_containers.append(parent_xpath)
    else:
        detector_logger.info("  未找到明显的父容器。")

    if not final_suggested_containers and candidate_containers:
        detector_logger.info("\n由于未找到强父容器信号，直接建议变化最频繁的元素作为容器:")
        final_suggested_containers.extend(candidate_containers[:3]) # Fallback to top direct changes

    detector_logger.info("\n--- 总结：建议的流式容器 XPath (按推断可能性排序) ---")
    if final_suggested_containers:
        for i, xpath_sugg in enumerate(list(dict.fromkeys(final_suggested_containers))[:5]): # Unique, top 5
            detector_logger.info(f"  建议 {i+1}: {xpath_sugg}")
    else:
        detector_logger.info("  未能自动推断出明确的流式容器 XPath。请检查详细的 XPath 变化频率统计，并结合 Playwright Trace 进行分析。")

    detector_logger.info("\n提示: 请将以上建议的 XPath 在浏览器开发者工具中测试，并结合 Playwright Trace (如果启用) 进行验证。")


async def run_observer(page: Page, context: BrowserContext):
    detector_logger.info(f"动态流容器探测器启动。请在浏览器中手动输入 Prompt 并提交。")
    input("完成手动提交后，请按 Enter 键开始监控 DOM 变化...")

    detector_logger.info(f"正在页面注入 MutationObserver 并开始监听 {MONITOR_DURATION_SECONDS} 秒...")
    await page.evaluate(JS_MUTATION_OBSERVER_SCRIPT, {"minTextChangeLength": MIN_TEXT_CHANGE_LENGTH})

    all_dom_changes = []
    monitor_start_time = time.time()

    try:
        while time.time() - monitor_start_time < MONITOR_DURATION_SECONDS:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            changes_batch = await page.evaluate(JS_GET_CHANGES_SCRIPT)
            if changes_batch:
                detector_logger.debug(f"获取到 {len(changes_batch)} 条 DOM 变化记录。")
                all_dom_changes.extend(changes_batch)
        detector_logger.info(f"监控时间达到 {MONITOR_DURATION_SECONDS} 秒。总共记录 {len(all_dom_changes)} 条原始变化。")

    except KeyboardInterrupt:
        detector_logger.info("用户中断监控。")
    except Exception as e:
        detector_logger.error(f"监控过程中发生错误: {e}", exc_info=True)
    finally:
        detector_logger.info("正在停止页面 MutationObserver...")
        await page.evaluate(JS_STOP_OBSERVER_SCRIPT)

    if all_dom_changes:
        await analyze_changes(all_dom_changes)
    else:
        detector_logger.info("在监控期间未记录到任何 DOM 变化。")


async def main():
    async with async_playwright() as p:
        detector_logger.info("正在启动 Firefox...")
        browser = await p.firefox.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        trace_path = None
        if ENABLE_PLAYWRIGHT_TRACING:
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            trace_path = TRACE_DIR / f"dynamic_detect_trace_{timestamp}.zip"
            await context.tracing.start(name="dynamic_detect", screenshots=True, snapshots=True, sources=True)
            detector_logger.info(f"Playwright Tracing 已启动。Trace 文件将保存到: {trace_path}")

        try:
            detector_logger.info(f"正在导航到: {TARGET_URL}")
            await page.goto(TARGET_URL, timeout=60000)
            detector_logger.info(f"页面加载完成。")

            await run_observer(page, context)

        except Exception as e:
            detector_logger.error(f"主流程发生错误: {e}", exc_info=True)
        finally:
            detector_logger.info("正在关闭浏览器...")
            if ENABLE_PLAYWRIGHT_TRACING and trace_path and context:
                try:
                    await context.tracing.stop(path=str(trace_path))
                    detector_logger.info(f"Playwright Trace 已保存到: {trace_path}")
                    detector_logger.info(f"你可以使用 'playwright show-trace {trace_path}' 来查看。")
                except Exception as e_trace:
                    detector_logger.error(f"保存 Playwright Trace 时出错: {e_trace}")
            
            await browser.close()
            detector_logger.info("浏览器已关闭。脚本结束。")

if __name__ == "__main__":
    asyncio.run(main())
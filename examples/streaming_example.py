#!/usr/bin/env python3
"""
流式响应示例脚本
演示如何使用流式响应功能与LLM进行交互
"""

import asyncio
import time
import sys
import json
import argparse
from typing import Dict, Any, List

# 添加项目根目录到路径
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from browser_automator import LLMWebsiteAutomator
from config_manager import get_llm_site_config

async def stream_demo(model_name: str, prompt: str, format_output: bool = True):
    """演示流式响应功能
    
    Args:
        model_name: 要使用的模型名称
        prompt: 要发送的提示
        format_output: 是否格式化输出（彩色、带时间戳）
    """
    # 初始化自动化器
    site_config = get_llm_site_config(model_name)
    automator = LLMWebsiteAutomator(site_config)
    
    try:
        # 连接到浏览器
        print(f"正在连接到 {model_name}...")
        await automator.initialize()
        
        # 准备性能指标
        start_time = time.time()
        chunk_count = 0
        total_chars = 0
        
        # 发送消息并处理流式响应
        print(f"\n>>> 用户: {prompt}\n")
        print(f">>> {model_name} (流式响应):")
        
        # 获取流式响应
        stream_gen = await automator.send_message(prompt, stream=True)
        
        # 处理流式响应
        async for chunk in stream_gen:
            chunk_count += 1
            total_chars += len(chunk)
            
            # 输出格式化或原始响应
            if format_output:
                # 计算已用时间
                elapsed = time.time() - start_time
                # 输出带有时间戳和颜色的文本
                sys.stdout.write(f"\033[32m{chunk}\033[0m")
                sys.stdout.flush()
            else:
                # 简单输出
                sys.stdout.write(chunk)
                sys.stdout.flush()
        
        # 输出性能统计
        elapsed = time.time() - start_time
        print(f"\n\n--- 性能统计 ---")
        print(f"总时间: {elapsed:.2f}秒")
        print(f"数据块数量: {chunk_count}")
        print(f"总字符数: {total_chars}")
        print(f"平均速度: {total_chars/elapsed:.2f} 字符/秒")
        print(f"平均块大小: {total_chars/chunk_count:.2f} 字符/块") if chunk_count > 0 else None
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭浏览器
        await automator.close()

async def compare_streaming_vs_regular(model_name: str, prompt: str):
    """比较流式响应和常规响应的性能差异
    
    Args:
        model_name: 要使用的模型名称
        prompt: 要发送的提示
    """
    # 初始化自动化器
    site_config = get_llm_site_config(model_name)
    automator = LLMWebsiteAutomator(site_config)
    
    try:
        # 连接到浏览器
        print(f"正在连接到 {model_name}...")
        await automator.initialize()
        
        # 测试流式响应
        print(f"\n>>> 测试流式响应")
        stream_start = time.time()
        
        stream_gen = await automator.send_message(prompt, stream=True)
        stream_response = ""
        first_token_time = None
        
        async for chunk in stream_gen:
            if not first_token_time and chunk.strip():
                first_token_time = time.time()
            stream_response += chunk
            # 输出进度指示
            sys.stdout.write(".")
            sys.stdout.flush()
        
        stream_end = time.time()
        stream_total = stream_end - stream_start
        stream_ttft = first_token_time - stream_start if first_token_time else None
        
        print(f"\n流式响应完成，总时间: {stream_total:.2f}秒")
        if stream_ttft:
            print(f"首个token时间: {stream_ttft:.2f}秒")
        
        # 测试常规响应
        print(f"\n>>> 测试常规响应")
        regular_start = time.time()
        
        regular_response = await automator.send_message(prompt, stream=False)
        
        regular_end = time.time()
        regular_total = regular_end - regular_start
        
        print(f"常规响应完成，总时间: {regular_total:.2f}秒")
        
        # 比较结果
        print(f"\n--- 性能比较 ---")
        print(f"流式响应总时间: {stream_total:.2f}秒")
        print(f"常规响应总时间: {regular_total:.2f}秒")
        print(f"差异: {regular_total - stream_total:.2f}秒 ({(regular_total/stream_total - 1)*100:.1f}%)")
        
        if stream_ttft:
            print(f"流式响应首个token时间: {stream_ttft:.2f}秒")
            print(f"用户体验改善: {regular_total - stream_ttft:.2f}秒 ({(regular_total/stream_ttft - 1)*100:.1f}%)")
        
        # 验证响应内容一致性
        content_match = stream_response.strip() == regular_response.strip()
        print(f"响应内容一致: {'是' if content_match else '否'}")
        
    except Exception as e:
        print(f"错误: {e}")
    finally:
        # 关闭浏览器
        await automator.close()

async def api_streaming_demo(prompt: str, api_url: str = "http://localhost:8000/v1/chat/completions"):
    """演示通过API使用流式响应
    
    Args:
        prompt: 要发送的提示
        api_url: API端点URL
    """
    import aiohttp
    
    # 准备请求数据
    request_data = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    
    print(f"通过API发送流式请求: {api_url}")
    print(f"提示: {prompt}")
    print("\n响应:")
    
    start_time = time.time()
    async with aiohttp.ClientSession() as session:
        async with session.post(api_url, json=request_data) as response:
            if response.status != 200:
                print(f"API错误: {response.status}")
                print(await response.text())
                return
                
            # 处理SSE流
            buffer = ""
            async for line in response.content:
                line = line.decode('utf-8')
                buffer += line
                
                if buffer.endswith('\n\n'):
                    for event in buffer.strip().split('\n\n'):
                        if not event.startswith('data: '):
                            continue
                            
                        data = event[6:]  # 移除 'data: ' 前缀
                        
                        if data == '[DONE]':
                            continue
                            
                        try:
                            json_data = json.loads(data)
                            content = json_data.get('choices', [{}])[0].get('delta', {}).get('content', '')
                            if content:
                                sys.stdout.write(content)
                                sys.stdout.flush()
                        except json.JSONDecodeError:
                            print(f"无法解析JSON: {data}")
                    
                    buffer = ""
    
    elapsed = time.time() - start_time
    print(f"\n\n总时间: {elapsed:.2f}秒")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="流式响应演示")
    parser.add_argument("--model", "-m", default="default", help="要使用的模型名称")
    parser.add_argument("--prompt", "-p", default="请写一篇关于人工智能的短文，包括其历史、现状和未来发展。", help="要发送的提示")
    parser.add_argument("--compare", "-c", action="store_true", help="比较流式和常规响应")
    parser.add_argument("--api", "-a", action="store_true", help="使用API进行流式响应")
    parser.add_argument("--api-url", default="http://localhost:8000/v1/chat/completions", help="API端点URL")
    parser.add_argument("--raw", "-r", action="store_true", help="输出原始响应，不带格式")
    
    args = parser.parse_args()
    
    if args.api:
        asyncio.run(api_streaming_demo(args.prompt, args.api_url))
    elif args.compare:
        asyncio.run(compare_streaming_vs_regular(args.model, args.prompt))
    else:
        asyncio.run(stream_demo(args.model, args.prompt, not args.raw))

if __name__ == "__main__":
    main()
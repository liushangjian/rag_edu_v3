import asyncio
from llm import LLMClient
import traceback

async def simple_test():
    client = LLMClient()
    
    print("开始测试...")
    
    try:
        # 测试 Claude
        print("\n测试 Claude:")
        response = await client.chat(
            model="claude-3-5-sonnet-20241022",
            prompt="你好，请用一句话介绍你自己"
        )
        print(f"Claude响应: {response['content']}")
        
        # 测试 GPT
        print("\n测试 GPT:")
        response = await client.chat(
            model="gpt-4o-mini",
            prompt="你好，请用一句话介绍你自己"
        )
        print(f"GPT响应: {response['content']}")
        
        # 测试 Deepseek
        print("\n测试 Deepseek:")
        response = await client.chat(
            model="deepseek-chat",
            prompt="你好，请用一句话介绍你自己"
        )
        print(f"Deepseek响应: {response['content']}")
        
    except Exception as e:
        print(f"测试出错: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(simple_test()) 
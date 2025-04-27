import asyncio

# 定义一个异步函数
async def task1():
    print("Task 1 started")
    await asyncio.sleep(1)  # 模拟耗时操作
    print("Task 1 finished")
    return 1

# 定义另一个异步函数
async def main():
    print("Main function started")
    result = await task1()  # 等待 task1 执行完毕
    print(f"Result from task1: {result}")
    print("Main function finished")

# 运行异步程序
asyncio.run(main())
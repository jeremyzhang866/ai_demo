import asyncio
import logging

# === 日志配置 ===
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(funcName)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


async def say(name: str, sec: int) -> None:
    logger.info(f"{name}：开始任务（等待 {sec} 秒）")
    await asyncio.sleep(sec)
    logger.info(f"{name}：结束任务")


async def main() -> None:
    logger.info("主任务开始")

    await asyncio.gather(
        say("任务A", 1),
        say("任务B", 2),
        say("任务C", 3),
    )

    logger.info("所有任务完成")


if __name__ == "__main__":
    asyncio.run(main())
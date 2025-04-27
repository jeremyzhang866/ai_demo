from fastapi import FastAPI, HTTPException, Query
import aiohttp
import asyncio
import async_timeout

app = FastAPI()


# 异步请求函数，支持重试和超时
async def fetch_with_retry(url: str, retries: int = 3, timeout: int = 5):
    for attempt in range(1, retries + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with async_timeout.timeout(timeout):
                    async with session.get(url) as response:
                        response.raise_for_status()
                        data = await response.text()
                        return data
        except Exception as e:
            if attempt == retries:
                raise HTTPException(status_code=500, detail=f"请求失败：{str(e)}")
            await asyncio.sleep(1)
            return None
    return None


# FastAPI 路由
@app.get("/fetch")
async def fetch_endpoint(
    url: str = Query(..., description="要请求的目标 URL"),
    retries: int = Query(3, ge=1, le=5),
    timeout: int = Query(5, ge=1, le=30)
):
    result = await fetch_with_retry(url, retries=retries, timeout=timeout)
    return {"url": url, "result": result[:1000]}
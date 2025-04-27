from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

# 如果你想在代码中直接启动 Uvicorn 服务器，可以使用以下代码
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8100)
"""
BTC 5min LLM 预测系统 V3 - 启动入口

通过 uvicorn 启动 FastAPI 应用，端口从 .env 读取（默认 8000）。
"""

import uvicorn

from binance_predict.config.settings import settings


def main():
    uvicorn.run(
        "binance_predict.main:app",
        host="0.0.0.0",
        port=settings.api_port,
        log_level=settings.log_level.lower(),
        reload=False,
    )


if __name__ == "__main__":
    main()

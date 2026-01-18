# 使用 Python 3.11 轻量版作为基础镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 1. 安装系统依赖: Tesseract OCR 及其中文语言包
# libgl1-mesa-glx 有时是图像处理库需要的依赖
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-chi-sim \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# 2. 复制并安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. 复制项目代码并启动
COPY . .
CMD ["python", "stock_bot.py"]
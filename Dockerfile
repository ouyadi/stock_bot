# 使用轻量级的 Python 3.11 镜像
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 复制依赖文件并安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制机器人代码
COPY stock_bot.py .

# 启动命令 (-u 参数保证日志实时输出)
CMD ["python", "-u", "stock_bot.py"]
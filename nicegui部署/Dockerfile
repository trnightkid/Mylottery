FROM ubuntu:latest
LABEL authors="zhuangyan"

ENTRYPOINT ["top", "-b"]

FROM python:3.11-slim
# 设置工作目录
WORKDIR /app
# 复制依赖文件
COPY requirements.txt .
# 安装依赖（使用阿里云镜像加速）
RUN pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
# 复制应用代码
COPY lottery_nicegui.py .
# 设置时区
ENV TZ=Asia/Shanghai
# 创建输出目录
RUN mkdir -p /app/output
# 暴露端口
EXPOSE 8080
# 启动命令
CMD ["python", "lottery_nicegui.py"]
# 截图金额修改器（云部署版）

上传截图 → 按指令修改图中金额 → 回传图片（同字号/字体/粗细/填充）。

## 部署（Render 免费档）

1. 推送到 GitHub 私有仓库（含 fonts/ 与 server.py）
2. Render 创建 Blueprint 或 Web Service，连接该仓库
3. 自动按 render.yaml 构建：`pip install -r requirements.txt`
4. 启动：`uvicorn server:app --host 0.0.0.0 --port $PORT`
5. 获得固定公网 URL，无需本机开机

## 本地运行

```
pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8000
```

## 说明

- OCR：RapidOCR（onnxruntime），模型随 pip 包自动安装，无需外网下载
- 字体：随仓库打包在 fonts/（Arial / 微软雅黑 / 黑体 / 宋体 常规+粗体）
- 结果图片输出到 static/output/，实例休眠后不保留（免费档无持久磁盘）

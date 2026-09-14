# 截图金额修改器（网页版）

上传截图 → 自动识别金额 → 填写新金额 → 回传图片（同字号 / 粗细 / 位置）。

纯浏览器运行，图片不上传任何服务器。

## 在线使用

<https://b-aoge.github.io/amount-editor/amount-editor-web.html>

## 本地使用

直接用浏览器打开 `amount-editor-web.html` 即可；或起一个静态服务：

```bash
python -m http.server 8000
```

然后访问 <http://127.0.0.1:8000/amount-editor-web.html>

## 使用步骤

1. 上传截图（APP 界面 / 凭证单据均可）
2. 自动识别并选中金额（可点击列表切换，或拖动框选）
3. 填写新金额
4. 点击「处理图片」，下载结果图

## 说明

- OCR：Tesseract.js，浏览器本地识别，首次加载约 10MB 识别引擎，之后有缓存
- 渲染：Canvas 逐像素分析原数字的颜色 / 背景 / 高度，自动匹配字号与粗细
- 部署：GitHub Pages 静态托管，免费、固定链接、无需服务器

# Vector RAG Step 2

本阶段将真实向量 RAG 的运行信息展示到 Web 页面。

## 修改内容

- `careerpilot/templates/index.html`
  - 动态展示 `hybrid / vector / keyword / keyword_fallback`
  - 展示综合相关度、向量相似度、关键词相关度
  - 展示 Embedding 模型、索引复用/重建状态
  - 向量检索降级时展示原因
- `careerpilot/static/style.css`
  - 新增向量 RAG 可观测性样式
- `tests/test_api_observability.py`
  - 检查 API 字段和 Web 展示字段

## 验证

```bash
pytest -q
python -m uvicorn careerpilot.api:app --host 0.0.0.0 --port 8001 --env-file .env
```

打开 `http://127.0.0.1:8001`，运行一次分析，在“RAG 检索来源”中确认：

- 检索模式为 `hybrid`
- Embedding 模型为 `BAAI/bge-small-zh-v1.5`
- 向量相似度不为 0
- 没有降级提示

# RAG Evaluation Step 3

本阶段比较 `keyword`、`vector`、`hybrid` 三种检索模式。

## 新增内容

- `careerpilot/rag_evaluation.py`：Hit@1、Hit@3、MRR、延迟统计和报告生成。
- `data/evaluation/rag_queries.json`：8 条固定检索评估样本。
- `scripts/evaluate_rag.py`：真实 BGE + FAISS 离线评估脚本。
- `tests/test_rag_evaluation.py`：不依赖模型下载的指标单元测试。

## 运行

```bash
pytest -q
python scripts/evaluate_rag.py --repeat 3
```

输出：

```text
outputs/rag_evaluation/rag_evaluation.json
outputs/rag_evaluation/rag_evaluation.md
```

评估脚本会先预热本地 BGE 模型和 FAISS 索引，正式延迟不包含首次模型下载时间。

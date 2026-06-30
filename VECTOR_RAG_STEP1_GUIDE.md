# Phase 5 Step 1：真实向量 RAG

本压缩包已经合并第一轮向量 RAG 修改，未包含 `.env`、`.git`、运行缓存和输出文件。

## 推荐使用方式

在原项目根目录执行：

```bash
cd ~/ZYX/实习项目/CareerPilot-LangGraph
cp .env ../CareerPilot.env.backup
```

解压本压缩包后，用其中同名目录内的文件覆盖当前项目文件。不要删除原项目中的 `.env`。

然后执行：

```bash
source ~/.venv/bin/activate
pytest -q
```

在 `.env` 中加入：

```dotenv
CAREERPILOT_RAG_MODE=hybrid
CAREERPILOT_RAG_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
CAREERPILOT_RAG_KNOWLEDGE_DIR=data/knowledge
CAREERPILOT_RAG_INDEX_DIR=data/vector_store
CAREERPILOT_RAG_TOP_K=4
CAREERPILOT_RAG_CHUNK_SIZE=420
CAREERPILOT_RAG_CHUNK_OVERLAP=60
CAREERPILOT_RAG_VECTOR_WEIGHT=0.75
CAREERPILOT_RAG_DEVICE=cpu
```

首次真实检索会建立 `data/vector_store` 索引。

# CareerPilot-LangGraph 求职匹配报告（样例）

## 1. 岗位定位
- 目标岗位：大模型应用 / Agent 工程实习生
- 候选人定位：适合定位为：有 CV/推理优化背景的大模型应用/Agent 工程候选人。
- 匹配分：**约 50-60 / 100**

## 2. 已匹配技能
- Python
- PyTorch
- Transformer
- DashScope
- Qwen
- Prompt Engineering
- CUDA
- TensorRT
- vLLM
- 量化

## 3. 需要补强的技能/信号
- LangGraph
- RAG
- Tool Calling / Function Calling
- MCP
- FAISS / Milvus
- FastAPI / Docker
- 开源 PR 经历

## 4. 推荐 GitHub 项目路线
仓库名：**CareerPilot-LangGraph**

### 核心功能
- 简历 PDF/文本解析：抽取教育背景、项目经历、技能关键词
- JD 结构化分析：抽取核心要求、加分项、技术栈和岗位关键词
- LangGraph 状态图编排：ProfileExtractor → JDAnalyzer → Matcher → ProjectPlanner → ResumeRewriter
- 结构化输出：将匹配分、缺口技能、项目建议输出为 JSON + Markdown 报告
- RAG 扩展：把公司 JD、面经、开源 issue 建成知识库，支持岗位定制检索
- 工具调用扩展：接入 GitHub issue 查询、项目 README 分析、面试题生成工具

## 5. 可写入简历的项目 bullet
- 基于 LangGraph 设计多节点 Agent 工作流，实现简历解析、JD 分析、技能匹配、项目规划与简历改写的端到端自动化。
- 接入 Qwen/DashScope OpenAI-compatible API，结合结构化输出约束生成可复用的 JSON/Markdown 求职分析报告。
- 设计确定性匹配工具与可扩展工具调用接口，为后续 RAG、GitHub issue 检索和 vLLM serving benchmark 预留扩展点。

# legal_diff_agent

法规检索、案例检索与多地对比的 LangGraph 智能体工程。**Agent 行为约定（官网优先入库等）**：见根目录 [`SKILLS.md`](SKILLS.md)。启动 CLI 时默认 **自动读入并拼接到系统提示**（`agent_skills_enabled`，见 `app/config.py`）；基础指令在 `app/graph/prompts.py`。数据流水线将公开法律文本解析、按「条/款」切分；检索层采用 **BM25 + 向量（FAISS）** 混合融合；**两地对比**在检索候选上使用 **句向量余弦相似度 + 贪心配对**，实现条文片段级语义对齐（非人工逐条标注）。LLM 默认通过 **本地 Transformers（GPU）** 直接加载模型（也可切换到 Ollama）。

## 环境（Conda 环境 `lawdif`）

依赖与 `requirements.txt` 一致；环境由 Conda 管理（`conda env list` 可见），不在项目目录下。

**首次创建（若尚未创建）：**

```bash
conda create -n lawdif python=3.11 -y
conda activate lawdif
pip install --upgrade pip setuptools wheel
cd /path/to/legal_diff_agent
pip install -r requirements.txt
```

**日常使用：**

```bash
conda activate lawdif
cd /path/to/legal_diff_agent
```

验证 LangGraph 等是否可用：

```bash
conda activate lawdif
python -c "from importlib.metadata import version; from langgraph.prebuilt import create_react_agent; print('langgraph', version('langgraph')); print(create_react_agent)"
cd /path/to/legal_diff_agent && PYTHONPATH=. python -m pytest tests/ -q
```

运行应用：

```bash
conda activate lawdif
export PYTHONPATH=.
python -m app.main -q "你的问题"
```

LLM 后端配置：

1. 默认：`LLM_BACKEND=transformers`（使用 `local_model_path` 加载本地模型）
2. 可选：`LLM_BACKEND=ollama`（使用 Ollama 调本地/远端 Ollama 服务）

环境变量可在项目根目录 `.env` 中配置（见 `app/config.py`），例如：

```bash
LLM_BACKEND=transformers
LOCAL_MODEL_PATH=/path/to/Qwen3-8B
LOCAL_DTYPE=bfloat16
LOCAL_MAX_NEW_TOKENS=256
```

若切到 Ollama，可设置：

```bash
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen2.5
```

## 运行

（先 `conda activate lawdif`）

```bash
export PYTHONPATH=.
python -m app.main -q "上海和深圳关于这个高空抛物的规定有什么区别，有典型的司法实践案例吗"
```

交互模式：

```bash
python -m app.main
```

交互模式会把每轮 `user/assistant` 历史写入 SQLite 缓存（默认 `data/outputs/chat_cache.sqlite`），后续同一 `--thread-id` 的多轮对话会自动读取上下文。

## 索引构建（概要）

1. 将法规原文（`.txt/.html/.htm`）放入 `data/raw/laws/`，案例原文（可下载页面）放入 `data/raw/cases/`。
2. 使用 `scripts/build_law_index.py` / `scripts/build_case_index.py` 生成 `data/index/` 下的 BM25、FAISS 与 `law_texts.json` / `case_texts.json`（脚本见 `scripts/`）。

### 一键抓取/导入（推荐）
法规（下载页面 -> 构建索引）：
```bash
# 1) 准备 URL 列表（每行一个 URL）
cp -n /dev/null data/raw/laws/urls.txt
# 编辑 urls.txt，填入公开法规页面链接

# 2) 导入到 raw
python scripts/import_laws_from_urls.py --urls-file data/raw/laws/urls.txt --out-dir data/raw/laws

# 3) 构建法规索引
python scripts/build_law_index.py data/raw/laws
```

案例（下载页面 -> 抽取文本 -> 生成 JSONL -> 构建索引）：
```bash
# 0) 准备 URL 列表
cp -n /dev/null data/raw/cases/urls.txt
# 编辑 urls.txt，填入公开裁判文书页面链接

# 1) 下载并生成案例 JSONL（用于 build_case_index）
python scripts/import_cases_from_urls.py \
  --urls-file data/raw/cases/urls.txt \
  --out-dir data/raw/cases \
  --output-jsonl data/processed/cases/case_snippets.jsonl

# 2) 构建案例索引
python scripts/build_case_index.py data/processed/cases/case_snippets.jsonl
```

无索引时仍可启动 CLI：工具会自动返回“缺哪类索引数据”和“下一步命令”。

## 布局

- `app/data_pipeline/`：解析与切分
- `app/retrieval/`：BM25、嵌入、FAISS、混合检索
- `app/tools/`：检索与导出工具
- `app/graph/`：LangGraph ReAct 流程
- `app/llm/`：LLM 模型适配（本地 Transformers / Ollama）

## 许可

示例代码按项目需要自行补充许可证。

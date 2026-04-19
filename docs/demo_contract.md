# 法律法规差异对比 Demo — 集成契约（最小可运行）

**仓库 / 分支**：`legal_diff_agent` · `demo-integration`（commit `5524eb6`）  
**角色**：主控/集成 agent 只做架构、契约、联调；业务实现由各子模块在约定边界内完成。

---

## 1. 目标与范围

**Demo 目标**：在**两份法规全文**上，自动**按条款切分**，并输出**相同点、差异点、可能冲突点**及**引用片段**。

**非目标（本阶段）**：不替代现有「法域 + 主题 + 本地索引检索」的 `compare_tool` 产品路径；Demo 可走**独立 API/脚本**，尽量少改 `app/graph` 与现有工具注册。

---

## 2. 与现有仓库的关系（复用）

| 能力 | 位置 | Demo 如何复用 |
|------|------|----------------|
| 按条/款切分 | `app/data_pipeline/chunker.py` · `app/data_pipeline/parser.py`（`law_text_to_chunks`） | Parser 直接产出 `LawChunkRecord` 列表 |
| 条号展示线索 | `app/services/article_alignment.py` · `extract_article_hint` | Compare 展示引用时复用 |
| 双列表片段语义对齐 | `app/services/article_alignment.py` · `semantic_align_jurisdictions` | 将两份解析结果转为 `hits` 形 `dict`（`id` / `text`）即可复用贪心配对 |
| 对比行模型 | `app/schema/models.py` · `CompareRow` | 对齐结果可先映射为行；**「相同/差异/冲突」分类**需 Compare 扩展字段或外层包装（见下） |
| 句向量 | `app/retrieval/embedding.py` | 与现网对齐逻辑一致时需同一嵌入后端 |

**现有 `compare_service.compare_jurisdictions`**：依赖 `LawService.search` + 索引，面向「主题 + 两法域」检索场景。**Demo 的两份全文**更适合：**Parser 切条 → 构造伪 hits → `semantic_align_jurisdictions`**，或独立实现「按条号顺序对齐」的轻量逻辑；二者二选一，本契约以**可复用嵌入对齐**为默认推荐路径。

---

## 3. 建议目录结构（增量）

在**少改动**前提下，建议仅增加：

```text
docs/
  demo_contract.md          # 本文件
app/demo/                   # 可选：Demo 专用编排（薄层）
  __init__.py
  router.py                 # 若使用 FastAPI，挂载 /api/demo/...
  service.py                # parse → compare 串联
tests/
  test_demo_contract.py     # 契约/序列化冒烟（可选）
```

**UI** 可放在仓库外或 `frontend/`（后续联调再定）；本阶段契约只规定 **HTTP JSON**，不规定框架。

---

## 4. 三模块边界与接口

### 4.1 Parser 模块

**职责**：输入原始法规材料 → 清洗/解析 → **按条（及必要时款）切分** → 结构化片段列表。  
**不负责**：语义对比、嵌入、冲突判定。

**输入（逻辑契约）**

| 字段 | 类型 | 说明 |
|------|------|------|
| `source` | `str` 或 `{ "type": "plain", "text": str }` / 文件路径 | 最小 Demo 可用「纯文本字符串」；扩展可为 `html` / `pdf` 路径 |
| `doc_id` | `str` | 文档标识，用于 chunk `id` 前缀 |
| `meta` | 可选 `LawDocumentMeta` 字段覆盖 | 如 `title`、`jurisdiction` |

**输出**

- `list[LawChunkRecord]`（定义见 `app/schema/models.py`）
- 或等价 JSON 可序列化结构：

```json
{
  "doc_id": "string",
  "meta": { "title": "", "jurisdiction": "", "source_type": "", "source_url": "", "raw_path": null },
  "chunks": [
    {
      "id": "doc_id:0",
      "doc_id": "doc_id",
      "article_label": "第十二条",
      "text": "……",
      "meta": {},
      "extra": {}
    }
  ]
}
```

**实现锚点**：`parse_file` / `parse_html_to_law_text` + `law_text_to_chunks`（纯文本可先构造 `LawDocumentMeta` + 全文再走 `law_text_to_chunks`）。

---

### 4.2 Compare 模块

**职责**：接收**两份** Parser 输出（或等价 chunk 列表），产出对齐与分类结果。  
**不负责**：HTTP、文件上传 UI、持久化索引构建。

**输入**

| 字段 | 类型 | 说明 |
|------|------|------|
| `chunks_a` / `chunks_b` | `list[dict]` 或 `list[LawChunkRecord]` | 至少含 `id`、`text`；建议含 `article_label` |
| `label_a` / `label_b` | `str` | 展示用，如「版本 A」「上海稿」 |
| `topic` | `str` | 可选；传入 `semantic_align_jurisdictions` 的 `topic` 以改善嵌入对齐 |
| `options` | 可选 object | 如 `min_similarity`、`max_pairs`（对齐 `settings` 或 Demo 覆盖） |

**对齐层（推荐复用）**：将 chunk 转为 `hits = [{"id": c.id, "text": c.text}, ...]`，调用 `semantic_align_jurisdictions(hits_a, hits_b, topic, label_a, label_b)` → `list[CompareRow]`。

**分类层（最小 Demo）**：在每条对齐或未对齐片段对上增加**启发式或 LLM** 标签（实现可后补）：

| `category` | 含义 |
|------------|------|
| `same` | 高度一致或仅标点/表述微调 |
| `different` | 实体性表述差异 |
| `possible_conflict` | 义务/责任/罚则等可能互斥或需人工复核 |
| `unpaired_a` / `unpaired_b` | 仅一侧存在的条文 |

**输出（逻辑结构）**

```json
{
  "summary": "可选：一两句总述",
  "pairs": [
    {
      "category": "same | different | possible_conflict",
      "similarity_score": 0.0,
      "article_hint_a": "",
      "article_hint_b": "",
      "quote_a": "引用片段（建议 ≤2000 字与现网一致）",
      "quote_b": "",
      "note": "",
      "chunk_id_a": "",
      "chunk_id_b": "",
      "alignment_method": "semantic_embedding_greedy | retrieval_rank_fallback | single_side | ..."
    }
  ],
  "only_in_a": [{ "chunk_id": "", "article_label": "", "text": "" }],
  "only_in_b": []
}
```

> **说明**：`CompareRow` 现有字段可覆盖 `pairs[]` 的大部分列；若需 `category`，建议在 Compare 服务内扩展 Pydantic 模型或在此 JSON 外层包装，**避免**大规模改动 `CompareRow` 全局引用时可先用 `extra` 字段（若团队同意）。

---

### 4.3 UI 模块

**职责**：收集两份文本（或上传）→ 调用后端 → 展示列表/高亮。  
**不负责**：切条与对齐算法。

**调用后端后的响应 JSON（与 Compare 输出一致 + 包装）**

建议统一 HTTP 层包裹：

```json
{
  "ok": true,
  "error": null,
  "parser": {
    "a": { "doc_id": "", "meta": {}, "chunks": [] },
    "b": { "doc_id": "", "meta": {}, "chunks": [] }
  },
  "result": {
    "summary": "",
    "pairs": [],
    "only_in_a": [],
    "only_in_b": []
  }
}
```

错误时：

```json
{
  "ok": false,
  "error": "人类可读说明",
  "error_code": "PARSE_FAILED | EMBED_FAILED | ..."
}
```

---

## 5. API 设计（建议）

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/demo/diff` | Body: `{ "text_a", "text_b", "label_a?", "label_b?", "topic?" }`；返回上一节「UI 包装」JSON |
| `GET` | `/api/health` | 可选，联调用 |

**内容类型**：`application/json`；UTF-8。

**鉴权**：Demo 阶段可省略；生产再补。

---

## 6. 集成与联调顺序（主控职责）

1. 冻结 Parser 输出 schema（本文 §4.1）。  
2. 冻结 Compare 输出 schema（本文 §4.2），先打通「仅对齐 + `category` 占位常量」。  
3. 实现或接通最小 HTTP 层（`app/demo/router.py` 或独立 `uvicorn` 入口）。  
4. UI 只对接 `/api/demo/diff` 的 JSON。  
5. 用两份短样例（各含「第×条」）做端到端冒烟。

---

## 7. 修订记录

| 日期 | 说明 |
|------|------|
| 2026-04-18 | 初稿：基于 `demo-integration` @ `5524eb6` |

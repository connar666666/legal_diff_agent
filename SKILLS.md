# legal_diff_agent — Agent 行为技能说明（官网优先入库）

本文档描述 **本工程里「法规智能体」应遵循的操作策略**。  
**启动 `build_agent_graph()` / `python -m app.main` 时**，若 `app/config.py` 中 `agent_skills_enabled=True`（默认），会把本文件全文 **追加到系统提示词** 中，模型在对话里可直接遵循。基础指令仍在 `app/graph/prompts.py` 的 `SYSTEM_PROMPT`。

---

## 技能：公开法规「官网一手 → 本地 → 索引 → 检索」

当用户需要 **查询公开资料并保存到本地、再可检索** 时：

1. **首选来源**：各级政府/人大/司法部等 **官方网站** 公布的法规正文页面或 **PDF**，不用论坛、自媒体、二次转载摘要当唯一依据。
2. **推荐工具链**（与代码中工具一致）：
   - `lookup_law_url_tool` → 查 `data/raw/law.txt` 是否已有 URL  
   - `discover_law_urls_tool` → **优先** `official_law_domains`（见 `app/config.py`），再通用搜索  
   - `fetch_law_primary_source_tool` → 写入 `data/raw/laws/` 并更新映射  
   - `build_law_index_tool` → 重建 BM25 + FAISS + `law_texts.json`  
   - `search_law_tool` → 基于本地索引回答  
3. **一键补齐**：缺口明显时可用 `auto_import_law_primary_source_tool(law_name, jurisdiction)`。
4. **`web_search_tool`**：仅作 **线索与事实核对**，**不**替代官网正文下载入库。

## 反模式

- 只靠网页搜索摘句，不 `fetch`、不建索引，却声称「已入库」。  
- 新文件落盘后跳过 `build_law_index_tool`，仍期望检索命中。

---

## 技能：快速检索（按意图选工具，少推理）

下列规则 **按顺序套用**，无需再「想该用哪种检索」：

### 1. 意图 → 首选工具（一览）

| 用户要什么 | 直接调用 | 前提 |
|------------|----------|------|
| 查**法规条文**、某法域规定 | `search_law_tool(query=用户问题或关键词, jurisdiction=地名或"")` | 法规索引已构建 |
| 查**案例**、判决、案号 | `search_case_tool(query=…)` | 案例索引已构建 |
| **两地法规**对比（异同） | `compare_tool(topic=主题, jurisdiction_a=, jurisdiction_b=)` | 两地相关法规**均已**在索引中；内部用句向量对**检索候选**做余弦配对 |
| 缺索引 / 工具返回「未加载」 | 先 `lookup` → `discover` → `fetch` → `build_law_index_tool`，或 `auto_import_law_primary_source_tool` | — |
| 只要**网页线索**、新闻、非入库事实 | `web_search_tool(query=…)` | **不**替代法规正文检索 |

### 2. 法规检索在系统里如何实现（心里有数即可）

- **混合检索**：**BM25（关键字）** + **FAISS向量（语义）** 加权融合，再取 Top 片段；不是单关键词。  
- **向量**：建索引与查询用同一套嵌入模型（见配置 `embedding_model_name`）。  
- **法域过滤**：`jurisdiction` 会过滤「片段正文里是否含该地名」；若正文无地名，可试 `jurisdiction=""` 或换 query 带地名。

### 3. 操作习惯（提速）

1. **先工具、后长答**：有索引时 **先** `search_law_tool` / `search_case_tool` / `compare_tool`，再组织回答。  
2. **失败看 JSON**：工具返回 `ok: false` 时读 `missing` / `next_steps`，按提示补数据或建索引，**不要**空口编条文。  
3. **对比不走两次 search**：跨地法规异同 **优先一次** `compare_tool`，不要两边各搜一次再自己拼（除非 `compare_tool` 不可用）。
4. **条文级引用**：`search_law_tool` 命中与 `compare_tool` 的 `rows[]` 中含 **`citation`**、**`law_title`**、**`article_label`**（及对比侧的 **`citation_a` / `citation_b`**）时，向用户作答须**显式写出**这些引用，与工具 JSON 一致；重建索引后才会生成完整 `citation`（见 `law_chunk_meta.json`）。

## 配置（关闭或换路径）

- 环境变量或 `.env`：`AGENT_SKILLS_ENABLED=false` 则只使用 `SYSTEM_PROMPT`，不读本文件。  
- `AGENT_SKILLS_PATH`：可指向其它 Markdown 路径（相对路径相对项目根）。

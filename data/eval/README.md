# 评测集（JSONL）

每行 **一个 JSON 对象**（可用 `#` 开头做注释行，解析时会跳过空行与注释行）。

## 字段说明

| 字段 | 必填 | 说明 |
|------|------|------|
| `id` | 推荐 | 样本唯一 id；省略时为 `line_N`。 |
| `question` | **是** | 用户自然语言问题。 |
| `task_type` | 否 | `retrieve` / `compare` / `general` / `refuse`，默认 `retrieve`。仅用于记录，不影响跑分逻辑。 |
| `gold_citations` | 否 | 字符串列表，金标准引用。可与回答中抽取的「《法名》第×条」或「第×条」做 **F1** 对齐。示例：`["《中华人民共和国民法典》第一千二百五十四条", "第一千二百五十四条"]`。 |
| `gold_keywords` | 否 | 期望搜索词应包含的关键词（评 **Baseline 检索词相关性** 时用）。 |
| `jurisdiction` | 否 | 备注用法域；当前脚本仅透传，可放在 `extra` 或后续扩展。 |

## 如何补充数据

1. 复制 `examples/sample_questions.jsonl` 为新文件（如 `my_gold.jsonl`）。
2. 每条至少写 `question`；有金标准时补 `gold_citations`（建议同时写法名+条号与单独条号，便于模糊匹配）。
3. **对比类**题可写长问题，并在 `gold_citations` 中列两侧应出现的关键条（或拆成两条样本）。
4. 运行：

```bash
export PYTHONPATH=.
python scripts/run_eval_experiment.py run --dataset data/eval/my_gold.jsonl \
  --systems baseline_web_rag,full_agent --out data/outputs/eval/my_run.jsonl
python scripts/run_eval_experiment.py aggregate --results data/outputs/eval/my_run.jsonl
```

## 多基座对比

本仓库 **Baseline** 与 **Agent（transformers 后端）** 均使用 `settings.local_model_path`（Qwen 等）。  
更换基座时：改 `.env` / 环境变量后 **分别输出不同 `--out`**，再对多个 JSONL 分别 `aggregate` 对比表格即可。  
若某基座仅能通过 API 提供，需另写薄适配器（本脚本未内置）。

## 指标说明（自动）

- `gold_metrics`：`precision` / `recall` / `f1`（抽取引用 vs `gold_citations`，启发式包含匹配）。
- `evidence_support`：抽取引用是否出现在当次 **证据文本**（Baseline 为搜索摘要；Agent 为所有 `ToolMessage` 拼接）。
- `search_query_relevance`（仅 Baseline）：检索首句与用户问题的字符重合度 + `gold_keywords` 命中率。

人工 Rubric 需离线另表记录，本脚本不替代法务审阅。

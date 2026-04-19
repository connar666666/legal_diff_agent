"""应用配置：路径、模型与检索参数。"""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """从环境变量或 `.env` 加载配置。"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    project_root: Path = Field(default_factory=_project_root)

    # 启动时拼入系统提示：项目根目录 SKILLS.md（可改路径或关闭）
    agent_skills_enabled: bool = True
    agent_skills_path: Path = Field(default_factory=lambda: _project_root() / "SKILLS.md")

    # LLM 后端选择：`ollama` 使用 HTTP；`transformers` 直接加载本地模型到 GPU
    llm_backend: str = "transformers"

    # 数据目录
    data_raw_laws: Path = Field(default_factory=lambda: _project_root() / "data" / "raw" / "laws")
    data_raw_cases: Path = Field(default_factory=lambda: _project_root() / "data" / "raw" / "cases")
    data_processed_laws: Path = Field(
        default_factory=lambda: _project_root() / "data" / "processed" / "laws"
    )
    data_processed_cases: Path = Field(
        default_factory=lambda: _project_root() / "data" / "processed" / "cases"
    )
    data_index: Path = Field(default_factory=lambda: _project_root() / "data" / "index")
    data_outputs: Path = Field(default_factory=lambda: _project_root() / "data" / "outputs")

    # 多轮对话 SQLite（thread_id -> 消息 JSON）
    thread_history_db_path: Path = Field(
        default_factory=lambda: _project_root() / "data" / "outputs" / "chat_cache.sqlite"
    )

    # 法典/法律 URL 归档（缺口时用于快速反查并可选重新抓取）
    # 说明：文件后缀为 .txt，但内容使用 JSONL（每行一个 JSON 对象），便于机器读取。
    law_url_registry_path: Path = Field(
        default_factory=lambda: _project_root() / "data" / "raw" / "law.txt"
    )
    # 用于“优先官方站点”的候选域名（逗号分隔）。如不可靠也会自动回退到通用搜索。
    official_law_domains: str = "npc.gov.cn,gov.cn,moj.gov.cn"

    # Web 搜索/抓取参数（用于发现/下载一手法律文本页面）
    web_search_timeout_s: float = 20.0
    web_search_max_results: int = 10

    # 本地 LLM（Ollama / 兼容 OpenAI 的 chat API）
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5"
    ollama_timeout_s: float = 120.0

    # 本地 Transformers（GPU/CPU）配置
    local_model_path: str = "/home/fangcong/codes_LBF_10/dynamic/multiagent_code/epymarl/src_lbf_v13/model/Qwen3-8B"
    local_trust_remote_code: bool = True
    local_dtype: str = "bfloat16"  # bfloat16 / float16 / float32
    local_max_new_tokens: int = 1024
    local_temperature: float = 0.2
    local_top_p: float = 0.95

    # 嵌入与检索
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    embedding_batch_size: int = 32
    bm25_top_k: int = 20
    vector_top_k: int = 20
    hybrid_fusion_top_k: int = 15
    bm25_weight: float = 0.45
    vector_weight: float = 0.55

    # 混合检索之后的 Cross-Encoder 精排（在融合候选上重算相关性）
    rerank_enabled: bool = False
    rerank_model_name: str = "BAAI/bge-reranker-base"
    rerank_pool_k: int = 40
    rerank_batch_size: int = 16
    rerank_max_length: int = 512
    rerank_max_passage_chars: int = 4000

    # 多地对比：条文级语义对齐（检索候选池 + 句向量配对）
    compare_retrieval_top_k: int = 32
    compare_semantic_min_similarity: float = 0.28
    compare_max_aligned_pairs: int = 12

    # 日志
    log_level: str = "INFO"

    # 工具调用/模型输出调试日志
    tool_debug_enabled: bool = False
    tool_debug_log_path: Path = Field(
        default_factory=lambda: _project_root() / "data" / "outputs" / "tool_call_debug.jsonl"
    )

    # 可选：指定已有索引路径（否则在 data/index 下按约定命名）
    law_bm25_path: Optional[Path] = None
    law_vector_path: Optional[Path] = None
    case_bm25_path: Optional[Path] = None
    case_vector_path: Optional[Path] = None

    def resolve_law_bm25(self) -> Path:
        return self.law_bm25_path or (self.data_index / "law_bm25.json")

    def resolve_law_vector(self) -> Path:
        return self.law_vector_path or (self.data_index / "law_faiss")

    def resolve_case_bm25(self) -> Path:
        return self.case_bm25_path or (self.data_index / "case_bm25.json")

    def resolve_case_vector(self) -> Path:
        return self.case_vector_path or (self.data_index / "case_faiss")


settings = Settings()

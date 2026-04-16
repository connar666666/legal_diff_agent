"""结构化业务模型（法规片段、案例、对比项）。"""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class LawDocumentMeta(BaseModel):
    """法规文档级元数据。"""

    title: str = ""
    jurisdiction: str = ""  # 法域，如「全国」「某省」
    source_type: str = ""  # 如「法律」「行政法规」「地方性法规」
    source_url: str = ""
    raw_path: Optional[str] = None


class LawChunkRecord(BaseModel):
    """单条可检索法规片段（与索引中的一条对应）。"""

    id: str
    doc_id: str
    article_label: str = ""  # 如「第十二条」
    text: str
    meta: LawDocumentMeta = Field(default_factory=LawDocumentMeta)
    extra: dict[str, Any] = Field(default_factory=dict)


class CaseChunkRecord(BaseModel):
    """案例片段。"""

    id: str
    case_id: str
    title: str = ""
    court: str = ""
    snippet: str
    source_url: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class RetrievalHit(BaseModel):
    """检索命中（法规或案例）。"""

    id: str
    score: float
    kind: Literal["law", "case"]
    payload: dict[str, Any] = Field(default_factory=dict)


class CompareRow(BaseModel):
    """多地法规对比中的一行。"""

    aspect: str
    jurisdiction_a: str
    jurisdiction_b: str
    content_a: str
    content_b: str
    note: str = ""
    # 条文级语义对齐（句向量配对）时的可选字段
    similarity_score: Optional[float] = None
    chunk_id_a: str = ""
    chunk_id_b: str = ""
    article_hint_a: str = ""
    article_hint_b: str = ""
    alignment_method: str = ""

import dataclasses

@dataclasses.dataclass
class GenerationConfig:

    candidate_count: int | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    response_mime_type: str | None = None
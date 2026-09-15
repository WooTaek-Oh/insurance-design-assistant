"""가입설계 상태(멀티턴 실시간 설계 수정) 공용 모듈.

대화 중 사용자가 "암진단비 특약 5천만원으로 넣어줘", "정기특약 빼줘" 같이
설계를 수정해달라고 하면, LLM이 자연어 답변 끝에 사람 눈에는 안 보이는
JSON 블록으로 "이번 요청을 반영한 전체 설계 상태"를 함께 출력하도록
프롬프트로 지시한다. 이 모듈은 그 블록을 답변 본문에서 분리하고,
파싱/검증해서 세션에 저장할 수 있는 형태로 만들어준다.

LLM이 매번 "전체 상태"를 다시 출력하게 해서 (변경분만 계산하는 게 아니라)
누적 diff를 우리 쪽에서 병합하는 로직을 없앴다 — 병합 버그 위험이 줄어든다.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

# 내용물이 유효한 JSON인지와 무관하게 펜스 블록 자체는 항상 잡아내야
# 화면에 노출되지 않는다 (JSON이 깨졌을 때를 대비해 (\{.*?\}) 대신 (.*?) 사용).
DESIGN_BLOCK_RE = re.compile(r"```design-update\s*(.*?)\s*```", re.DOTALL)

DEFAULT_DESIGN = {"base_product": None, "riders": {}}

SYSTEM_PROMPT_ADDITION = """

[가입설계 상태 관리 규칙]
- 아래 "[현재 설계 상태]"는 지금까지 대화로 합의된 가입설계 내용입니다.
- 사용자가 주계약 선택/변경, 특약 추가·삭제, 가입금액 변경 등 설계를
  수정해달라고 요청하면:
  1. 무엇을 어떻게 바꿨는지 자연스러운 말로 설명하세요.
  2. 답변 맨 마지막 줄에 반드시 이번 요청까지 반영한 "전체" 설계 상태를
     아래 형식의 JSON 코드블록으로 출력하세요 (이전 상태를 그대로 복사하지
     말고, 이번 변경을 합친 최신 전체 상태를 출력):

```design-update
{"base_product": "상품명 또는 null", "riders": {"특약명": 가입금액_만원_숫자}}
```

- 설계 수정 요청이 아닌 일반 질문이나 잡담에는 이 블록을 절대 출력하지 마세요.
- 이 블록은 사용자에게 보이지 않고 시스템이 자동으로 읽어서 화면에 표로
  보여주므로, 반드시 유효한 JSON만 담아야 합니다 (설명 텍스트 금지)."""


def format_design_state(design: dict) -> str:
    return json.dumps(design, ensure_ascii=False)


@dataclass
class ExtractResult:
    clean_answer: str
    update: dict | None = None
    parse_error: bool = False


def extract_design_update(answer_text: str) -> ExtractResult:
    """답변 텍스트에서 design-update JSON 블록을 분리해낸다.

    블록이 없으면 원문 그대로 반환. 블록은 있는데 JSON이 깨졌으면
    (LLM이 형식을 안 지킨 경우) 블록만 제거하고 update=None, parse_error=True.
    """
    match = DESIGN_BLOCK_RE.search(answer_text)
    if not match:
        return ExtractResult(clean_answer=answer_text.strip())

    clean = DESIGN_BLOCK_RE.sub("", answer_text).strip()
    try:
        data = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ExtractResult(clean_answer=clean, update=None, parse_error=True)

    if not isinstance(data, dict):
        return ExtractResult(clean_answer=clean, update=None, parse_error=True)

    return ExtractResult(clean_answer=clean, update=data)


def sanitize_design(raw: dict, fallback: dict) -> dict:
    """LLM이 출력한 design-update를 검증/정제해서 안전한 상태로 만든다."""
    base_product = raw.get("base_product")
    if not isinstance(base_product, str) or not base_product.strip():
        base_product = fallback.get("base_product")

    riders_raw = raw.get("riders")
    riders: dict[str, float] = {}
    if isinstance(riders_raw, dict):
        for name, amount in riders_raw.items():
            if not isinstance(name, str) or not name.strip():
                continue
            if isinstance(amount, (int, float)) and amount > 0:
                riders[name.strip()] = amount

    return {"base_product": base_product, "riders": riders}

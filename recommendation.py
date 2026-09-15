"""AI 가입설계 추천 알고리즘 (RFP 핵심기능 03).

실제 고객 이력 데이터가 없으므로, FC가 상담 중 파악한 고객 정보를 입력하면
더미 페르소나 데이터와 규칙 기반(rule-based) 점수 계산으로 상품/특약을
추천한다. ML 모델이 아니라 규칙 기반인 이유: (1) 실제 이력 데이터가 없어
학습이 불가능하고, (2) 왜 추천했는지 FC가 고객에게 바로 설명할 수 있어야
하는데 규칙 기반이 설명 가능성(explainability)이 훨씬 높다.

추천 대상 상품은 지금 실제로 RAG에 인덱싱된 상품에 한정한다 (아직 인덱싱
안 된 상품을 추천하면 챗봇이 그 상품에 대해 근거 있는 답변을 못 하므로).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CustomerProfile:
    age: int
    gender: str  # "남성" | "여성"
    has_children: bool
    family_history: list[str] = field(default_factory=list)  # 예: ["암", "심장질환"]
    existing_products: list[str] = field(default_factory=list)  # 예: ["실손의료비"]
    concerns: list[str] = field(default_factory=list)  # 예: ["의료비", "노후자금", "사망보장"]
    income_level: str = "중간"  # "낮음" | "중간" | "높음"


@dataclass
class Recommendation:
    product: str
    rider: str | None
    suggested_amount: int | None  # 만원 단위
    score: float
    reasons: list[str]


# 실제로 RAG에 인덱싱된 상품만 추천 대상으로 등록한다.
# (건강보장보험/종신보험은 아직 docs/_pending/에 있어 인덱싱 전이므로 제외)
PRODUCT_CATALOG = [
    {
        "product": "무배당우리WON하는암보험",
        "rider": "암진단비특약",
        "base_amount": 3000,
    },
    {
        "product": "무배당우리WON하는암보험",
        "rider": "보험료납입면제특약",
        "base_amount": None,
    },
    {
        "product": "무배당우리WON하는급여실손의료비보장보험(갱신형)",
        "rider": None,
        "base_amount": None,
    },
    {
        "product": "무배당우리WON하는누구나행복연금보험",
        "rider": None,
        "base_amount": None,
    },
]


def _score_cancer_rider(profile: CustomerProfile) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    if "암" in profile.family_history:
        score += 40
        reasons.append("가족력에 암이 있어 암 보장 필요성이 높음")
    if profile.age >= 40:
        score += 25
        reasons.append("40대 이상은 암 발병률이 증가하는 연령대")
    if "암진단비특약" not in profile.existing_products and "무배당우리WON하는암보험" not in profile.existing_products:
        score += 15
        reasons.append("현재 암 관련 보장이 없음")
    if "의료비" in profile.concerns:
        score += 10
        reasons.append("의료비 부담 경감을 원함")
    return score, reasons


def _score_health_waiver_rider(profile: CustomerProfile) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    if profile.income_level == "낮음":
        score += 30
        reasons.append("소득 수준을 고려할 때 납입면제 특약으로 위험 대비 필요")
    if profile.family_history:
        score += 20
        reasons.append("가족력이 있어 진단 시 보험료 납입 부담을 줄일 필요")
    return score, reasons


def _score_medical_expense(profile: CustomerProfile) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    if "실손의료비" not in profile.existing_products:
        score += 45
        reasons.append("실손의료비 보장이 없어 의료비 실비 보장 공백 존재")
    if profile.age < 40:
        score += 15
        reasons.append("젊을 때 가입하면 보험료 부담이 낮음")
    if "의료비" in profile.concerns:
        score += 15
        reasons.append("의료비 부담에 대한 우려가 있음")
    return score, reasons


def _score_pension(profile: CustomerProfile) -> tuple[float, list[str]]:
    score, reasons = 0.0, []
    if "노후자금" in profile.concerns:
        score += 40
        reasons.append("노후자금 마련에 대한 니즈가 있음")
    if profile.age >= 40:
        score += 25
        reasons.append("40대 이상은 은퇴 준비를 본격적으로 고려할 시점")
    if profile.income_level in ("중간", "높음"):
        score += 15
        reasons.append("연금보험 납입 여력이 있는 소득 수준")
    return score, reasons


_SCORERS = {
    ("무배당우리WON하는암보험", "암진단비특약"): _score_cancer_rider,
    ("무배당우리WON하는암보험", "보험료납입면제특약"): _score_health_waiver_rider,
    ("무배당우리WON하는급여실손의료비보장보험(갱신형)", None): _score_medical_expense,
    ("무배당우리WON하는누구나행복연금보험", None): _score_pension,
}


def recommend(profile: CustomerProfile, top_n: int = 3) -> list[Recommendation]:
    """고객 프로필을 받아 점수 높은 순으로 상품/특약을 추천한다."""
    results: list[Recommendation] = []
    for item in PRODUCT_CATALOG:
        key = (item["product"], item["rider"])
        scorer = _SCORERS.get(key)
        if scorer is None:
            continue
        score, reasons = scorer(profile)
        if score <= 0:
            continue
        suggested_amount = item["base_amount"]
        if suggested_amount and "암" in profile.family_history and item["rider"] == "암진단비특약":
            suggested_amount = int(suggested_amount * 1.5)  # 가족력 있으면 증액 제안
        results.append(
            Recommendation(
                product=item["product"],
                rider=item["rider"],
                suggested_amount=suggested_amount,
                score=score,
                reasons=reasons,
            )
        )
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:top_n]

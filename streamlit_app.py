import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import design_state
import rag
import recommendation as reco

st.set_page_config(page_title="가입설계 챗봇", page_icon="💬")
st.title("가입설계 챗봇")
st.caption("동양생명 FC를 위한 AI 가입설계 도우미 (프로토타입) · 약관 RAG 기반")


# Gemini 챗 모델 초기화 (키는 Secrets에서 읽어옴)
#
# 주의: "gemini-flash-latest" 같은 -latest 별칭은 쓰지 않는다. 이 별칭은
# 시간이 지나면 자동으로 최신 모델을 가리키게 바뀌는데, 실제로 어느 시점에
# "gemini-3.8-flash"(신규 프리뷰 성격 모델)로 넘어가면서 무료 등급 일일
# 한도가 단 20건으로 뚝 떨어지는 걸 확인했다 (라이브 데모 중 20턴만 넘어가도
# 챗봇이 멈추는 셈). 대신 검증된 안정 버전을 고정해서 쓴다.
CHAT_MODEL = "gemini-3.6-flash"


@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL,
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3,
    )


# 미리 인덱싱해둔 chroma_db/ 를 로드 (없으면 None → 일반 대화 모드로 동작)
@st.cache_resource
def get_vectorstore():
    try:
        return rag.load_vectorstore(st.secrets["GOOGLE_API_KEY"])
    except Exception as e:
        st.warning(f"약관 검색 인덱스를 불러오지 못했습니다 ({e}). 일반 대화 모드로 동작합니다.")
        return None


llm = get_llm()
vectorstore = get_vectorstore()

if vectorstore is None:
    st.warning(
        "⚠️ 약관 데이터베이스가 아직 준비되지 않았습니다. "
        "`scripts/build_index.py`로 인덱싱 후 다시 배포해주세요. "
        "지금은 근거 문서 없이 일반적인 답변만 제공됩니다."
    )

# 시스템 프롬프트 (챗봇 성격 + RAG 근거 활용 + 할루시네이션 차단 규칙)
SYSTEM_PROMPT = """당신은 동양생명 보험설계사(FC)를 돕는 가입설계 어시스턴트입니다.
설계사가 상품, 특약, 가입설계에 대해 물으면 명확하고 실무적으로 답하세요.

[근거 문서 활용 규칙 - 반드시 지킬 것]
- 아래 "검색된 약관 근거"가 제공되면, 그 내용에 근거해서만 사실을 답변하세요.
- 근거 문서에 없는 내용을 추측하거나 지어내지 마세요.
- 근거 문서가 없거나 질문과 관련성이 낮다고 표시된 경우, 보험 상품/약관에 대한
  사실 관계 질문에는 반드시 "확인이 필요합니다. 정확한 내용은 약관 원문이나
  상품 담당 부서를 통해 확인해주세요."라고 답하세요.
- 단, 인사말이나 챗봇 자체에 대한 질문 등 약관 근거가 필요 없는 일반적인
  대화는 자연스럽게 응답하세요.""" + design_state.SYSTEM_PROMPT_ADDITION

RELEVANCE_WARNING = (
    "\n\n(주의: 위 근거는 질문과 관련성이 낮을 수 있습니다. "
    "약관 관련 사실 질문이라면 근거 없이 답하지 말고 확인이 필요하다고 안내하세요.)"
)

# 대화 기록 및 가입설계 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "design" not in st.session_state:
    st.session_state.design = {"base_product": None, "riders": {}}


def render_design_sidebar():
    """현재까지 대화로 합의된 가입설계 상태를 사이드바에 실시간으로 보여준다."""
    with st.sidebar:
        st.subheader("📋 현재 가입설계")
        design = st.session_state.design
        if design["base_product"]:
            st.markdown(f"**주계약**: {design['base_product']}")
        if design["riders"]:
            st.markdown("**특약**")
            for name, amount in design["riders"].items():
                st.markdown(f"- {name}: {amount:,.0f}만원")
        if not design["base_product"] and not design["riders"]:
            st.caption("아직 설계된 내용이 없습니다. 채팅으로 상품이나 특약을 요청해보세요.")
        if st.button("🔄 설계 초기화"):
            st.session_state.design = {"base_product": None, "riders": {}}
            st.rerun()


def render_recommendation_sidebar():
    """고객 정보를 입력받아 규칙 기반으로 상품/특약을 추천한다 (RFP 핵심기능 03).

    실제 이력 데이터가 없어 더미 프로필 입력 + 점수 기반 매칭으로 대체했다.
    추천 결과는 근거(왜 추천했는지)와 함께 보여주고, 버튼 한 번으로 04번
    가입설계 상태에 바로 반영할 수 있게 연결했다.
    """
    with st.sidebar:
        with st.expander("🤖 AI 가입설계 추천 (베타)"):
            st.caption("고객 정보를 입력하면 규칙 기반으로 상품·특약을 추천합니다.")
            age = st.number_input("나이", min_value=15, max_value=100, value=35, key="reco_age")
            has_children = st.checkbox("자녀 있음", key="reco_children")
            family_history = st.multiselect(
                "가족력", ["암", "심장질환", "뇌혈관질환"], key="reco_family"
            )
            existing = st.multiselect(
                "기존 가입 상품", ["실손의료비", "암보험", "연금보험"], key="reco_existing"
            )
            concerns = st.multiselect(
                "고객 관심사", ["의료비", "사망보장", "노후자금"], key="reco_concerns"
            )
            income = st.selectbox("소득 수준", ["낮음", "중간", "높음"], index=1, key="reco_income")

            if st.button("🔍 추천 받기", key="reco_button"):
                profile = reco.CustomerProfile(
                    age=age,
                    gender="",
                    has_children=has_children,
                    family_history=family_history,
                    existing_products=existing,
                    concerns=concerns,
                    income_level=income,
                )
                st.session_state.recommendations = reco.recommend(profile)

            recs = st.session_state.get("recommendations")
            if recs:
                st.markdown("---")
                st.markdown("**추천 결과**")
                for i, r in enumerate(recs):
                    label = r.product + (f" - {r.rider}" if r.rider else "")
                    amount_label = f" ({r.suggested_amount:,}만원)" if r.suggested_amount else ""
                    st.markdown(f"**{i + 1}. {label}{amount_label}**")
                    for reason in r.reasons:
                        st.caption(f"· {reason}")
                    if st.button("➕ 설계에 추가", key=f"add_reco_{i}"):
                        design = st.session_state.design
                        if not design["base_product"]:
                            design["base_product"] = r.product
                        if r.rider:
                            design["riders"][r.rider] = r.suggested_amount or 1000
                        st.session_state.design = design
                        st.rerun()


render_design_sidebar()
render_recommendation_sidebar()

# 이전 대화 표시 (근거 페이지가 있으면 함께 표시)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("confidence_label"):
            st.caption(msg["confidence_label"])
        if msg.get("sources"):
            st.caption(f"📄 참고한 약관 페이지: {msg['sources']}")

# 사용자 입력
if prompt := st.chat_input("가입설계에 대해 물어보세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("약관 검색 중..."):
            # 1) 벡터 검색으로 근거 청크 조회 (임베딩 API 쿼터 초과 등으로 실패해도
            #    앱이 죽지 않고 "근거 없음"으로 안전하게 넘어가게 처리)
            retrieved = []
            if vectorstore:
                try:
                    retrieved = rag.retrieve(vectorstore, prompt, k=4)
                except Exception as e:
                    reason = "API 사용량 한도 초과" if "429" in str(e) else "일시적 오류"
                    st.caption(f"⚠️ 약관 검색 실패({reason})로 근거 없이 답변합니다.")
            has_evidence = rag.has_relevant_evidence(retrieved)
            context_text = rag.format_context(retrieved) if retrieved else "(검색된 근거 없음)"
            sources = rag.format_sources(retrieved) if has_evidence else None

            # 확신도(%)는 LLM이 말로 지어내는 게 아니라, 실제 검색 거리값을
            # 그대로 환산한 값이다 (rag.evidence_confidence_percent 참고).
            confidence_label = (
                rag.confidence_label(rag.evidence_confidence_percent(retrieved))
                if retrieved
                else None
            )

            # 2) 시스템 프롬프트 + 근거 + 현재 설계 상태를 포함해 대화 맥락 구성
            design_json = design_state.format_design_state(st.session_state.design)
            system_content = (
                f"{SYSTEM_PROMPT}\n\n[검색된 약관 근거]\n{context_text}"
                f"\n\n[현재 설계 상태]\n{design_json}"
            )
            if not has_evidence:
                system_content += RELEVANCE_WARNING

            chat_history = [SystemMessage(content=system_content)]
            for m in st.session_state.messages:
                if m["role"] == "user":
                    chat_history.append(HumanMessage(content=m["content"]))
                else:
                    chat_history.append(AIMessage(content=m["content"]))

        with st.spinner("생각 중..."):
            response = llm.invoke(chat_history)
            # content가 리스트(블록 구조)로 오면 텍스트만 뽑아냄
            if isinstance(response.content, list):
                answer = "".join(
                    block.get("text", "") if isinstance(block, dict) else str(block)
                    for block in response.content
                )
            else:
                answer = response.content

            # 3) 답변에 design-update 블록이 있으면 분리해서 설계 상태에 반영
            #    (사용자에게는 자연어 설명만 보이고, JSON은 화면에 노출하지 않음)
            extracted = design_state.extract_design_update(answer)
            answer = extracted.clean_answer
            design_changed = False
            if extracted.update is not None:
                st.session_state.design = design_state.sanitize_design(
                    extracted.update, st.session_state.design
                )
                design_changed = True

            st.markdown(answer)
            if confidence_label:
                st.caption(confidence_label)
            if sources:
                st.caption(f"📄 참고한 약관 페이지: {sources}")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence_label": confidence_label,
        }
    )

    if design_changed:
        st.rerun()  # 사이드바의 가입설계 요약을 즉시 갱신

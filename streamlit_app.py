import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

import design_state
import rag
import recommendation as reco

st.set_page_config(page_title="가입설계 챗봇", page_icon="🛡️")

USER_AVATAR = "🧑‍💼"
ASSISTANT_AVATAR = "🛡️"

st.markdown("## 🛡️ 가입설계 챗봇")
st.caption("동양생명 FC를 위한 AI 가입설계 도우미 (프로토타입) · 약관 근거 기반 답변")
st.caption("📚 현재 답변 가능한 상품: 암보험 · 연금보험 · 실손의료비 (그 외 상품은 근거가 없어 \"확인이 필요합니다\"로 응답)")
st.divider()


def render_confidence_badge(percent: int) -> str:
    """확신도(%)를 색이 있는 뱃지 HTML로 렌더링한다 (등급 기준은 rag.confidence_label과 동일)."""
    if percent >= 70:
        color, bg, label = "#166534", "#DCFCE7", "근거 확신도"
    elif percent >= 50:
        color, bg, label = "#854D0E", "#FEF9C3", "근거 확신도 (참고용)"
    else:
        color, bg, label = "#991B1B", "#FEE2E2", "근거 확신도 (관련 근거 부족)"
    return (
        f'<span style="background:{bg};color:{color};padding:2px 10px;'
        f'border-radius:999px;font-size:0.8em;font-weight:600;">'
        f"{label} {percent}%</span>"
    )


def render_sources_tag(sources: str) -> str:
    """참고 페이지 목록을 작은 태그 스타일로 렌더링한다."""
    return (
        '<span style="color:#6B7280;font-size:0.85em;">📄 참고한 약관 페이지: '
        f"{sources}</span>"
    )


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
    """챗 모델을 초기화한다.

    get_vectorstore()와 달리 이건 실패하면 앱이 아예 기능을 할 수 없으므로
    (LLM 없이는 대화 자체가 불가능), "일반 모드로 낮추기"가 아니라 원인을
    명확히 알려주고 st.stop()으로 깔끔하게 멈춘다. 예전에는 예외처리가
    전혀 없어서 GOOGLE_API_KEY가 비어있으면 Streamlit 기본 에러 화면(원인을
    알 수 없는 트레이스백)이 모든 사용자에게 매번 노출됐다.
    """
    try:
        api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        st.error(
            "⚠️ GOOGLE_API_KEY가 설정되지 않았습니다. 로컬은 "
            "`.streamlit/secrets.toml`, 배포 환경은 Streamlit Cloud의 "
            "App settings → Secrets에 등록해주세요."
        )
        st.stop()

    try:
        return ChatGoogleGenerativeAI(model=CHAT_MODEL, google_api_key=api_key, temperature=0.3)
    except Exception as e:
        st.error(f"⚠️ Gemini 챗 모델 초기화에 실패했습니다: {e}")
        st.stop()


# 미리 인덱싱해둔 chroma_db/ 를 로드 (없으면 None → 일반 대화 모드로 동작)
@st.cache_resource
def get_vectorstore():
    try:
        return rag.load_vectorstore(st.secrets["GOOGLE_API_KEY"])
    except Exception as e:
        st.warning(f"약관 검색 인덱스를 불러오지 못했습니다 ({e}). 일반 대화 모드로 동작합니다.")
        return None


# 캐싱: 같은 질문이 반복되면(데모 리허설, 재질문 등) 매번 임베딩 API를 다시
# 부르지 않고 캐시된 결과를 즉시 반환한다 — 응답 속도도 빨라지고, 안 그래도
# 빠듯한 무료 등급 쿼터도 아낄 수 있다. 캐시 키는 질문 텍스트뿐이라 벡터
# 인덱스가 안 바뀌는 한 안전하다 (인덱스를 새로 빌드해 재배포하면 컨테이너가
# 새로 뜨면서 캐시도 함께 초기화된다).
@st.cache_data(show_spinner=False, ttl=3600)
def cached_retrieve(_vectorstore, query: str, k: int = 4):
    return rag.retrieve(_vectorstore, query, k)


def _history_cache_key(chat_history) -> str:
    """chat_history(LangChain 메시지 리스트)를 캐시 키로 쓸 문자열로 직렬화한다."""
    return "\x1f".join(f"{m.type}:{m.content}" for m in chat_history)


# 채팅 응답도 똑같은 이유로 캐싱한다. 캐시 키에는 시스템 프롬프트에 실린
# 검색 근거·설계 상태까지 전부 들어있어서, 근거나 설계가 조금이라도 달라지면
# 자동으로 새 키가 되어 다시 호출된다 — 오래된 답이 잘못 재사용될 위험은 없다.
@st.cache_data(show_spinner=False, ttl=3600)
def cached_llm_invoke(_llm, cache_key: str, _chat_history):
    return _llm.invoke(_chat_history)


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

# 세션 메시지 전체를 매 턴 통째로 API에 보내면 상담이 길어질수록 비용·지연이
# 계속 늘어난다. 최근 몇 턴만 컨텍스트로 사용한다 (사용자+어시스턴트 합산 개수).
MAX_HISTORY_MESSAGES = 10

RELEVANCE_WARNING = (
    "\n\n(주의: 위 근거는 질문과 관련성이 낮을 수 있습니다. "
    "약관 관련 사실 질문이라면 근거 없이 답하지 말고 확인이 필요하다고 안내하세요.)"
)

# 대화 기록 및 가입설계 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "design" not in st.session_state:
    st.session_state.design = {"base_products": [], "riders": {}}


def render_design_sidebar():
    """현재까지 대화로 합의된 가입설계 상태를 사이드바에 실시간으로 보여준다.

    고객이 암보험 + 연금보험처럼 서로 다른 상품을 동시에 여러 개 가입할 수
    있어서 base_products는 리스트다.
    """
    with st.sidebar:
        with st.container(border=True):
            st.markdown("#### 📋 현재 가입설계")
            design = st.session_state.design
            if design["base_products"]:
                st.markdown("**주계약**")
                for product in design["base_products"]:
                    st.markdown(f"　· {product}")
            if design["riders"]:
                st.markdown("**특약**")
                for name, amount in design["riders"].items():
                    if amount is not None:
                        st.markdown(f"　· {name} — **{amount:,.0f}만원**")
                    else:
                        st.markdown(f"　· {name}")
            if not design["base_products"] and not design["riders"]:
                st.caption("아직 설계된 내용이 없습니다. 채팅으로 상품이나 특약을 요청해보세요.")
            if st.button("🔄 설계 초기화", use_container_width=True):
                st.session_state.design = {"base_products": [], "riders": {}}
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
                try:
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
                except Exception as e:
                    st.error(f"추천 계산 중 오류가 발생했습니다: {e}")

            recs = st.session_state.get("recommendations")
            if recs:
                st.markdown("---")
                st.markdown("**🏆 추천 결과**")
                medals = ["🥇", "🥈", "🥉"]
                for i, r in enumerate(recs):
                    label = r.product + (f" - {r.rider}" if r.rider else "")
                    amount_label = f" ({r.suggested_amount:,}만원)" if r.suggested_amount else ""
                    medal = medals[i] if i < len(medals) else "▪️"
                    with st.container(border=True):
                        st.markdown(f"**{medal} {label}{amount_label}**")
                        for reason in r.reasons:
                            st.caption(f"· {reason}")
                        if st.button("➕ 설계에 추가", key=f"add_reco_{i}", use_container_width=True):
                            design = st.session_state.design
                            if r.product not in design["base_products"]:
                                design["base_products"].append(r.product)
                            if r.rider:
                                # 보험료납입면제특약처럼 애초에 "가입금액" 개념이 없는
                                # 특약(base_amount=None)에 임의로 1000만원을 넣지 않는다
                                # — None은 design_state에서 "금액 없는 특약"으로 처리됨.
                                design["riders"][r.rider] = r.suggested_amount
                            st.session_state.design = design
                            st.rerun()


render_design_sidebar()
render_recommendation_sidebar()

# 이전 대화 표시 (근거 페이지가 있으면 함께 표시)
for msg in st.session_state.messages:
    avatar = USER_AVATAR if msg["role"] == "user" else ASSISTANT_AVATAR
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg.get("confidence_percent") is not None:
            st.markdown(render_confidence_badge(msg["confidence_percent"]), unsafe_allow_html=True)
        if msg.get("sources"):
            st.markdown(render_sources_tag(msg["sources"]), unsafe_allow_html=True)

# 사용자 입력
if prompt := st.chat_input("가입설계에 대해 물어보세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(prompt)

    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        with st.spinner("약관 검색 중..."):
            # 1) 벡터 검색으로 근거 청크 조회 (임베딩 API 쿼터 초과 등으로 실패해도
            #    앱이 죽지 않고 "근거 없음"으로 안전하게 넘어가게 처리)
            retrieved = []
            if vectorstore:
                try:
                    retrieved = cached_retrieve(vectorstore, prompt, k=4)
                except Exception as e:
                    reason = "API 사용량 한도 초과" if "429" in str(e) else "일시적 오류"
                    st.caption(f"⚠️ 약관 검색 실패({reason})로 근거 없이 답변합니다.")
            has_evidence = rag.has_relevant_evidence(retrieved)
            context_text = rag.format_context(retrieved) if retrieved else "(검색된 근거 없음)"
            sources = rag.format_sources(retrieved) if has_evidence else None

            # 확신도(%)는 LLM이 말로 지어내는 게 아니라, 실제 검색 거리값을
            # 그대로 환산한 값이다 (rag.evidence_confidence_percent 참고).
            confidence_percent = rag.evidence_confidence_percent(retrieved) if retrieved else None

            # 2) 시스템 프롬프트 + 근거 + 현재 설계 상태를 포함해 대화 맥락 구성
            design_json = design_state.format_design_state(st.session_state.design)
            system_content = (
                f"{SYSTEM_PROMPT}\n\n[검색된 약관 근거]\n{context_text}"
                f"\n\n[현재 설계 상태]\n{design_json}"
            )
            if not has_evidence:
                system_content += RELEVANCE_WARNING

            chat_history = [SystemMessage(content=system_content)]
            for m in st.session_state.messages[-MAX_HISTORY_MESSAGES:]:
                if m["role"] == "user":
                    chat_history.append(HumanMessage(content=m["content"]))
                else:
                    chat_history.append(AIMessage(content=m["content"]))

        design_changed = False
        with st.spinner("생각 중..."):
            # 검색(rag.retrieve)과 인덱싱(_add_in_batches)에는 429/일시적 오류
            # 예외 처리가 있는데 정작 이 호출엔 없어서, 실패하면 Streamlit
            # 기본 에러 화면이 FC에게 그대로 노출되던 문제를 같은 패턴으로 수정.
            try:
                cache_key = _history_cache_key(chat_history)
                response = cached_llm_invoke(llm, cache_key, chat_history)
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
                if extracted.update is not None:
                    st.session_state.design = design_state.sanitize_design(
                        extracted.update, st.session_state.design
                    )
                    design_changed = True
            except Exception as e:
                reason = "API 사용량 한도 초과" if "429" in str(e) else "일시적 오류"
                st.caption(f"⚠️ 답변 생성 실패({reason})")
                answer = "일시적인 오류로 답변을 생성하지 못했습니다. 잠시 후 다시 시도해주세요."
                # 실패한 호출의 결과이므로 근거/확신도는 이 메시지에 붙이지 않는다.
                confidence_percent = None
                sources = None

            st.markdown(answer)
            if confidence_percent is not None:
                st.markdown(render_confidence_badge(confidence_percent), unsafe_allow_html=True)
            if sources:
                st.markdown(render_sources_tag(sources), unsafe_allow_html=True)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "confidence_percent": confidence_percent,
        }
    )

    if design_changed:
        st.rerun()  # 사이드바의 가입설계 요약을 즉시 갱신

import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

st.set_page_config(page_title="가입설계 챗봇", page_icon="💬")
st.title("가입설계 챗봇")
st.caption("동양생명 FC를 위한 AI 가입설계 도우미 (프로토타입)")

# Gemini 모델 초기화 (키는 Secrets에서 읽어옴)
@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=st.secrets["GOOGLE_API_KEY"],
        temperature=0.3,
    )

llm = get_llm()

# 시스템 프롬프트 (챗봇 성격 정의)
SYSTEM_PROMPT = """당신은 동양생명 보험설계사(FC)를 돕는 가입설계 어시스턴트입니다.
설계사가 상품, 특약, 가입설계에 대해 물으면 명확하고 실무적으로 답하세요.
확실하지 않은 내용은 지어내지 말고 '확인이 필요합니다'라고 답하세요."""

# 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 이전 대화 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 사용자 입력
if prompt := st.chat_input("가입설계에 대해 물어보세요"):
    # 사용자 메시지 표시 + 저장
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Gemini 호출
    with st.chat_message("assistant"):
        with st.spinner("생각 중..."):
            # 대화 맥락 구성
            chat_history = [SystemMessage(content=SYSTEM_PROMPT)]
            for m in st.session_state.messages:
                if m["role"] == "user":
                    chat_history.append(HumanMessage(content=m["content"]))
                else:
                    chat_history.append(AIMessage(content=m["content"]))

            response = llm.invoke(chat_history)
            answer = response.content
            st.markdown(answer)

    # 답변 저장
    st.session_state.messages.append({"role": "assistant", "content": answer})


# import streamlit as st
# import google.generativeai as genai
#
# st.title("모델 확인용")
#
# genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
#
# st.write("사용 가능한 모델 목록:")
# for m in genai.list_models():
#     if "generateContent" in m.supported_generation_methods:
#         st.write(m.name)
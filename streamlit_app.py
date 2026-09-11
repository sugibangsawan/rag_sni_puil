import streamlit as st

from rag_sni_puil.config import LLM_MODEL, vector_store
from rag_sni_puil.rag import ask

st.set_page_config(page_title="SNI PUIL & Sutrado Chat", page_icon="⚡", layout="centered")


@st.cache_resource
def get_store():
    return vector_store()


if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("SNI PUIL & Sutrado")
st.caption(f"Tanya jawab berdasarkan SNI PUIL 2011 dan spesifikasi kabel Sutrado, model {LLM_MODEL}")

with st.sidebar:
    st.subheader("Chat")
    if st.button("Clear history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Tulis pertanyaan tentang SNI PUIL atau kabel Sutrado..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Mencari di dokumen..."):
            answer = ask(
                prompt,
                history=st.session_state.messages[:-1],
                store=get_store(),
            )
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})

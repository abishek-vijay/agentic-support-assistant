import streamlit as st
from agent import chat

st.title("Support Agent")
ms = st.session_state.setdefault("m", [])
for m in ms:
    if m["role"] in ("user", "assistant") and m.get("content"):
        st.chat_message(m["role"]).write(m["content"])
if p := st.chat_input("Ask something"):
    st.chat_message("user").write(p)
    ms.append({"role": "user", "content": p})
    st.chat_message("assistant").write(chat(ms))

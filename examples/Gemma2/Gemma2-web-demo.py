from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st

with st.sidebar:
    st.markdown("## Gemma2.0 LLM")

st.title("💬 Gemma2.0 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

path = "google/gemma-2-2b-it"


@st.cache_resource
def get_model():
    print("Creat tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(path)
    print("Creat model...")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )

    return tokenizer, model


tokenizer, model = get_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

    st.chat_message("user").write(prompt)

    inputs = tokenizer.apply_chat_template(
        st.session_state.messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer.encode(inputs, add_special_tokens=False, return_tensors="pt")
    outputs = model.generate(input_ids=inputs.to(model.device), max_new_tokens=150)
    outputs = tokenizer.decode(outputs[0])
    response = outputs.split("model")[-1].replace("<end_of_turn>\n<eos>", "")

    st.session_state.messages.append({"role": "model", "content": response})

    st.chat_message("model").write(response)

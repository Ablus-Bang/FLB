from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import streamlit as st


with st.sidebar:
    st.markdown("## Phi-3 LLM")

st.title("💬 Phi-3 Chatbot")
st.caption("🚀 A streamlit chatbot powered by Self-LLM")

mode_name_or_path = '/home/zxd/workspace/models/phi3-4k-mini'

@st.cache_resource
def get_model():
    tokenizer = AutoTokenizer.from_pretrained(mode_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(mode_name_or_path, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
  
    return tokenizer, model

def bulid_input(prompt, history=[]):
    system_format='<s><|system|>\n{content}<|end|>\n'
    user_format='<|user|>\n{content}<|end|>\n'
    assistant_format='<|assistant|>\n{content}<|end|>\n'
    history.append({'role':'user','content':prompt})
    prompt_str = ''
    for item in history:
        if item['role']=='user':
            prompt_str+=user_format.format(content=item['content'])
        else:
            prompt_str+=assistant_format.format(content=item['content'])
    return prompt_str + '<|assistant|>\n'

tokenizer, model = get_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    
    st.chat_message("user").write(prompt)    
    input_str = bulid_input(prompt=prompt, history=st.session_state["messages"])
    input_ids = tokenizer.encode(input_str, add_special_tokens=False, return_tensors='pt').cuda()
    outputs = model.generate(
        input_ids=input_ids, max_new_tokens=512, do_sample=True,
        top_p=0.9, temperature=0.5, repetition_penalty=1.1, eos_token_id=tokenizer.encode('<|endoftext|>')[0]
        )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    
    response = tokenizer.decode(outputs)

    response = response.split('<|end|>')[0]
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").write(response)
    print(st.session_state)
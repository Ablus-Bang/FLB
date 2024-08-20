from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch

# Setting device parameters
DEVICE = "cuda"  # USE CUDA
DEVICE_ID = "0"  # CUDA device ID, or None if not set
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE  # Combine CUDA device information

# Clean up GPU memory function
def torch_gc():
    if torch.cuda.is_available():  # Check if CUDA is available
        with torch.cuda.device(CUDA_DEVICE):  # Specifying CUDA Devices
            torch.cuda.empty_cache()  # Clear CUDA cache
            torch.cuda.ipc_collect()  # Collecting CUDA memory fragments


app = FastAPI()

@app.post("/")
async def chat(request: Request):
    global model, tokenizer  # Declare global variables to use the model and tokenizer inside the function
    json_post_raw = await request.json()  
    json_post = json.dumps(json_post_raw)  
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get('prompt') 

    messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
    ]

    input_ids = tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    model_inputs = tokenizer([input_ids], return_tensors="pt").to('cuda')
    generated_ids = model.generate(model_inputs.input_ids,max_new_tokens=512)
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    now = datetime.datetime.now() 
    time = now.strftime("%Y-%m-%d %H:%M:%S")  
    answer = {
        "response": response,
        "status": 200,
        "time": time
    }
    torch_gc()  
    return answer

if __name__ == '__main__':
    model_name_or_path = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
   
    uvicorn.run(app, host='0.0.0.0', port=8080, workers=1) 
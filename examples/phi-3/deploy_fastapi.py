from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import json
import datetime
import torch


DEVICE = "cuda"
DEVICE_ID = "0"
CUDA_DEVICE = f"{DEVICE}:{DEVICE_ID}" if DEVICE_ID else DEVICE


def torch_gc():
    if torch.cuda.is_available():
        with torch.cuda.device(CUDA_DEVICE):
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()


app = FastAPI()


@app.post("/")
async def create_item(request: Request):
    global model, tokenizer
    json_post_raw = await request.json()
    json_post = json.dumps(json_post_raw)
    json_post_list = json.loads(json_post)
    prompt = json_post_list.get("prompt")

    print(prompt)
    messages = [{"role": "user", "content": prompt}]
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    output_ids = model.generate(input_ids.to("cuda"), max_new_tokens=2048)

    response = tokenizer.decode(
        output_ids[0][input_ids.shape[1] :], skip_special_tokens=True
    )

    now = datetime.datetime.now()
    time = now.strftime("%Y-%m-%d %H:%M:%S")

    answer = {"response": response, "status": 200, "time": time}

    log = (
        "["
        + time
        + "] "
        + '", prompt:"'
        + prompt
        + '", response:"'
        + repr(response)
        + '"'
    )
    print(log)
    torch_gc()
    return answer


if __name__ == "__main__":

    model_name_or_path = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
    ).eval()

    uvicorn.run(app, host="0.0.0.0", port=8080, workers=1)

from fastapi import FastAPI
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import torch
from logging import getLogger
logger = getLogger("uvicorn.error")

class Request(BaseModel):
    query: str

app = FastAPI()

model_name = "philschmid/instruct-igel-001"
# model = LlamaForCausalLM.from_pretrained(base_model_path,torch_dtype=torch.float16, device_map="auto")
# tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)

# model = AutoModelForCausalLM.from_pretrained("philschmid/instruct-igel-001", device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "sequential", torch_dtype=torch.half)
tokenizer = AutoTokenizer.from_pretrained(model_name)

generator = model.generate

def get_output(query:str):
    query = f"### Anweisung:\n{query}\n\n### Antwort:"
    gen_in = tokenizer.encode(query, return_tensors="pt").cuda()
    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            temperature=1.0,
            max_new_tokens=512,
            # repetition_penalty=1.0,
            top_p=0.95,
            length_penalty=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # parse output
    return tokenizer.batch_decode(generated_ids)[0]

def parse_output(model_output:str):
    parsed_output = model_output.split('### Antwort:')[1]
    parsed_output = parsed_output.split("<|endoftext|>")[0]
    return parsed_output

@app.post("/ask-model/")
async def ask_model(request: Request):
    start_time = time.time()
    model_output = get_output(request.query)
    parsed_text = parse_output(model_output)
    end_time = time.time()
    return {"question":request.query, "answer": parsed_text, "inference_time": end_time - start_time
}
# @app.post("/inference-time/")
# async def calc_inf_time(query: Request):
#     output = generator(f"### Anweisung:\n{query}\n\n### Antwort:")
#     parsed_text = output[0]["generated_text"].split("### Antwort:")[1]
#     return {"answer": parsed_text, "inference_time": inf_time}

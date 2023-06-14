from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import torch

base_model_path = "llama-13b-hf-out"
model = AutoModelForCausalLM.from_pretrained(base_model_path, torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=False, device_map="auto")
generator = model.generate

def get_output(user_input:str):

    generated_text = ""
    gen_in = tokenizer.encode(user_input, return_tensors="pt").cuda()
    in_tokens = len(gen_in)
    with torch.no_grad():
        generated_ids = generator(
            gen_in,
            max_new_tokens=2048,
            use_cache=True,
            pad_token_id=tokenizer.eos_token_id,
            num_return_sequences=1,
            do_sample=False,
            repetition_penalty=1.0,
            num_beams=1,
            early_stopping=False,
        )
        generated_text = tokenizer.batch_decode(generated_ids)[0]

    return generated_text
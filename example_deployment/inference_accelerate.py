## HuggingFace Transformers FAST API
from fastapi import FastAPI
import time
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, LlamaForCausalLM, LlamaTokenizer
from pydantic import BaseModel
import torch
from optimum.pipelines import pipeline
from optimum.bettertransformer import BetterTransformer
from logging import getLogger
from accelerate import infer_auto_device_map
import torch_tensorrt as torchtrt




logger = getLogger("uvicorn.error")

class Request(BaseModel):
    query: str

# app = FastAPI()

model = AutoModelForCausalLM.from_pretrained("philschmid/instruct-igel-001", device_map="sequential", torch_dtype=torch.half)
tokenizer = AutoTokenizer.from_pretrained("philschmid/instruct-igel-001")
print(model.config)
print(model.dtype)
model.eval()
# dummy_model_input = tokenizer("This is a sample input", return_tensors="pt")
# # COMPILE TRT module using Torch-TensorRT
# trt_module = torchtrt.compile(model,
# inputs=dummy_model_input,enabled_precisions={torch.half})
# # RUN optimized inference with Torch-TensorRT
# print(trt_module("This is a sample input"))

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
            top_p=0.9,
            length_penalty=1,
            pad_token_id=tokenizer.eos_token_id,
        )
    # parse output
    return tokenizer.batch_decode(generated_ids)[0]

questions = ["Schreibe eine Produktbeschreibung für einen LG 43UQ75009LF 109 cm (43 Zoll) UHD Fernseher (Active HDR, 60 Hz, Smart TV) [Modelljahr 2022]", "Wie ist das Wetter morgen?", "Wie ist das Wetter in 3 Tagen?", "Beantworten Sie die Frage am Ende des Textes anhand der folgenden Zusammenhänge. Wenn Sie die Antwort nicht wissen, sagen Sie, dass Sie es nicht wissen, versuchen Sie nicht, eine Antwort zu erfinden. Das Unternehmen wurde 2016 von den französischen Unternehmern Clément Delangue, Julien Chaumond und Thomas Wolf gegründet und entwickelte ursprünglich eine Chatbot-App, die sich an Teenager richtete.[2] Nachdem das Modell hinter dem Chatbot offengelegt wurde, konzentrierte sich das Unternehmen auf eine Plattform für maschinelles Lernen. Im März 2021 sammelte Hugging Face in einer Serie-B-Finanzierungsrunde 40 Millionen US-Dollar ein[3]. Am 28. April 2021 rief das Unternehmen in Zusammenarbeit mit mehreren anderen Forschungsgruppen den BigScience Research Workshop ins Leben, um ein offenes großes Sprachmodell zu veröffentlichen.[4] Im Jahr 2022 wurde der Workshop mit der Ankündigung von BLOOM abgeschlossen, einem mehrsprachigen großen Sprachmodell mit 176 Milliarden Parametern.[5]' Frage: Wann wurde Hugging Face gegründet?","Kannst du mir sagen, wie das Wetter in 3 Tagen ist?"]
for q in questions:
    start_time = time.time()
    print(get_output(q))
    end_time = time.time()
    print(f"Time: {end_time - start_time}")

# def parse_output(model_output:str):
#     parsed_output = model_output.split('### Antwort:')[1]
#     parsed_output = parsed_output.split("<|endoftext|>")[0]
#     return parsed_output

# @app.post("/ask-model/")
# async def ask_model(request: Request):
#     start_time = time.time()
#     model_output = get_output(request.query)
#     parsed_text = parse_output(model_output)
#     end_time = time.time()
#     return {"question":request.query, "answer": parsed_text, "inference_time": end_time - start_time
# }
# # @app.post("/inference-time/")
# # async def calc_inf_time(query: Request):
# #     output = generator(f"### Anweisung:\n{query}\n\n### Antwort:")
# #     parsed_text = output[0]["generated_text"].split("### Antwort:")[1]
# #     return {"answer": parsed_text, "inference_time": inf_time}

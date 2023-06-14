## HuggingFace Transformers FAST API
import time
from transformers import AutoTokenizer, AutoModelForCausalLM


tokenizer = AutoTokenizer.from_pretrained("philschmid/instruct-igel-001")
model = AutoModelForCausalLM.from_pretrained("philschmid/instruct-igel-001")
#


def get_output(query: str):
    tokenizer.encode_plus(query, return_tensors="pt")
    output = model.generate(
        **tokenizer.encode_plus(query, return_tensors="pt"),
        max_length=100,
        temperature=0.8,
        no_repeat_ngram_size=2,
        top_p=0.92,
        penalty=1.0,
    )

import torch
from peft import PeftModel
from transformers import LlamaForCausalLM, LlamaTokenizer, GenerationConfig

from utils.prompter import Prompter

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def load_model(
    base_model: str = "decapoda-research/llama-7b-hf",
    lora_weights: str = 'howardchen123/alpaca-lora-llama-sentiment'
):
    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    if device == "cuda":
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=False,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            torch_dtype=torch.float16,
        )
    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model, device_map={"": device}, low_cpu_mem_usage=True
        )
        model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map={"": device},
        )

    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    model.half()
    model.eval()
    model = torch.compile(model)

    return model, tokenizer

print("Loading Model from Hugging Face")
llm_model, tokenizer = load_model()
prompter = Prompter()


def generate_response(input: str):

    instruction = "Detect the sentiment of the tweet."
    prompt = prompter.generate_prompt(instruction, input)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=0.1,
        top_p=0.75,
        top_k=0.4,
        num_beams=4,
    )

    with torch.no_grad():
        generation_output = llm_model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=128
        )
    
    s = generation_output.sequences[0]
    output = tokenizer.decode(s)
    yield prompter.get_response(output)
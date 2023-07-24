# Greatly Inspired by Stanford's Alpaca-LoRA

import torch
from transformers import (
    LlamaTokenizer,
    LlamaForCausalLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, prepare_model_for_int8_training

class Configuration:
    # Model configuration
    MODEL_PATH = 'decapoda-research/llama-7b-hf'
    LORA_R = 8
    LORA_ALPHA = 16
    LORA_DROPOUT = 0.05
    LORA_TARGET_MODULES = ["q_proj", "v_proj"]

    # Training configuration
    BATCH_SIZE = 128
    MICRO_BATCH_SIZE = 4
    GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
    LEARNING_RATE = 3e-4
    TRAIN_STEPS = 300
    OUTPUT_DIR = "experiments"

def setup_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def setup_model(model_path):
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    model = LlamaForCausalLM.from_pretrained(
        model_path, load_in_8bit=True, torch_dtype=torch.float16, device_map="auto"
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"
    return model, tokenizer

def tokenize_prompt(tokenizer, prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        padding=False,
        return_tensors=None,
    )
    if result["input_ids"][-1] != tokenizer.eos_token_id:
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result

def generate_prompt(instruction, input, label):
    return f"""Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
                ### Instruction:
                {instruction}
                ### Input:
                {input}
                ### Response:
                {label}
            """

def generate_and_tokenize_prompt(tokenizer, data_point):
    full_prompt = generate_prompt(
        data_point["instruction"],
        data_point["input"],
        data_point["output"],
    )
    tokenized_full_prompt = tokenize_prompt(tokenizer, full_prompt)
    return tokenized_full_prompt

def prepare_datasets(data_path, tokenizer):
    data = load_dataset("csv", data_files=data_path)
    train_val = data["train"].train_test_split(test_size=10000, shuffle=True, seed=0)
    train_data = train_val["train"].map(lambda x: generate_and_tokenize_prompt(tokenizer, x))
    val_data = train_val["test"].map(lambda x: generate_and_tokenize_prompt(tokenizer, x))
    return train_data, val_data

def setup_peft_model(model):
    config = LoraConfig(
        r=Configuration.LORA_R,
        lora_alpha=Configuration.LORA_ALPHA,
        target_modules=Configuration.LORA_TARGET_MODULES,
        lora_dropout=Configuration.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model

def setup_training_arguments():
    training_args = {
        "per_device_train_batch_size": Configuration.MICRO_BATCH_SIZE,
        "gradient_accumulation_steps": Configuration.GRADIENT_ACCUMULATION_STEPS,
        "warmup_steps": 100,
        "max_steps": Configuration.TRAIN_STEPS,
        "learning_rate": Configuration.LEARNING_RATE,
        "fp16": True,
        "logging_steps": 10,
        "optim": "adamw_torch",
        "evaluation_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": 50,
        "save_steps": 50,
        "output_dir": Configuration.OUTPUT_DIR,
        "save_total_limit": 3,
        "load_best_model_at_end": True,
        "report_to": "tensorboard",
    }
    return TrainingArguments(**training_args)

def setup_data_collator(tokenizer):
    return DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

def setup_trainer(model, train_data, val_data, training_arguments, data_collator):
    trainer = Trainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        args=training_arguments,
        data_collator=data_collator
    )
    return trainer

def main():
    DEVICE = setup_device()

    model, tokenizer = setup_model(Configuration.MODEL_PATH)
    train_data, val_data = prepare_datasets(data_path="/content/drive/MyDrive/repo/LLM-Sentimental-Analysis/data/alpaca-news-sentiment-dataset.csv", tokenizer=tokenizer)

    model = setup_peft_model(model)
    training_arguments = setup_training_arguments()
    data_collator = setup_data_collator(tokenizer)

    trainer = setup_trainer(model, train_data, val_data, training_arguments, data_collator)

    model.config.use_cache = False
    old_state_dict = model.state_dict()
    model.load_state_dict(get_peft_model_state_dict(model, old_state_dict))

    model = torch.compile(model)

    trainer.train()
    # Save model if training was successful
    if trainer.is_world_process_zero():
        model.save_pretrained(Configuration.OUTPUT_DIR)

if __name__ == "__main__":
    main()
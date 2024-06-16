import os
import sys
from typing import Tuple, Union

import fire
import torch
from datasets import load_dataset
from handler import DataHandler
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    LlamaForCausalLM,
    LlamaTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

class ProactiveTrainingCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        control.should_save = True

class ProactiveTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        needs_more_information = inputs.pop("needs_more_information", None)
        outputs = model(**inputs)
        logits = outputs.get("logits")

        if needs_more_information is not None:
            loss = proactive_loss(logits, labels, needs_more_information)
        else:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))

        return (loss, outputs) if return_outputs else loss

def proactive_loss(predictions, labels, needs_more_information):
    """
    Custom loss function for proactive LLM training.
    Args:
        predictions: Model predictions.
        labels: Ground truth labels (float values).
        needs_more_information: Float mask indicating the quality of information at that point.
    Returns:
        Loss value.
    """
    # Standard Cross Entropy Loss
    ce_loss = torch.nn.functional.cross_entropy(predictions.view(-1, predictions.size(-1)), labels.view(-1), reduction='none')

    # Proactive penalty: penalize if the model doesn't ask questions when it should
    proactive_penalty = ce_loss * (1 - needs_more_information.view(-1))

    # Final loss is a combination of standard CE loss and proactive penalty
    loss = ce_loss.mean() + proactive_penalty.mean()

    return loss


def main(
    model: str,
    val_set_size: Union[int, float] = 0.1,
    prompt_template: str = "/u/ybkim95/proactive-llm/medAlpaca/medalpaca/prompt_templates/medalpaca.json",
    model_max_length: int = 256,
    train_on_inputs: bool = True,
    data_path: str = "/u/ybkim95/proactive-llm/data/MDDial/augmented_medical_dialogues2.json",
    train_in_8bit: bool = False,
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    lora_target_modules: Tuple[str] = ("q_proj", "v_proj"),
    per_device_batch_size: int = 1,
    num_epochs: int = 3,
    learning_rate: float = 2e-5,
    global_batch_size: int = 1,
    output_dir: str = "./output",
    save_total_limit: int = 3,
    eval_steps: int = 200,
    device_map: str = "auto",
    wandb_run_name: str = "test",
    use_wandb: bool = False,
    wandb_project: str = "medalpaca",
    optim: str = "adamw_torch",
    lr_scheduler_type: str = "cosine",
    fp16: bool = True,
    bf16: bool = False,
    gradient_checkpointing: bool = False,  # Disable gradient checkpointing for LoRA
    warmup_steps: int = 100,
    **kwargs
):
    # Setting environment variable to control memory allocation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    model_name = model
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    gradient_accumulation_steps = max(1, global_batch_size // per_device_batch_size)
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = max(1, gradient_accumulation_steps // world_size)

    if train_in_8bit and not use_lora:
        raise ValueError("8bit training without LoRA is not supported")

    if "llama" in model_name:
        load_model = LlamaForCausalLM
    else:
        load_model = AutoModelForCausalLM

    model = load_model.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if any([use_lora, bf16, fp16]) else torch.float32,
        device_map=device_map,
    )

    if use_lora:
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    if "llama" in model_name.lower():
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

    data_handler = DataHandler(
        tokenizer=tokenizer,
        prompt_template=prompt_template,
        model_max_length=model_max_length,
        train_on_inputs=train_on_inputs,
    )

    data = load_dataset("json", data_files=data_path)

    if val_set_size > 0:
        data = (
            data["train"]
            .train_test_split(test_size=val_set_size, shuffle=True, seed=42)
            .map(data_handler.generate_and_tokenize_prompt)
        )
    else:
        data = data.shuffle(seed=42).map(data_handler.generate_and_tokenize_prompt)

    # Print out some samples for debugging
    for i in range(3):
        print(f"Sample {i}:", data["train"][i])

    if not ddp and torch.cuda.device_count() > 1:
        model.is_parallelizable = True
        model.model_parallel = True

    training_args = TrainingArguments(
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        optim=optim,
        lr_scheduler_type=lr_scheduler_type,
        evaluation_strategy="steps" if val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if val_set_size > 0 else None,
        save_steps=eval_steps,
        output_dir=output_dir,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True if val_set_size > 0 else False,
        ddp_find_unused_parameters=False if ddp else None,
        group_by_length=False,
        report_to="wandb" if use_wandb else None,
        run_name=wandb_run_name if use_wandb else None,
        gradient_checkpointing=gradient_checkpointing,
        **kwargs
    )

    trainer = ProactiveTrainer(
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["test"] if val_set_size > 0 else None,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
        compute_metrics=None,
        callbacks=[ProactiveTrainingCallback()],
    )

    model.config.use_cache = False

    if use_lora:
        old_state_dict = model.state_dict
        model.state_dict = (
            lambda self, *_, **__: get_peft_model_state_dict(self, old_state_dict())
        ).__get__(model, type(model))

    try:
        trainer.train()
    except:
        pass

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    fire.Fire(main)

import re
import json
import fire
import string
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline

sampling = {
    "do_sample": True,
    "top_k": 50,
    "num_beams": 1,
    "max_new_tokens": 128,
    "early_stopping": True,
    "temperature": 0.4,
    "top_p": 0.9
}

class Inferer:
    def __init__(self, checkpoint_path, base_model, peft):
        self.checkpoint_path = checkpoint_path
        self.base_model = base_model
        self.peft = peft
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained('yahma/llama-13b-hf')
        
        # Load the model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        
        # Ensure the model is in evaluation mode
        self.model.eval()
        
        # Define the inference pipeline
        self.text_generator = TextGenerationPipeline(model=self.model, tokenizer=self.tokenizer, device=0 if torch.cuda.is_available() else -1)
    
    def __call__(self, instruction, input, output, **generation_kwargs):
        prompt = f"{instruction}\n\n{input}\n\n{output}"

        print("prompt:", prompt)
        print()

        return self.text_generator(prompt, **generation_kwargs)[0]['generated_text']

def strip_special_chars(input_str):
    "Remove special characters from string start/end"
    if not input_str:
        return input_str
    
    start_index = 0
    end_index = len(input_str) - 1

    while start_index < len(input_str) and input_str[start_index] not in string.ascii_letters + string.digits:
        start_index += 1

    while end_index >= 0 and input_str[end_index] not in string.ascii_letters + string.digits:
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index:end_index + 1]
    else:
        return ""

def main(
    checkpoint_path: str = "/u/ybkim95/proactive-llm/medAlpaca/lora-alpaca-13b", 
    base_model: str = "yahma/llama-13b-hf",
    peft: bool = True,
):
    model = Inferer(
        checkpoint_path=checkpoint_path,
        base_model=base_model,
        peft=peft,
    )

    question = "Patient: Recently, I am experiencing Cough."

    formatted_question = question
    response = model(
        instruction="Based on the patient's query, make a diagnosis.",
        input=formatted_question,
        output="Answer:",
        **sampling
    )
    response = strip_special_chars(response)
    
    print(f"Answer:\n{response}")
    # else:
        # print(f"Generated Answer does not start with a capital letter: {response}")

if __name__ == "__main__":
    fire.Fire(main)

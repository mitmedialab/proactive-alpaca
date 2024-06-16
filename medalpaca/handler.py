import json
import logging
from typing import Dict, Optional

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

def load_json(fn: str):
    with open(fn, "r") as fp:
        d = json.load(fp)
    return d

class DataHandler:
    """Helper class to handle prompt generation and data tokenization."""

    def __init__(
        self,
        tokenizer,
        prompt_template: str = "/u/ybkim95/proactive-llm/medAlpaca/medalpaca/prompt_templates/medalpaca.json",
        model_max_length: int = 256,
        train_on_inputs: bool = True,
    ) -> None:
        if model_max_length > 2048:
            logger.warn(f"{model_max_length} exceeds the max token length LLaMA was trained with.")
        self.prompt_template = load_json(prompt_template)
        self.model_max_length = model_max_length
        self.train_on_inputs = train_on_inputs
        self.tokenizer = tokenizer

    def tokenize(self, prompt: str, add_eos_token: bool = True, return_tensors: str = None, truncation: bool = True) -> Dict[str, list]:
        result: Dict = self.tokenizer(
            prompt,
            truncation=truncation,
            max_length=self.model_max_length,
            padding=False,
            return_tensors=return_tensors,
            add_special_tokens=False,
        )
        if (
            result["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(result["input_ids"]) < self.model_max_length
            and add_eos_token
        ):
            result["input_ids"].append(self.tokenizer.eos_token_id)
            result["attention_mask"].append(1)

        result["labels"] = result["input_ids"].copy()

        return result

    def generate_and_tokenize_prompt(self, data_point: Dict):
        prompt: str = self.generate_prompt(
            initial_patient_statement=data_point.get("initial_patient_statement", ""),
            dialogue=data_point.get("dialogue", []),
            final_diagnosis=data_point.get("final_diagnosis", ""),
        )
        tokenized_prompt: Dict = self.tokenize(prompt)
        tokenized_prompt["needs_more_information"] = data_point.get("needs_more_information", 0.0)
        if not self.train_on_inputs:
            user_prompt: str = self.generate_prompt(
                initial_patient_statement=data_point.get("initial_patient_statement", ""),
                dialogue=[d for d in data_point.get("dialogue", []) if d['type'] == 'question']
            )
            tokenized_user_prompt: Dict = self.tokenize(user_prompt, add_eos_token=False)
            user_prompt_len = len(tokenized_user_prompt["input_ids"])
            tokenized_prompt["labels"] = [
                -100 if i < user_prompt_len else label
                for i, label in enumerate(tokenized_prompt["labels"])
            ]
        return tokenized_prompt

    def generate_prompt(
        self,
        initial_patient_statement: Optional[str] = None,
        dialogue: Optional[list] = None,
        final_diagnosis: Optional[str] = None,
    ) -> str:
        if not any([initial_patient_statement, dialogue, final_diagnosis]):
            raise ValueError("At least one of `initial_patient_statement`, `dialogue`, `final_diagnosis` should be defined")

        prompt = (
            f'{self.prompt_template["primer"]}'
            f'{self.prompt_template["instruction"]}The patient says: {initial_patient_statement}\n'
            f'{self.prompt_template["input"]}Dialogue:\n'
        )

        for turn in dialogue:
            for k,v in turn.items():
                prompt += f"({k}): {v}"
            prompt += "\n"

        prompt += f'{self.prompt_template["output"]}The final diagnosis is: {final_diagnosis}\n'

        return prompt

    def resolve_output(self, output: str): 
        pass

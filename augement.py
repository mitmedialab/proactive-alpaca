import json
import itertools
from typing import List, Dict, Tuple

def generate_turn_subsets(dialog: List[Dict]) -> List[Tuple[List[Dict], float]]:
    turn_subsets = []
    num_turns = len(dialog)
    for r in range(1, num_turns):  # Ensure at least the first turn is included, so start from 1
        for subset in itertools.combinations(dialog[1:], r):  # Skip the first turn in combinations
            subset_with_first_turn = [dialog[0]] + list(subset)
            label = len(subset_with_first_turn) / num_turns
            turn_subsets.append((subset_with_first_turn, label))
    # Add the full dialog as well
    turn_subsets.append((dialog, 1.0))
    return turn_subsets

def augment_dialogs(dialogs: Dict[str, List[Dict]]) -> List[Dict]:
    augmented_dialogs = []
    for dialog_id, dialog in dialogs.items():
        # Convert to expected format
        dialogue_list = []
        for turn in dialog:
            for speaker, text in turn.items():
                dialogue_list.append({"type": speaker, "text": text})
        
        turn_subsets = generate_turn_subsets(dialogue_list)
        for subset, label in turn_subsets:
            augmented_dialogs.append({
                "initial_patient_statement": dialogue_list[0]["text"],  # Assuming first turn is patient statement
                "dialogue": subset,
                "final_diagnosis": "",  # Assuming final diagnosis is not provided in the original data
                "needs_more_information": label
            })
    return augmented_dialogs

def load_and_augment_data(input_path: str, output_path: str):
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    augmented_data = augment_dialogs(data)
    
    with open(output_path, 'w') as f:
        json.dump(augmented_data, f, indent=4)

# Example usage
input_path = '/data/healthy-ml/scratch/xoxu/projects/data/train.json'
output_path = '/data/healthy-ml/scratch/xoxu/projects/data/augmented_medical_dialogues2.json'
load_and_augment_data(input_path, output_path)

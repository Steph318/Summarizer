import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    
    def convert_examples_to_features(self, example_batch):
        # Ensure dialogue and summary are lists of strings
        dialogues = example_batch['dialogue']
        summaries = example_batch['summary']
        
        for dialogue in example_batch['dialogue']:
            if not isinstance(dialogue, str):
                print("Invalid dialogue entry:", dialogue)  # Debugging

        if not all(isinstance(d, str) for d in example_batch['dialogue']):
            raise TypeError("All dialogues must be strings.")
        if not all(isinstance(s, str) for s in example_batch['summary']):
            raise TypeError("All summaries must be strings.")


        # Tokenize input (dialogues)
        input_encodings = self.tokenizer(
            dialogues, 
            max_length=1024, 
            truncation=True, 
            padding=True
        )

        # Tokenize target (summaries)
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                summaries, 
                max_length=128, 
                truncation=True, 
                padding=True
            )
        
        # Return encodings in the expected format
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    

    def convert(self):
        dataset_samsum = load_dataset(self.config.data_path)  # For Hugging Face hub datasets

        
        def is_valid_example(example):
            return isinstance(example['dialogue'], str) and isinstance(example['summary'], str)

        dataset_samsum = dataset_samsum.filter(is_valid_example)

        dataset_samsum_pt = dataset_samsum.map(
            self.convert_examples_to_features,
            batched=True
        )
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum-dataset")
        )
        print("Dataset processed and saved.")

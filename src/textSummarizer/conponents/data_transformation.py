import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_dataset, load_from_disk
from textSummarizer.entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)


    def clean_dataset(self, dataset):
        # Retain only 'dialogue' and 'summary' columns
        necessary_columns = ['input_ids', 'attention_mask', 'labels']
        for split in dataset.keys():  # Iterate over all splits (train, validation, test)
            dataset[split] = dataset[split].remove_columns(
                [col for col in dataset[split].column_names if col not in necessary_columns]
            )
        return dataset


    def convert_examples_to_features(self, example_batch):
        """
        Converts a batch of examples into features suitable for model training.

        Args:
            example_batch (dict): A batch of examples containing 'dialogue' and 'summary'.

        Returns:
            dict: A dictionary containing 'input_ids', 'attention_mask', and 'labels'.
        """
        # Extract dialogues and summaries
        dialogues = example_batch.get('dialogue', [])
        summaries = example_batch.get('summary', [])

        # Ensure dialogues and summaries are lists of strings
        if not isinstance(dialogues, list) or not all(isinstance(d, str) for d in dialogues):
            raise TypeError("All dialogues must be strings.")
        if not isinstance(summaries, list) or not all(isinstance(s, str) for s in summaries):
            raise TypeError("All summaries must be strings.")

        # Tokenize input dialogues
        input_encodings = self.tokenizer(
            dialogues,
            max_length=1024,
            truncation=True,
            padding=True
        )

        # Tokenize target summaries
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                summaries,
                max_length=128,
                truncation=True,
                padding=True
            )

        # Return tokenized features in the required format
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
        dataset_samsum = self.clean_dataset(dataset_samsum)

        print("Dataset columns:", dataset_samsum_pt.column_names)

        dataset_samsum_pt = dataset_samsum_pt.remove_columns(["id", "dialogue", "summary"])

        print("Dataset columns:", dataset_samsum_pt.column_names)
        
        dataset_samsum_pt.save_to_disk(
            os.path.join(self.config.root_dir, "samsum-dataset")
        )
        print("Dataset processed and saved.")

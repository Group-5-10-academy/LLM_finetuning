from transformers import LlamaTokenizer, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, DatasetDict
import torch

model_name = "path/to/pretrained-llama-model"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

data = {
"train": [
{"instruction": "Translate the following sentence to Amharic:", "input": "How are you?", "output": "እእእእ እእ/እእ?"},
# Add more examples
],
"validation": [
{"instruction": "Translate the following sentence to Amharic:", "input": "Good morning", "output": "እእእእእ እእእእ?"},
# Add more examples
],
}

dataset = DatasetDict({"train": load_dataset("json", data_files={"train": "path/to/train_data.json"}),
"validation": load_dataset("json", data_files={"validation": "path/to/validation_data.json"})})

def tokenize_function(examples):
    return tokenizer(
examples["instruction"] + " " + examples["input"],
padding="max_length",
truncation=True,
max_length=512
)
tokenized_datasets = dataset.map(tokenize_function, batched=True)
def format_data(examples):
    return {"input_ids": examples["input_ids"], "labels": examples["input_ids"]}

lm_datasets = tokenized_datasets.map(format_data, batched=True)

training_args = TrainingArguments(
output_dir="./results",
evaluation_strategy="epoch",
learning_rate=5e-5,
per_device_train_batch_size=4,
per_device_eval_batch_size=4,
num_train_epochs=3,
weight_decay=0.01,
save_total_limit=3,
logging_dir='./logs',
)

eval_results = trainer.evaluate()
print(eval_results)
fine_tuned_model = LlamaForCausalLM.from_pretrained("path/to/save-model")
fine_tuned_tokenizer = LlamaTokenizer.from_pretrained("path/to/save-model")

inputs = fine_tuned_tokenizer("Translate the following sentence to Amharic: How are you?", return_tensors="pt")
outputs = fine_tuned_model.generate(**inputs)
print(fine_tuned_tokenizer.decode(outputs[0], skip_special_tokens=True))
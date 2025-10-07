from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


'''
what is model_checkpoint = "distilbert-base-uncased"
what is imdb_dataset.map(preprocess_function, batched=True), how does it split batch, why in the function is examples["text"], not example? can it apply process to all entries at the same time?
"`input_ids`: The numerical representation of the text.", how can you use a number to represent a text and why
how attention_mask is calculated
what is from_pretrained
pls explain all params in TrainingArguments
explain np.argmax(logits, axis = -1)
'''
# 1. Loading Data From Hugging Face

imdb_dataset = load_dataset('imdb', cache_dir="hf_cache")

print("---Dataset Structure---")
print(imdb_dataset)

print('---Training Data Example---')
example = imdb_dataset['train'][-1]
print(example)

# 2. Tokenization
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, cache_dir="hf_cache")

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation = True, padding = True)

tokenized_datasets = imdb_dataset.map(preprocess_function, batched=True)

print("\n--- Tokenized Dataset Structure ---")
print(tokenized_datasets)

# 3. Fine-tuning the model
model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=2, cache_dir="hf_cache")

training_args = TrainingArguments(
    output_dir="results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_eval_batch_size=16,
    per_device_train_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis = -1)
    return {"accuracy" : accuracy_score(labels, predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(1000)),
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(1000)),  # Using a smaller subset for speed
    tokenizer = tokenizer,
    compute_metrics=compute_metrics
)

print("\n--- Starting Training ---")
trainer.train()
print("--- Training Finished ---")

# 11. Evaluate the model
print("\n--- Evaluating Model ---")
evaluation_results = trainer.evaluate()
print(evaluation_results)
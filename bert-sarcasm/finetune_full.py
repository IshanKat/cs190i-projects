from config import *
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer, EvalPrediction, TrainerCallback
from datasets import load_dataset
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from transformers import default_data_collator
from progress_callback import TQDMProgressBar
import matplotlib.pyplot as plt
import time
import os

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR_FULL, exist_ok=True)

# Set up logging
log_file = os.path.join(OUTPUT_DIR_FULL, 'training_log.txt')
with open(log_file, 'w') as f:
    f.write("=== Training Log ===\n\n")

def log_message(message):
    print(message)
    with open(log_file, 'a') as f:
        f.write(message + "\n")

# Track training metrics
training_losses = []
training_times = []
start_time = time.time()

class MetricsCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            training_losses.append(logs["loss"])
            training_times.append(time.time() - start_time)
            log_message(f"Step {state.global_step}: loss = {logs['loss']:.4f}")

log_message("Loading dataset...")
dataset = load_dataset("tweet_eval", "irony")

# === Tokenizer and preprocessing ===
tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128
    )

tokenized = dataset.map(preprocess_function, batched=True)
tokenized = tokenized.remove_columns(["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# Load model (default: all layers trainable)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

# Metrics
def compute_metrics(p: EvalPrediction):
    pred = np.argmax(p.predictions, axis=1)
    return {"accuracy": accuracy_score(p.label_ids, pred), "f1": f1_score(p.label_ids, pred)}

# Evaluate base model before training
log_message("\nEvaluating base model before training...")
base_trainer = Trainer(
    model=model,
    args=TrainingArguments(output_dir=OUTPUT_DIR_FULL, per_device_eval_batch_size=EVAL_BATCH_SIZE),
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics
)
base_metrics = base_trainer.evaluate()
log_message(f"Base model metrics: {base_metrics}")

# Training arguments
args = TrainingArguments(
    output_dir=OUTPUT_DIR_FULL,
    eval_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=EVAL_BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=WEIGHT_DECAY,
    logging_dir="./logs",
    logging_steps=LOGGING_STEPS
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    tokenizer=tokenizer,
    data_collator=default_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[TQDMProgressBar(), MetricsCallback()]
)

log_message("\nStarting training...")
trainer.train()

log_message("\nEvaluating after training...")
final_metrics = trainer.evaluate()
log_message(f"Final metrics: {final_metrics}")

# Plot training metrics
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss on left y-axis
color = 'tab:blue'
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Loss', color=color)
ax1.plot(range(len(training_losses)), training_losses, color=color, label='Loss')
ax1.tick_params(axis='y', labelcolor=color)

# Create second y-axis for training time
ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Training Time (seconds)', color=color)
ax2.plot(range(len(training_times)), training_times, color=color, label='Time')
ax2.tick_params(axis='y', labelcolor=color)

# Add title and legend
plt.title('Training Loss and Time')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR_FULL, 'training_metrics.png'))
plt.close()

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path

import datasets
from datasets import Dataset, DatasetDict

import torch
from datasets import ClassLabel, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import evaluate
import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    PretrainedConfig,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

device = "gpu" if torch.cuda.is_available() else "cpu"

# data ################################################
aspect = 1

data = {'tokens': [],
        'labels': []}

with open("../data/BeerAdvocate/annotations.json") as fin:
    for line in fin:
        sample = json.loads(line)
        data['tokens'].append(sample['x'])

        tokens_mask = torch.zeros((len(sample['x']),), dtype=torch.int)
        for rng in sample[str(aspect)]:
            tokens_mask[rng[0]:rng[1]] = 1

        data['labels'].append(tokens_mask.tolist())

# Whole data as a dataset object
raw_dataset = Dataset.from_dict(data)

# 90% train, 10% (test + validation)
train_test_valid = raw_dataset.train_test_split(test_size=0.1)
test_valid = train_test_valid['test'].train_test_split(test_size=0.5)

# Generating divided dataset dictionary
dataset = DatasetDict({
    'train': train_test_valid['train'],
    'validation': test_valid['train'],
    'test': test_valid['test']})

label_list = ['O', 'B-SELECTED', 'I-SELECTED']
num_labels = len(label_list)

# label2id = {l: i for i, l in enumerate(label_list)}
# id2label = {i: l for i, l in enumerate(label_list)}

label2id = {i: i for i in range(len(label_list))}
id2label = {i: i for i in range(len(label_list))}


# Map that sends B-Xxx label to its I-Xxx counterpart
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and label.replace("B-", "I-") in label_list:
        b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    else:
        b_to_i_label.append(idx)

model_name_or_path = 'bert-base-uncased'
tokenizer_name_or_path = model_name_or_path

# XXX: what if we have binary token classification instead?
# Do we need to introduce a new task? is this optional?
task_name = 'ner'
cache_dir = None
model_revision = 'main'
use_auth_token = False

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path,
    use_fast=True)

model = AutoModelForTokenClassification.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config)

# #######################################################################
padding = False
max_length = 128
label_all_tokens = False

# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        max_length=max_length,
        padding=padding,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )

    labels = []
    for i, label in enumerate(examples['labels']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label2id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                if label_all_tokens:
                    label_ids.append(b_to_i_label[label2id[label[word_idx]]])
                else:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# #######################################################################
processed_raw_datasets = dataset.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_raw_datasets["train"]
eval_dataset = processed_raw_datasets["validation"]

# Data collator
use_fp16 = False
data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if use_fp16 else None)

batch_size = 64
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

# #######################################################################
# Optimizer

weight_decay = 0.
learning_rate = 5e-5

# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
    },
    {
        "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    },
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

# # Use the device given by the `accelerator` object.
# device = accelerator.device
# model.to(device)

# #######################################################################
# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
gradient_accumulation_steps = 1
max_train_steps = None
num_train_epochs = 3
lr_scheduler_type = 'linear'
num_warmup_steps = 0
checkpointing_steps = None

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if max_train_steps is None:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

lr_scheduler = get_scheduler(
    name=lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=max_train_steps,
)

# We need to recalculate our total training steps as the size of the training dataloader may have changed.
num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
if overrode_max_train_steps:
    max_train_steps = num_train_epochs * num_update_steps_per_epoch
# Afterwards we recalculate our number of training epochs
num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

# Figure out how many steps we should save the Accelerator states
if checkpointing_steps is not None and checkpointing_steps.isdigit():
    checkpointing_steps = int(checkpointing_steps)

# #######################################################################
import numpy as np

# Metrics
metric = evaluate.load('seqeval')

return_entity_level_metrics = False

def get_labels(predictions, references):
    # Transform predictions and references tensos to numpy arrays
    if device == "cpu":
        y_pred = predictions.detach().clone().numpy()
        y_true = references.detach().clone().numpy()
    else:
        y_pred = predictions.detach().cpu().clone().numpy()
        y_true = references.detach().cpu().clone().numpy()

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(pred, gold_label) if l != -100]
        for pred, gold_label in zip(y_pred, y_true)
    ]
    return true_predictions, true_labels


def compute_metrics():
    results = metric.compute()
    if return_entity_level_metrics:
        # Unpack nested dictionaries
        final_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                for n, v in value.items():
                    final_results[f"{key}_{n}"] = v
            else:
                final_results[key] = value
        return final_results
    else:
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }


# #######################################################################
per_device_train_batch_size = 8
num_processes = 3
pad_to_max_length = False


# Only show the progress bar once on each machine.
# progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0

# Train
resume_from_checkpoint = None
for epoch in range(starting_epoch, num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / gradient_accumulation_steps

        loss.backward()
        # print(loss)
        if step % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            # progress_bar.update(1)
            completed_steps += 1
        if completed_steps >= max_train_steps:
            break

    model.eval()
    samples_seen = 0
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        # if not pad_to_max_length:  # necessary to pad predictions and labels for being gathered
        #     predictions = torch.nn.utils.rnn.pad_sequences(predictions, batch_first=True, pad_index=-100)
        #     labels = torch.nn.utils.rnn.pad_sequences(labels, batch_first=True, pad_index=-100)
        # predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
        # # If we are in a multiprocess environment, the last batch has duplicates
        # if accelerator.num_processes > 1:
        #     if step == len(eval_dataloader) - 1:
        #         predictions_gathered = predictions_gathered[: len(eval_dataloader.dataset) - samples_seen]
        #         labels_gathered = labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
        #     else:
        #         samples_seen += labels_gathered.shape[0]
        preds, refs = get_labels(predictions, labels)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )  # predictions and preferences are expected to be a nested list of labels, not label_ids

    eval_metric = compute_metrics()
    print(f"epoch {epoch}:", eval_metric)

# #######################################################################
# Test
print('Evaluating Test Set ...')

eval_dataset = processed_raw_datasets["test"]

batch_size = len(eval_dataset)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

model.eval()
with torch.no_grad():
    outputs = model(**batch)
predictions = outputs.logits.argmax(dim=-1)
labels = batch["labels"]
# if not pad_to_max_length:  # necessary to pad predictions and labels for being gathered
#     predictions = torch.nn.utils.rnn.pad_sequences(predictions, batch_first=True, pad_index=-100)
#     labels = torch.nn.utils.rnn.pad_sequences(labels, batch_first=True, pad_index=-100)
# predictions_gathered, labels_gathered = accelerator.gather((predictions, labels))
# # If we are in a multiprocess environment, the last batch has duplicates
# if accelerator.num_processes > 1:
#     if step == len(eval_dataloader) - 1:
#         predictions_gathered = predictions_gathered[: len(eval_dataloader.dataset) - samples_seen]
#         labels_gathered = labels_gathered[: len(eval_dataloader.dataset) - samples_seen]
#     else:
#         samples_seen += labels_gathered.shape[0]
preds, refs = get_labels(predictions, labels)
metric.add_batch(
    predictions=preds,
    references=refs,
)  # predictions and preferences are expected to be a nested list of labels, not label_ids

test_metric = compute_metrics()
print(f"Test Metrics:", test_metric)
# #######################################################################
print('Experiment completed.')
# #######################################################################

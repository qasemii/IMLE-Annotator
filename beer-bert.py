import json
import math
import numpy as np
from datasets import Dataset, DatasetDict

import torch
from torch import optim, Tensor
from torch.utils.data import DataLoader

import evaluate
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    DataCollatorWithPadding,
    get_scheduler,
)

from typing import Optional, Tuple, Callable

from imle.imle import imle
from bert_imle.modules import DifferentiableSelectKModel, Model
from bert_imle.solvers import mathias_select_k
from bert_imle.target import TargetDistribution
from bert_imle.noise import SumOfGammaNoiseDistribution

device = "cuda" if torch.cuda.is_available() else "cpu"

# arguments
aspect = 1
embedding_dims = 200
# maxlen = 350
batch_size = 128
kernel_size = 3
hidden_dims = 250
epochs = 20
select_k = 5
checkpoint_path = 'models/model.pt'

# data loading
# input_path_train = "data/BeerAdvocate/reviews.aspect" + str(aspect) + ".train.txt"
# input_path_validation = "data/BeerAdvocate/reviews.aspect" + str(aspect) + ".heldout.txt"

# # Preparing train data
# train_data = {'tokens': [], 'labels': []}
# with open(input_path_train) as fin:
#     for line in fin:
#         y, sep, text = line.partition("\t")
#         tokens = text.split(" ")
#         train_data['tokens'].append(tokens)
#         labels = [float(v) for v in y.split()]
#         train_data['labels'].append(labels[aspect])
#
# # Preparing train data
# validation_data = {'tokens': [], 'labels': []}
# with open(input_path_validation) as fin:
#     for line in fin:
#         y, sep, text = line.partition("\t")
#         tokens = text.split(" ")
#         validation_data['tokens'].append(tokens)
#         labels = [float(v) for v in y.split()]
#         validation_data['labels'].append(labels[aspect])
#
#
# # Prepare data as Dataset object
# train_dataset = Dataset.from_dict(train_data)
# validation_dataset = Dataset.from_dict(validation_data)
# test_valid_dataset = validation_dataset.train_test_split(test_size=0.5)
#
# # Creating Dataset object
# dataset = DatasetDict({
#     'train': train_dataset,
#     'validation': test_valid_dataset['train'],
#     'test': test_valid_dataset['test']})

data = {'tokens': [],
        'labels': []}

with open("data/BeerAdvocate/annotations.json") as fin:
    for line in fin:
        sample = json.loads(line)
        data['tokens'].append(sample['x'])
        data['labels'].append(sample['y'][aspect])

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

label2id = {i: i for i in range(len(label_list))}
id2label = {i: i for i in range(len(label_list))}

# Map that sends B-SELECTED label to its I-SELECTED counterpart
b_to_i_label = []
for idx, label in enumerate(label_list):
    if label.startswith("B-") and label.replace("B-", "I-") in label_list:
        b_to_i_label.append(label_list.index(label.replace("B-", "I-")))
    else:
        b_to_i_label.append(idx)

# Initializing the bert model
# model_name_or_path = 'bert-base-uncased'
# model_name_or_path = 'albert-base-v1'
model_name_or_path = "prajjwal1/bert-tiny"


config = AutoConfig.from_pretrained(
    model_name_or_path,
    output_hidden_states=True,
    num_labels=num_labels)

tokenizer = AutoTokenizer.from_pretrained(
    model_name_or_path,
    use_fast=True)

# #######################################################################

max_length = 350
padding = False
label_all_tokens = False


def tokenize(example):
    tokenized_inputs = tokenizer(
        example['tokens'],
        max_length=max_length,
        padding=padding,
        truncation=True,
        is_split_into_words=True,
    )
    tokenized_inputs["scores"] = example['labels']
    return tokenized_inputs


# tokenize and align the whole dataset
processed_raw_datasets = dataset.map(
    tokenize,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Running tokenizer on dataset",
)

train_dataset = processed_raw_datasets["train"]
eval_dataset = processed_raw_datasets["validation"]

# Data collator
use_fp16 = False
data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if use_fp16 else None)

# Data Loader
batch_size = 64
train_dataloader = DataLoader(train_dataset, shuffle=True, collate_fn=data_collator, batch_size=batch_size)
eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)

# #######################################################################
# Model

# initializing differentiable select_k model
blackbox_function = lambda logits: mathias_select_k(logits, select_k=select_k)

nb_samples = 1
imle_input_temp = 0.0
imle_output_temp = 10.0
imle_lambda = 1000.0
gradient_scaling = False

target_distribution = TargetDistribution(alpha=1.0, beta=imle_lambda, do_gradient_scaling=gradient_scaling)
noise_distribution = SumOfGammaNoiseDistribution(select_k=select_k, nb_iterations=10, device=device)


@imle(
    nb_samples=nb_samples,
    target_distribution=target_distribution,
    noise_distribution=noise_distribution,
    theta_noise_temperature=imle_input_temp,
    target_noise_temperature=imle_output_temp)
def imle_select_k(logits: Tensor) -> Tensor:
    return mathias_select_k(logits, select_k=select_k)


select_k_model = DifferentiableSelectKModel(imle_select_k, blackbox_function)

# Initializing model
model = Model(
    model_name_or_path=model_name_or_path,
    config=config,
    hidden_dims=hidden_dims,
    select_k=select_k,
    differentiable_select_k=select_k_model)

# #######################################################################
# Optimizer

weight_decay = 0.
learning_rate = 5e-5

# Split weights in two groups, one with weight decay and the other not.
no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
    {
        "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay
    }
]
optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=learning_rate)

# optimizer = optim.Adam(model.parameters(), lr=0.001, eps=1e-7)

# # Use the device given by the `accelerator` object.
# device = accelerator.device
# model.to(device)

# #######################################################################
# Scheduler and math around the number of training steps.
overrode_max_train_steps = False
gradient_accumulation_steps = 1
max_train_steps = None
num_train_epochs = 5
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

# #######################################################################
import numpy as np

# Metrics
metric = torch.nn.MSELoss()
mse_metric = evaluate.load('mse')

# #######################################################################
per_device_train_batch_size = 8
num_processes = 3
pad_to_max_length = False

# Only show the progress bar once on each machine.
# progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
completed_steps = 0
starting_epoch = 0

# Train
print('Training Process ...')
resume_from_checkpoint = None
for epoch in range(starting_epoch, num_train_epochs):
    model.train()
    for step, batch in enumerate(train_dataloader):
        scores = batch.data['scores']
        batch.data.pop('scores')
        # [B, ]
        outputs = model(**batch).squeeze()

        loss = metric(outputs, scores)
        loss = loss / gradient_accumulation_steps
        loss.backward()
        print(loss)

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
        scores = batch.data['scores']
        batch.data.pop('scores')
        with torch.no_grad():
            # [B, ]
            outputs = model(**batch).squeeze()

        mse_metric.add_batch(
            predictions=outputs,
            references=scores,
        )

    print(f"epoch {epoch}: {mse_metric.compute()}\n-----------------")

# #######################################################################
# # Test
# print('Evaluating Test Set ...')
# metric = evaluate.load('seqeval')
#
# eval_dataset = processed_raw_datasets["test"]
#
# batch_size = eval_dataset.__len__()
# eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=batch_size)
# # getting the whole dataset in one batch
# batch = next(iter(eval_dataloader))
#
# model.eval()
# with torch.no_grad():
#     outputs = model(**batch)
#
# pass
# #######################################################################
print('Experiment completed.')
# #######################################################################

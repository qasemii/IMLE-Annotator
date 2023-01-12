import torch
from datasets import Dataset, DatasetDict

import json
import transformers
import evaluate

from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    set_seed,
)
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
    'valid': test_valid['train'],
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
task_name = 'bi-ner'  # binary name entity classification
cache_dir = None
model_revision = 'main'
use_auth_token = False

config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label,
    finetuning_task=task_name,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None)

tokenizer = AutoTokenizer.from_pretrained(
    tokenizer_name_or_path,
    cache_dir=cache_dir,
    use_fast=True,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None)

model = AutoModelForTokenClassification.from_pretrained(
    model_name_or_path,
    from_tf=bool(".ckpt" in model_name_or_path),
    config=config,
    cache_dir=cache_dir,
    revision=model_revision,
    use_auth_token=True if use_auth_token else None)

# #######################################################################
padding = False
max_seq_length = None
label_all_tokens = False


# Tokenize all texts and align the labels with them.
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        padding=padding,
        truncation=True,
        max_length=max_seq_length,
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
train_dataset = dataset["train"]
eval_dataset = dataset["valid"]

use_fp16 = False
overwrite_cache = False

train_dataset = train_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=1,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on train dataset")

eval_dataset = eval_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=1,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on validation dataset")
# #######################################################################
# Data collator
data_collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8 if use_fp16 else None)

# Metrics
metric = evaluate.load('seqeval')

import numpy as np

return_entity_level_metrics = False


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


# #######################################################################
do_train = True
do_eval = True

argv = ['--output_dir', '/tmp/test-ner', '--do_train', '--do_eval', '--overwrite_output_dir', '--num_train_epochs',
        '1', '--report_to', 'none']

parser = HfArgumentParser((TrainingArguments,))
training_args, = parser.parse_args_into_dataclasses(args=argv)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset if do_train else None,
    eval_dataset=eval_dataset if do_eval else None,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics)

checkpoint = None

train_result = trainer.train(resume_from_checkpoint=checkpoint)

metrics = train_result.metrics
trainer.save_model()

metrics["train_samples"] = len(train_dataset)

trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
# #######################################################################
metrics = trainer.evaluate()

metrics["eval_samples"] = len(eval_dataset)

trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)
# #######################################################################
overwrite_cache = False

predict_dataset = dataset["test"]

predict_dataset = predict_dataset.map(
    tokenize_and_align_labels,
    batched=True,
    num_proc=1,
    load_from_cache_file=not overwrite_cache,
    desc="Running tokenizer on prediction dataset")

predictions, labels, metrics = trainer.predict(predict_dataset, metric_key_prefix="predict")

predicted_labels = torch.from_numpy(np.argmax(predictions, axis=2))
predicted_probs = torch.nn.Softmax(dim=2)(torch.from_numpy(predictions))[:, :, 1]

# Remove ignored index (special tokens)
# true_predictions = [
#     [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
#     for prediction, label in zip(predictions, labels)
# ]

trainer.log_metrics("predict", metrics)
trainer.save_metrics("predict", metrics)
# #######################################################################
print('Experiment completed.')
# #######################################################################

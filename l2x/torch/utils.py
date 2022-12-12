# -*- coding: utf-8 -*-

import json
import pickle

import numpy as np
import random

import matplotlib.pyplot as plt
import seaborn as sns

import torch
from torch import nn, Tensor
from torch.distributions.gamma import Gamma

from torch.distributions import Uniform

import math

from l2x.utils import pad_sequences

from snippets.explore_data import nltk_word_tokenize

from typing import Optional, Tuple, Callable

import logging

logger = logging.getLogger(__name__)


def set_seed(seed: int, is_deterministic: bool = True):
    # set the seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if is_deterministic is True:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return


def plot_stats(data, title='Plot'):
    plt.figure(figsize=(20, 8))
    sns.countplot(x=[token for token in data if token != '<PAD>'])
    plt.title(f'{title}', size=15)
    plt.show()


def subset_precision(model, aspect, id_to_word, word_to_id, select_k, device: torch.device, max_len: int = 350):
    data = []
    num_annotated_reviews = 0
    with open("data/annotations.json") as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
            num_annotated_reviews = num_annotated_reviews + 1

    highlights = []

    selected_word_counter = 0
    correct_selected_counter = 0

    for anotr in range(num_annotated_reviews):
        ranges = data[anotr][str(aspect)]  # the aspect id
        text_list = data[anotr]['x']
        review_length = len(text_list)

        list_test = []
        tokenid_list = [word_to_id.get(token, 0) for token in text_list]
        list_test.append(tokenid_list)

        # X_test_subset = np.asarray(list_test)
        # X_test_subset = sequence.pad_sequences(X_test_subset, maxlen=350)

        X_test_subset = pad_sequences(list_test, max_len=max_len)
        X_test_subset_t = torch.tensor(X_test_subset, dtype=torch.long, device=device)

        with torch.inference_mode():
            model.eval()
            prediction = model.z(X_test_subset_t)

        x_val_selected = prediction[0].cpu().numpy() * X_test_subset

        # [L,]
        selected_words = np.vectorize(id_to_word.get)(x_val_selected)[0][-review_length:]
        selected_nonpadding_word_counter = 0
        for i, w in enumerate(selected_words):
            if w != '<PAD>':  # we are nice to the L2X approach by only considering selected non-pad tokens
                selected_nonpadding_word_counter = selected_nonpadding_word_counter + 1
                for r in ranges:
                    if i in range(r[0], r[1]):
                        correct_selected_counter = correct_selected_counter + 1

                        # highlight the correct selected tokens
                        text_list[i] = '}\hlc[purple!30]{' + text_list[i] + \
                                       '}\hlc[cyan!30]{'
                        selected_words[i] = '<PAD>'
                        break

        for i, w in enumerate(selected_words):
            if w != '<PAD>':
                # highlight the wrong selected tokens
                text_list[i] = '\hlc[red!60]{' + text_list[i] + '}'

        for r in ranges:
            # highlight the ground truth tokens
            text_list[r[0]] = '\hlc[cyan!30]{' + text_list[r[0]]
            text_list[r[1] - 1] = text_list[r[1] - 1] + '}'

        highlights.append(' '.join(text_list) + '\\\\')

        # we make sure that we select at least 10 non-padding words
        # if we have more than select_k non-padding words selected, we allow it but count that in
        selected_word_counter = selected_word_counter + max(selected_nonpadding_word_counter, select_k)

    with open("highlights.txt", "w") as f:
        f.write('\n\n'.join(highlights))

    return correct_selected_counter / selected_word_counter


def subset_precision_esnli(model, data, id_to_word, word_to_id, select_k, device: torch.device, max_len: int = 350):
    # tokenize using nltk word tokenizer
    tokenized_sentence = data['sentence']['merged']

    marked_highlights_list = []
    entailment_dist, contradiction_dist, neutral_dist = [], [], []

    label_to_id = {'entailment': 2, 'neutral': 1, 'contradiction': 0}
    id_to_label = {value: key for key, value in label_to_id.items()}

    selected_word_counter, correct_selected_counter = 0, 0
    for anotr in range(len(tokenized_sentence)):
        text_list = tokenized_sentence[anotr]
        review_length = len(text_list)

        list_test = []
        tokenid_list = [word_to_id.get(token, 0) for token in text_list]
        list_test.append(tokenid_list)

        X_test_subset = pad_sequences(list_test, max_len=max_len)
        X_test_subset_t = torch.tensor(X_test_subset, dtype=torch.long, device=device)

        with torch.inference_mode():
            model.eval()
            prediction = model.z(X_test_subset_t)

            label_score = model(X_test_subset_t)
            predicted_idx = torch.argmax(label_score, dim=1).tolist()
            predicted_label = [id_to_label[i] for i in predicted_idx]

        x_val_selected = prediction[0].cpu().numpy() * X_test_subset
        # [L,]
        selected_words = np.vectorize(id_to_word.get)(x_val_selected)[0][-review_length:]
        selected_nonpadding_word = []
        selected_nonpadding_word_counter = 0

        # premise_highlights = data['highlight']['premise'][anotr]
        # hypothesis_highlights = data['highlight']['hypothesis'][anotr]

        highlights_idx = data['highlight']['merged'][anotr]
        for i, w in enumerate(selected_words):
            if w != '<PAD>':  # we are nice to the L2X approach by only considering selected non-pad tokens
                selected_nonpadding_word.append(w)
                selected_nonpadding_word_counter = selected_nonpadding_word_counter + 1
                if i in highlights_idx:
                    # correct selected words
                    correct_selected_counter = correct_selected_counter + 1
                    text_list[i] = '\hlc[purple!30]{' + text_list[i] + '}'
                    highlights_idx.remove(i)
                else:
                    # wrong selected words
                    text_list[i] = '\hlc[red!60]{' + text_list[i] + '}'
                # exclude explored words
                selected_words[i] = '<PAD>'

        for i in highlights_idx:
            # highlight the wrong selected tokens
            text_list[i] = '\hlc[cyan!30]{' + text_list[i] + '}'
            selected_words[i] = '<PAD>'

        # check if the predicted label is true or not
        label = data['label'][anotr]
        if label == predicted_label[0]:
            label = f'\\textbf{ {label}} \\cmark\\\\'
        else:
            label = f'\\textbf{ {label}} \\xmark\\\\'

        if label == 'entailment':
            entailment_dist += selected_nonpadding_word
        elif label == 'neutral':
            neutral_dist += selected_nonpadding_word
        else:
            contradiction_dist += selected_nonpadding_word

        marked_highlights_list.append(' '.join(text_list) + label)

        # we make sure that we select at least 10 non-padding words
        # if we have more than select_k non-padding words selected, we allow it but count that in
        selected_word_counter = selected_word_counter + max(selected_nonpadding_word_counter, select_k)

    with open("highlights.txt", "w") as f:
        f.write('\n\n'.join(marked_highlights_list))

    # plot_stats(entailment_dist, 'Entailment')

    stats = {'entailment': entailment_dist, 'neutral': neutral_dist, 'contradiction': contradiction_dist}
    with open('statistics.pkl', 'wb') as file:
        pickle.dump(stats, file)

    return correct_selected_counter / selected_word_counter


if __name__ == '__main__':
    pass


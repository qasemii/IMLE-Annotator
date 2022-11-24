import csv
import nltk
from nltk.tokenize import word_tokenize


# from utils.preprocess_eSNLI import csv_to_txt


def get_data(file_dir):
    file = open(file_dir)
    rows = csv.DictReader(file)

    premise, hypothesis, sentence, label, premise_highlight_idx, hypothesis_highlight_idx, highlight = [], [], [], [], [], [], []
    for row in rows:
        s1, s2 = row['Sentence1'], row['Sentence2']
        sentence_merged = s1 + ' ' + s2

        premise_marked, hypothesis_marked = row['Sentence1_marked_1'], row['Sentence2_marked_1']
        sentence_marked = premise_marked + premise_marked

        lbl = row['gold_label']

        s1_highlight, s2_highlight = [], []
        # for i, s in enumerate(premise_marked.split()):
        #     if len(s.split('*')) != 1:
        #         s1_highlight.append(i)
        #
        # for i, s in enumerate(hypothesis_marked.split()):
        #     if len(s.split('*')) != 1:
        #         s2_highlight.append(i)

        for i, s in enumerate(premise_marked.split()):
            if len(s.split('*')) != 1:
                s1_highlight.append(i)
                temp = s.split('*')[1]
                if len(temp.split(',')) != 1:
                    i = i + 1

        for i, s in enumerate(hypothesis_marked.split()):
            if len(s.split('*')) != 1:
                s2_highlight.append(i)
                temp = s.split('*')[1]
                if len(temp.split(',')) != 1:
                    i = i + 1

        premise.append(s1)
        hypothesis.append(s2)
        sentence.append(sentence_merged)

        label.append(lbl)

        premise_highlight_idx.append(s1_highlight)
        hypothesis_highlight_idx.append(s2_highlight)

        h = s1_highlight + [h + len(word_tokenize(s1)) for h in s2_highlight]
        highlight.append(h)

    return {'sentence': {'merged': sentence,
                         'premise': premise,
                         'hypothesis': hypothesis},
            'labels': label,
            'highlight': {'merged': highlight,
                          'premise': premise_highlight_idx,
                          'hypothesis': hypothesis_highlight_idx}}


def nltk_word_tokenize(input_list):
    # tokenize using nltk word tokenizer
    tokenized_sentence = []
    for i in range(len(input_list)):
        tokenized_sentence.append(word_tokenize(input_list[i]))
    return tokenized_sentence


# # get data dictionary
# TRAIN_INPUT_PATH = '../data/esnli_test.csv'
# train_data = get_data(TRAIN_INPUT_PATH)
# print('Done')

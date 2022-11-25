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

        s1_tokens = word_tokenize(premise_marked)
        s2_tokens = word_tokenize(hypothesis_marked)

        # getting highlights for premise
        j = 0
        h_detected = False
        for i, w in enumerate(s1_tokens):
            if w == '*' and not h_detected:
                s1_highlight.append(j)
                h_detected = True
            elif w == '*' and h_detected:
                h_detected = False
            else:
                j = j + 1

        # getting highlights for hypothesise
        j = 0
        h_detected = False
        for i, w in enumerate(s2_tokens):
            if w == '*' and not h_detected:
                s2_highlight.append(j)
                h_detected = True
            elif w == '*' and h_detected:
                h_detected = False
            else:
                j = j + 1

        premise.append(s1)
        hypothesis.append(s2)
        sentence.append(sentence_merged)

        label.append(lbl)

        premise_highlight_idx.append(s1_highlight)
        hypothesis_highlight_idx.append(s2_highlight)

        x = s1_highlight + [h + len(word_tokenize(s1)) for h in s2_highlight]
        # temp = word_tokenize(sentence_merged)
        # for i in x:
        #     ttt = temp[i]
        highlight.append(x)

    return {'sentence': {'merged': sentence,
                         'premise': premise,
                         'hypothesis': hypothesis},
            'label': label,
            'highlight': {'merged': highlight,
                          'premise': premise_highlight_idx,
                          'hypothesis': hypothesis_highlight_idx}}


def nltk_word_tokenize(input_list):
    # tokenize using nltk word tokenizer
    tokenized_sentence = []
    for i in range(len(input_list)):
        tokenized_sentence.append(word_tokenize(input_list[i]))
    return tokenized_sentence


if __name__ == '__main__':
    TEST_INPUT_PATH = '../data/esnli_test.csv'
    get_data(TEST_INPUT_PATH)
    print('Done')


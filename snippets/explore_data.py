import csv
import pickle
from nltk.tokenize import word_tokenize


# from utils.preprocess_eSNLI import csv_to_txt


def get_data(file_dir):
    file = open(file_dir)
    rows = csv.DictReader(file)

    premise, hypothesis, sentence, explanation, label, premise_highlight_idx, hypothesis_highlight_idx, highlight = [], [], [], [], [], [], [], []
    for row in rows:
        s1, s2 = row['Sentence1'], row['Sentence2']
        sentence_merged = s1 + ' ' + s2
        exp = row['Explanation_1']

        premise_marked, hypothesis_marked = row['Sentence1_marked_1'], row['Sentence2_marked_1']
        sentence_marked = premise_marked + ' ' + hypothesis_marked

        lbl = row['gold_label']

        s1_highlight, s2_highlight, s_highlight = [], [], []

        s1_tokens = word_tokenize(premise_marked)
        s2_tokens = word_tokenize(hypothesis_marked)
        s_tokens = word_tokenize(sentence_marked)

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

        # getting highlights for hypothesis
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

        # getting highlights for merged sentence
        j = 0
        h_detected = False
        for i, w in enumerate(s_tokens):
            if w == '*' and not h_detected:
                s_highlight.append(j)
                h_detected = True
            elif w == '*' and h_detected:
                h_detected = False
            else:
                j = j + 1

        s1_tokenized = word_tokenize(s1)
        s2_tokenized = word_tokenize(s2)
        s_tokenized = s1_tokenized + s2_tokenized
        exp_tokenized = s_tokenized + word_tokenize(exp)

        premise.append(s1_tokenized)
        hypothesis.append(s2_tokenized)
        sentence.append(s_tokenized)
        explanation.append(exp_tokenized)

        label.append(lbl)

        premise_highlight_idx.append(s1_highlight)
        hypothesis_highlight_idx.append(s2_highlight)
        highlight.append(s_highlight)

    return {'sentence': {'merged': sentence,
                         'premise': premise,
                         'hypothesis': hypothesis,
                         'explanation': explanation},
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
    TRAIN_INPUT_PATH = '../data/eSNLI/esnli_train_1.csv'
    train_data_1 = get_data(TRAIN_INPUT_PATH)
    # with open('../data/eSNLI/esnli_train_1_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(train_data_1, file)
    #
    # TRAIN_INPUT_PATH = '../data/eSNLI/esnli_train_2.csv'
    # train_data_2 = get_data(TRAIN_INPUT_PATH)
    # with open('../data/eSNLI/esnli_train_2_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(train_data_2, file)
    #
    # train_data = {'sentence': {'merged': train_data_1['sentence']['merged'] + train_data_2['sentence']['merged'],
    #                            'premise': train_data_1['sentence']['premise'] + train_data_2['sentence']['premise'],
    #                            'hypothesis': train_data_1['sentence']['hypothesis'] + train_data_2['sentence']['hypothesis']},
    #               'label': train_data_1['label'] + train_data_2['label'],
    #               'highlight': {'merged': train_data_1['highlight']['merged'] + train_data_2['highlight']['merged'],
    #                             'premise': train_data_1['highlight']['premise'] + train_data_2['highlight']['premise'],
    #                             'hypothesis': train_data_1['highlight']['hypothesis'] + train_data_2['highlight']['hypothesis']}}
    #
    # with open('../data/eSNLI/esnli_train_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(train_data, file)
    #
    # VALIDATION_INPUT_PATH = '../data/eSNLI/esnli_dev.csv'
    # val_data = get_data(VALIDATION_INPUT_PATH)
    # with open('../data/eSNLI/esnli_val_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(val_data, file)
    #
    # TEST_INPUT_PATH = '../data/eSNLI/esnli_test.csv'
    # test_data = get_data(TEST_INPUT_PATH)
    # with open('../data/eSNLI/esnli_test_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(test_data, file)
    #
    # print('Done')

    # ############################ read the generated files
    # TRAIN_INPUT_PATH = '../data/eSNLI/esnli_train_preprocessed.pkl'
    # with open(TRAIN_INPUT_PATH, 'rb') as file:
    #     train_data = pickle.load(file)
    #
    # print('Done')

    pass

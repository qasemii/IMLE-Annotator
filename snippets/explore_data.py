import csv
import pickle
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
        sentence_marked = premise_marked + ' ' + hypothesis_marked

        lbl = row['gold_label']

        s1_highlight, s2_highlight, s_highlight = [], [], []

        # s1_tokens = word_tokenize(premise_marked)
        # s2_tokens = word_tokenize(hypothesis_marked)
        s_tokens = word_tokenize(sentence_marked)

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

        premise.append(s1)
        hypothesis.append(s2)
        sentence.append(sentence_merged)

        label.append(lbl)

        premise_highlight_idx.append(s1_highlight)
        hypothesis_highlight_idx.append(s2_highlight)
        highlight.append(s_highlight)

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
    # TRAIN_INPUT_PATH = '../data/esnli_train_1.csv'
    # train_data = get_data(TRAIN_INPUT_PATH)
    # train_tokenized = nltk_word_tokenize(train_data['sentence']['merged'])
    # train_data['sentence']['merged'] = train_tokenized
    #
    # with open('../data/eSNLI/esnli_train_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(train_data, file)
    #
    # VALIDATION_INPUT_PATH = '../data/esnli_dev.csv'
    # val_data = get_data(VALIDATION_INPUT_PATH)
    # val_tokenized = nltk_word_tokenize(val_data['sentence']['merged'])
    # val_data['sentence']['merged'] = val_tokenized
    # with open('../data/eSNLI/esnli_val_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(val_data, file)
    #
    # TEST_INPUT_PATH = '../data/esnli_test.csv'
    # test_data = get_data(TEST_INPUT_PATH)
    # test_tokenized = nltk_word_tokenize(test_data['sentence']['merged'])
    # test_data['sentence']['merged'] = test_tokenized
    # with open('../data/eSNLI/esnli_test_preprocessed.pkl', 'wb') as file:
    #     pickle.dump(test_data, file)

    PATH = '../data/eSNLI/esnli_test_preprocessed.pkl'
    with open(PATH, 'rb') as file:
        stats = pickle.load(file)

    pass
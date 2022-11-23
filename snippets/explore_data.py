import csv
from nltk.tokenize import word_tokenize
# from utils.preprocess_eSNLI import csv_to_txt


file_dir = "esnli_train_1.csv"


def get_data(file_dir):
    global highlights_idx
    file = open(file_dir)
    rows = csv.DictReader(file)

    sentences, labels, highlights, highlights_idx = [], [], [], []
    for row in rows:
        premise, hypothesis = row['Sentence1'], row['Sentence2']
        sentence = premise + ' ' + hypothesis
        label = row['gold_label']

        highlighted_premise, highlighted_hypothesis = row['Sentence1_marked_1'], row['Sentence2_marked_1']
        l1, l2 = len(highlighted_premise.split('*')) // 2, len(highlighted_hypothesis.split('*')) // 2

        premise_highlights = [highlighted_premise.split('*')[2 * i + 1] for i in range(l1)]
        hypothesis_highlights = [highlighted_hypothesis.split('*')[2 * i + 1] for i in range(l2)]
        highlight = premise_highlights + hypothesis_highlights
        # we remove punctuations from highlights - otherwise subset precision would be wrong
        for i, h in enumerate(highlight):
            if len(h.split()) != 1:
                highlight = h.split()[0]

        premise_highlights = highlighted_premise.split()
        hypothesis_highlights = highlighted_hypothesis.split()
        sentence_marked = premise_highlights + hypothesis_highlights
        highlight_idx = []
        for i, s in enumerate(sentence_marked):
            temp = s.split('*')
            if len(s.split('*')) != 1:
                highlight_idx.append(i)

        sentences.append(sentence)
        labels.append(label)
        highlights.append(highlight)
        highlights_idx.append(highlight_idx)

    return {'sentences': sentences, 'labels': labels, 'highlights': highlights, 'highlights_idx': highlights_idx}


def nltk_word_tokenize(input_list):
    # tokenize using nltk word tokenizer
    tokenized_sentence = []
    for i in range(len(input_list)):
        tokenized_sentence.append(word_tokenize(input_list[i]))
    return tokenized_sentence

# # get data dictionary
# TRAIN_INPUT_PATH = '../data/esnli_train_1.csv'
# train_data = get_data(TRAIN_INPUT_PATH)
# print('Done')
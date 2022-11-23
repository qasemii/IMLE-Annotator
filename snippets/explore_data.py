import csv
from nltk.tokenize import word_tokenize
# from utils.preprocess_eSNLI import csv_to_txt


file_dir = "esnli_train_1.csv"


def get_data(file_dir):
    file = open(file_dir)
    rows = csv.DictReader(file)

    sentences, labels, highlights = [], [], []
    for row in rows:
        premise, hypothesis = row['Sentence1'], row['Sentence2']
        sentence = premise + ' ' + hypothesis
        label = row['gold_label']

        highlighted_premise, highlighted_hypothesis = row['Sentence1_marked_1'], row['Sentence2_marked_1']
        l1, l2 = len(highlighted_premise.split('*')) // 2, len(highlighted_hypothesis.split('*')) // 2

        premise_highlights = [highlighted_premise.split('*')[2 * i + 1] for i in range(l1)]
        hypothesis_highlights = [highlighted_hypothesis.split('*')[2 * i + 1] for i in range(l2)]
        highlight = premise_highlights + hypothesis_highlights

        sentences.append(sentence)
        labels.append(label)
        highlights.append(highlight)

    return {'sentences': sentences, 'labels': labels, 'highlights': highlights}


def nltk_word_tokenize(input_list):
    # tokenize using nltk word tokenizer
    tokenized_sentence = []
    for i in range(len(input_list)):
        tokenized_sentence.append(word_tokenize(input_list[i]))
    return tokenized_sentence

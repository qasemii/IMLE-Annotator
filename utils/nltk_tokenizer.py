import nltk
from nltk import word_tokenize
from snippets.explore_data import get_data

# nltk.download('punkt')

data_path = '../data/eSNLI/esnli_train_1.csv'
data = get_data(data_path)

sample = data['sentences'][23]
print(sample)
print(word_tokenize(sample))

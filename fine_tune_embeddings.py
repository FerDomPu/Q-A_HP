from flair.data import Corpus
from flair.datasets import ColumnCorpus
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

# 1. get the corpus
# define columns
columns = {0: 'text', 1: 'pos', 3: 'ner'}

# this is the folder in which train, test and dev files reside
data_folder = 'C:/Users/User/Desktop/HarryPotter-Q&A/data'

# init a corpus using column format, data folder and the names of the train, dev and test files
corpus: Corpus = ColumnCorpus(data_folder, columns,
                              train_file='conllpp_train.txt',
                              test_file='conllpp_test.txt',
                              dev_file='conllpp_dev.txt')

# 2. what label do we want to predict?
label_type = 'ner'

# 3. make the label dictionary from the corpus
label_dict = corpus.make_label_dictionary(label_type=label_type, add_unk=False)
print(label_dict)

# 4. initialize embedding stack with Flair and GloVe
embedding_types = [
    WordEmbeddings('glove'),
    FlairEmbeddings('news-forward'),
    FlairEmbeddings('news-backward'),
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

# 5. initialize sequence tagger
tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type)

# 6. initialize trainer
trainer = ModelTrainer(tagger, corpus)

# 7. start training
trainer.train('resources/taggers/conllpp_harry',
              learning_rate=0.1,
              mini_batch_size=32,
              max_epochs=10)

# load the model you trained
from flair.models import SequenceTagger
model = SequenceTagger.load("resources/taggers/conllpp_harry/best-model.pt")

# create example sentence
from flair.data import Sentence
sentence = Sentence("What is the species of Rubeus Hagrid?")

# predict tags and print
model.predict(sentence)
sentence

import os
import pandas as pd
import json
from torch.utils.data import Dataset
import torch
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
import string
from vocab import Vocabulary
import matplotlib.pyplot as plt


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


# class LyricsDataset(Dataset):

#     def __init__(self, root_dir):
#         self.files = [f for f in os.listdir(root_dir) if f.endswith('.json')]
#         self.root_dir = root_dir
#     def __len__(self):
#         return len(self.files)
    
#     def process_lyrics(self, lyrics):
#         lemmatizer = WordNetLemmatizer()
#         stemmer = PorterStemmer()
#         STOPWORDS = set(stopwords.words('english'))
#         PUNCTUATION = set(string.punctuation)
#         tokens = word_tokenize(lyrics)
#         tokens = [a_word.lower() for a_word in tokens]
#         tokens = [a_word for a_word in tokens if a_word.isalpha()]
#         tokens = [a_word for a_word in tokens if a_word not in STOPWORDS]
#         tokens = [a_word for a_word in tokens if a_word not in PUNCTUATION]
#         tokens = [lemmatizer.lemmatize(a_word) for a_word in tokens]
#         tokens = [stemmer.stem(a_word) for a_word in tokens]

#         processed_lyrics = " ".join(tokens)
#         return processed_lyrics

#     def __getitem__(self, idx):
#         filename = self.files[idx]
#         artist, title, mood = filename.split('_', 2)
#         mood = mood.replace('.json', '')
#         file_path = os.path.join(self.root_dir, filename)
#         with open(file_path, 'r', encoding='utf-8') as file:
#             data = json.load(file)
#             lyrics = data['lyrics']
#             processed_lyrics = self.process_lyrics(lyrics)

#         sample = {'Artist': artist, 'Title': title, 'Mood': mood, 'Lyrics': processed_lyrics}
#         return sample
    
#     def count_moods(self):
#         mood_counts = {}
#         for idx in range(len(self.files)):
#             sample = self[idx]
#             mood = sample['Mood']
#             if mood in mood_counts:
#                 mood_counts[mood] += 1
#             else:
#                 mood_counts[mood] = 1
#         return mood_counts


class LyricsDataset(Dataset):

    def __init__(self, data, vocab, max_length):
        self.data = data
        self.vocab = vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        # convert caption (string) to word ids.
        tokens = sample["Lyrics"].split()
        lyric = []

        # build the Tensor version of the caption, with token words
        lyric.extend([self.vocab(token) for token in tokens])
        # # pad the sequence
        # if len(lyric) < self.max_length:
        #     lyric.extend([self.vocab('<pad>')] * (self.max_length - len(lyric)))
        # else:
        #     lyric = lyric[:self.max_length]
        
        lyric = torch.Tensor(lyric)
        mood = torch.Tensor(sample["Mood"]).float()
        return lyric.int(), mood

        
# dataset = LyricsDataset(root_dir='/Users/zhangmingyin/Desktop/lyric/ml_lyrics')
# a, b, c= dataset[0]
# print(a, b)
# print(c)
# print(len(dataset))

def process_lyrics(lyrics):
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    STOPWORDS = set(stopwords.words('english'))
    PUNCTUATION = set(string.punctuation)
    tokens = word_tokenize(lyrics)
    tokens = [a_word.lower() for a_word in tokens]
    tokens = [a_word for a_word in tokens if a_word.isalpha()]
    tokens = [a_word for a_word in tokens if a_word not in STOPWORDS]
    tokens = [a_word for a_word in tokens if a_word not in PUNCTUATION]
    tokens = [lemmatizer.lemmatize(a_word) for a_word in tokens]
    tokens = [stemmer.stem(a_word) for a_word in tokens]

    processed_lyrics = " ".join(tokens)
    return processed_lyrics

def build_lyrics_df(root_dir):
    lyric_data = []
    #classes ={"happy":[1, 0, 0, 0], "angry": [0, 1, 0, 0], "sad": [0, 0, 1, 0], "relaxed":[0, 0, 0, 1]}
    
    for file in os.listdir(root_dir):
        if file.endswith('.json'):
            artist, title, mood = file.split('_', 2)
            mood = mood.replace('.json',"")
            #mood = classes.get(mood)
            file_path = os.path.join(root_dir, file)
            with open(file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                lyrics = data['lyrics']
                processed_lyrics = process_lyrics(lyrics)
            sample = {'Artist': artist, 'Title': title, 'Mood': mood, 'Lyrics': processed_lyrics}
            lyric_data.append(sample)
    
    return lyric_data

root_dir = '/Users/zhangmingyin/Desktop/lyric/ml_lyrics'
data = build_lyrics_df(root_dir)
df = pd.DataFrame(data)

total_count = len(df)
print(f"Total number of samples: {total_count}")

mood_counts = df['Mood'].value_counts()
print("Counts of each type:")
print(mood_counts)

mood_counts.plot.pie(autopct='%.2f%%', startangle=90, counterclock=False)
plt.title('Distribution of Moods in the Dataset')
plt.ylabel('')
plt.show()
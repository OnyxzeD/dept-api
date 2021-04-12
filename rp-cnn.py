import collections
import re
import string
import unicodedata

import nltk
import pandas as pd
from keras import layers
from keras.backend import clear_session
from keras.layers import (
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    Flatten,
    GlobalMaxPooling1D,
    Input,
    Reshape,
    concatenate,
)
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split


# function to removing extra whitespaces and tabs
def remove_extra_whitespace_tabs(text):
    # pattern = r'^\s+$|\s+$'
    pattern = r"^\s*|\s\s*"
    return re.sub(pattern, " ", text).strip()


# function to remove punctuation
def remove_punctuation(text):
    text = "".join([c for c in text if c not in string.punctuation])
    return text


# function to remove numbers
def remove_numbers(text):
    # define the pattern to keep
    pattern = r"[^a-zA-z.,!?/:;\"\'\s]"
    return re.sub(pattern, "", text)


# function to remove special characters
def remove_special_characters(text):
    # define the pattern to keep
    pat = r"[^a-zA-z0-9.,!?/:;\"\'\s]"
    return re.sub(pat, "", text)


# function to remove accented characters
def remove_accented_chars(text):
    new_text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )
    return new_text


# function to remove mention
def remove_mention(text):
    words = text.split(" ")
    # initializing char list
    char_list = ["@", "/rlt/", "https://", "pic.twitter.com"]
    # Remove words containing list characters
    res = [ele for ele in words if all(ch not in ele for ch in char_list)]
    # replace \n (newline)
    text2 = " ".join(res)
    clean = text2.replace("\n", " ")
    return clean


# function to remove Emoji
def deEmojify(text):
    return text.encode("ascii", "ignore").decode("ascii")


def cleaning(text):
    de = deEmojify(text)
    lc = de.lower()
    rm = remove_mention(lc)
    rewt = remove_extra_whitespace_tabs(rm)
    rp = remove_punctuation(rewt)
    rn = remove_numbers(rp)
    rsc = remove_special_characters(rn)
    rac = remove_accented_chars(rsc)
    return rac


filepath_dict = {
    "tweety65": "tweety65.txt",
}

df_list = []
for source, filepath in filepath_dict.items():
    df = pd.read_csv(filepath, names=["sentence", "label"], sep="\t")
    df["source"] = source  # Add another column filled with the source name
    df_list.append(df)

df = pd.concat(df_list)

# pick dataset
df_source = df[df["source"] == "tweety65"]
sentences = df_source["sentence"].values
sentences = [cleaning(s) for s in sentences]

# remove stopwords
def removeStopWords(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)


filtered_words = [removeStopWords(sen) for sen in sentences]
sentences = ["".join(sen) for sen in filtered_words]

y = df_source["label"].values
sentences_train, sentences_test, y_train, y_test = train_test_split(
    sentences, y, test_size=0.25, random_state=1000
)

# count dataset words
num_words = [len(sentence.split()) for sentence in sentences]
maxsentences = max(sentences, key=len)
# maxlen = len(max(sentences, key=len).split())
maxlen = 300
# print("dataset words length is : %s" % sum(num_words))
# print("Longest sentence is : %s" % maxsentences)
# print("Longest word is : %s" % maxlen)

print("Training data : ")
print("Max sentence length is : %s" % len(max(sentences_train, key=len).split()))
print("Tes data : ")
print("Max sentence length is : %s" % len(max(sentences_test, key=len).split()))

# Word Embeddings
tokenizer = Tokenizer(num_words=20000)
tokenizer.fit_on_texts(sentences_train)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1

X_train = tokenizer.texts_to_sequences(sentences_train)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

embedding_dim = 300
X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)


# create embedding matrix
import numpy as np


def create_embedding_matrix(filepath, word_index, embedding_dim):
    vocab_size = len(word_index) + 1  # Adding again 1 because of reserved 0 index
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, *vector = line.split()
            if word in word_index:
                idx = word_index[word]
                embedding_matrix[idx] = np.array(vector, dtype=np.float32)[
                    :embedding_dim
                ]

    return embedding_matrix


embedding_matrix = create_embedding_matrix(
    "w2vec_wiki_id_300.txt", tokenizer.word_index, embedding_dim
)
nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
# print(nonzero_elements / vocab_size)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV

# CNN with w2vec
cnn_model = Sequential()
cnn_model.add(
    layers.Embedding(
        vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen
    )
)
cnn_model.add(layers.Conv1D(200, 5, activation="relu"))
cnn_model.add(layers.GlobalMaxPooling1D())
cnn_model.add(layers.Dense(10, activation="relu"))
cnn_model.add(layers.Dense(1, activation="sigmoid"))
cnn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
cnn_model.summary()

history = cnn_model.fit(
    X_train,
    y_train,
    epochs=100,
    verbose=False,
    validation_data=(X_test, y_test),
    batch_size=10,
)
loss, accuracy = cnn_model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = cnn_model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

# save cnn_model
cnn_model.save("my_cnn_model")  # creates a HDF5 file 'my_model.h5'
# del cnn_model  # deletes the existing model


# new_data = [
# "@jisxchu Apa salahku apa salah ibuku hidupku dirundung pilu.....",
# "@taeviIc Leh opo salahku su",
# "Semua ini bukan salahku."
# "@ROSExGBK @hemanisnya Ap salahku azriel",
# "@zjaehqyun Apa salahku bebi",
# "Gusti salahku apa sih, dia berubah banget, sampe gk kenal aku 😭",
# "Bila ku tau Apa salahku Hingga kau pergi, tinggalkan aku Apa salahku  Hingga kau pergi, tinggalkan aku Yang mencintaimuu",
# "@chawrelia Aku jg pernah dibonceng di motor, tiba2 ada motor mepet. Salahku karena dibonceng? Mantanku cm diem dan bilang dia takut kalau ngelawan. Putusin langsung",
# "@miracle__13 ora popo hahaha salahku lek kui, meh tak hapus yo raiso sayangnya",
# "@nadiasuroyo @rahamida @WanandaRai Apa sii salahku.. wkwkwk",
# "Satu lagi nih pelaku UKM Indonesia yg berhasil menembus pasar mancanegara adalah PT Indoto Tirta Mulia yg sukses mencatatkan ekspor perdana produk. Ini membuktikan bahwa pandemi tidak menyurutkan langkah UKM untuk terus berkarya dan berprestasi.\n@Kemendag\nhttps://t.co/gbQdQ6A277"
# ]

# new_data = [
# "Dunia gak perlu tau soal kamu siapa. kamu kenapa. Tidak akan ada banyak orang yang peduli. Sebagian hanya sebatas penasaran dan kemudian memilih meninggalkan.",
# "Seperti kebanyakan pemuda yg beranjak usia melewati 25 tahun. Diriku berada di ambang antara keputusan,seperti apa kelanjutannya dimasa depan. Merangkai terlalu banyak angan. Sampai tidak satupun dapat tergapai. Mulai berkeinginan untuk putus asa, namun menahan diri sekuat tenaga",
# "Hari hari itu pun aku pertama kalinya aku merasa kesepian lagi. Gelisah sana sini tak beraturan. Mungkin ungkapanmu tsb terbukti adanya",
# "Aku tipe orang yang gasuka cerita masalah ke orang”. Tapi klo udah gakuat malah jadi depresi/stress aku pasti cerita kok ke orang tertentu",
# "Idup gue dah kesepian, temen dikit,  temen itu itu aja, malah ssekarang ada lu covid corona. Tambah gak punya temen akunya 😭😭😭",
# "Kadang sering banget gua ngomong sm orang tua yang enggak-enggak saat gua depresi, setelah gua balik sadar lagi gua nyesel karena udah bikin orang tua gua cemas ke gua. Padahal Ini cuma perkara kalian yang nakut-nakutin gua kalo semua yang gua takutin bakal kejadian bangsat.",
# "Lagi Depresi.Tiba2 ketawa, tiba2 mau nangis",
# "Bersedih adalah suatu hal yang wajar,tapi jangan sampe kesedihan itu melemahkan hati kita dan bikin kita jadi putus asa. #BOT",
# "Putus asa bukanlah cara yang tepat dalam perjalanan anda menuju cita-cita",
# "Apakah kamu pernah ngerasain kesepian yang kesepian banget sampe ngerasa sendiri dan pengen pergi aja dari bumi?"
# ]

# new_test = [cleaning(d) for d in new_data]
# f_words = [removeStopWords(n) for n in new_test]
# new_test = [''.join(n) for n in f_words]
# new_test = tokenizer.texts_to_sequences(new_test)
# new_test = pad_sequences(new_test, padding='post', maxlen=maxlen)

# ynew = model.predict(new_test)
# for x in ynew:
# if x > 0.6:
# print("Data Depressive")
# else:
# print("Data Non Depressive")

# print(ynew)
# print(new_test)

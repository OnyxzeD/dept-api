import nltk
import twint
import os
from app import db, response
from app.models import Detection, History, Users, Tweets
from flask import request


import pandas as pd
from pandas import DataFrame
import collections
import re
import string
import unicodedata
import random

import numpy as np

np.random.seed(90)
import tensorflow as tf

tf.random.set_seed(96)

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
from keras.regularizers import l2
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split

import matplotlib.pyplot as plt

plt.style.use("ggplot")

embedding_matrix = np.zeros(shape=(3820, 300))


def index():
    try:
        detection = (
            Detection.query.join(Users).order_by(Detection.created_at.desc()).all()
        )
        result = transform(detection)

        return response.ok(result, "Detection Lists")

    except Exception as e:
        return response.badRequest([], str(e))


def show(detectionId):
    try:
        detection = Detection.query.filter_by(id=detectionId).first()
        if not detection:
            return response.badRequest([], "Data not found")

        data = singleTransform(detection)

        detail = detection.details.all()
        array = []
        for d in detail:
            array.append(
                {
                    "raw_tweet": d.raw_tweet,
                    "processed_tweet": d.processed_tweet,
                    "timestamp": d.timestamp.strftime("%Y-%m-%d %H:%M"),
                    "depressive": d.depressive,
                }
            )
        data["details"] = array

        return response.ok(data, "Detail")
    except Exception as e:
        return response.badRequest([], str(e))


def byUser(user_id):
    try:
        detection = (
            Detection.query.filter_by(user_id=user_id)
            .order_by(Detection.created_at.desc())
            .all()
        )
        data = transform(detection)
        return response.ok(data, "Detection Lists")

    except Exception as e:
        return response.badRequest([], str(e))


def store():
    try:
        user_id = request.json["user_id"]
        account = request.json["account"]
        result = request.json["result"]
        start = request.json["start"]
        end = request.json["end"]
        tweets = request.json["tweets"]

        detection = Detection(
            user_id=user_id, account=account, result=result, start=start, end=end
        )
        db.session.add(detection)
        db.session.commit()

        bulk = []
        for t in tweets:
            bulk.append(
                History(
                    detection_id=detection.id,
                    raw_tweet=t["raw"],
                    processed_tweet=t["processed"],
                    timestamp=t["timestamp"],
                    depressive=t["label"],
                )
            )

        db.session.add_all(bulk)
        db.session.commit()

        data = {
            "id": detection.id,
            "user_id": user_id,
            "account": account,
            "start": start,
            "end": end,
            "tweets": tweets,
        }

        return response.ok(data, "Successfully create data!")

    except Exception as e:
        return response.badRequest([], str(e))


def update(detectionId):
    try:
        user_id = request.json["user_id"]
        account = request.json["account"]
        result = request.json["result"]
        start = request.json["start"]
        end = request.json["end"]

        detection = Detection.query.filter_by(id=detectionId).first()

        if user_id.strip():
            detection.user_id = user_id

        if account.strip():
            detection.account = account

        if result.strip():
            detection.result = result

        if start.strip():
            detection.start = start

        if end.strip():
            detection.end = end

        detection.setUpdated()
        db.session.commit()

        return response.ok("", "Successfully update data!")

    except Exception as e:
        return response.badRequest([], str(e))


def delete(detectionId):
    try:
        detection = Detection.query.filter_by(id=detectionId).first()
        if not detection:
            return response.badRequest([], "Data not found")

        db.session.delete(detection)
        db.session.commit()

        return response.ok("", "Successfully delete data!")
    except Exception as e:
        return response.badRequest([], str(e))


def singleTransform(data):
    user = Users.query.filter_by(id=data.user_id).first()
    data = {
        "id": data.id,
        "user_id": data.user_id,
        "account": data.account,
        "result": data.result,
        "start": data.start.strftime("%Y-%m-%d"),
        "end": data.end.strftime("%Y-%m-%d"),
        "created_at": data.created_at.strftime("%Y-%m-%d %H:%M"),
        "tester": user.name,
    }

    return data


def transform(datas):
    array = []
    for d in datas:
        array.append(singleTransform(d))
    return array


def scrape(username, start, end):
    try:
        # username = request.json["username"]
        # start = request.json["start"]
        # end = request.json["end"]

        username = username
        start = start
        end = end

        arr = []
        c = twint.Config()
        c.Username = username
        c.Profile_full = True
        c.Since = start
        c.Until = end
        c.Store_object = True
        c.Store_object_tweets_list = arr
        c.Hide_output = True

        twint.run.Profile(c)

        data = []
        for t in arr:
            data.append(
                {
                    "id": t.id,
                    "username": t.username,
                    "timestamp": t.datestamp + " " + t.timestamp,
                    "tweet": t.tweet,
                }
            )

        # ascending sort
        data.sort(key=mySort)

        # return response.ok(data, "Successfully scrape data!")
        return data

    except Exception as e:
        return response.badRequest([], str(e))


def classify():
    try:
        # scraping tweet
        scrape_data = scrape(
            request.json["username"], request.json["start"], request.json["end"]
        )
        if not scrape_data:
            return response.badRequest([], "Sorry, no tweet for the entered period")

        # load dataset
        tweets = Tweets.query.all()
        tlists = []
        for t in tweets:
            tlists.append([t.content, int(t.label)])

        df = DataFrame(tlists, columns=["sentence", "label"])

        sentences = df["sentence"].values
        labels = df["label"].values
        sentences = [cleaning(s) for s in sentences]

        filtered_words = [removeStopWords(sen) for sen in sentences]
        sentences = ["".join(sen) for sen in filtered_words]
        sentences_train, sentences_test, y_train, y_test = train_test_split(
            sentences, labels, test_size=0.25, random_state=1000
        )

        # count dataset words
        num_words = [len(sentence.split()) for sentence in sentences]
        maxsentences = max(sentences, key=len)
        maxlen = 50

        # tokenizing
        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(sentences)

        # Adding 1 because of reserved 0 index
        vocab_size = len(tokenizer.word_index) + 1

        # load cnn model
        from keras.models import load_model

        cnn_model = load_model("model-0.872acc-0.362loss", compile=True)

        new_data = [s["tweet"] for s in scrape_data]

        # preprocess new data
        new_test = [cleaning(d) for d in new_data]
        f_words = [removeStopWords(n) for n in new_test]
        new_test = ["".join(n) for n in f_words]

        predict_data = tokenizer.texts_to_sequences(new_test)
        predict_data = pad_sequences(predict_data, padding="post", maxlen=50)

        # predict
        ynew = cnn_model.predict(predict_data)

        detection = []
        total = 0

        for x in range(0, len(new_data)):
            label = 0
            if ynew[x] > 0.6:
                label = 1
                total += 1

            detection.append(
                {
                    "raw": new_data[x],
                    "processed": new_test[x],
                    "label": label,
                    "id": scrape_data[x]["id"],
                    "username": scrape_data[x]["username"],
                    "timestamp": scrape_data[x]["timestamp"],
                }
            )

        data = {
            "detection": detection,
            "result": (total / len(new_test)) * 100,
        }

        return response.ok(data, "Successfully!")

    except Exception as e:
        return response.badRequest([], str(e))


def plot_history(history):
    accuracy = history.history["accuracy"]
    val_accuracy = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    x = range(1, len(accuracy) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, accuracy, "b", label="Training accuracy")
    plt.plot(x, val_accuracy, "r", label="Validation accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, "b", label="Training loss")
    plt.plot(x, val_loss, "r", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()


def train():
    clear_session()
    percentage = 0.25

    array = []
    tlists = []

    tweets = Tweets.query.all()
    for t in tweets:
        tlists.append([t.content, int(t.label)])

    # tweets = Tweets.query.filter_by(label=1).limit(400).all()
    # for t in tweets:
    #     array.append(
    #         {
    #             "id": t.id,
    #             "account": t.account,
    #             "content": t.content,
    #             "label": t.label,
    #             "created_at": t.created_at,
    #         }
    #     )

    # tweets = Tweets.query.filter_by(label=0).limit(400).all()
    # for t in tweets:
    #     array.append(
    #         {
    #             "id": t.id,
    #             "account": t.account,
    #             "content": t.content,
    #             "label": t.label,
    #             "created_at": t.created_at,
    #         }
    #     )

    array.sort(key=secondSort)
    for a in array:
        tlists.append([a["content"], int(a["label"])])

    # shuffle dataset
    random.shuffle(tlists)

    df = DataFrame(tlists, columns=["sentence", "label"])

    sentences = df["sentence"].values
    labels = df["label"].values
    sentences = [cleaning(s) for s in sentences]

    # tokenizing
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(sentences)

    filtered_words = [removeStopWords(sen) for sen in sentences]
    sentences = ["".join(sen) for sen in filtered_words]
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=percentage, random_state=1000
    )

    # count dataset words
    num_words = [len(sentence.split()) for sentence in sentences]
    maxsentences = max(sentences, key=len)
    maxlen = 50

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

    # Word Embeddings
    embedding_dim = 300
    embedding_matrix = create_embedding_matrix(
        "w2vec_wiki_id_300.txt", tokenizer.word_index, embedding_dim
    )
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

    # my own hyperparameter tuning
    num_filter = [32, 64, 96, 128, 160, 192, 224, 256]
    num_kernel = [3, 4, 5, 6]
    data = []
    best_acc = 0.0
    best = []

    for f in num_filter:
        for k in num_kernel:
            # CNN with w2vec
            cnn_model = Sequential()
            cnn_model.add(
                layers.Embedding(
                    vocab_size,
                    embedding_dim,
                    weights=[embedding_matrix],
                    input_length=maxlen,
                    trainable=True,
                )
            )
            cnn_model.add(
                layers.Conv1D(
                    f,
                    k,
                    activation="relu",
                    kernel_regularizer=l2(0.01),
                    bias_regularizer=l2(0.01),
                )
            )
            cnn_model.add(layers.GlobalMaxPooling1D())
            cnn_model.add(layers.Dense(64, activation="relu"))
            cnn_model.add(Dropout(0.5))
            cnn_model.add(Flatten())
            cnn_model.add(layers.Dense(1, activation="sigmoid"))
            cnn_model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
            cnn_model.summary()

            history = cnn_model.fit(
                X_train,
                y_train,
                epochs=20,
                verbose=False,
                validation_data=(X_test, y_test),
                batch_size=32,
                shuffle=False,
            )

            train_loss, train_accuracy = cnn_model.evaluate(
                X_train, y_train, verbose=False
            )
            test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=False)

            if test_accuracy > best_acc:
                best_acc = test_accuracy
                best = {
                    "best filter size": f,
                    "best kernel size": k,
                    "Test Loss": test_loss,
                    "Test Accuracy": test_accuracy,
                }
                plot_history(history)
                # cnn_model.save("cnn_model" + str(test_accuracy))
                cnn_model.save("cnn_model_fix")

    p = 100 - (percentage * 100)
    filename = (
        "1000dt9-"
        + str(best["best filter size"])
        + "f-"
        + str(best["best kernel size"])
        + "k-"
        + str(round(best["Test Accuracy"], 3))
        + "acc-"
        + str(round(best["Test Loss"], 3))
        + "loss.png"
    )
    best["img-plot"] = filename
    data.append(best)
    plt.savefig("static/sampling/" + filename)

    return response.ok(data, "Successfully Train Model!")


def tester():
    clear_session()
    tweets = Tweets.query.all()
    tlists = []
    for t in tweets:
        tlists.append([t.content, int(t.label)])

    # shuffle dataset
    random.shuffle(tlists)

    df = DataFrame(tlists, columns=["sentence", "label"])

    sentences = df["sentence"].values
    labels = df["label"].values
    sentences = [cleaning(s) for s in sentences]

    # tokenizing
    tokenizer = Tokenizer(num_words=20000)
    tokenizer.fit_on_texts(sentences)

    filtered_words = [removeStopWords(sen) for sen in sentences]
    sentences = ["".join(sen) for sen in filtered_words]
    sentences_train, sentences_test, y_train, y_test = train_test_split(
        sentences, labels, test_size=0.25, random_state=1000
    )

    # count dataset words
    num_words = [len(sentence.split()) for sentence in sentences]
    maxsentences = max(sentences, key=len)
    maxlen = 50

    # Adding 1 because of reserved 0 index
    vocab_size = len(tokenizer.word_index) + 1

    X_train = tokenizer.texts_to_sequences(sentences_train)
    X_test = tokenizer.texts_to_sequences(sentences_test)

    vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index

    X_train = pad_sequences(X_train, padding="post", maxlen=maxlen)
    X_test = pad_sequences(X_test, padding="post", maxlen=maxlen)

    # Word Embeddings
    embedding_dim = 300
    embedding_matrix = create_embedding_matrix(
        "w2vec_wiki_id_300.txt", tokenizer.word_index, embedding_dim
    )
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))

    # CNN model with w2vec
    cnn_model = Sequential()
    cnn_model.add(
        layers.Embedding(
            vocab_size,
            embedding_dim,
            weights=[embedding_matrix],
            input_length=maxlen,
            trainable=True,
        )
    )
    cnn_model.add(
        layers.Conv1D(
            64,
            6,
            activation="relu",
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    cnn_model.add(layers.GlobalMaxPooling1D())
    cnn_model.add(layers.Dense(64, activation="relu"))
    cnn_model.add(Dropout(0.5))
    cnn_model.add(Flatten())
    cnn_model.add(layers.Dense(1, activation="sigmoid"))
    cnn_model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    cnn_model.summary()

    history = cnn_model.fit(
        X_train,
        y_train,
        epochs=20,
        verbose=False,
        validation_data=(X_test, y_test),
        batch_size=64,
        shuffle=False,
    )

    train_loss, train_accuracy = cnn_model.evaluate(X_train, y_train, verbose=False)
    test_loss, test_accuracy = cnn_model.evaluate(X_test, y_test, verbose=False)

    plot_history(history)
    # cnn_model.save("test_cnn_model")
    # plt.savefig("static/sampling/qwerty.png")

    # set test data
    new_data = [
        "Apa gunanya ngungkapin pembelaan diri. Kalau yg disimpulkan ttp saya yg bersalah. Lebih baik diam. 🤐",
        "Dimanakah kebahagiaan yg dulu kau janjikan? nyatanya aku kesepian sekarang :'(",
        "Di titik tiba tiba putus asa buat semuanya, terutama masa depan",
        "Dunia gak perlu tau soal kamu siapa. kamu kenapa. Tidak akan ada banyak orang yang peduli. Sebagian hanya sebatas penasaran dan kemudian memilih meninggalkan.",
        "Seperti kebanyakan pemuda yg beranjak usia melewati 25 tahun. Diriku berada di ambang antara keputusan,seperti apa kelanjutannya dimasa depan. Merangkai terlalu banyak angan. Sampai tidak satupun dapat tergapai. Mulai berkeinginan untuk putus asa, namun menahan diri sekuat tenaga",
        "kalo bunuh diri ga dosa, aku lakuin  deh daripada hidup tapi kya org mau mati tp ga mati gini 🙂",
        "Bersedih adalah suatu hal yang wajar,tapi jangan sampe kesedihan itu melemahkan hati kita dan bikin kita jadi putus asa. #BOT",
        "Putus asa bukanlah cara yang tepat dalam perjalanan anda menuju cita-cita",
        "Apakah kamu pernah ngerasain kesepian yang kesepian banget sampe ngerasa sendiri dan pengen pergi aja dari bumi?",
        "Jangan terburu buru mencari peganti untuk menghapuskan luka semalam, jangan terburu buru menerima orang baru kerana takut kesepian sebaliknya rawat dulu luka luka tu sampai sembuh dengan sempurna",
    ]

    # preprocess new data
    new_test = [cleaning(d) for d in new_data]
    f_words = [removeStopWords(n) for n in new_test]
    new_test = ["".join(n) for n in f_words]

    predict_data = tokenizer.texts_to_sequences(new_test)
    predict_data = pad_sequences(predict_data, padding="post", maxlen=50)

    # predict
    ynew = cnn_model.predict(predict_data)

    detection = []
    total = 0

    for x in range(0, len(new_data)):
        label = 0
        if ynew[x] > 0.6:
            label = 1
            total += 1

        detection.append(
            {
                "raw": new_data[x],
                "processed": new_test[x],
                "label": label,
            }
        )

    data = {
        "Training Loss": train_loss,
        "Test Loss": test_loss,
        "Training Accuracy": train_accuracy,
        "Test Accuracy": test_accuracy,
        "detection": detection,
    }

    return response.ok(data, "Successfully Train Model!")


def tester2():
    try:
        # load cnn model
        from keras.models import load_model

        cnn_model = load_model("model-0.872acc-0.362loss", compile=True)

        # set test data
        new_data = [
            "Berbulan2 saya nangis tiap malam, sampe rasanya lelah sama jalan hidup. Sangat berbanding terbalik kan dengan kalian yg sedang merayakan kebahagiaan? Saya tidak akan membalas apapun meskipun kalian sangat sangat sangat bikin saya sakit hati. Semoga Allah segera membalas kalian 🙂",
            'buat diriku. "maaf untuk malam2 panjang yang sudah lalu! mata susah terpejam, pikiran yang lelah, sakit hati yang terpaksa/dipaksa bungkam dll. terima kasih tubuh sudah terlihat baik2 saja dihadapan orang2" :\')',
            "Aku baru kebeli motor aja udh seneng",
            "hari ini banyak bgt yg ngasih makanan Alhamdulillah",
            "Karena hadir adalah hadiah paling indah,bagi rindu yang membuncah",
            "Selamat ulang tahun bhayangkara!!",
            "Rindu indomie",
            "ngelukain sendiri. Ngobatin sendiri. Mati rasa kemudian. Ingin menangis sudah tak bisa, aku seperti orang lemah yang tak berguna",
            "pergelangan tangan saya satu kali, dua kali, tiga kali, sampai air mata saya terhenti.\n\nPertanyaan yang sering muncul adalah, “Apakah saya ingin mati”?\n\nTidak. Saya tidak ingin mati saat itu. Saya hanya ingin menyalurkan luka emosi dan batin saya ke fisik saya. Agar mungkin,",
            "Seandainya nikah bukan ibadah, saya memilih sendiri dan tidak ingin bersanding dengan laki laki manapun.  Dan seandainya bunuh diri tidak dosa, mungkin saya sudah lama tidak ada.",
            "Ku akhiri hidupku dengan bunuh diri",
        ]

        # load dataset
        tweets = Tweets.query.all()
        sentences = []
        for t in tweets:
            sentences.append(t.content)

        # tokenizing
        sentences = [cleaning(s) for s in sentences]
        filtered_words = [removeStopWords(sen) for sen in sentences]
        sentences = ["".join(sen) for sen in filtered_words]

        tokenizer = Tokenizer(num_words=20000)
        tokenizer.fit_on_texts(sentences)

        # preprocess new data
        new_test = [cleaning(d) for d in new_data]
        f_words = [removeStopWords(n) for n in new_test]
        new_test = ["".join(n) for n in f_words]
        maxwords = len(max(new_test, key=len).split())
        new_test = tokenizer.texts_to_sequences(new_test)
        new_test = pad_sequences(new_test, padding="post", maxlen=50)

        # predict
        ynew = cnn_model.predict(new_test)
        lblnew = []

        for x in ynew:
            a = float(x)
            lblnew.append(round(a, 3))

        return response.ok(lblnew, "Detection Result")

    except Exception as e:
        return response.badRequest([], str(e))


# create embedding matrix
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


# k-folds cross-validation
def create_model(num_filters, kernel_size, vocab_size, embedding_dim, maxlen):
    model = Sequential()
    model.add(
        layers.Embedding(
            vocab_size, embedding_dim, weights=[embedding_matrix], input_length=maxlen
        )
    )
    model.add(
        layers.Conv1D(
            num_filters,
            kernel_size,
            activation="relu",
            kernel_regularizer=l2(0.01),
            bias_regularizer=l2(0.01),
        )
    )
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def mySort(e):
    return e["timestamp"]


def secondSort(e):
    return e["created_at"]


def dashboard():
    try:
        total_users = Users.query.count()
        total_depressed = (
            Detection.query.filter(Detection.result > 50)
            .group_by(Detection.account)
            .count()
        )
        total_account = Detection.query.group_by(Detection.account).count()

        last = Detection.query.order_by(Detection.id.desc()).first()
        last_detection = singleTransform(last)
        detail = len(last.details.all())
        last_detection["total_tweets"] = detail

        data = {
            "total_account": str(total_account),
            "total_depressed": str(total_depressed),
            "total_users": str(total_users),
            "last_detection": last_detection,
        }

        return response.ok(data, "Dashboard")

    except Exception as e:
        return response.badRequest([], str(e))


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
    char_list = ["@", "/rlt/", "https://", "pic.twitter.com", "#"]
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


# remove stopwords
def removeStopWords(text):
    factory = StopWordRemoverFactory()
    stopword = factory.create_stop_word_remover()
    return stopword.remove(text)

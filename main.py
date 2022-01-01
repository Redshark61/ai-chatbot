# pylint: disable=wrong-import-position
import json
import random
import os
import time
import pickle
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
import tflearn
import tensorflow as tf
from nltk.stem.lancaster import LancasterStemmer
import nltk
import numpy as np

nltk.download('punkt')
stemmer = LancasterStemmer()

with open("./intents.json", encoding="utf-8") as file:
    data = json.load(file)

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            tokenizedWord = nltk.word_tokenize(pattern, language="french")
            words.extend(tokenizedWord)
            docs_x.append(tokenizedWord)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []

    outEmpty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        tokenizedWord = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in tokenizedWord:
                bag.append(1)
            else:
                bag.append(0)

        outputRow = outEmpty[:]
        outputRow[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(outputRow)

    training = np.array(training)
    output = np.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tf.compat.v1.reset_default_graph()
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    begin = time.time()
    model = tflearn.DNN(net)

    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

    end = time.time()
    print("Training time: ", end - begin)

def bagOfWords(s, words):
    bag = [0 for _ in range(len(words))]

    sWords = nltk.word_tokenize(s, language="french")
    sWords = [stemmer.stem(word.lower()) for word in sWords]

    for word in sWords:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1

    return np.array(bag)

def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bagOfWords(inp, words)])[0]
        resultsIndices = np.argmax(results)
        tag = labels[resultsIndices]

        if results[resultsIndices] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            print(random.choice(responses))
        else:
            print("I didn't understand that, try again!")

chat()

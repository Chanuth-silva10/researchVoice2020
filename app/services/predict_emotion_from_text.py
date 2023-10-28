import json
import re
import nltk
import spacy
import string
from numpy import argmax
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_preprocessing.text import tokenizer_from_json
from keras.models import load_model


class PredictEmotionFromText:
    def __init__(
        self, text_emo_model_path, text_emo_tokenizer_path, text_suicidal_model_path
    ) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.emo_tokenizer = Tokenizer(num_words=10000)
        self.suicidal_tokenizer = Tokenizer(
            num_words=50000, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True
        )
        self.punc_list = list(string.punctuation)
        stopwords_list = set(stopwords.words("english"))
        stopwords_temp = [re.sub("[']", "", stopword) for stopword in stopwords_list]
        self.stopwords_set = set(stopwords_temp)
        self.emo_classes_dict = {
            0: "anger",
            1: "fear",
            2: "happy",
            3: "love",
            4: "sadness",
            5: "surprise",
        }
        self.suicidal_classes_dict = {0: "nonsuicidal", 1: "suicidal"}
        self.emo_model = load_model(text_emo_model_path)
        with open(text_emo_tokenizer_path) as f:
            data = json.load(f)
            self.emo_tokenizer = tokenizer_from_json(data)
        self.suicidal_model = load_model(text_suicidal_model_path)

    def process_text(self, text) -> str:
        global i
        i += 1
        if i == 1000:
            print(i)
        text_new = re.sub(r"['#@\$\"\.,!?~]", "", text)
        text_new = re.sub(r"[-_/]", " ", text_new)
        text_new = re.sub(r" +", " ", text_new)
        text_new = re.sub(r"(.)\1{2,}", r"\1", text_new)
        text_new = re.sub(r"[\U00010000-\U0010ffff]", "", text_new)  # removing emojis
        word_splitted = nltk.word_tokenize(text_new)  # word splitting
        word_splitted = [word.lower() for word in word_splitted]  # lowercase
        word_splitted = [
            word
            for word in word_splitted
            if word not in self.punc_list and word not in self.stopwords_set
        ]
        text_new = " ".join(word_splitted)
        doc = self.nlp(text_new)
        text_new = [token.lemma_ for token in doc]  # lemmatization
        text_new = [token.lower() for token in text_new]
        return " ".join(text_new)

    def emo_prediction(self, text):
        global i
        i = 1
        processed_text = self.process_text(text)
        emo_text_seq = self.emo_tokenizer.texts_to_sequences([processed_text])
        emo_text_pad = pad_sequences(emo_text_seq, maxlen=64)
        emo_prediction = self.emo_model(emo_text_pad)
        emo_predicted_label = self.emo_classes_dict[argmax(emo_prediction[0])]
        return emo_predicted_label

    def suicidal_prediction(self, text):
        self.suicidal_tokenizer.fit_on_texts([text])
        suicidal_text_seq = self.suicidal_tokenizer.texts_to_sequences([text])
        suicidal_text_pad = pad_sequences(suicidal_text_seq, maxlen=300)
        suicidal_prediction = self.suicidal_model(suicidal_text_pad)
        suicidal_predicted_label = self.suicidal_classes_dict[
            argmax(suicidal_prediction, axis=1)[0]
        ]
        return suicidal_predicted_label

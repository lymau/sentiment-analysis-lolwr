from flask import Flask, render_template, request
import os
import re
import nltk
import pickle
import string
import emoji
import re
from googletrans import Translator
string.punctuation
nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

app = Flask(__name__)
app.debug = True

@app.route("/")
def main():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        text = request.form['input']

        result = preprocessing(text)

        current_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_path, 'model/model.pkl')
        tfidf_path = os.path.join(current_path, 'model/tfidf_model.pkl')

        model = load_model(model_path)

        tfidf_model = load_model(tfidf_path)
        result = tfidf_model.transform([result])

        sentiment = model.predict(result)

        emoji = '😄' if sentiment == [1] else '😠'

        sentiment = 'Sentimen Positif' if sentiment == [1] else 'Sentimen Negatif'

        prob = model.predict_proba(result)

        prob = prob[0][0] if prob[0][0] > prob[0][1] else prob[0][1]

        return render_template('index.html', sentiment=sentiment, prob=prob, emoji=emoji)

def load_model(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model

def sanitize(text):
    emoticon_str = r"""
        (?:
        [<>]?
        [:;=8]                     # eyes
        [\-o\*\']?                 # optional nose
        [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth      
        |
        [\)\]\(\[dDpP/\:\}\{@\|\\] # mouth
        [\-o\*\']?                 # optional nose
        [:;=8]                     # eyes
        [<>]?
        )"""

    nonalphabet = re.compile('[^a-zA-Z ]')
    rpt = re.compile(r'(.)\1{1,}', re.IGNORECASE)

    text = emoji.replace_emoji(text, replace='')  # removing emoji
    
    text = re.sub(emoticon_str, '', text)  # removing emoticon symbol
    text = re.sub(r"""(?:@[\w_]+)""", '', text)  # removing hashtag
    text = re.sub(r'http://\/[^\s]+', '', text,
                    flags=re.MULTILINE)  # removing web links
    text = re.sub(r'https://\/[^\s]+', '', text,
                    flags=re.MULTILINE)  # removing web links
    text = re.sub(r'www.\/[^\s]+', '', text,
                    flags=re.MULTILINE)  # removing web links
    text = nonalphabet.sub('', text)
    text = rpt.sub(r'\1', text)
    text = text.lower()

    return text

def translate(text):  
    translator = Translator()

    return translator.translate(text, dest='id').text

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return tokens

def remove_stopwords(text):
    list_stopword =  set(stopwords.words('indonesian'))
    custom_stopwords = ['game', 'bagu', 'nya', 'gamenya']
    for stopword in custom_stopwords:
        list_stopword.add(stopword)
    output = [i for i in text if i not in list_stopword]
    return output

def preprocessing(text):
    text = sanitize(text)
    text = translate(text)
    text = tokenize(text)
    text = remove_stopwords(text)
    return ' '.join(text)

if __name__ == '__main__':
    app.run(debug=True)
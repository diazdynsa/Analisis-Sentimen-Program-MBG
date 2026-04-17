from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import pickle
import re
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

print("Memuat AI Model & Radar...")
model = load_model('model_sentimen_mbg.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
with open('label_classes.pkl', 'rb') as f:
    label_classes = pickle.load(f)

MAX_LENGTH = 100

def bersihkan_teks(teks):
    teks = str(teks).lower()
    teks = re.sub(r'http\S+|www\S+|https\S+', '', teks)
    teks = re.sub(r'\@\w+|\#', '', teks)
    teks = re.sub(r'[^a-zA-Z\s]', ' ', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()
    return teks

def tebak_sentimen(kalimat):
    kalimat_bersih = bersihkan_teks(kalimat)
    seq = tokenizer.texts_to_sequences([kalimat_bersih])
    padded = pad_sequences(seq, maxlen=MAX_LENGTH, padding='pre', truncating='pre')
    prediksi = model.predict(padded, verbose=0)[0]
    ind_max = np.argmax(prediksi)
    
    # Fitur Unik: Keyword X-Ray (Ambil kata penting yang lebih dari 3 huruf)
    kata_kunci = [kata for kata in kalimat_bersih.split() if len(kata) > 3]
    kata_kunci = list(set(kata_kunci))[:5] # Ambil maksimal 5 kata unik
    
    return label_classes[ind_max], prediksi[ind_max] * 100, kata_kunci

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text_input')
    if text:
        hasil, skor, keywords = tebak_sentimen(text)
        return render_template('index.html', text_input=text, hasil=hasil, skor=round(skor, 2), keywords=keywords)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
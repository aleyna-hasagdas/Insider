from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model and other necessary components
model = load_model('neural_network_model.h5')
vectorizer = joblib.load('vectorizer.pkl')

# Manual category dictionary
category_dict = {
    0: "Aksesuar",
    1: "Aksesuar & Sarf Malz.",
    2: "Aksesuar Ürünleri",
    3: "Antenler / Kablolar",
    4: "Bilgisayar",
    5: "Bilgi Teknolojileri",
    6: "Cep Telefonu",
    7: "Dijital Yaşam",
    8: "Elektrikli Ev Aletleri",
    9: "Elektronik",
    10: "Elektronik & TV",
    11: "Elektronik ve Televizyon",
    12: "Elektronik/TV",
    13: "Ev Aletleri & Elektronik",
    14: "Ev Elektroniği",
    15: "Foto & Kamera",
    16: "Foto&Kamera",
    17: "Fotoğraf / Elektronik",
    18: "Hobi & Oyun Konsolları",
    19: "Kişisel Bilgisayarlar",
    20: "Küçük Ev Aletleri",
    21: "Led Tv",
    22: "Monitör",
    23: "Mürekkep Kartuşları",
    24: "Network Ürünleri",
    25: "OEM",
    26: "OEM & Çevre Birimleri",
    27: "OEM Ürünleri",
    28: "Other",
    29: "Oto Aks & Navigasyon",
    30: "Oyun & Oyun Konsolu",
    31: "Oyun - Hobi",
    32: "Oyun Dünyası",
    33: "PC / Monitör",
    34: "Soğutucu(Fan)",
    35: "Telefon",
    36: "Televizyon",
    37: "Tüketici Elektroniği",
    38: "Tüketim Malzemeleri",
    39: "Tüketim Ürünleri",
    40: "Yazılım",
    41: "Çevre Birimler",
    42: "Çevre Birimleri"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form['description']
    print(f"Received description: {data}")  # Debugging
    preprocessed = vectorizer.transform([data]).toarray()
    print(f"Transformed input: {preprocessed}")  # Debugging
    prediction = model.predict(preprocessed)
    print(f"Raw prediction: {prediction}")  # Debugging
    predicted_class = np.argmax(prediction, axis=1)[0]
    print(f"Predicted class index: {predicted_class}")  # Debugging

    # Convert predicted class index to category using the dictionary
    predicted_label = category_dict.get(predicted_class, "Unknown Category")
    print(f"Predicted label: {predicted_label}")  # Debugging

    return render_template('index.html', prediction_text='Predicted Category: {}'.format(predicted_label))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)

from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import func
from PIL import Image
import numpy as np
import io

app = Flask(__name__)

# Konfigurasi Database (SQLite untuk testing lokal)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///db.sqlite3'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Model Database
class PlantClassification(db.Model):
    plant_id = db.Column(db.String(100), primary_key=True)
    plant_name = db.Column(db.String(100), nullable=False)
    scientific_name = db.Column(db.String(100))
    origin_place = db.Column(db.String(100))
    plant_description = db.Column(db.Text)
    climate = db.Column(db.String(50))
    fertilizer = db.Column(db.String(100))
    uses = db.Column(db.Text)
    common_disease = db.Column(db.String(100))
    harvest_time = db.Column(db.String(50))
    watering_frequency = db.Column(db.String(50))
    harvest_time_days = db.Column(db.Integer)
    watering_interval = db.Column(db.String(50))

# Load model klasifikasi gambar
model = load_model('my_model2.h5')

# Daftar kelas sesuai model (plant_name yang dikembalikan model)
classes = [
    'Aloe Vera', 'Apple', 'Areca Palm', 'Birds Nest Fern', 'Blueberry', 'Cherry', 'Chinese Evergreen', 'Corn', 'Dracaena', 'Dumb Cane', 'Elephant Ear', 'Grape', 'Monstera Deliciosa', 'Peach', 'Pepper Bell', 'Polka Dot Plant', 'Ponytail Palm', 'Potato', 'Raspberry', 'Snake Plant', 'Soybean', 'Strawberry', 'Tomato'
]

# classes = [
#     'plant_01', 'plant_02', 'plant_03', 'plant_04', 'plant_05', 'plant_06', 'plant_07', 'plant_08', 'plant_09', 'plant_10', 'plant_11', 'plant_12', 'plant_13', 'plant_14', 'plant_15', 'plant_16', 'plant_17', 'plant_18', 'plant_19', 'plant_20', 'plant_21', 'plant_22'
# ]

# Fungsi untuk preprocessing gambar
def preprocess_image(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0  # Normalisasi piksel
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Endpoint untuk klasifikasi gambar
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ambil gambar dari request
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read()))
        processed_img = preprocess_image(img)

        # Prediksi dengan model
        predictions = model.predict(processed_img)[0]  # Dapatkan array probabilitas
        predicted_index = np.argmax(predictions)  # Dapatkan indeks prediksi tertinggi
        highest_probability = predictions[predicted_index]  # Ambil nilai probabilitas tertinggi
        predicted_class = classes[predicted_index]  # Ambil nama kelas berdasarkan indeks

        # Debugging: Print informasi prediksi
        print(f"Predicted class: {predicted_class}")
        print(f"Highest probability: {highest_probability:.2f}")

        # Tentukan ambang batas probabilitas (misalnya 50% atau 0.50)
        threshold = 0.50

        if highest_probability >= threshold:
            # Normalisasi predicted_class untuk pencarian di database
            predicted_class_normalized = predicted_class.strip().lower()

            # Cari data di database berdasarkan plant_name
            plant_data = PlantClassification.query.filter(
                func.lower(func.trim(PlantClassification.plant_name)) == predicted_class_normalized
            ).first()

            if plant_data:
                # Response dengan data lengkap dari database dan tingkat kemiripan
                response_data = {
                    'plant_id': plant_data.plant_id,
                    'plant_name': plant_data.plant_name,
                    'scientific_name': plant_data.scientific_name,
                    'origin_place': plant_data.origin_place,
                    'plant_description': plant_data.plant_description,
                    'climate': plant_data.climate,
                    'fertilizer': plant_data.fertilizer,
                    'uses': plant_data.uses,
                    'common_disease': plant_data.common_disease,
                    'harvest_time': plant_data.harvest_time,
                    'watering_frequency': plant_data.watering_frequency,
                    'harvest_time_days': plant_data.harvest_time_days,
                    'watering_interval': plant_data.watering_interval,
                    'probability': f"{highest_probability:.2%}"  # Format sebagai persentase
                }
            else:
                response_data = {'error': f'Data for class "{predicted_class}" not found in database.'}
        else:
            # Jika probabilitas tertinggi di bawah ambang batas
            response_data = {
                'error': 'Gambar tidak dikenali. Probabilitas tidak memenuhi ambang batas.',
                'probability': f"{highest_probability:.2%} tampak seperti {predicted_class}" 
            }

        return jsonify(response_data)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Debug: Cetak semua data dari database
    with app.app_context():
        all_plants = PlantClassification.query.all()
        for plant in all_plants:
            print(f"Database plant_name: {plant.plant_name}, plant_id: {plant.plant_id}")

    app.run(debug=True)

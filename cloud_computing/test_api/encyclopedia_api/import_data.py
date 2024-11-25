import pandas as pd
from app import db, PlantClassification, app  # Import 'app' dari app.py

# Baca spreadsheet
df = pd.read_excel('plant_encyclopedia.xlsx')  # Sesuaikan dengan nama file

# Masukkan data ke database SQLite dalam application context
with app.app_context():
    for index, row in df.iterrows():
        plant = PlantClassification(
            plant_id=row['plant_id'],
            plant_name=row['plant_name'],
            scientific_name=row['scientific_name'],
            origin_place=row['origin_place'],
            plant_description=row['plant_description'],
            climate=row['climate'],
            fertilizer=row['fertilizer'],
            uses=row['uses'],
            common_disease=row['common_disease'],
            harvest_time=row['harvest_time'],
            watering_frequency=row['watering_frequency'],
            harvest_time_days=row['harvest_time_days'],
            watering_interval=row['watering_interval']
        )
        db.session.add(plant)  # Tambahkan data ke session

    db.session.commit()  # Commit perubahan ke database
    print("Data berhasil diimport ke database SQLite!")

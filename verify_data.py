import pickle
import os

# Check if the data files exist
faces_data_path = 'data/faces_data.pkl'
names_data_path = 'data/names.pkl'

if os.path.exists(faces_data_path) and os.path.exists(names_data_path):
    with open(faces_data_path, 'rb') as f:
        FACES = pickle.load(f)
    with open(names_data_path, 'rb') as w:
        LABELS = pickle.load(w)

    print(f"Shape of FACES: {FACES.shape}")
    print(f"Number of LABELS: {len(LABELS)}")
    print(f"First 10 LABELS: {LABELS[:10]}")
else:
    print("Data files not found. Make sure to run add_faces.py first.")

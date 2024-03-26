import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

directories = ['Sounds/ambulance', 'Sounds/firetruck']

X = []
y = []

# Process each file in each directory
for label, directory in enumerate(directories):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            # Load the .wav file
            file_path = os.path.join(directory, filename)
            audio, sr = librosa.load(file_path, sr=None)
            
            mfccs = librosa.feature.mfcc(audio, sr=sr)
            mfccs_processed = np.mean(mfccs.T, axis=0)
            
            # Append features and label
            X.append(mfccs_processed)
            y.append(label)

X = np.array(X)
y = np.array(y)

# Splitting dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training a Support Vector Machine classifier
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)


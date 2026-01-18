import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Load data
df = pd.read_csv('glaucoma[1].csv')

# Drop unused columns
df.drop([
    'Patient ID',
    'Medication Usage',
    'Visual Field Test Results',
    'Optical Coherence Tomography (OCT) Results',
    'Visual Symptoms',
    'Medical History'
], axis=1, inplace=True)

# Encode target
le = LabelEncoder()
df['Glaucoma Type'] = le.fit_transform(df['Glaucoma Type'])

# One-hot encode features
X = pd.get_dummies(df.drop('Glaucoma Type', axis=1), drop_first=True)
y = df['Glaucoma Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Handle imbalance
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

# Save everything
pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))
pickle.dump(X.columns, open('features.pkl', 'wb'))

print("Model training complete and saved")

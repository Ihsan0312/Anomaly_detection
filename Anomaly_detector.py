import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import IsolationForest

import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

import matplotlib.pyplot as plt

# Step 1: Load the Dataset
file_path = '/home/ihsan/Documents/Intern/archive/output1.csv'  # Replace with your actual path
df = pd.read_csv(file_path)

# Step 2: Rename Columns
df = df.rename(columns={
    'Source IP': 'src_ip',
    'Destination IP': 'dst_ip',
    'Protocol': 'protocol',
    'Length': 'packet_size'
})

# Step 3: Handle Missing Values
print("Missing values before cleaning:")
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

print("Missing values after cleaning:")
print(df.isnull().sum())

# Step 4: Encode Categorical Columns
label_encoder = LabelEncoder()
df['src_ip_encoded'] = label_encoder.fit_transform(df['src_ip'])
df['dst_ip_encoded'] = label_encoder.fit_transform(df['dst_ip'])
df['protocol_encoded'] = label_encoder.fit_transform(df['protocol'])

# Step 5: Scale the packet_size Column
scaler = StandardScaler()
df['packet_size_scaled'] = scaler.fit_transform(df[['packet_size']])

# Step 6: Prepare Features for Model Training
X = df[['src_ip_encoded', 'dst_ip_encoded', 'packet_size_scaled', 'protocol_encoded']]

# Step 7: Train Isolation Forest Model
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

# Step 8: Predict Anomalies
y_pred = model.predict(X)

# Convert Predictions (-1: anomaly, 1: normal) to (1: anomaly, 0: normal)
df['anomaly'] = (y_pred == -1).astype(int)

# Step 9: Results
print("Anomaly Count:")
print(df['anomaly'].value_counts())

print("\nExample Anomalies:")
print(df[df['anomaly'] == 1].head())

# Step 10: Visualize Results
# Bar chart for anomaly counts
df['anomaly'].value_counts().plot(kind='bar', color=['blue', 'red'])
plt.title('Anomaly Count')
plt.xlabel('Class (0: Normal, 1: Anomaly)')
plt.ylabel('Number of Observations')
plt.show()

# Step 11: Save Results
output_path = '/home/ihsan/Documents/Intern/archive/output_with_anomalies.csv'  # Replace with your actual path
df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

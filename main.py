import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from utils.anomaly import detect_anomalies
from config import *
from utils.data_loader import load_data, create_sequences
from models.model import NeuroTwinLatent
from training.train import train_model
from inference.predict import evaluate
from inference.what_if import predict_future, what_if_simulation

# Load data
data_scaled, scaler = load_data()

X, y = create_sequences(data_scaled, SEQ_LENGTH)

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Validation
val_split = int(0.9 * len(X_train))
X_val = X_train[val_split:]
y_val = y_train[val_split:]
X_train = X_train[:val_split]
y_train = y_train[:val_split]

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

# Model
model = NeuroTwinLatent()
print(model)

# Train
train_model(model, train_loader, X_val, y_val, EPOCHS, LR)

torch.save(model.state_dict(), "models/neurotwin.pth")

# Evaluate
predictions, pred_np, actual_np, pred_real, actual_real = evaluate(model, X_test, y_test, scaler)

anomalies = detect_anomalies(pred_np, actual_np)
print("Anomaly indices:", anomalies)

# Plot
import numpy as np
plt.figure()
plt.title("NeuroTwin Prediction")
actual = actual_np[:100]
pred = pred_np[:100]
plt.plot(actual, label="Actual")
plt.plot(pred, label="Predicted")

# 🔥 Confidence band add
uncertainty = np.std(pred) * 0.5
upper = pred + uncertainty
lower = pred - uncertainty
plt.fill_between(range(len(pred)), lower, upper, alpha=0.2)
anomalies = detect_anomalies(pred, actual)
plt.scatter(anomalies, pred[anomalies], color='red', label='Anomalies', zorder=5)
plt.legend()
plt.show()

# Future
sample_seq = X_test[0]
future = predict_future(model, sample_seq)
print("Future:", future)

# What-if
orig, mod, _ = what_if_simulation(model, sample_seq)
plt.figure()
plt.title("What-If Simulation")
plt.plot(orig, label="Original Future")
plt.plot(mod, label="Modified Future")
plt.legend()
plt.show()


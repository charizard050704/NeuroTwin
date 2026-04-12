import torch
import numpy as np
from config import SEQ_LENGTH
from utils.data_loader import load_data, create_sequences
from models.model import NeuroTwinLatent
from inference.predict import evaluate
from inference.what_if import predict_future, what_if_simulation
from utils.anomaly import detect_anomalies
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


MODEL_PATH = "models/neurotwin.pth"


class NeuroTwinBackend:
    def __init__(self):
        self.model = NeuroTwinLatent()
        self.model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        self.model.eval()

        self.data_scaled, self.scaler = load_data()

        X, y = create_sequences(self.data_scaled, SEQ_LENGTH)

        split = int(0.8 * len(X))
        self.X_test = torch.tensor(X[split:], dtype=torch.float32)
        self.y_test = torch.tensor(y[split:], dtype=torch.float32)

    # 🔥 Prediction
    def run_prediction(self):
        preds, pred_np, actual_np, pred_real, actual_real = evaluate(
            self.model, self.X_test, self.y_test, self.scaler
        )

        return actual_real.flatten(), pred_real.flatten()

    # 🔥 Anomaly
    def run_anomaly(self):
        actual, pred = self.run_prediction()
        indices = detect_anomalies(pred, actual)
        return actual, pred, indices

    # 🔥 Forecast
    def run_forecast(self):
        sample_seq = self.X_test[0]
        future = predict_future(self.model, sample_seq)
        return future

    # 🔥 What-if
    def run_whatif(self):
        sample_seq = self.X_test[0]
        orig, mod, _ = what_if_simulation(self.model, sample_seq)
        return orig, mod
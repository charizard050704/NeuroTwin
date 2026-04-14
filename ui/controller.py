import torch
from utils.data_loader import load_data, create_sequences
from config import SEQ_LENGTH
from models.model import NeuroTwinLatent
from inference.predict import evaluate
from inference.what_if import predict_future, what_if_simulation
from utils.anomaly import detect_anomalies

class Controller:
    def __init__(self):
        # ===== LOAD DATA =====
        data_scaled, self.scaler = load_data()
        X, y = create_sequences(data_scaled, SEQ_LENGTH)

        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        self.X_test = X[-200:]
        self.y_test = y[-200:]

        # ===== LOAD TRAINED MODEL =====
        self.model = NeuroTwinLatent()
        self.model.load_state_dict(
            torch.load("models/neurotwin.pth", map_location="cpu")
        )
        self.model.eval()

    # ===== PREDICTION =====
    def prediction(self):
        _, pred_np, actual_np, pred_real, actual_real = evaluate(
            self.model, self.X_test, self.y_test, self.scaler
        )

        return actual_real.flatten(), pred_real.flatten()

    # ===== ANOMALY =====
    def anomaly(self):
        _, pred_np, actual_np, pred_real, actual_real = evaluate(
            self.model, self.X_test, self.y_test, self.scaler
        )

        actual = actual_real.flatten()
        pred = pred_real.flatten()

        idx = detect_anomalies(pred, actual)

        return actual, pred, idx

    # ===== FORECAST =====
    def forecast(self):
        sample = self.X_test[0]
        future = predict_future(self.model, sample)
        return future

    # ===== WHAT IF =====
    def what_if(self):
        sample = self.X_test[0]
        result = what_if_simulation(self.model, sample)

        # handle both return formats
        if len(result) == 3:
            orig, mod, _ = result
        else:
            orig, mod = result

        return orig, mod
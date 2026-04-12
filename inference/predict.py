import torch
import numpy as np
from utils.metrics import compute_metrics

def evaluate(model, X_test, y_test, scaler):

    model.eval()
    with torch.no_grad():
        predictions, _, _ = model(X_test, target_len=1)
        predictions = predictions.squeeze()

    pred_np = predictions.numpy()
    actual_np = y_test.squeeze().numpy()

    compute_metrics(actual_np, pred_np)

    pred_real = scaler.inverse_transform(pred_np.reshape(-1,1))
    actual_real = scaler.inverse_transform(actual_np.reshape(-1,1))

    print("Real predictions:", pred_real[:5])
    print("Real actual:", actual_real[:5])

    return predictions,pred_np,actual_np,pred_real,actual_real
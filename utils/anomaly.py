def detect_anomalies(pred, actual, top_k=15):
    import numpy as np

    pred = np.array(pred).flatten()
    actual = np.array(actual).flatten()

    errors = np.abs(pred - actual)

    indices = np.argsort(errors)[-top_k:]
    return indices
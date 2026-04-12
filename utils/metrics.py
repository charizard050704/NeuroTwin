from sklearn.metrics import mean_squared_error, mean_absolute_error

def compute_metrics(actual, pred):
    mse = mean_squared_error(actual, pred)
    mae = mean_absolute_error(actual, pred)

    print("MSE:", mse)
    print("MAE:", mae)

    return mse, mae
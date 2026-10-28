import numpy as np

def error_stats(pred, ref):
    """pred, ref: массивы (N, 2) в пикселях. Возвращает словарь."""
    e = np.hypot(pred[:, 0] - ref[:, 0], pred[:, 1] - ref[:, 1])
    return {
        "mean": e.mean(), 
        "rmse": np.sqrt((e**2).mean()),
        "p95": np.percentile(e, 95), 
        "max": e.max()
    }
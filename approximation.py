"""
Пункт 3.7 – Аппроксимация (E,N)->(B,L) по значащим пикселям TIFF.
Используется JAX, SGD с моментом, ранняя остановка.
"""

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap, random, value_and_grad
from functools import partial
from osgeo import gdal

# ------------------------------------------------------------
#  Инициализация MLP
# ------------------------------------------------------------
def init_mlp_params(rng_key, layer_sizes):
    params = []
    keys = random.split(rng_key, len(layer_sizes))
    for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
        w = random.normal(keys[0], (n_in, n_out)) * jnp.sqrt(2.0 / n_in)
        b = jnp.zeros((n_out,))
        params.append((w, b))
        keys = keys[1:]
    return params

def mlp_forward(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w_last, b_last = params[-1]
    return x @ w_last + b_last

# ------------------------------------------------------------
#  Функция потерь
# ------------------------------------------------------------
def mse_loss(params, x, y):
    y_pred = vmap(partial(mlp_forward, params))(x)
    return jnp.mean((y_pred - y) ** 2)

# ------------------------------------------------------------
#  SGD с моментом
# ------------------------------------------------------------
def init_momentum(params):
    return [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]

def momentum_update(params, grads, velocity, lr=1e-3, momentum=0.9):
    new_params, new_vel = [], []
    for (w, b), (dw, db), (vw, vb) in zip(params, grads, velocity):
        vw = momentum * vw + lr * dw
        vb = momentum * vb + lr * db
        w_new = w - vw
        b_new = b - vb
        new_params.append((w_new, b_new))
        new_vel.append((vw, vb))
    return new_params, new_vel

# ------------------------------------------------------------
#  Точное преобразование UTM -> (B,L) через GDAL
# ------------------------------------------------------------
def exact_bl(utm_points, utm_to_wgs84):
    lats, lons = [], []
    for E, N in utm_points:
        lon, lat, _ = utm_to_wgs84.TransformPoint(E, N)
        lats.append(lat)
        lons.append(lon)
    return np.column_stack([lats, lons])

# ------------------------------------------------------------
#  Получение значащих пикселей из TIFF
# ------------------------------------------------------------
def load_valid_pixels_from_tiff(tiff_path, geotf):
    """
    Возвращает координаты центров значащих пикселей (E,N).
    geotf — GeoTransform в формате GDAL (список из 6 элементов).
    Считаем значащими все пиксели, не равные 0 и не NoData.
    """
    ds = gdal.Open(tiff_path)
    if ds is None:
        raise FileNotFoundError(f"Не удалось открыть {tiff_path}")
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    ds = None

    # Маска значащих пикселей
    if nodata is not None:
        valid = (data != 0) & (data != nodata)
    else:
        valid = data != 0

    rows, cols = np.where(valid)
    # Координаты центров пикселей
    x_coords = geotf[0] + (cols + 0.5) * geotf[1] + (rows + 0.5) * geotf[2]
    y_coords = geotf[3] + (cols + 0.5) * geotf[4] + (rows + 0.5) * geotf[5]
    return np.column_stack([x_coords, y_coords])

# ------------------------------------------------------------
#  Генерация выборок из значащих пикселей
# ------------------------------------------------------------
def generate_datasets_from_tiff(tiff_path, geotf, coef = 8, max_points=256):
    all_points = load_valid_pixels_from_tiff(tiff_path, geotf)
    print(f"Найдено значащих пикселей: {len(all_points)}")
    if len(all_points) == 0:
        raise RuntimeError("Нет значащих пикселей в изображении")
    # Перемешиваем и ограничиваем количество
    idx = np.random.permutation(len(all_points))
    all_points = all_points[idx[:]]
    train_points = all_points[:max_points]
    test_points = all_points[:max_points*coef]
    return train_points, test_points

# ------------------------------------------------------------
#  Основная функция
# ------------------------------------------------------------
def run_approximation(rpc, height, utm_to_wgs84, wgs84_to_utm, grid_info):
    # Ищем tiff-файл в текущей папке
    import glob
    tiff_files = glob.glob("*.tif") + glob.glob("*.tiff")
    if not tiff_files:
        raise FileNotFoundError("TIFF не найден, положите его рядом с main.py")
    tiff_path = tiff_files[0]
    print(f"\nИспользую TIFF: {tiff_path}")

    geotf = grid_info['gt']

    print("\n" + "="*60)
    print("3.7 Аппроксимация обратного проектирования по значащим пикселям")
    print("="*60)

    # Генерация выборок
    train_points, test_points = generate_datasets_from_tiff(tiff_path, geotf, coef = 8, max_points = 256)
    print(f"Обучающих точек: {len(train_points)}")
    print(f"Тестовых точек: {len(test_points)}")

    y_train = exact_bl(train_points, utm_to_wgs84)
    y_test  = exact_bl(test_points, utm_to_wgs84)

    # Нормализация
    E_min, E_max = train_points[:,0].min(), train_points[:,0].max()
    N_min, N_max = train_points[:,1].min(), train_points[:,1].max()
    E_center, E_scale = (E_min + E_max)/2, (E_max - E_min)/2
    N_center, N_scale = (N_min + N_max)/2, (N_max - N_min)/2

    B_mean, B_std = y_train[:,0].mean(), y_train[:,0].std()
    L_mean, L_std = y_train[:,1].mean(), y_train[:,1].std()

    def norm_utm(points):
        E_n = (points[:,0] - E_center) / E_scale
        N_n = (points[:,1] - N_center) / N_scale
        return np.column_stack([E_n, N_n])

    def norm_bl(y):
        return np.column_stack([(y[:,0] - B_mean) / B_std,
                                 (y[:,1] - L_mean) / L_std])

    def denorm_bl(y_norm):
        return np.column_stack([y_norm[:,0] * B_std + B_mean,
                                 y_norm[:,1] * L_std + L_mean])

    X_tr = norm_utm(train_points)
    X_te = norm_utm(test_points)
    y_tr_n = norm_bl(y_train)
    y_te_n = norm_bl(y_test)

    # Сеть
    rng_key = random.PRNGKey(42)
    layer_sizes = [2, 8, 8, 8, 2]
    params = init_mlp_params(rng_key, layer_sizes)
    velocity = init_momentum(params)

    lr = 0.001
    momentum = 0.5
    epochs = 100000
    batch_size = 16

    @jit
    def predict(params, X):
        return vmap(partial(mlp_forward, params))(X)

    # Функция для расчёта пиксельной ошибки
    def compute_pixel_error(params, points_orig, X_norm):
        y_pred_norm = predict(params, jnp.array(X_norm))
        y_pred = denorm_bl(np.array(y_pred_norm))
        errs = []
        for (E, N), (Bp, Lp) in zip(points_orig, y_pred):
            n_pred, m_pred = rpc.forward(Bp, Lp, height)
            lon_true, lat_true, _ = utm_to_wgs84.TransformPoint(E, N)
            n_true, m_true = rpc.forward(lat_true, lon_true, height)
            errs.append(np.hypot(m_pred - m_true, n_pred - n_true))
        errs = np.array(errs)
        return errs.mean(), errs.max()

    print(f"Обучение MLP ({epochs} эпох, lr={lr}, momentum={momentum})...")
    best_test_mean = float('inf')
    best_params = params
    best_epoch = 0

    for epoch in range(epochs):
        perm = np.random.permutation(len(X_tr))
        X_shuf = jnp.array(X_tr[perm])
        y_shuf = jnp.array(y_tr_n[perm])
        total_loss = 0.0
        for i in range(0, len(X_shuf), batch_size):
            xb = X_shuf[i:i+batch_size]
            yb = y_shuf[i:i+batch_size]
            loss, grads = value_and_grad(mse_loss)(params, xb, yb)
            params, velocity = momentum_update(params, grads, velocity, lr, momentum)
            total_loss += loss * len(xb)

        if (epoch+1) % 100 == 0:
            train_mean, train_max = compute_pixel_error(params, train_points, X_tr)
            test_mean, test_max = compute_pixel_error(params, test_points, X_te)
            print(f"  эпоха {epoch+1:5d}: loss={total_loss/len(X_shuf):.8f} "
                  f"train_err(mean/max)={train_mean:.4f}/{train_max:.4f} px "
                  f"test_err(mean/max)={test_mean:.4f}/{test_max:.4f} px")
            if test_mean < best_test_mean:
                best_test_mean = test_mean
                best_params = params
                best_epoch = epoch+1

    params = best_params
    print(f"\nЛучшая модель на эпохе {best_epoch}, test_mean_error = {best_test_mean:.4f} px")

    # Итоговая оценка
    def compute_errors(points_orig, y_pred_norm):
        y_pred = denorm_bl(np.array(y_pred_norm))
        y_true = exact_bl(points_orig, utm_to_wgs84)
        rmse_B = np.sqrt(np.mean((y_pred[:,0]-y_true[:,0])**2))
        rmse_L = np.sqrt(np.mean((y_pred[:,1]-y_true[:,1])**2))
        pix_errs = []
        for (E,N), (Bp,Lp) in zip(points_orig, y_pred):
            n_pred, m_pred = rpc.forward(Bp, Lp, height)
            lon_true, lat_true, _ = utm_to_wgs84.TransformPoint(E, N)
            n_true, m_true = rpc.forward(lat_true, lon_true, height)
            pix_errs.append(np.hypot(m_pred-m_true, n_pred-n_true))
        pix_errs = np.array(pix_errs)
        rmse_pix = np.sqrt(np.mean(pix_errs**2))
        max_pix = np.max(pix_errs)
        return rmse_B, rmse_L, rmse_pix, max_pix

    results = {}
    for name, pts, X_n in [('train', train_points, X_tr),
                           ('test', test_points, X_te)]:
        y_pred_norm = predict(params, jnp.array(X_n))
        b, l, p, mp = compute_errors(pts, y_pred_norm)
        results[name] = (len(pts), b, l, p, mp)

    print("\nРезультаты аппроксимации:")
    print(f"{'Выборка':<10} {'N':<8} {'RMSE_B (град)':<15} {'RMSE_L (град)':<15} "
          f"{'RMSE пикс':<12} {'Макс. ошибка, пикс':<20}")
    for name in ['train', 'test']:
        n, b, l, p, mp = results[name]
        print(f"{name:<10} {n:<8} {b:<15.6e} {l:<15.6e} {p:<12.6f} {mp:<20.6f}")

    return results
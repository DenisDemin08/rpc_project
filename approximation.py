"""
Пункт 3.7 – Аппроксимация (E,N)->(B,L) по значащим пикселям TIFF.
PyTorch + CUDA, ранняя остановка, выбор оптимизатора SGD/Adam.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
from osgeo import gdal

# ------------------------------------------------------------
#  MLP на PyTorch
# ------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self, layer_sizes):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.activation = nn.Tanh()

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)   # без активации на выходе
        return x

# ------------------------------------------------------------
#  Нормализация
# ------------------------------------------------------------
def normalize_utm(points, E_center, E_scale, N_center, N_scale):
    E_n = (points[:, 0] - E_center) / E_scale
    N_n = (points[:, 1] - N_center) / N_scale
    return torch.stack([E_n, N_n], dim=1)

def normalize_bl(y, B_mean, B_std, L_mean, L_std):
    B_n = (y[:, 0] - B_mean) / B_std
    L_n = (y[:, 1] - L_mean) / L_std
    return torch.stack([B_n, L_n], dim=1)

def denormalize_bl(y_norm, B_mean, B_std, L_mean, L_std):
    B = y_norm[:, 0] * B_std + B_mean
    L = y_norm[:, 1] * L_std + L_mean
    return torch.stack([B, L], dim=1)

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
#  Загрузка значащих пикселей из TIFF
# ------------------------------------------------------------
def load_valid_pixels_from_tiff(tiff_path, geotf):
    ds = gdal.Open(tiff_path)
    if ds is None:
        raise FileNotFoundError(f"Не удалось открыть {tiff_path}")
    band = ds.GetRasterBand(1)
    data = band.ReadAsArray()
    nodata = band.GetNoDataValue()
    ds = None

    if nodata is not None:
        valid = (data != 0) & (data != nodata)
    else:
        valid = data != 0

    rows, cols = np.where(valid)
    x_coords = geotf[0] + (cols + 0.5) * geotf[1] + (rows + 0.5) * geotf[2]
    y_coords = geotf[3] + (cols + 0.5) * geotf[4] + (rows + 0.5) * geotf[5]
    return np.column_stack([x_coords, y_coords])

# ------------------------------------------------------------
#  Генерация выборок
# ------------------------------------------------------------
def generate_datasets_from_tiff(tiff_path, geotf, coef=8, max_points=16384):
    all_points = load_valid_pixels_from_tiff(tiff_path, geotf)
    print(f"Найдено значащих пикселей: {len(all_points)}")
    if len(all_points) == 0:
        raise RuntimeError("Нет значащих пикселей в изображении")
    idx = np.random.permutation(len(all_points))
    all_points = all_points[idx[:]]
    train_points = all_points[:max_points]
    test_points = all_points[:max_points * coef]
    return train_points, test_points

# ------------------------------------------------------------
#  Пиксельная ошибка (средняя / максимальная)
# ------------------------------------------------------------
def compute_pixel_error(model, points_orig, X_norm, rpc, height, utm_to_wgs84,
                        denorm_params, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
        y_pred_norm = model(X_t).cpu().numpy()
    B_mean, B_std, L_mean, L_std = denorm_params
    y_pred = y_pred_norm * np.array([B_std, L_std]) + np.array([B_mean, L_mean])

    errs = []
    for (E, N), (Bp, Lp) in zip(points_orig, y_pred):
        n_pred, m_pred = rpc.forward(Bp, Lp, height)
        lon_true, lat_true, _ = utm_to_wgs84.TransformPoint(E, N)
        n_true, m_true = rpc.forward(lat_true, lon_true, height)
        errs.append(np.hypot(m_pred - m_true, n_pred - n_true))
    errs = np.array(errs)
    return errs.mean(), errs.max()

# ------------------------------------------------------------
#  Вычисление градусных и пиксельных метрик
# ------------------------------------------------------------
def compute_metrics(model, points_orig, X_norm, y_true, denorm_params, rpc, height,
                    utm_to_wgs84, device):
    model.eval()
    with torch.no_grad():
        X_t = torch.as_tensor(X_norm, dtype=torch.float32, device=device)
        y_pred_norm = model(X_t).cpu().numpy()
    B_mean, B_std, L_mean, L_std = denorm_params
    y_pred = y_pred_norm * np.array([B_std, L_std]) + np.array([B_mean, L_mean])

    # градусные ошибки
    rmse_B = np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2))
    rmse_L = np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2))

    # пиксельные ошибки
    pix_errs = []
    for (E, N), (Bp, Lp) in zip(points_orig, y_pred):
        n_pred, m_pred = rpc.forward(Bp, Lp, height)
        lon_true, lat_true, _ = utm_to_wgs84.TransformPoint(E, N)
        n_true, m_true = rpc.forward(lat_true, lon_true, height)
        pix_errs.append(np.hypot(m_pred - m_true, n_pred - n_true))
    pix_errs = np.array(pix_errs)
    rmse_pix = np.sqrt(np.mean(pix_errs ** 2))
    max_pix = np.max(pix_errs)

    return rmse_B, rmse_L, rmse_pix, max_pix

# ------------------------------------------------------------
#  Основная функция
# ------------------------------------------------------------
def run_approximation(rpc, height, utm_to_wgs84, wgs84_to_utm, grid_info,
                      optimizer_type='sgd', lr=0.001, momentum=0.95, weight_decay=0.0,
                      epochs=100000, batch_size=8, patience=1000,
                      coef=4, max_points=16384*2, model_save_path='best_mlp.pth'):
    # Поиск TIFF
    tiff_files = glob.glob("*.tif") + glob.glob("*.tiff")
    if not tiff_files:
        raise FileNotFoundError("TIFF не найден, положите его рядом с main.py")
    tiff_path = tiff_files[0]
    print(f"\nИспользую TIFF: {tiff_path}")

    geotf = grid_info['gt']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Устройство: {device}")

    # Генерация данных
    train_points, test_points = generate_datasets_from_tiff(tiff_path, geotf,
                                                           coef=coef,
                                                           max_points=max_points)
    print(f"Обучающих точек: {len(train_points)}")
    print(f"Тестовых точек: {len(test_points)}")

    y_train = exact_bl(train_points, utm_to_wgs84)
    y_test = exact_bl(test_points, utm_to_wgs84)

    # Нормализация входов
    E_min, E_max = train_points[:, 0].min(), train_points[:, 0].max()
    N_min, N_max = train_points[:, 1].min(), train_points[:, 1].max()
    E_center = (E_min + E_max) / 2.0
    E_scale = (E_max - E_min) / 2.0
    N_center = (N_min + N_max) / 2.0
    N_scale = (N_max - N_min) / 2.0

    # Нормализация выходов (B, L)
    B_mean = y_train[:, 0].mean()
    B_std = y_train[:, 0].std()
    L_mean = y_train[:, 1].mean()
    L_std = y_train[:, 1].std()

    # Подготовка тензоров
    X_tr = np.column_stack([
        (train_points[:, 0] - E_center) / E_scale,
        (train_points[:, 1] - N_center) / N_scale
    ])
    X_te = np.column_stack([
        (test_points[:, 0] - E_center) / E_scale,
        (test_points[:, 1] - N_center) / N_scale
    ])
    y_tr_norm = np.column_stack([
        (y_train[:, 0] - B_mean) / B_std,
        (y_train[:, 1] - L_mean) / L_std
    ])
    y_te_norm = np.column_stack([
        (y_test[:, 0] - B_mean) / B_std,
        (y_test[:, 1] - L_mean) / L_std
    ])

    # Переводим в torch
    X_tr_t = torch.tensor(X_tr, dtype=torch.float32)
    y_tr_t = torch.tensor(y_tr_norm, dtype=torch.float32)
    X_te_t = torch.tensor(X_te, dtype=torch.float32)
    y_te_t = torch.tensor(y_te_norm, dtype=torch.float32)

    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Модель
    layer_sizes = [2, 64, 64, 64, 64, 2]
    model = MLP(layer_sizes).to(device)
    criterion = nn.MSELoss()

    # Оптимизатор
    if optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError("optimizer_type должен быть 'sgd' или 'adam'")

    # Параметры для денормализации
    denorm_params = (B_mean, B_std, L_mean, L_std)

    # Ранняя остановка
    best_val_metric = float('inf')
    best_epoch = 0
    patience_counter = 0
    best_state = None

    print(f"Обучение MLP ({epochs} эпох, optim={optimizer_type}, lr={lr}, batch={batch_size})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)

        avg_loss = total_loss / len(train_dataset)

        # Оценка каждые 100 эпох
        if epoch % 100 == 0:
            # Ошибки на train
            train_rmse_B, train_rmse_L, train_rmse_pix, train_max_pix = compute_metrics(
                model, train_points, X_tr, y_train, denorm_params,
                rpc, height, utm_to_wgs84, device)
            # Ошибки на test
            test_rmse_B, test_rmse_L, test_rmse_pix, test_max_pix = compute_metrics(
                model, test_points, X_te, y_test, denorm_params,
                rpc, height, utm_to_wgs84, device)

            print(f"Эпоха {epoch:5d}: loss={avg_loss:.6f} | "
                  f"train B RMSE={train_rmse_B:.6e}° L RMSE={train_rmse_L:.6e}° "
                  f"pix err mean={train_rmse_pix:.4f} max={train_max_pix:.4f} | "
                  f"test  B RMSE={test_rmse_B:.6e}° L RMSE={test_rmse_L:.6e}° "
                  f"pix err mean={test_rmse_pix:.4f} max={test_max_pix:.4f}")

            # Метрика для ранней остановки: сумма градусных RMSE на тесте
            val_metric = test_rmse_B + test_rmse_L
            if val_metric < best_val_metric:
                best_val_metric = val_metric
                best_epoch = epoch
                patience_counter = 0
                # Сохраняем лучшую модель
                best_state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_metric': best_val_metric,
                    'denorm_params': denorm_params,
                    'normalization': {
                        'E_center': E_center, 'E_scale': E_scale,
                        'N_center': N_center, 'N_scale': N_scale
                    }
                }
                torch.save(best_state, model_save_path)
                print(f"  -> сохранена лучшая модель (метрика B+L = {best_val_metric:.6e}°)")
            else:
                patience_counter += 1
                if patience_counter >= patience // 100:  # терпение в периодах по 100 эпох
                    print(f"Ранняя остановка на эпохе {epoch}: нет улучшений {patience_counter * 100} эпох.")
                    break

    # Загружаем лучшую модель
    if best_state is not None:
        model.load_state_dict(best_state['model_state_dict'])
        print(f"\nЛучшая модель с эпохи {best_epoch}, метрика B+L = {best_val_metric:.6e}°")
    else:
        print("\nЛучшая модель не найдена, используется последняя.")

    # Финальные метрики
    print("\nРезультаты аппроксимации:")
    print(f"{'Выборка':<10} {'N':<8} {'RMSE_B (град)':<15} {'RMSE_L (град)':<15} "
          f"{'RMSE пикс':<12} {'Макс. ошибка, пикс':<20}")
    for name, pts, X_n, y_true in [
        ('train', train_points, X_tr, y_train),
        ('test', test_points, X_te, y_test)
    ]:
        b, l, p, mp = compute_metrics(model, pts, X_n, y_true, denorm_params,
                                      rpc, height, utm_to_wgs84, device)
        print(f"{name:<10} {len(pts):<8} {b:<15.6e} {l:<15.6e} {p:<12.6f} {mp:<20.6f}")

    return {
        'train': (len(train_points), b, l, p, mp),
        'test': (len(test_points), b, l, p, mp)
    }
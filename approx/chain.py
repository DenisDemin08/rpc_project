import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ReferenceChain:
    def __init__(self, rpc, height, utm_to_wgs84, gt):
        self.rpc = rpc
        self.height = height
        self.utm_to_wgs84 = utm_to_wgs84
        self.gt = np.asarray(gt, dtype=float)

    def step1_affine(self, m_tr, n_tr):
        """(m_tr, n_tr) -> (E, N)"""
        m_tr = np.asarray(m_tr, dtype=float)
        n_tr = np.asarray(n_tr, dtype=float)
        E = self.gt[0] + m_tr * self.gt[1] + n_tr * self.gt[2]
        N = self.gt[3] + m_tr * self.gt[4] + n_tr * self.gt[5]
        return E, N

    def step2_utm_to_bl(self, E, N):
        """(E, N) -> (B, L) одним вызовом TransformPoints на весь массив."""
        E = np.asarray(E, dtype=float).ravel()
        N = np.asarray(N, dtype=float).ravel()

        pts = np.column_stack([E, N, np.zeros_like(E)])
        # Векторизованный вызов GDAL
        transformed = self.utm_to_wgs84.TransformPoints(pts.tolist())
        
        arr = np.asarray(transformed, dtype=float)
        lon = arr[:, 0]
        lat = arr[:, 1]
        return lat, lon

    def step3_forward(self, lat, lon):
        """(B, L) -> (m, n) через прямое RPC."""
        lat = np.asarray(lat, dtype=float).ravel()
        lon = np.asarray(lon, dtype=float).ravel()
        n, m = self.rpc.forward(lat, lon, self.height)
        return m, n

    def __call__(self, m_tr, n_tr):
        """Полная цепочка: (m_tr, n_tr) -> (N, 2), столбцы (m, n)."""
        E, N = self.step1_affine(m_tr, n_tr)
        lat, lon = self.step2_utm_to_bl(E, N)
        m, n = self.step3_forward(lat, lon)
        return np.column_stack([
            np.asarray(m, dtype=float).ravel(),
            np.asarray(n, dtype=float).ravel(),
        ])

# Функции для генерации выборок и графиков

def create_samples(rpc, height, utm_to_wgs84, grid_data, path="data/samples.npz"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    chain = ReferenceChain(rpc, height, utm_to_wgs84, grid_data["gt"])
    width, height_tr = int(grid_data["width_tr"]), int(grid_data["height_tr"])

    # 1. Train (шаг 16)
    m_tr_train, n_tr_train = np.meshgrid(np.arange(0, width, 16), np.arange(0, height_tr, 16))
    m_tr_train, n_tr_train = m_tr_train.ravel(), n_tr_train.ravel()

    # 2. Test (шаг 4, сдвиг 2)
    m_tr_test, n_tr_test = np.meshgrid(np.arange(2, width, 4), np.arange(2, height_tr, 4))
    m_tr_test, n_tr_test = m_tr_test.ravel(), n_tr_test.ravel()

    # 3. Border (периметр)
    m_b1, n_b1 = np.arange(0, width, 2), np.zeros_like(np.arange(0, width, 2))
    m_b2, n_b2 = np.arange(0, width, 2), np.full_like(np.arange(0, width, 2), height_tr - 1)
    n_b3 = np.arange(2, height_tr - 2, 2)
    m_b3, m_b4 = np.zeros_like(n_b3), np.full_like(n_b3, width - 1)
    m_tr_border = np.concatenate([m_b1, m_b2, m_b3, m_b4])
    n_tr_border = np.concatenate([n_b1, n_b2, n_b3, n_b3])

    specs = {"train": (m_tr_train, n_tr_train), 
             "test": (m_tr_test, n_tr_test), 
             "border": (m_tr_border, n_tr_border)}

    arrays = {"gt": np.asarray(grid_data["gt"]), "width_tr": width, "height_tr": height_tr}
    info = {}

    for name, (m_tr, n_tr) in specs.items():
        mn = chain(m_tr, n_tr)
        m_ref, n_ref = mn[:, 0], mn[:, 1]
        
        inside = (m_ref >= 0.5) & (m_ref < rpc.width - 0.5) & (n_ref >= 0.5) & (n_ref < rpc.height - 0.5)

        arrays[f"{name}_m_tr"] = m_tr
        arrays[f"{name}_n_tr"] = n_tr
        arrays[f"{name}_m"] = m_ref
        arrays[f"{name}_n"] = n_ref
        arrays[f"{name}_inside"] = inside

        info[name] = {"total": len(m_tr), "inside": int(inside.sum()), 
                      "frac": float(inside.sum()/len(m_tr)) if len(m_tr) else 0}

    np.savez_compressed(path, **arrays)
    return info

def plot_error_map(m_tr, n_tr, err, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(m_tr, n_tr, c=err, s=4, cmap="viridis")
    ax.set_xlabel("m_tr"); ax.set_ylabel("n_tr")
    ax.set_aspect("equal"); ax.invert_yaxis()
    cb = fig.colorbar(sc, ax=ax); cb.set_label("Ошибка, пикс.")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)

def measure_timing(chain, width, height, repeats=5):
    rng = np.random.default_rng(42)
    sizes = (10**3, 10**4, 10**5, 10**6)
    
    # Прогрев
    _ = chain(rng.uniform(0, width, 100), rng.uniform(0, height, 100))
    
    results = []
    for n_pts in sizes:
        times = []
        for _ in range(repeats):
            m, n = rng.uniform(0, width, n_pts), rng.uniform(0, height, n_pts)
            t0 = time.perf_counter()
            _ = chain(m, n)
            times.append(time.perf_counter() - t0)
        results.append((n_pts, float(np.median(times))))

    # Шаги для N=10^6
    m, n = rng.uniform(0, width, 10**6), rng.uniform(0, height, 10**6)
    E, N = chain.step1_affine(m, n)
    lat, lon = chain.step2_utm_to_bl(E, N)

    def med_time(func, *args):
        ts = [time.perf_counter() - (t0:=time.perf_counter()) or time.perf_counter() - t0 for _ in range(repeats) if func(*args) or True] # упрощенно
        ts = []
        for _ in range(repeats):
            t0 = time.perf_counter(); func(*args); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))

    t1 = med_time(chain.step1_affine, m, n)
    t2 = med_time(chain.step2_utm_to_bl, E, N)
    t3 = med_time(chain.step3_forward, lat, lon)
    total = t1 + t2 + t3

    return results, {"t1": t1, "t2": t2, "t3": t3, "fracs": [100*t/total for t in (t1, t2, t3)]}

def plot_timing(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Ns, ts = [r[0] for r in results], [r[1] for r in results]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(Ns, ts, "o-")
    ax.set_xlabel("N, число точек"); ax.set_ylabel("Время, с")
    ax.grid(True, which="both", linestyle=":")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)
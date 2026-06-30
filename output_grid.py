import numpy as np

def compute_pixel_size_in_utm(rpc, height, wgs84_to_utm):
    m_c = rpc.width / 2.0
    n_c = rpc.height / 2.0
    # inverse(n, m, height) -> lat, lon, it, ok
    lat_c, lon_c, _, _ = rpc.inverse(n_c, m_c, height)
    lat_m1, lon_m1, _, _ = rpc.inverse(n_c, m_c + 1, height)
    lat_n1, lon_n1, _, _ = rpc.inverse(n_c + 1, m_c, height)

    E_c, N_c, _ = wgs84_to_utm.TransformPoint(lon_c, lat_c)
    E_m1, N_m1, _ = wgs84_to_utm.TransformPoint(lon_m1, lat_m1)
    E_n1, N_n1, _ = wgs84_to_utm.TransformPoint(lon_n1, lat_n1)

    dist_m = np.hypot(E_m1 - E_c, N_m1 - N_c)
    dist_n = np.hypot(E_n1 - E_c, N_n1 - N_c)
    return float(dist_m), float(dist_n)

def run(rpc, height, utm_to_wgs84, wgs84_to_utm):
    w, h = rpc.width, rpc.height

    # Углы: (19,0), (2533,23), (2514,2522), (0,2499)
    corners = [
        (19, 0),
        (0, 2499),
        (2514, 2522),
        (2533, 23)
    ]
    center = (w / 2.0, h / 2.0)
    all_points = corners + [center]
    labels = ["левый верхний", "правый верхний", "правый нижний",
              "левый нижний", "центр"]

    print("=" * 60)
    print("3.4 Задание выходной сетки (UTM)")
    print("=" * 60)

    print("\nКоординаты опорных точек в UTM:")
    print(f"{'Точка':<20} {'m':<10} {'n':<10} {'B':<14} {'L':<14} "
          f"{'E (UTM)':<14} {'N (UTM)':<14}")
    utm_points = []
    for label, (m, n) in zip(labels, all_points):
        lat, lon, _, _ = rpc.inverse(n, m, height)      # исправлено
        E, N, _ = wgs84_to_utm.TransformPoint(lon, lat)
        utm_points.append((E, N))
        print(f"{label:<20} {m:<10.3f} {n:<10.3f} {lat:<14.8f} "
              f"{lon:<14.8f} {E:<14.3f} {N:<14.3f}")

    E_vals = [p[0] for p in utm_points]
    N_vals = [p[1] for p in utm_points]
    E_min, E_max = min(E_vals), max(E_vals)
    N_min, N_max = min(N_vals), max(N_vals)
    print(f"\nГраницы в UTM: E=[{E_min:.3f}, {E_max:.3f}] м, "
          f"N=[{N_min:.3f}, {N_max:.3f}] м")

    res_E, res_N = compute_pixel_size_in_utm(rpc, height, wgs84_to_utm)
    print(f"Размер пикселя по центральной точке: res_E = {res_E:.6f} м, "
          f"res_N = {res_N:.6f} м")

    width_tr = int(np.ceil((E_max - E_min) / res_E)) + 1
    height_tr = int(np.ceil((N_max - N_min) / res_N)) + 1
    print(f"Размеры выходного изображения: {width_tr} x {height_tr} пикселей")

    gt = [float(E_min), float(res_E), 0.0,
          float(N_max), 0.0, -float(res_N)]
    print(f"Геотрансформация: {gt}")

    test_output = [
        (0, 0, "левый верхний"),
        (width_tr - 1, 0, "правый верхний"),
        (0, height_tr - 1, "левый нижний"),
        (width_tr - 1, height_tr - 1, "правый нижний"),
        (width_tr // 2, height_tr // 2, "центр")
    ]
    print("\nПроверка покрытия выходной сетки "
          "(попадание в исходное изображение):")
    print(f"{'Точка':<20} {'m_tr':<8} {'n_tr':<8} {'E':<12} {'N':<12} "
          f"{'B':<14} {'L':<14} {'m':<10} {'n':<10} {'внутри':<8}")
    for m_tr, n_tr, lbl in test_output:                 # имя исправлено
        E = gt[0] + m_tr * gt[1] + n_tr * gt[2]
        N = gt[3] + m_tr * gt[4] + n_tr * gt[5]
        lon, lat, _ = utm_to_wgs84.TransformPoint(E, N)
        # forward(lat, lon, height) -> n, m
        n_est, m_est = rpc.forward(lat, lon, height)    # исправлено
        inside = (0.5 <= m_est < w - 0.5) and (0.5 <= n_est < h - 0.5)
        print(f"{lbl:<20} {m_tr:<8} {n_tr:<8} {E:<12.3f} {N:<12.3f} "
              f"{lat:<14.8f} {lon:<14.8f} {m_est:<10.3f} {n_est:<10.3f} "
              f"{'да' if inside else 'нет'}")

    return {
        'E_min': E_min, 'E_max': E_max,
        'N_min': N_min, 'N_max': N_max,
        'res_E': res_E, 'res_N': res_N,
        'width_tr': width_tr, 'height_tr': height_tr,
        'gt': gt
    }
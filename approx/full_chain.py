def run(rpc, height, utm_to_wgs84, grid_data):
    width_tr = grid_data['width_tr']
    height_tr = grid_data['height_tr']
    gt = grid_data['gt']

    step = max(1, min(width_tr // 20, height_tr // 20))
    m_tr_list = list(range(0, width_tr, step))
    n_tr_list = list(range(0, height_tr, step))
    total_points = len(m_tr_list) * len(n_tr_list)

    inside_count = 0
    m_vals, n_vals = [], []

    for m_tr in m_tr_list:
        for n_tr in n_tr_list:
            E = gt[0] + m_tr * gt[1] + n_tr * gt[2]
            N = gt[3] + m_tr * gt[4] + n_tr * gt[5]
            lon, lat, _ = utm_to_wgs84.TransformPoint(E, N)
            m_est, n_est = rpc.forward_mn(lat, lon, height)
            if (0.5 <= m_est < rpc.width-0.5) and (0.5 <= n_est < rpc.height-0.5):
                inside_count += 1
                m_vals.append(m_est)
                n_vals.append(n_est)

    print("=" * 60)
    print("3.5 Полная цепочка преобразований (m_tr,n_tr) -> (E,N) -> (B,L) -> (m,n)")
    print("=" * 60)
    print(f"Проверено точек: {total_points}")
    print(f"Точек внутри исходного изображения: {inside_count}")
    print(f"Точек вне: {total_points - inside_count}")
    if m_vals:
        print(f"min m = {min(m_vals):.3f}, max m = {max(m_vals):.3f}")
        print(f"min n = {min(n_vals):.3f}, max n = {max(n_vals):.3f}")
    else:
        print("Нет точек внутри изображения – проверьте параметры выходной сетки.")
    return inside_count, total_points
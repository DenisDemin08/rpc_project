import numpy as np
from approx.chain import ReferenceChain

def run(rpc, height, utm_to_wgs84, grid_data):
    width_tr = grid_data['width_tr']
    height_tr = grid_data['height_tr']
    gt = grid_data['gt']

    print("=" * 60)
    print("3.5 Полная цепочка преобразований (m_tr,n_tr) -> (E,N) -> (B,L) -> (m,n)")
    print("=" * 60)

    # Генерируем сетку точек
    step = max(1, min(width_tr // 20, height_tr // 20))
    m_tr_list = np.arange(0, width_tr, step)
    n_tr_list = np.arange(0, height_tr, step)
    mm, nn = np.meshgrid(m_tr_list, n_tr_list)
    m_tr = mm.ravel()
    n_tr = nn.ravel()
    total_points = len(m_tr)

    # Используем векторизованную цепочку
    chain = ReferenceChain(rpc, height, utm_to_wgs84, gt)
    mn = chain(m_tr, n_tr)
    m_vals = mn[:, 0]
    n_vals = mn[:, 1]

    inside = (m_vals >= 0.5) & (m_vals < rpc.width - 0.5) & (n_vals >= 0.5) & (n_vals < rpc.height - 0.5)
    inside_count = int(inside.sum())

    print(f"Проверено точек: {total_points}")
    print(f"Точек внутри исходного изображения: {inside_count}")
    print(f"Точек вне: {total_points - inside_count}")
    
    if inside_count > 0:
        print(f"min m = {m_vals[inside].min():.3f}, max m = {m_vals[inside].max():.3f}")
        print(f"min n = {n_vals[inside].min():.3f}, max n = {n_vals[inside].max():.3f}")
    else:
        print("Нет точек внутри изображения – проверьте параметры выходной сетки.")
    
    return inside_count, total_points
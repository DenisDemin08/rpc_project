import numpy as np
from osgeo import gdal

gdal.UseExceptions()

def run(rpc, height, utm_to_wgs84, grid_data):
    tmp_ds = gdal.GetDriverByName('MEM').Create('', 1, 1, 1)
    rpc_meta = {
        'LINE_OFF': str(rpc.LINE_OFF), 'SAMP_OFF': str(rpc.SAMP_OFF),
        'LAT_OFF': str(rpc.LAT_OFF), 'LONG_OFF': str(rpc.LONG_OFF),
        'HEIGHT_OFF': str(rpc.HEIGHT_OFF),
        'LINE_SCALE': str(rpc.LINE_SCALE), 'SAMP_SCALE': str(rpc.SAMP_SCALE),
        'LAT_SCALE': str(rpc.LAT_SCALE), 'LONG_SCALE': str(rpc.LONG_SCALE),
        'HEIGHT_SCALE': str(rpc.HEIGHT_SCALE),
        'LINE_NUM_COEFF': ' '.join(map(str, rpc.c)),
        'LINE_DEN_COEFF': ' '.join(map(str, rpc.d)),
        'SAMP_NUM_COEFF': ' '.join(map(str, rpc.e)),
        'SAMP_DEN_COEFF': ' '.join(map(str, rpc.f)),
    }
    tmp_ds.SetMetadata(rpc_meta, 'RPC')
    transformer_rpc = gdal.Transformer(tmp_ds, None, ['METHOD=RPC'])

    width_tr = grid_data['width_tr']
    height_tr = grid_data['height_tr']
    gt = grid_data['gt']
    step = max(1, min(width_tr // 20, height_tr // 20))
    m_tr_list = list(range(0, width_tr, step))
    n_tr_list = list(range(0, height_tr, step))

    errors = []
    errors_m = []
    errors_n = []
    max_err = 0.0
    max_point = None

    for m_tr in m_tr_list:
        for n_tr in n_tr_list:
            E = gt[0] + m_tr * gt[1] + n_tr * gt[2]
            N = gt[3] + m_tr * gt[4] + n_tr * gt[5]
            lon, lat, _ = utm_to_wgs84.TransformPoint(E, N)
            m_own, n_own = rpc.forward_mn(lat, lon, height)
            if (0.5 <= m_own < rpc.width-0.5) and (0.5 <= n_own < rpc.height-0.5):
                succ, pnt = transformer_rpc.TransformPoint(1, lon, lat, height)
                m_gdal, n_gdal = pnt[0], pnt[1]
                err = np.hypot(m_own - m_gdal, n_own - n_gdal)
                errors.append(err)
                errors_m.append(m_own - m_gdal)
                errors_n.append(n_own - n_gdal)
                if err > max_err:
                    max_err = err
                    max_point = (m_tr, n_tr, m_own, n_own, m_gdal, n_gdal)

    K = len(errors)
    if K == 0:
        print("Нет общих точек внутри изображения – невозможно сравнение.")
        return

    rmse_m = np.sqrt(np.mean(np.array(errors_m)**2))
    rmse_n = np.sqrt(np.mean(np.array(errors_n)**2))
    rmse_total = np.sqrt(np.mean(np.array(errors)**2))
    mean_err = np.mean(errors)

    print("=" * 60)
    print("3.6 Сравнение полной цепочки (собственная vs GDAL)")
    print("=" * 60)
    print(f"Количество точек сравнения: {K}")
    print(f"RMSE по m : {rmse_m:.6f} пикс.")
    print(f"RMSE по n : {rmse_n:.6f} пикс.")
    print(f"RMSE общая: {rmse_total:.6f} пикс.")
    print(f"Максимальная ошибка: {max_err:.6f} пикс.")
    print(f"Средняя ошибка: {mean_err:.6f} пикс.")
    if max_point:
        print("\nТочка с максимальной ошибкой:")
        print(f"  (m_tr, n_tr) = ({max_point[0]}, {max_point[1]})")
        print(f"  собственная: (m,n) = ({max_point[2]:.3f}, {max_point[3]:.3f})")
        print(f"  GDAL:        (m,n) = ({max_point[4]:.3f}, {max_point[5]:.3f})")
        print(f"  ошибка = {max_err:.6f} пикс.")
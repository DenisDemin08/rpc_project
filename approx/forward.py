import numpy as np
from osgeo import gdal

gdal.UseExceptions()

def run(rpc, height):
    center_lat = rpc.LAT_OFF
    center_lon = rpc.LONG_OFF
    step = 0.01
    lats = [center_lat - step, center_lat, center_lat + step]
    lons = [center_lon - step, center_lon, center_lon + step]

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
    transformer = gdal.Transformer(tmp_ds, None, ['METHOD=RPC'])

    print("=" * 60)
    print("3.1 Прямое RPC-преобразование и сравнение с GDAL")
    print("=" * 60)
    print(f"{'№':<4} {'B (широта)':<14} {'L (долгота)':<14} "
          f"{'m_own':<10} {'n_own':<10} {'m_gdal':<10} {'n_gdal':<10} {'err_pix':<10}")
    idx = 1
    for lat in lats:
        for lon in lons:
            m_own, n_own = rpc.forward_mn(lat, lon, height)
            succ, pnt = transformer.TransformPoint(1, lon, lat, height)
            m_gdal, n_gdal = pnt[0], pnt[1]
            err = np.hypot(m_own - m_gdal, n_own - n_gdal)
            print(f"{idx:<4} {lat:<14.6f} {lon:<14.6f} "
                  f"{m_own:<10.3f} {n_own:<10.3f} {m_gdal:<10.3f} {n_gdal:<10.3f} {err:<10.6f}")
            idx += 1
    print()
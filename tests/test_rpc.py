import numpy as np
import pytest
from osgeo import gdal
from approx.rpc_model import RPC

gdal.UseExceptions()

@pytest.fixture
def rpc():
    # Файл rpc.txt должен лежать в папке data/
    return RPC('data/rpc.txt')

@pytest.fixture
def gdal_transformer(rpc):
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
    return gdal.Transformer(tmp_ds, None, ['METHOD=RPC'])

def test_forward_vs_gdal(rpc, gdal_transformer):
    """Прямое RPC совпадает с GDAL на 100 случайных точках."""
    np.random.seed(42)
    lats = rpc.LAT_OFF + np.random.rand(100) * rpc.LAT_SCALE
    lons = rpc.LONG_OFF + np.random.rand(100) * rpc.LONG_SCALE
    heights = rpc.HEIGHT_OFF + np.random.rand(100) * rpc.HEIGHT_SCALE
    
    # Векторизованный вызов
    n_own, m_own = rpc.forward(lats, lons, heights)
    
    m_gdal = []
    n_gdal = []
    for lat, lon, h in zip(lats, lons, heights):
        succ, pnt = gdal_transformer.TransformPoint(1, lon, lat, h)
        m_gdal.append(pnt[0])
        n_gdal.append(pnt[1])
        
    m_gdal = np.array(m_gdal)
    n_gdal = np.array(n_gdal)
    
    # Проверка точности
    assert np.allclose(m_own, m_gdal, atol=1e-6), "Ошибка по m превышает 1e-6"
    assert np.allclose(n_own, n_gdal, atol=1e-6), "Ошибка по n превышает 1e-6"

def test_inverse_roundtrip(rpc):
    """Обратное RPC, а затем прямое возвращают исходный пиксель."""
    np.random.seed(42)
    m = np.random.uniform(0, rpc.width, 100)
    n = np.random.uniform(0, rpc.height, 100)
    height = rpc.HEIGHT_OFF
    
    for i in range(100):
        lat, lon, iters, ok = rpc.inverse(n[i], m[i], height)
        assert ok, f"Метод Ньютона не сошелся для точки ({m[i]}, {n[i]})"
        
        n_back, m_back = rpc.forward(lat, lon, height)
        assert np.isclose(n_back, n[i], atol=1e-6), "Ошибка round-trip по n"
        assert np.isclose(m_back, m[i], atol=1e-6), "Ошибка round-trip по m"
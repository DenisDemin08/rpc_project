from osgeo import osr

def determine_utm_zone(lat, lon):
    zone = int((lon + 180) // 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    hemisphere = 'north' if lat >= 0 else 'south'
    return zone, hemisphere, epsg

def create_transformers(epsg):
    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromEPSG(epsg)
    srs_wgs84 = osr.SpatialReference()
    srs_wgs84.ImportFromEPSG(4326)
    srs_utm.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    srs_wgs84.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    utm_to_wgs84 = osr.CoordinateTransformation(srs_utm, srs_wgs84)
    wgs84_to_utm = osr.CoordinateTransformation(srs_wgs84, srs_utm)
    return utm_to_wgs84, wgs84_to_utm

def run(rpc, height):
    m_center = rpc.width / 2.0
    n_center = rpc.height / 2.0
    lat_center, lon_center, _, _ = rpc.inverse_mn(m_center, n_center, height)

    print("=" * 60)
    print("3.3 Определение зоны UTM, EPSG, проверка преобразований")
    print("=" * 60)
    print(f"Центр изображения: m={m_center:.1f}, n={n_center:.1f}")
    print(f"Геодезические координаты центра: B={lat_center:.10f}°, L={lon_center:.10f}°")

    zone, hem, epsg = determine_utm_zone(lat_center, lon_center)
    print(f"\nЗона UTM: {zone} ({hem})")
    print(f"EPSG код: {epsg}")

    utm_to_wgs84, wgs84_to_utm = create_transformers(epsg)

    test_points = [
        (lat_center, lon_center),
        (lat_center + 0.01, lon_center + 0.01),
        (lat_center - 0.01, lon_center - 0.01),
    ]
    print("\nПроверка преобразований (B,L) -> (E,N) -> (B',L'):")
    print(f"{'№':<4} {'B':<14} {'L':<14} {'E':<14} {'N':<14} {'B\'':<14} {'L\'':<14} {'ошибка B°':<12} {'ошибка L°':<12}")
    for i, (B, L) in enumerate(test_points, 1):
        E, N, _ = wgs84_to_utm.TransformPoint(L, B)
        L2, B2, _ = utm_to_wgs84.TransformPoint(E, N)
        err_B = B - B2
        err_L = L - L2
        print(f"{i:<4} {B:<14.8f} {L:<14.8f} {E:<14.3f} {N:<14.3f} {B2:<14.8f} {L2:<14.8f} {err_B:<12.3e} {err_L:<12.3e}")

    return {
        'epsg': epsg,
        'utm_to_wgs84': utm_to_wgs84,
        'wgs84_to_utm': wgs84_to_utm,
        'lat_center': lat_center,
        'lon_center': lon_center
    }
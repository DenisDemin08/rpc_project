import numpy as np

def run(rpc, height):
    w, h = rpc.width, rpc.height
    m_vals = np.linspace(0.5, w - 0.5, 5)
    n_vals = np.linspace(0.5, h - 0.5, 5)

    print("=" * 60)
    print("3.2 Обратное RPC -> round-trip проверка")
    print("=" * 60)
    print(f"{'№':<4} {'m':<8} {'n':<8} {'B':<16} {'L':<16} "
          f"{'m\'':<8} {'n\'':<8} {'err_pix':<12} {'Iter':<6} {'Status'}")
    idx = 1
    for n in n_vals:
        for m in m_vals:
            lat, lon, iters, ok = rpc.inverse_mn(m, n, height)
            m_back, n_back = rpc.forward_mn(lat, lon, height)
            err = np.hypot(m_back - m, n_back - n)
            status = 'OK' if ok else 'FAIL'
            print(f"{idx:<4} {m:<8.3f} {n:<8.3f} {lat:<16.10f} {lon:<16.10f} "
                  f"{m_back:<8.3f} {n_back:<8.3f} {err:<12.3e} {iters:<6} {status}")
            idx += 1

    corners = [(19, 0), (2533, 23), (2514, 2522), (0, 2499)]
    print("\nУглы изображения:")
    for m_int, n_int in corners:
        m = m_int + 0.5
        n = n_int + 0.5
        lat, lon, it, ok = rpc.inverse_mn(m, n, height)
        print(f"Угол (m={m_int}, n={n_int}): B={lat:.10f}, L={lon:.10f}, сходимость: {ok}, итераций: {it}")
import numpy as np

class RPC:
    def __init__(self, filepath):
        self._parse(filepath)
        self.width = 2368
        self.height = 2644
        self._init_derivatives()

    def _parse(self, path):
        data = {}
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if ':' in line:
                    key, val = line.split(':')
                    key = key.strip()
                    val = val.strip().split()[0]
                    data[key] = float(val)
        self.LINE_OFF = data['LINE_OFF']
        self.SAMP_OFF = data['SAMP_OFF']
        self.LAT_OFF = data['LAT_OFF']
        self.LONG_OFF = data['LONG_OFF']
        self.HEIGHT_OFF = data['HEIGHT_OFF']
        self.LINE_SCALE = data['LINE_SCALE']
        self.SAMP_SCALE = data['SAMP_SCALE']
        self.LAT_SCALE = data['LAT_SCALE']
        self.LONG_SCALE = data['LONG_SCALE']
        self.HEIGHT_SCALE = data['HEIGHT_SCALE']
        self.c = np.array([data[f'LINE_NUM_COEFF_{i}'] for i in range(1, 21)])
        self.d = np.array([data[f'LINE_DEN_COEFF_{i}'] for i in range(1, 21)])
        self.e = np.array([data[f'SAMP_NUM_COEFF_{i}'] for i in range(1, 21)])
        self.f = np.array([data[f'SAMP_DEN_COEFF_{i}'] for i in range(1, 21)])

    def _monomials(self, P, L, H):
        return np.array([
            np.ones_like(P), L, P, H,
            L*P, L*H, P*H,
            L**2, P**2, H**2,
            P*L*H, L**3, L*P**2, L*H**2,
            L**2*P, P**3, P*H**2, L**2*H,
            P**2*H, H**3
        ])

    def _init_derivatives(self):
        def du_dP(P, L, H):
            return np.array([
                0,0,1,0, L,0,H,0,2*P,0,
                L*H,0,2*P*L,0,L**2,3*P**2, H**2,0,2*P*H,0
            ])
        def du_dL(P, L, H):
            return np.array([
                0,1,0,0, P,H,0,2*L,0,0,
                P*H,3*L**2,P**2,H**2,2*L*P,0,0,2*L*H,0,0
            ])
        self._du_dP = du_dP
        self._du_dL = du_dL

    def normalize(self, lat, lon, height):
        lat, lon, height = np.broadcast_arrays(lat, lon, height)
        P = (lat - self.LAT_OFF) / self.LAT_SCALE
        L = (lon - self.LONG_OFF) / self.LONG_SCALE
        H = (height - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        return P, L, H

    def forward(self, lat, lon, height):
        """Прямой RPC → (n, m) – координаты строки и столбца (центры пикселей)"""
        P, L, H = self.normalize(lat, lon, height)
        u = self._monomials(P, L, H)
        Y = np.dot(self.c, u) / np.dot(self.d, u)
        X = np.dot(self.e, u) / np.dot(self.f, u)

        n = Y * self.LINE_SCALE + self.LINE_OFF + 0.5
        m = X * self.SAMP_SCALE + self.SAMP_OFF + 0.5
        return n, m

    def forward_mn(self, lat, lon, height):
        """Возвращает (m, n) – столбец, строка"""
        n, m = self.forward(lat, lon, height)
        return m, n

    def forward_with_jacobian(self, lat, lon, height):
        P, L, H = self.normalize(lat, lon, height)
        u = self._monomials(P, L, H)
        du_dP = self._du_dP(P, L, H)
        du_dL = self._du_dL(P, L, H)

        num_l = np.dot(self.c, u)
        den_l = np.dot(self.d, u)
        Y = num_l / den_l
        dY_dP = (np.dot(self.c, du_dP)*den_l - num_l*np.dot(self.d, du_dP)) / (den_l**2)
        dY_dL = (np.dot(self.c, du_dL)*den_l - num_l*np.dot(self.d, du_dL)) / (den_l**2)

        num_s = np.dot(self.e, u)
        den_s = np.dot(self.f, u)
        X = num_s / den_s
        dX_dP = (np.dot(self.e, du_dP)*den_s - num_s*np.dot(self.f, du_dP)) / (den_s**2)
        dX_dL = (np.dot(self.e, du_dL)*den_s - num_s*np.dot(self.f, du_dL)) / (den_s**2)

        dLat = 1.0 / self.LAT_SCALE
        dLon = 1.0 / self.LONG_SCALE
        dn_dlat = self.LINE_SCALE * dY_dP * dLat
        dn_dlon = self.LINE_SCALE * dY_dL * dLon
        dm_dlat = self.SAMP_SCALE * dX_dP * dLat
        dm_dlon = self.SAMP_SCALE * dX_dL * dLon

        J = np.array([[dn_dlat, dn_dlon],
                      [dm_dlat, dm_dlon]])
        
        n = Y * self.LINE_SCALE + self.LINE_OFF + 0.5
        m = X * self.SAMP_SCALE + self.SAMP_OFF + 0.5
        return n, m, J

    def inverse(self, n, m, height, lat_init=None, lon_init=None, max_iter=100, tol=1e-6):
        """Обратный RPC: (n,m) – центры пикселей → (lat, lon)"""
        if lat_init is None or lon_init is None:
            lat_init = self.LAT_OFF
            lon_init = self.LONG_OFF
        lat, lon = lat_init, lon_init
        for it in range(max_iter):
            n_est, m_est, J = self.forward_with_jacobian(lat, lon, height)
            res = np.array([n - n_est, m - m_est])
            if np.max(np.abs(res)) < tol:
                return lat, lon, it+1, True
            try:
                delta = np.linalg.solve(J, res)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(J, res, rcond=None)[0]
            lat += delta[0]
            lon += delta[1]
        return lat, lon, max_iter, False

    def inverse_mn(self, m, n, height, lat_init=None, lon_init=None, max_iter=100, tol=1e-6):
        """Обертка inverse с порядком (m,n)"""
        return self.inverse(n, m, height, lat_init, lon_init, max_iter, tol)
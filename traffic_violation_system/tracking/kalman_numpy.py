# tracking/kalman_numpy.py
# Pure-numpy Kalman filter — no filterpy required
# Drop-in replacement for filterpy.kalman.KalmanFilter

import numpy as np


class KalmanFilter:
    """
    Minimal Kalman filter using only numpy.
    Matches the filterpy API used in tracker.py so the tracker
    needs zero changes.

    State vector x: [cx, cy, s, r, vx, vy, vs]
      cx, cy = centre of bbox
      s      = scale (area)
      r      = aspect ratio (w/h)
      vx, vy = velocity of centre
      vs     = velocity of scale
    """

    def __init__(self, dim_x: int, dim_z: int):
        self.dim_x = dim_x
        self.dim_z = dim_z

        # State / observation matrices (set by caller)
        self.F = np.eye(dim_x)          # State transition
        self.H = np.zeros((dim_z, dim_x))  # Observation
        self.R = np.eye(dim_z)          # Measurement noise
        self.Q = np.eye(dim_x)          # Process noise
        self.P = np.eye(dim_x)          # Covariance estimate

        # State estimate
        self.x = np.zeros((dim_x, 1))

    # ── Core KF steps ──────────────────────────────────────────────────────────

    def predict(self):
        """x = F·x,  P = F·P·Fᵀ + Q"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z: np.ndarray):
        """
        z: measurement vector, shape (dim_z, 1)
        Standard Kalman update equations.
        """
        if z is None:
            return

        # Innovation
        y = z - self.H @ self.x

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        self.x = self.x + K @ y

        # Covariance update  (Joseph form for numerical stability)
        I_KH = np.eye(self.dim_x) - K @ self.H
        self.P = I_KH @ self.P

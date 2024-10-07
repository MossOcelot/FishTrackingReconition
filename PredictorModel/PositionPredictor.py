import torch

class KalmanFilter:
    def __init__(self, dt, process_noise, measurement_noise, initial_position):
        # Time step
        self.dt = dt

        # State vector [x, y, vx, vy] as a tensor
        self.state = torch.tensor([initial_position[0], initial_position[1], 0, 0], dtype=torch.float32)

        # State transition matrix (F)
        self.F = torch.tensor([[1, 0, self.dt, 0],
                               [0, 1, 0, self.dt],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=torch.float32)

        # Process noise covariance (Q)
        self.Q = torch.eye(4) * process_noise

        # Measurement matrix (H) maps the state to position measurements [x, y]
        self.H = torch.tensor([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=torch.float32)

        # Measurement noise covariance (R)
        self.R = torch.eye(2) * measurement_noise

        # Covariance matrix (P)
        self.P = torch.eye(4)

    def predict(self):
        # Predict the next state
        self.state = self.F @ self.state

        # Predict the covariance matrix
        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[:2]  # Return the predicted position [x, y]

    def update(self, z):
        # z is the new measurement [x, y] as a tensor
        z = torch.tensor(z, dtype=torch.float32)

        # Innovation (residual)
        y = z - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ torch.linalg.inv(S)

        # Update the state
        self.state = self.state + K @ y

        # Update the covariance matrix
        I = torch.eye(self.F.shape[0])
        self.P = (I - K @ self.H) @ self.P

        return self.state[:2]  # Return the updated position [x, y]

    def predict_next_n_steps(self, n):
        predictions = []
        for _ in range(n):
            next_state = self.F @ self.state
            predictions.append(next_state[:2])  # Store predicted position
            self.state = next_state  # Update state for next prediction
        return predictions

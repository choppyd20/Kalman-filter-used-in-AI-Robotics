import numpy as np

# model parameters
A = np.array([[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]])
B = np.array([[1.0, 0, 0],
              [0, 1.0, 0],
              [0, 0, 1.0]])
C = np.array([[1, 0, 0],
              [0, 1, 0]])

#  noise parameters
pos_sigma = 0.2
rot_sigma = 0.02
obs_sigma = 0.01
R = np.diag([pos_sigma, pos_sigma, rot_sigma]) ** 2 # motion noise
Q = np.diag([obs_sigma, obs_sigma]) ** 2 # observation noise

def control_input(x):
    dx = np.cos(x[2]) * 0.1
    dy = np.sin(x[2]) * 0.1
    dtheta = 0.01
    u = np.array([dx, dy, dtheta])
    return u

def motion_model(x, u):
    return A @ x + B @ u

def motion_model_with_noise(x, u):
    return motion_model(x, u) + np.array([pos_sigma, pos_sigma, rot_sigma]) * np.random.randn(3)

def observation_model(x):
    return C @ x

def observation_model_with_noise(x):
    return observation_model(x) + np.array([obs_sigma, obs_sigma]) * np.random.randn(2)

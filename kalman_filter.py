"""
Kalman filter for robot localization
Adapted from the following project:
    https://github.com/AtsushiSakai/PythonRobotics/blob/master/Localization/extended_kalman_filter/extended_kalman_filter.py
"""

import matplotlib.pyplot as plt
import numpy as np
from robot_model import control_input, motion_model, motion_model_with_noise, observation_model_with_noise
from robot_model import A, B, C, R, Q

MAX_TIMESTEPS = 500

def kalman_filter(xEst, sigmaEst, z, u):
    #TODO: Motion update
    xEst = A @ xEst + B @ u
    sigmaEst = A @ sigmaEst @ A.transpose() + R

    #TODO: Measurement update
    K = sigmaEst @ C.transpose() @ \
        np.linalg.inv(C @ sigmaEst @ C.transpose() + Q)
    xEst = xEst + K @ (z - C @ xEst)
    sigmaEst = (np.eye(3) - K @ C) @ sigmaEst
    return xEst, sigmaEst

def main():
    # Initial State Vector [x y yaw]
    xEst = np.zeros(3)
    xTrue = np.zeros(3)
    sigmaEst = np.eye(3)
    xDR = np.zeros(3)  # dead reckoning estimate

    # history
    xEst_array = [xEst]
    xTrue_array = [xTrue]
    xDR_array = [xDR]
    z_array = []

    for t in range(MAX_TIMESTEPS):
        # apply control and get observation
        u = control_input(xTrue)
        xTrue = motion_model_with_noise(xTrue, u)
        z = observation_model_with_noise(xTrue)

        #TODO: localization by dead reckoning
        xDR = A @ xDR + B @ u 

        #TODO: localization by Kalman Filtering
        xEst, sigmaEst = kalman_filter(xEst, sigmaEst, z, u)
        print('Time %3d: xTrue=[%5.2f,%5.2f], xEst=[%5.2f,%5.2f]' % (t, xTrue[0], xTrue[1], xEst[0], xEst[1]))

        # store data history
        xEst_array.append(xEst)
        xTrue_array.append(xTrue)
        xDR_array.append(xDR)
        z_array.append(z)

        # show animation
        plt.cla()
        plt.plot([z[0] for z in z_array], [z[1] for z in z_array], ".g", label="GPS")
        plt.plot([x[0] for x in xTrue_array], [x[1] for x in xTrue_array], "-b", label="Ground Truth")
        plt.plot([x[0] for x in xDR_array], [x[1] for x in xDR_array], "-k", label="DR Estimate")
        plt.plot([x[0] for x in xEst_array], [x[1] for x in xEst_array], "-r", label="KF Estimate")
        plt.axis("equal")
        plt.grid(True)
        plt.legend()
        plt.pause(0.01)
    plt.show()

if __name__ == '__main__':
    main()

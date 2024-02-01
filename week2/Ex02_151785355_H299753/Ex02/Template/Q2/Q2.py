# Complete the parts denoted by ???
# import csv
import rclpy
from rclpy.node import Node
import numpy as np
from math import sin, cos, pi, sqrt, pow
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Pose
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Constants
dt = 0.1    # Time increment (s)
D = 0.4     # Wheel distance (m)
R = 0.1     # Wheel radiius (m)
UNCERTAINTY_MODEL = [20, 6, 25, 8]
MESSAGE_LIMIT = 2216
coordinates_line = []
coordinates_cloud = []


class Simulator(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')

        # Logged variables
        # The index 0 is for the mean (no noise) robot
        self.x_euler = np.zeros((101, MESSAGE_LIMIT + 1))
        self.y_euler = np.zeros((101, MESSAGE_LIMIT + 1))
        self.yaw_euler = np.zeros((101, MESSAGE_LIMIT + 1))

        # Counter for indexing
        self.message_count = 0

        # Subscriber
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/Wheels_data',
            self.listener_callback,
            10)
        self.subscription  # prevent unused variable warning



    def plot(self):
        fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        plt.grid(color='grey', linestyle='-', linewidth=0.1)
        plt.title("Trajectories")
        plt.xlabel("X-Axis")
        plt.ylabel("Y-Axis")
        plt.plot(self.x_euler[0, :], self.y_euler[0, :], 'b', label='Expected position')

        # Calculate the mean trajectory
        x_mean = (self.x_euler.sum(axis=0) - self.x_euler[0, :]) / 100
        y_mean = (self.y_euler.sum(axis=0) - self.y_euler[0, :]) / 100
        # print(x_mean.shape, y_mean.shape)
        plt.plot(x_mean, y_mean, 'r', label='Mean position')

        # Plot the Confidence Ellipses
        indices = [700, 1400, 2100]
        for index in indices:

            # Calculate the covariances of each sample
            plt.plot(self.x_euler[:, index - 1], self.y_euler[:, index - 1], linestyle='None', marker=".")
            cov = np.cov(self.x_euler[:, index - 1], self.y_euler[:, index - 1])

            # Compute the eigenvalues and eigenvectors
            lambda_, v = np.linalg.eig(cov)
            lambda_ = np.sqrt(lambda_)

            # Plot 3 layers of confidence
            for level in range(1, 4):
                ell = Ellipse(xy=(x_mean[index - 1], y_mean[index - 1]),
                              width=lambda_[0] * level * 2, height=lambda_[1] * level * 2,
                              angle=np.rad2deg(np.arccos(v[0, 0])))
                ell.set_facecolor('none')
                ell.set_edgecolor('black')
                ax.add_artist(ell)
        ax.legend()
        plt.show()

    def listener_callback(self, msg):
        # Increase the counter and process the prediction
        self.message_count += 1
        self.step_calculation(msg.data[0], msg.data[1])

    def step_calculation(self, wl, wr):
        # Calculate the mean position
        wl = wl * 2 * np.pi / 60
        wr = wr * 2 * np.pi / 60
        rot_head = 0.5 * R / D * (wr - wl) * dt
        trans_head = 0.5 * R * (wl + wr) * dt

        # Make normal distribution for noise
        epsilon_rot = UNCERTAINTY_MODEL[0] * (rot_head ** 2) + UNCERTAINTY_MODEL[1] * (trans_head ** 2)
        epsilon_trans = UNCERTAINTY_MODEL[2] * (trans_head ** 2) + UNCERTAINTY_MODEL[3] * 2 * (rot_head ** 2)

        # Predict the robot position
        for i in range(0, 101):
            state_k = np.matrix([[self.x_euler[i, self.message_count - 1]],
                                 [self.y_euler[i, self.message_count - 1]],
                                 [self.yaw_euler[i, self.message_count - 1]]])

            # With noise
            if i > 0:
                sigma_rot_1 = rot_head + np.random.normal(0.0, epsilon_rot)
                sigma_rot_2 = rot_head + np.random.normal(0.0, epsilon_rot)
                sigma_trans = trans_head + np.random.normal(0.0, epsilon_trans)
                x_euler = state_k[0] + sigma_trans * cos(state_k[2] + sigma_rot_1)
                y_euler = state_k[1] + sigma_trans * sin(state_k[2] + sigma_rot_1)
                yaw_euler = state_k[2] + sigma_rot_1 + sigma_rot_2

            # Without noise
            else:
                x_euler = state_k[0] + trans_head * cos(state_k[2] + rot_head)
                y_euler = state_k[1] + trans_head * sin(state_k[2] + rot_head)
                yaw_euler = state_k[2] + 2 * rot_head

            # Regulate the psi
            if yaw_euler > pi:
                yaw_euler -= 2 * pi
            elif yaw_euler < -pi:
                yaw_euler += 2 * pi

            self.x_euler[i, self.message_count] = x_euler
            self.y_euler[i, self.message_count] = y_euler
            self.yaw_euler[i, self.message_count] = yaw_euler


def main(args=None):
    rclpy.init(args=args)

    simulator = Simulator()
    print("Connected!")

    # Spin until the last message to plot
    while simulator.message_count < MESSAGE_LIMIT:
        rclpy.spin_once(simulator)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    simulator.destroy_node()
    rclpy.shutdown()

    simulator.plot()

if __name__ == '__main__':
    main()
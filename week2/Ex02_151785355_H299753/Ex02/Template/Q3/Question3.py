### Insert your code at places denoted by ???

import rclpy
from rclpy.node import Node
import numpy as np
from math import sin, cos, sqrt
from sensor_msgs.msg import JointState
import matplotlib.pyplot as plt
from numpy import linalg as LA

# Constants
dt = 0.1    # Time (s)
D = 0.4     # 
Radius = 0.1
ODOM_LIMIT = 413
SENSOR_LIMIT = 82
L0 = np.array([7.3, -4.5]).T
L1 = np.array([-5.5, 8.3]).T
L2 = np.array([-7.5, -6.3]).T


class Server(Node):
    def __init__(self):
        super().__init__('Question3')
        self.state_size = 3
        self.num_of_lms = 3
        self.R = np.eye(self.state_size) * 0.05
        self.M = np.eye(self.state_size) * 0.25

        # Declaring Variables
        self.odom_count = 0
        self.sensor_count = 0
        self.is_update = False

        self.x_hat_now_minus = np.zeros((self.state_size,1))
        self.x_hat_now_plus = np.zeros((self.state_size,1)) 
        self.x_hat_last_plus = np.zeros((self.state_size,1))
        self.p_now_minus = np.diag(np.ones(self.state_size)*0.0025)
        self.p_now_plus = np.diag(np.ones(self.state_size)*0.0025)
        self.p_last_plus = np.diag(np.ones(self.state_size)*0.0025)

        self.z = np.zeros((self.num_of_lms,1))
        self.z_hat = np.zeros((self.num_of_lms,1))

        self.log_x = np.empty(shape=(1,3))
        self.log_x.fill(0)

        # Subscriber
        self.sensor_subscription = self.create_subscription(
            JointState,
            '/Landmark_dist',
            self.sensor_callback,
            20)
        self.odom_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.odom_callback,
            20)

    def plot(self):
        plt.scatter(self.log_x[:,0], self.log_x[:,1])
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.show()

    def odom_callback(self, msg):
        # Increase the counter and process the prediction
        self.odom_count += 1
        #self.get_logger().info('Odom says: #%d "%s"' % (self.odom_count, msg.velocity))
        self.step_calculation(msg.velocity[0], msg.velocity[1])

    def sensor_callback(self, msg):
        # Increase the counter and process the update
        self.sensor_count += 1
        self.is_update = True
        #self.get_logger().info('Sensor says: #%d "%s"' % (self.sensor_count, msg.position))
        self.z[0] = msg.position[0]
        self.z[1] = msg.position[1]
        self.z[2] = msg.position[2]

    def update(self):
        location = np.array([self.x_hat_now_minus[0][0], self.x_hat_now_minus[1][0]])
        self.z_hat[0] = LA.norm(location - L0)
        self.z_hat[1] = LA.norm(location - L1)
        self.z_hat[2] = LA.norm(location - L2)

        H = np.zeros((3, 3))
        H[0, 0] = 2*(location[0] - L0[0])
        H[0, 1] = 2*(location[1] - L0[1])
        H[1, 0] = 2*(location[0] - L1[0])
        H[1, 1] = 2*(location[1] - L1[1])
        H[2, 0] = 2*(location[0] - L2[0])
        H[2, 1] = 2*(location[1] - L2[1])

        K = self.p_now_minus @ H.T @ LA.inv(H @ self.p_now_minus @ H.T + self.R)

        self.p_now_plus = (np.eye(self.state_size) - K @ H) @ self.p_now_minus
        self.x_hat_now_plus = self.x_hat_now_minus + K @ (self.z - self.z_hat)


    def step_calculation(self, wl, wr):
        rot_head = dt * Radius * (wr - wl) / (2 * D)
        trans_head = dt * Radius * (wr + wl) / 2
        x_last_plus = self.x_hat_last_plus[0]
        y_last_plus = self.x_hat_last_plus[1]
        psi_last_plus = self.x_hat_last_plus[2]
        #self.get_logger().info(str(self.x_hat_last_plus.shape))

        self.x_hat_now_minus[0] = x_last_plus + trans_head*np.cos(psi_last_plus + rot_head)
        self.x_hat_now_minus[1] = y_last_plus + trans_head*np.sin(psi_last_plus + rot_head)
        self.x_hat_now_minus[2] = psi_last_plus + 2*rot_head

        A = np.eye(self.state_size)
        A[0, 2] = - trans_head * sin(psi_last_plus + rot_head)
        A[1, 2] = trans_head * cos(psi_last_plus + rot_head)

        L = np.eye(self.state_size)
        L[0, 0] = cos(psi_last_plus + rot_head)
        L[0, 1] = -trans_head * sin(psi_last_plus + rot_head)
        L[1, 0] = sin(psi_last_plus + rot_head)
        L[1, 1] = trans_head * cos(psi_last_plus + rot_head)
        L[2, 1] = 1
        L[2, 2] = 1
        
        self.p_now_minus = A @ self.p_last_plus @ A.T + L @ self.M @ L.T

        if self.is_update:
            self.update()
            self.is_update = False
        else:
            self.x_hat_now_plus = self.x_hat_now_minus
            self.p_now_plus = self.p_now_minus

        self.x_hat_last_plus = self.x_hat_now_plus
        self.p_last_plus = self.p_now_plus

        self.log_x = np.vstack((self.log_x, self.x_hat_now_plus.T))

        #self.get_logger().info(str(self.x_hat_last_plus[0]))
 


def main(args=None):
    rclpy.init(args=args)

    server = Server()
    print("Connected!")

    try:
        while True:
            rclpy.spin_once(server)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    server.plot()

    server.destroy_node()
    rclpy.shutdown()

    


if __name__ == '__main__':
    main()

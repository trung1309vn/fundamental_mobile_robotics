import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot
from library.detect_obstacle import DetectObstacle

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 10 # np.pi # total simulation duration in seconds
# Set initial state
# init_state = np.array([1.5, -1., 0.]) # px, py, theta
init_state = np.array([2., -0.5, 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

robot_radius = 0.21
d_safe = 0.5
eps = 0.15

# Define Obstacles 
# obst_vertices = np.array( [ [-1., -1.5], [1., -1.5], [1., 1.5], [-1., 1.5], \
#         [-1., 1.], [0.5, 1.], [0.5, -1.], [-1., -1.], [-1., -1.5] ]) 
t = 0
obst_vertices = []
while t < 2*np.pi: 
    obst_vertices.append([0.5*np.cos(t), 0.5*np.sin(t)])
    t += 0.05
obst_vertices = np.asarray(obst_vertices)

# Define sensor's sensing range and resolution
sensing_range = 1. # in meter
sensor_resolution = np.pi/8 # angle between sensor data in radian


# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_sensor_endpoint(robot_state, sensors_dist):
    # assuming sensor position is in the robot's center
    sens_N = round(2*np.pi/sensor_resolution)
    sensors_theta = [i*2*np.pi/sens_N for i in range(sens_N)]
    obst_points = np.zeros((3,sens_N))

    # robot pose in world frame
    R_WB = np.array([[np.cos(robot_state[2]), -np.sin(robot_state[2]), robot_state[0]], 
                     [np.sin(robot_state[2]),  np.cos(robot_state[2]), robot_state[1]], 
                     [                     0,                       0,              1]])
    for i in range(sens_N):
        # sensor in robot frame
        R_BS = np.array([[np.cos(sensors_theta[i]), -np.sin(sensors_theta[i]), 0],
                         [np.sin(sensors_theta[i]),  np.cos(sensors_theta[i]), 0],
                         [                       0,                         0, 1]])
        # transform sensor reading from sensor frame to world frame
        temp = R_WB @ R_BS @ np.array([sensors_dist[i], 0, 1])
        obst_points[:,i] = temp

    return obst_points[:2,:]


def compute_control_input(desired_state, robot_state, obst_points, current_controller, current_time):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()

    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.])
    k_gain = 2   # k value for get to goal control
    k0 = None    # k value for obstacle avoidance control

    # calculate obstacle distance to robot center
    obst_distances = np.linalg.norm(obst_points.T - robot_state[:2], axis=1)

    # switching control
    # if currently use gtg control
    if (current_controller == "gtg"):
        # Check if any sensing in obstacle d_safe zone
        is_in_d_safe = np.sum(obst_distances < d_safe)
        if (is_in_d_safe):
            current_controller = "avo"
    # if currently use avo control
    else:
        # Check if any sensing in obstacle d_safe + eps zone
        is_in_d_safe_eps = np.sum(obst_distances < d_safe + eps)
        if (not is_in_d_safe_eps):
            current_controller = "gtg"

    # Compute the control input based on current control
    if (current_controller == "gtg"):
        state_diff = desired_state - robot_state
        current_input = k_gain * state_diff
    else:
        # use only sensing in d_safe + eps zone
        obst_dist = obst_distances[obst_distances < d_safe + eps]
        c = 0.5
        k0 = 1 / obst_dist * (c / (obst_dist**2 + eps)) 
        state_diff = robot_state[:2] - obst_points.T[obst_distances < d_safe + eps]
        for k, state in zip(k0, state_diff):
            print(k, state)
            current_input[:2] += k * state

        # averaging the avo control vector
        current_input[:2] /= k0.shape[0]

    return current_input, current_controller


# MAIN SIMULATION COMPUTATION
#---------------------------------------------------------------------
def simulate_control():
    sim_iter = round(t_max/Ts) # Total Step for simulation

    # Initialize robot's state (Single Integrator)
    robot_state = init_state.copy() # numpy array for [px, py, theta]
    desired_state = np.array([-2., 0.5, 0.]) # numpy array for goal / the desired [px, py, theta]

    # Store the value that needed for plotting: total step number x data length
    state_history = np.zeros( (sim_iter, len(robot_state)) ) 
    goal_history = np.zeros( (sim_iter, len(desired_state)) ) 
    input_history = np.zeros( (sim_iter, 3) ) # for [vx, vy, omega] vs iteration time

    # Initiate the Obstacle Detection
    range_sensor = DetectObstacle( sensing_range, sensor_resolution)
    # range_sensor.register_obstacle_bounded( obst_vertices )
    range_sensor.register_obstacle_bounded( obst_vertices )


    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)

        # Display the obstacle
        # sim_visualizer.ax.plot( obst_vertices[:,0], obst_vertices[:,1], '--r' )
        sim_visualizer.ax.add_patch(plt.Circle((0,0), 0.5, color="r"))
        sim_visualizer.ax.add_patch(plt.Circle((0,0), d_safe, color="r", fill=False))
        sim_visualizer.ax.add_patch(plt.Circle((0,0), d_safe + eps, color="g", fill=False))

        # get sensor reading
        # Index 0 is in front of the robot. 
        # Index 1 is the reading for 'sensor_resolution' away (counter-clockwise) from 0, and so on for later index
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        # compute and plot sensor reading endpoint
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)
        pl_sens, = sim_visualizer.ax.plot(obst_points[0], obst_points[1], '.') #, marker='X')
        pl_txt = [sim_visualizer.ax.text(obst_points[0,i], obst_points[1,i], str(i)) for i in range(len(distance_reading))]

    current_controller = "gtg"
    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # Get information from sensors
        distance_reading = range_sensor.get_sensing_data( robot_state[0], robot_state[1], robot_state[2])
        obst_points = compute_sensor_endpoint(robot_state, distance_reading)

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, current_controller = compute_control_input(desired_state, robot_state, obst_points, 
                                                                  current_controller, current_time)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data
            # update sensor visualization
            pl_sens.set_data(obst_points[0], obst_points[1])
            for i in range(len(distance_reading)): pl_txt[i].set_position((obst_points[0,i], obst_points[1,i]))
        
        #--------------------------------------------------------------------------------
        # Update new state of the robot at time-step t+1
        # using discrete-time model of single integrator dynamics for omnidirectional robot
        robot_state = robot_state + Ts*current_input # will be used in the next iteration
        robot_state[2] = ( (robot_state[2] + np.pi) % (2*np.pi) ) - np.pi # ensure theta within [-pi pi]

        # Update desired state if we consider moving goal position
        #desired_state = desired_state + Ts*(-1)*np.ones(len(robot_state))

    # End of iterations
    # ---------------------------
    # return the stored value for additional plotting or comparison of parameters
    return state_history, goal_history, input_history


if __name__ == '__main__':
    
    # Call main computation for robot simulation
    state_history, goal_history, input_history = simulate_control()


    # ADDITIONAL PLOTTING
    #----------------------------------------------
    t = [i*Ts for i in range( round(t_max/Ts) )]

    # Plot historical data of control input
    fig2 = plt.figure(2)
    ax = plt.gca()
    ax.plot(t, input_history[:,0], label='vx [m/s]')
    ax.plot(t, input_history[:,1], label='vy [m/s]')
    ax.plot(t, input_history[:,2], label='omega [rad/s]')
    ax.set(xlabel="t [s]", ylabel="control input")
    plt.legend()
    plt.grid()

    # Plot historical data of state
    fig3 = plt.figure(3)
    ax = plt.gca()
    ax.plot(t, state_history[:,0], label='px [m]')
    ax.plot(t, state_history[:,1], label='py [m]')
    ax.plot(t, state_history[:,2], label='theta [rad]')
    ax.plot(t, goal_history[:,0], ':', label='goal px [m]')
    ax.plot(t, goal_history[:,1], ':', label='goal py [m]')
    ax.plot(t, goal_history[:,2], ':', label='goal theta [rad]')
    ax.set(xlabel="t [s]", ylabel="state")
    plt.legend()
    plt.grid()

    plt.show()

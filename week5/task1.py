import numpy as np
import matplotlib.pyplot as plt
from library.visualize_mobile_robot import sim_mobile_robot

# Constants and Settings
Ts = 0.01 # Update simulation every 10ms
t_max = 12 #np.pi # total simulation duration in seconds
# Set initial state
init_state = np.array([1.5, -1., 0.]) # px, py, theta
IS_SHOWING_2DVISUALIZATION = True

# obstacle avoidance
radius = 0.21
max_trans = 0.5 # m/s
max_rot = 5     # rad/s
obst_radius = 0.5 # m
d_safe = 0.75     # m
eps = 0.15        # m
p = np.array([[0.],[0.]]) # obstacle origin (0, 0)

# Define Field size for plotting (should be in tuple)
field_x = (-2.5, 2.5)
field_y = (-2, 2)

# IMPLEMENTATION FOR THE CONTROLLER
#---------------------------------------------------------------------
def compute_control_input(desired_state, robot_state, current_controller, current_time):
    # Feel free to adjust the input and output of the function as needed.
    # And make sure it is reflected inside the loop in simulate_control()

    # initial numpy array for [vx, vy, omega]
    current_input = np.array([0., 0., 0.])
    k_gain = 2   # k value for get to goal control
    k0 = None    # k value for obstacle avoidance control

    # calculate obstacle distance to robot center
    obst_dist = np.linalg.norm(p.T - robot_state[:2])

    # switching control
    # if currently use gtg control
    if (current_controller == "gtg"):
        # Check if robot in obstacle d_safe zone
        if (obst_dist < d_safe):
            current_controller = "avo"
    # if currently use avo control
    else:
        # Check if robot in obstacle (d_safe + eps) zone
        if (not (obst_dist < d_safe + eps)):
            current_controller = "gtg"

    # Compute the control input based on current control
    if (current_controller == "gtg"):
        state_diff = desired_state - robot_state
        current_input = k_gain * state_diff
    else:
        c = 1.0
        k0 = 1 / obst_dist * (c / (obst_dist**2 + eps)) 
        state_diff = robot_state[:-1] - p.T
        current_input[:-1] = k0 * state_diff

    # clipping control input
    max_vx = max_trans * abs(current_input[0]) / np.linalg.norm(current_input[:-1])
    max_vy = max_trans * abs(current_input[1]) / np.linalg.norm(current_input[:-1])
    control_signs = np.sign(current_input)
    current_input[0] = control_signs[0] * min(abs(current_input[0]), max_vx)
    current_input[1] = control_signs[1] * min(abs(current_input[1]), max_vy)
    current_input[2] = control_signs[2] * min(abs(current_input[2]), max_rot)
    print(current_input)
    print()
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

    if IS_SHOWING_2DVISUALIZATION: # Initialize Plot
        sim_visualizer = sim_mobile_robot( 'omnidirectional' ) # Omnidirectional Icon
        #sim_visualizer = sim_mobile_robot( 'unicycle' ) # Unicycle Icon
        sim_visualizer.set_field( field_x, field_y ) # set plot area
        sim_visualizer.show_goal(desired_state)
        sim_visualizer.ax.add_patch(plt.Circle((0,0), 0.5, color="r"))
        sim_visualizer.ax.add_patch(plt.Circle((0,0), d_safe, color="r", fill=False))
        sim_visualizer.ax.add_patch(plt.Circle((0,0), d_safe + eps, color="g", fill=False))

    current_controller = "gtg"
    for it in range(sim_iter):
        current_time = it*Ts
        # record current state at time-step t
        state_history[it] = robot_state
        goal_history[it] = desired_state

        # COMPUTE CONTROL INPUT
        #------------------------------------------------------------
        current_input, current_controller = compute_control_input(desired_state, robot_state, current_controller, current_time)
        #------------------------------------------------------------

        # record the computed input at time-step t
        input_history[it] = current_input

        if IS_SHOWING_2DVISUALIZATION: # Update Plot
            sim_visualizer.update_time_stamp( current_time )
            sim_visualizer.update_goal( desired_state )
            sim_visualizer.update_trajectory( state_history[:it+1] ) # up to the latest data

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

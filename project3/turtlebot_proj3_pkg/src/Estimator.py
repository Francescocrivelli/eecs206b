import rospy
from std_msgs.msg import Float32MultiArray
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['FreeSans', 'Helvetica', 'Arial']
plt.rcParams['font.size'] = 14
import time


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        d : float
            Half of the track width (m) of TurtleBot3 Burger.
        r : float
            Wheel radius (m) of the TurtleBot3 Burger.
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][0] is timestamp (s),
            u[i][1] is left wheel rotational speed (rad/s), and
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is timestamp (s),
            x[i][1] is bearing (rad),
            x[i][2] is translational position in x (m),
            x[i][3] is translational position in y (m),
            x[i][4] is left wheel rotational position (rad), and
            x[i][5] is right wheel rotational position (rad).
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][0] is timex_hat stamp (s),
            y[i][1] is translational position in x (m) when freeze_bearing:=true,
            y[i][1] is distance to the landmark (m) when freeze_bearing:=false,
            y[i][2] is translational position in y (m) when freeze_bearing:=true, and
            y[i][2] is relative bearing (rad) w.r.t. the landmark when
            freeze_bearing:=false.
        x_hat : list
            A list of estimated system states. It should follow the same format
            as x.
        dt : float
            Update frequency of the estimator.
        fig : Figure
            matplotlib Figure for real-time plotting.
        axd : dict
            A dictionary of matplotlib Axis for real-time plotting.
        ln* : Line
            matplotlib Line object for ground truth states.
        ln_*_hat : Line
            matplotlib Line object for estimated states.
        canvas_title : str
            Title of the real-time plot, which is chosen to be estimator type.
        sub_u : rospy.Subscriber
            ROS subscriber for system inputs.
        sub_x : rospy.Subscriber
            ROS subscriber for system states.
        sub_y : rospy.Subscriber
            ROS subscriber for system outputs.
        tmr_update : rospy.Timer
            ROS Timer for periodically invoking the estimator's update method.

    Notes
    ----------
        The frozen bearing is pi/4 and the landmark is positioned at (0.5, 0.5).
    """
    # noinspection PyTypeChecker
    def __init__(self):
        self.d = 0.08
        self.r = 0.033
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.dt = 0.1
        self.fig, self.axd = plt.subplot_mosaic(
            [['xy', 'phi'],
             ['xy', 'x'],
             ['xy', 'y'],
             ['xy', 'thl'],
             ['xy', 'thr']], figsize=(20.0, 10.0))
        self.ln_xy, = self.axd['xy'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xy_hat, = self.axd['xy'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_y, = self.axd['y'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_y_hat, = self.axd['y'].plot([], 'o-c', label='Estimated')
        self.ln_thl, = self.axd['thl'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thl_hat, = self.axd['thl'].plot([], 'o-c', label='Estimated')
        self.ln_thr, = self.axd['thr'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_thr_hat, = self.axd['thr'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'
        self.sub_u = rospy.Subscriber('u', Float32MultiArray, self.callback_u)
        self.sub_x = rospy.Subscriber('x', Float32MultiArray, self.callback_x)
        self.sub_y = rospy.Subscriber('y', Float32MultiArray, self.callback_y)
        self.tmr_update = rospy.Timer(rospy.Duration(self.dt), self.update)



        # New attributes for quantitative measurements
        self.position_errors = []  # To store estimation errors
        self.orientation_errors = []
        self.running_times = []  # To store per-step running times

        # Register a shutdown hook to print metrics
        rospy.on_shutdown(self.print_metrics)


    def print_metrics(self):
        """Print accuracy and running time metrics when the node shuts down."""
        if len(self.position_errors) == 0 and len(self.orientation_errors) == 0 and len(self.running_times) == 0:
            print("No metrics computed yet.")
            return

        # Compute accuracy metrics
        # mse = np.mean(np.square(self.errors))
        # rmse = np.sqrt(mse)
        # mae = np.mean(np.abs(self.errors))
        mse_position = np.mean(np.square(self.position_errors))
        rmse_position = np.sqrt(mse_position)
        mse_orientation = np.mean(np.square(self.orientation_errors))
        rmse_orientation = np.sqrt(mse_orientation)
        print(f"Estimation Accuracy Metrics:")
        # print(f"MSE: {mse:.6f}")
        print(f"RMSE Position: {rmse_position:.6f}")
        # print(f"MAE: {mae:.6f}")
        print(f"RMSE Orientation: {rmse_orientation:.6f}")

        # Compute average running time
        avg_running_time = np.mean(self.running_times)
        print(f"Average Per-Step Running Time: {avg_running_time:.6f} seconds")



    def callback_u(self, msg):
        self.u.append(msg.data)

    def callback_x(self, msg):
        self.x.append(msg.data)
        if len(self.x_hat) == 0:
            self.x_hat.append(msg.data)

    def callback_y(self, msg):
        self.y.append(msg.data)

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xy'].set_title(self.canvas_title)
        self.axd['xy'].set_xlabel('x (m)')
        self.axd['xy'].set_ylabel('y (m)')
        self.axd['xy'].set_aspect('equal', adjustable='box')
        self.axd['xy'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].legend()
        self.axd['y'].set_ylabel('y (m)')
        self.axd['y'].legend()
        self.axd['thl'].set_ylabel('theta L (rad)')
        self.axd['thl'].legend()
        self.axd['thr'].set_ylabel('theta R (rad)')
        self.axd['thr'].set_xlabel('Time (s)')
        self.axd['thr'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xyline(self.ln_xy, self.x)
        self.plot_xyline(self.ln_xy_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_yline(self.ln_y, self.x)
        self.plot_yline(self.ln_y_hat, self.x_hat)
        self.plot_thlline(self.ln_thl, self.x)
        self.plot_thlline(self.ln_thl_hat, self.x_hat)
        self.plot_thrline(self.ln_thr, self.x)
        self.plot_thrline(self.ln_thr_hat, self.x_hat)

    def plot_xyline(self, ln, data):
        if len(data):
            x = [d[2] for d in data]
            y = [d[3] for d in data]
            ln.set_data(x, y)
            self.resize_lim(self.axd['xy'], x, y)

    def plot_philine(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            phi = [d[1] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            x = [d[2] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_yline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            y = [d[3] for d in data]
            ln.set_data(t, y)
            self.resize_lim(self.axd['y'], t, y)

    def plot_thlline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thl = [d[4] for d in data]
            ln.set_data(t, thl)
            self.resize_lim(self.axd['thl'], t, thl)

    def plot_thrline(self, ln, data):
        if len(data):
            t = [d[0] for d in data]
            thr = [d[5] for d in data] 
            ln.set_data(t, thr)
            self.resize_lim(self.axd['thr'], t, thr)

    # noinspection PyMethodMayBeStatic
    def resize_lim(self, ax, x, y):
        xlim = ax.get_xlim()
        ax.set_xlim([min(min(x) * 1.05, xlim[0]), max(max(x) * 1.05, xlim[1])])
        ylim = ax.get_ylim()
        ax.set_ylim([min(min(y) * 1.05, ylim[0]), max(max(y) * 1.05, ylim[1])])


class OracleObserver(Estimator):
    """Oracle observer which has access to the true state.

    This class is intended as a bare minimum example for you to understand how
    to work with the code.

    Example
    ----------
    To run the oracle observer:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=oracle_observer \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is            u_L = self.u[i][0]

    free of noise. You may use this knowledge to verify your implementation.
ing_times.append(end_time - start_time)
    Example
    ----------
    To run dead reckoning:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=dead_reckoning \
            noise_injection:=true \
            freeze_bearing:=false
    For debugging, you can simulate a noise-free unicycle model by setting
    noise_injection:=false.
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Dead Reckoning'
        self.current_step = 0

    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            start_time = time.time()
            # Get the current state estimate
            x_prev = self.x_hat[-1]
            
            # Extract the current state variables and control inputs
            phi_prev = x_prev[1]
            x_prev_pos = x_prev[2]
            y_prev_pos = x_prev[3]
            theta_L_prev = x_prev[4]
            theta_R_prev = x_prev[5]
        
            if self.current_step >= len(self.u):
                return 
            _, u_L, u_R = self.u[self.current_step]
            
            # Compute the state derivatives using the dynamics model
            phi_dot = (-self.r / (2 * self.d)) * u_L + (self.r / (2 * self.d)) * u_R
            x_dot = (self.r / 2) * np.cos(phi_prev) * u_L + (self.r / 2) * np.cos(phi_prev) * u_R
            y_dot = (self.r / 2) * np.sin(phi_prev) * u_L + (self.r / 2) * np.sin(phi_prev) * u_R
            theta_L_dot = u_L
            theta_R_dot = u_R
            
            # Update the state estimate using the forward Euler method
            phi_next = phi_prev + phi_dot * self.dt
            x_next = x_prev_pos + x_dot * self.dt
            y_next = y_prev_pos + y_dot * self.dt
            theta_L_next = theta_L_prev + theta_L_dot * self.dt
            theta_R_next = theta_R_prev + theta_R_dot * self.dt
            
    
            x_next_state = np.array([x_prev[0] + self.dt, phi_next, x_next, y_next, theta_L_next, theta_R_next])
            self.x_hat.append(x_next_state)


            self.current_step += 1
            # You may ONLY use self.u and self.x[0] for estimation


            
            end_time = time.time()
            self.running_times.append(end_time - start_time)

            # Compute estimation error
            if len(self.x) > self.current_step:
                error_position = np.linalg.norm(self.x[self.current_step][2:4] - x_next_state[2:4])  # Error function in terms of x and y
                error_orientation = np.linalg.norm(self.x[self.current_step][1] - x_next_state[1]) # phi 
                self.position_errors.append(error_position)
                self.orientation_errors.append(error_orientation)
            
          


class KalmanFilter(Estimator):
    """Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    linear unicycle model at the default bearing of pi/4. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the r1:6]ecursive Kalman filter update rule.

    Attributes:
    ----------
        phid : floating_times
            Default bearing of the turtlebot fixed at pi / 4.

    Example
    ----------
    To run the Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=kalman_filter \
            noise_injection:=true \
            freeze_bearing:=true
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Kalman Filter'
        self.phid = np.pi / 4
        # TODO: Your implementation goes here!
        # You may define the A, C, Q, R, and P matrices below.
        # Define model matrices
      
        self.A = np.eye(4)  # State transition matrix   (x, y, thetaL, thetaR)
        self.B = np.array([
            [self.r * 0.5 * np.cos(self.phid), self.r * 0.5 * np.cos(self.phid)],
            [self.r * 0.5 * np.sin(self.phid), self.r * 0.5 * np.sin(self.phid)],
            [1, 0],
            [0, 1]
        ]) * self.dt # Coself.current_step = 0ntrol input matrix
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]])
        # Define noise covariance matrices
        self.Q = np.diag([0.01, 0.01, 0.01, 0.01]) * 1 # Process noise covariance           (x, y, thetaL, thetaR)
        self.R = np.diag([0.1, 0.1]) * 0.1 # Measurement noise covariance                     (x_hat, y_hat, thetaL_hat, thetaR_hat)
        self.P = np.diag([1, 1, 1, 1])  * 0.01 # Initial state covariance  (Quickly or Slowly Adjust estimates based on measurements)

        # Observation Note: Seems like 0.01 is pretty good!             DO NOT TOUCH THESE TUNES PLS. JUST THE EKF
        
        self.current_step = 0
    # noinspection DuplicatedCode
    # noinspection PyPep8Naming
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here! print(error_position)
            # You may use self.u, self.y, and self.x[0] for estimation
            start_time = time.time()

            # Get the latest control input and measurement
            if self.current_step >= len(self.u):
                return 
            _, u_L, u_R = self.u[self.current_step] 
            u = np.array([u_L, u_R])
            y = self.y[self.current_step][1:] 

            # Get the previous state estimate and covariance
            x_prev = self.x_hat[-1][2:6] # Extract [x, y, thetaL, thetaR]  
            P_pred = self.A @ self.P @ self.A.T + self.Q # Covaraince extrapolation (4x4)                      

            # Prediction step
            x_pred = self.A @ x_prev + self.B @ u # State extrapolation (4x1)
            K = P_pred @ self.C.T @ np.linalg.inv(self.C @ P_pred @ self.C.T + self.R)  # Kalman gain

            y_pred = self.C @ x_pred # Predicted measurement 2x1

            x_updated = x_pred + K @ (y - y_pred)  # State update
            P_updated = (np.eye(4) - K @ self.C) @ P_pred  # Covariance update

            x_updated_full = np.concatenate(([self.x_hat[-1][0] + self.dt], [self.x_hat[-1][1]], x_updated)) # Add timestamp and phi back
            self.x_hat.append(x_updated_full)
            self.P = P_updated

            self.current_step += 1

            end_time = time.time()
            self.running_times.append(end_time - start_time)
            if len(self.x) > self.current_step:
                error_position = np.linalg.norm(self.x[self.current_step][2:4] - x_updated_full[2:4])  # Error function in terms of x and y
                error_orientation = np.linalg.norm(self.x[self.current_step][1] - x_updated_full[1]) # phi is constant so we don't need to measure orientation error
                self.position_errors.append(error_position)
                self.orientation_errors.append(error_orientation)
          


# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):                      # THIS PART IS THE EXTRA CREDIT     
    """Extended Kalman filter estimator.

    Your task is to implement the update method of this class using the u
    attribute, y attribute, and x0. You will need to build a model of the
    unicycle model and linearize it at every operating point. After building the
    model, use the provided inputs and outputs to estimate system state over
    time via the recursive extended Kalman filter update rule.

    Hint: You may want to reuse your code from DeadReckoning class and
    KalmanFilter class.

    Attributes:
    ----------
        landmark : tuple
            A tuple of the coordinates of the landmark.
            landmark[0] is the x coordinate.
            landmark[1] is the y coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ roslaunch proj3_pkg unicycle_bringup.launch \
            estimator_type:=extended_kalman_filter \
            noise_injection:=true \
            freeze_bearing:=false
    """
    def __init__(self):
        super().__init__()
        self.canvas_title = 'Extended Kalman Filter'
        self.landmark = (0.5, 0.5)

        self.current_step = 0
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.
      
        # Define noise covariance matrices
        # self.Q = np.diag([0.01, 0.01, 0.01, 0.001, 0.01]) * 0.1 # Process noise covariance                [phi, x, y, thetaL, thetaR]         
        self.Q = np.array([
            [0.001, 0, 0, 0, 0],     # phi noise and correlations
            [0, 0.001, 0, 0, 0],     # x noise and correlations    phi corr? 0.001
            [0, 0, 0.001, 0, 0],      # y noise and correlations
            [0, 0, 0, 0.0001, 0],       # thetaL noise
            [0, 0, 0, 0, 0.001]      # thetaR noise
        ])
        # self.R = np.diag([0.1, 0.1]) * 0.01 # Measurement noise covariance                          
        self.R = np.array([
            [0.001, 0],           
            [0, 0.001]])
        # self.P = np.diag([0.1, 10, 10, 1, 1]) * 1  # Initial state covariance                   
        self.P = np.array([[0.1, 0.01, 0.01, 0, 0],            # phi uncertainty and correlations
                           [0.01, 10, 0.01, 0, 0],             # x uncertainty and correlations
                           [0.01, 0.01, 10, 0, 0],             # y uncertainty and correlations
                           [0, 0, 0, 1, 0],                 # theta_L uncertainty
                           [0, 0, 0, 0, 1]])             # theta_R uncertainty

    def g(self, x, u):
        """Nonlinear dynamic self.current_step >= len(self.u):
                return ics model for the unicycle."""
    
        phi, x_pos, y_pos, theta_L, theta_R = x[1:6]
        u_L, u_R = u[0], u[1]

        # State update. So far, this is just the dead reckoning dynamic model 
        phi_new = phi                                                                           + (self.r / (2 * self.d)) * (u_R - u_L) * self.dt
        x_pos_new = x_pos    - (self.r * phi * np.sin(phi)) * (u_L + u_R) * self.dt             + (self.r / 2) * np.cos(phi) * (u_L + u_R) * self.dt
        y_pos_new = y_pos    +  (self.r * phi * np.cos(phi)) * (u_L + u_R) * self.dt            + (self.r / 2) * np.sin(phi) * (u_L + u_R) * self.dt
        theta_L_new = theta_L                                                                   + u_L * self.dt
        theta_R_new = theta_R                                                                   + u_R * self.dt

        return np.array([phi_new, x_pos_new, y_pos_new, theta_L_new, theta_R_new]) 

    def h(self, x):
        """Nonlinear measurement mod  print(f"U contains these: {u}")el."""
        phi, x_pos, y_pos, _, _ = x
        lx, ly = self.landmark

        # Distance to landmark
        distance = np.sqrt((lx - x_pos)**2 + (ly - y_pos)**2)
        # Bearing angle
        bearing = np.arctan2(ly - y_pos, lx - x_pos) - phi
    

        return np.array([distance, bearing])

    def approx_A(self, x, u):
        """Linearize the dynamics model around the current state and input."""
        phi, _, _, _, _ = x[1:6]        # [phi, x, y, thetaL, thetaR] ignore timestamp
        u_L, u_R = u[0], u[1]

        # Jacobian of the dynamics (A matrix)
        # A = np.eye(5)  # Identity matrix for the state transition
        # A[0, 3] = (-self.r / (2 * self.d)) * self.dt  # Partial derivative of phi w.r.t. theta_L
        # A[0, 4] = (self.r / (2 * self.d)) * self.dt  # Partial derivative of phi w.r.t. theta_R
        # A[1, 0] = (-self.r / 2) * np.sin(phi) * (u_L + u_R) * self.dt  # Partial derivative of x_pos w.r.t. phi
        # A[1, 3] = (self.r / 2) * np.cos(phi) * self.dt  # Partial derivative of x_pos w.r.t. theta_L
        # A[1, 4] = (self.r / 2) * np.cos(phi) * self.dt  # Partial derivative of x_pos w.r.t. theta_R
        # A[2, 0] = (self.r / 2) * np.cos(phi) * (u_L + u_R) * self.dt  # Partial derivative of y_pos w.r.t. phi
        # A[2, 3] = (self.r / 2) * np.sin(phi) * self.dt  # Partial derivative of y_pos w.r.t. theta_L
        # A[2, 4] = (self.r / 2) * np.sin(phi) * self.dt  # Partial derivative of y_pos w.r.t. theta_R
        # return A
        A = np.zeros((5,5))
        A[1,0] = (-self.r / 2) * np.sin(phi) * (u_L + u_R) * self.dt
        A[2,0] = (self.r / 2) * np.cos(phi) * (u_L + u_R) * self.dt 
        return A

    def approx_C(self, x):
        """Linearize the measurement model around the current state."""
        _, x_pos, y_pos, _, _ = x
        lx, ly = self.landmark

        # Distance to landmark
        distance = np.sqrt((lx - x_pos)**2 + (ly - y_pos)**2)

        # Jacobian of the measurement model (C matrix)
        C = np.zeros((2, 5))
        C[0, 1] = -(lx - x_pos) / distance  # Partial derivative of distance w.r.t. x_pos
        C[0, 2] = -(ly - y_pos) / distance  # Partial derivative of distance w.r.t. y_pos
        C[1, 1] = (ly - y_pos) / (distance**2)  # Partial derivative of bearing w.r.t. x_pos
        C[1, 2] = -(lx - x_pos) / (distance**2)  # Partial derivative of bearing w.r.t. y_pos
        C[1, 0] = -1  # Partial derivative of bearing w.r.t. phi

        return C

    # noinspection DuplicatedCode
    def update(self, _):
        if len(self.x_hat) > 0 and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            start_time = time.time()
    
            
            if self.current_step >= len(self.u):
                return 
            _, u_L, u_R = self.u[self.current_step]
            u = np.array([u_L, u_R])
            y = self.y[self.current_step][1:] 

            x_prev = self.x_hat[-1]
            x_pred = self.g(x_prev, u)      

         
            A = self.approx_A(x_prev, u)     # Linearize the dynamics model
            P_pred = A @ self.P @ A.T + self.Q
            C = self.approx_C(x_pred)   # Linets anarize the measurement model
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + self.R)

            y_pred = self.h(x_pred)
            x_updated = x_pred + K @ (y - y_pred)
            # self.x_hat.append(x_updated)
            x_updated_full = np.concatenate(([self.x_hat[-1][0] + self.dt], x_updated)) # Add back timestamp
            self.x_hat.append(x_updated_full)
            self.P = (np.eye(5) - K @ C) @ P_pred

            self.current_step += 1

            end_time = time.time()
            self.running_times.append(end_time - start_time)

            if len(self.x) > self.current_step:
                error_position = np.linalg.norm(self.x[self.current_step][2:4] - x_updated_full[2:4])  # Error function in terms of x and y
                error_orientation = np.linalg.norm(self.x[self.current_step][1] - x_updated_full[1]) # phi 
                self.position_errors.append(error_position)
                self.orientation_errors.append(error_orientation)
            


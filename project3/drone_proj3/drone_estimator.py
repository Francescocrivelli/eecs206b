import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = ['Arial']
plt.rcParams['font.size'] = 14


class Estimator:
    """A base class to represent an estimator.

    This module contains the basic elements of an estimator, on which the
    subsequent DeadReckoning, Kalman Filter, and Extended Kalman Filter classes
    will be based on. A plotting function is provided to visualize the
    estimation results in real time.

    Attributes:
    ----------
        u : list
            A list of system inputs, where, for the ith data point u[i],
            u[i][1] is the thrust of the quadrotor
            u[i][2] is right wheel rotational speed (rad/s).
        x : list
            A list of system states, where, for the ith data point x[i],
            x[i][0] is translational position in x (m),
            x[i][1] is translational position in z (m),
            x[i][2] is the bearing (rad) of the quadrotor
            x[i][3] is translational velocity in x (m/s),
            x[i][4] is translational velocity in z (m/s),
            x[i][5] is angular velocity (rad/s),
        y : list
            A list of system outputs, where, for the ith data point y[i],
            y[i][1] is distance to the landmark (m)
            y[i][2] is relative bearing (rad) w.r.t. the landmark
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

    Notes
    ----------
        The landmark is positioned at (0, 5, 5).
    """
    # noinspection PyTypeChecker
    def __init__(self, is_noisy=False):
        self.u = []
        self.x = []
        self.y = []
        self.x_hat = []  # Your estimates go here!
        self.t = []
        self.fig, self.axd = plt.subplot_mosaic(
            [['xz', 'phi'],
             ['xz', 'x'],
             ['xz', 'z']], figsize=(20.0, 10.0))
        self.ln_xz, = self.axd['xz'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_xz_hat, = self.axd['xz'].plot([], 'o-c', label='Estimated')
        self.ln_phi, = self.axd['phi'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_phi_hat, = self.axd['phi'].plot([], 'o-c', label='Estimated')
        self.ln_x, = self.axd['x'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_x_hat, = self.axd['x'].plot([], 'o-c', label='Estimated')
        self.ln_z, = self.axd['z'].plot([], 'o-g', linewidth=2, label='True')
        self.ln_z_hat, = self.axd['z'].plot([], 'o-c', label='Estimated')
        self.canvas_title = 'N/A'

        # Defined in dynamics.py for the dynamics model
        # m is the mass and J is the moment of inertia of the quadrotor 
        self.gr = 9.81 
        self.m = 0.92
        self.J = 0.0023
        # These are the X, Y, Z coordinates of the landmark
        self.landmark = (0, 5, 5)

        # This is a (N,12) where it's time, x, u, then y_obs 
        if is_noisy:
            with open('noisy_data.npy', 'rb') as f:
                self.data = np.load(f)
        else:
            with open('data.npy', 'rb') as f:
                self.data = np.load(f)

        self.dt = self.data[-1][0]/self.data.shape[0]


    def run(self):
        for i, data in enumerate(self.data):
            self.t.append(np.array(data[0]))
            self.x.append(np.array(data[1:7]))
            self.u.append(np.array(data[7:9]))
            self.y.append(np.array(data[9:12]))
            if i == 0:
                self.x_hat.append(self.x[-1])
            else:
                self.update(i)
        return self.x_hat

    def update(self, _):
        raise NotImplementedError

    def plot_init(self):
        self.axd['xz'].set_title(self.canvas_title)
        self.axd['xz'].set_xlabel('x (m)')
        self.axd['xz'].set_ylabel('z (m)')
        self.axd['xz'].set_aspect('equal', adjustable='box')
        self.axd['xz'].legend()
        self.axd['phi'].set_ylabel('phi (rad)')
        self.axd['phi'].set_xlabel('t (s)')
        self.axd['phi'].legend()
        self.axd['x'].set_ylabel('x (m)')
        self.axd['x'].set_xlabel('t (s)')
        self.axd['x'].legend()
        self.axd['z'].set_ylabel('z (m)')
        self.axd['z'].set_xlabel('t (s)')
        self.axd['z'].legend()
        plt.tight_layout()

    def plot_update(self, _):
        self.plot_xzline(self.ln_xz, self.x)
        self.plot_xzline(self.ln_xz_hat, self.x_hat)
        self.plot_philine(self.ln_phi, self.x)
        self.plot_philine(self.ln_phi_hat, self.x_hat)
        self.plot_xline(self.ln_x, self.x)
        self.plot_xline(self.ln_x_hat, self.x_hat)
        self.plot_zline(self.ln_z, self.x)
        self.plot_zline(self.ln_z_hat, self.x_hat)

    def plot_xzline(self, ln, data):
        if len(data):
            x = [d[0] for d in data]
            z = [d[1] for d in data]
            ln.set_data(x, z)
            self.resize_lim(self.axd['xz'], x, z)

    def plot_philine(self, ln, data):
        if len(data):
            t = self.t
            phi = [d[2] for d in data]
            ln.set_data(t, phi)
            self.resize_lim(self.axd['phi'], t, phi)

    def plot_xline(self, ln, data):
        if len(data):
            t = self.t
            x = [d[0] for d in data]
            ln.set_data(t, x)
            self.resize_lim(self.axd['x'], t, x)

    def plot_zline(self, ln, data):
        if len(data):
            t = self.t
            z = [d[1] for d in data]
            ln.set_data(t, z)
            self.resize_lim(self.axd['z'], t, z)

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
        $ python drone_estimator_node.py --estimator oracle_observer
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Oracle Observer'

    def update(self, _):
        self.x_hat.append(self.x[-1])


class DeadReckoning(Estimator):
    """Dead reckoning estimator.

    Your task is to implement the update method of this class using only the
    u attribute and x0. You will need to build a model of the unicycle model
    with the parameters provided to you in the lab doc. After building the
    model, use the provided inputs to estimate system state over time.

    The method should closely predict the state evolution if the system is
    free of noise. You may use this knowledge to verify your implementation.

    Example
    ----------
    To run dead reckoning:
        $ python drone_estimator_node.py --estimator dead_reckoning
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Dead Reckoning'

    def update(self, i):
        if len(self.x_hat) > 0:
            # TODO: Your implementation goes here!
            # Get the current state estimate
            x_prev = self.x_hat[-1]
            # u_prev = self.u[i - 1]  # Input at the previous time step

            # Extract state variables
            x, z, phi, x_dot, z_dot, phi_dot = x_prev
            u1, u2 = self.u[i]

            # Compute the derivatives using the dynamics model
            x_ddot = (-u1 * np.sin(phi)) / self.m
            z_ddot = -self.gr + (u1 * np.cos(phi)) / self.m
            phi_ddot = u2 / self.J

            # Update the state using the forward Euler method
            x_new = x + x_dot * self.dt
            z_new = z + z_dot * self.dt
            phi_new = phi + phi_dot * self.dt
            x_dot_new = x_dot + x_ddot * self.dt
            z_dot_new = z_dot + z_ddot * self.dt
            phi_dot_new = phi_dot + phi_ddot * self.dt

            # Append the new state estimate
            self.x_hat.append(np.array([x_new, z_new, phi_new, x_dot_new, z_dot_new, phi_dot_new]))
            # You may ONLY use self.u and self.x[0] for estimation
            # raise NotImplementedError

# noinspection PyPep8Naming
class ExtendedKalmanFilter(Estimator):
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
            landmark[2] is the z coordinate.

    Example
    ----------
    To run the extended Kalman filter:
        $ python drone_estimator_node.py --estimator extended_kalman_filter
    """
    def __init__(self, is_noisy=False):
        super().__init__(is_noisy)
        self.canvas_title = 'Extended Kalman Filter'
        # TODO: Your implementation goes here!
        # You may define the Q, R, and P matrices below.
        self.A = None
        self.B = None
        self.C = None


        # Process noise covariance
        self.Q = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Tune as needed
        # Measurement noise covariance
        self.R = np.diag([0.5, 0.5])  # Adjust based on sensor noise
        # State covariance matrix
        self.P = np.eye(6) * 1  # Initial state uncertainty


    # noinspection DuplicatedCode
    def update(self, i):
        if len(self.x_hat) > 0: #and self.x_hat[-1][0] < self.x[-1][0]:
            # TODO: Your implementation goes here!
            # You may use self.u, self.y, and self.x[0] for estimation
            # raise NotImplementedError
           

            # Get the latest control input and measurement
            u = self.u[i]
            y = self.y[i]

            # Get the previous state estimate and covariance
            x_prev = self.x_hat[-1]
            P_prev = self.P

            # State extrapolation (using nonlinear dynamics)
            x_pred = self.g(x_prev, u)

            # Linearize the dynamics model
            A = self.approx_A(x_prev, u)

            # Covariance extrapolation
            P_pred = A @ P_prev @ A.T + self.Q

            # Linearize the measurement model
            C = self.approx_C(x_pred)

            # Kalman gain
            K = P_pred @ C.T @ np.linalg.inv(C @ P_pred @ C.T + self.R)

            # Measurement update
            y_pred = self.h(x_pred)
            x_updated = x_pred + K @ (y - y_pred)

            # Covariance update
            P_updated = (np.eye(6) - K @ C) @ P_pred

            # Save the updated state and covariance
            self.x_hat.append(x_updated)
            self.P = P_updated


    def g(self, x, u):
        # raise NotImplementedError
        """Nonlinear dynamics model."""
        x_pos, z_pos, phi, x_vel, z_vel, phi_vel = x
        u1, u2 = u

        # Dynamics equations
        x_acc = (-u1 * np.sin(phi)) / self.m
        z_acc = -self.gr + (u1 * np.cos(phi)) / self.m
        phi_acc = u2 / self.J

        # State update
        x_pos_new = x_pos + x_vel * self.dt
        z_pos_new = z_pos + z_vel * self.dt
        phi_new = phi + phi_vel * self.dt
        x_vel_new = x_vel + x_acc * self.dt
        z_vel_new = z_vel + z_acc * self.dt
        phi_vel_new = phi_vel + phi_acc * self.dt

        return np.array([x_pos_new, z_pos_new, phi_new, x_vel_new, z_vel_new, phi_vel_new])





        # """Nonlinear dynamics model for the planar quadrotor."""
        # x_pos, z_pos, phi, vx, vz, omega = x
        # f1, f2 = u  # Control inputs (thrust from rotors)

        # # Operating point (x^*, u^*)
        # x_star = self.x_hat[-1] if len(self.x_hat) > 0 else x
        # u_star = self.u[-1] if len(self.u) > 0 else u

        # # Total thrust and net moment at operating point
        # u1_star = u_star[0] + u_star[1]
        # u2_star = (self.L / 2) * (u_star[0] - u_star[1])

        # # Dynamics at operating point
        # g_star = np.array([
        #     x_star[0] + x_star[3] * self.dt,  # x position
        #     x_star[1] + x_star[4] * self.dt,  # z position
        #     x_star[2] + x_star[5] * self.dt,  # phi (pitch angle)
        #     x_star[3] + (-u1_star * np.sin(x_star[2])) / self.mass * self.dt,  # vx
        #     x_star[4] + (-self.gr + (u1_star * np.cos(x_star[2])) / self.m) * self.dt,  # vz
        #     x_star[5] + (u2_star / self.J) * self.dt  # omega
        # ])

        # # Jacobian A (∂g/∂x)
        # A = self.approx_A(x_star, u_star)

        # # Jacobian B (∂g/∂u)
        # B = self.approx_B(x_star, u_star)

        # # Linearized dynamics
        # x_next = g_star + A @ (x - x_star) + B @ (u - u_star)

        # return x_next

    def h(self, x):
        # raise NotImplementedError
        """Nonlinear measurement model."""
        x_pos, z_pos, phi, _, _, _ = x
        lx, ly, lz = self.landmark

        # Distance to landmark
        distance = np.sqrt((lx - x_pos)**2 + ly**2 + (lz - z_pos)**2)
        # Bearing of the landmark
        bearing = phi

        return np.array([distance, bearing])

        
    def approx_A(self, x, u):
        # raise NotImplementedError
        """Linearize the dynamics model around the current state and input."""
        x_pos, z_pos, phi, x_vel, z_vel, phi_vel = x
        u1, u2 = u

        # Jacobian of the dynamics (A matrix)
        A = np.eye(6)  # Identity matrix for the state transition
        A[0, 3] = self.dt  # x_pos depends on x_vel
        A[1, 4] = self.dt  # z_pos depends on z_vel
        A[2, 5] = self.dt  # phi depends on phi_vel
        A[3, 2] = (-u1 * np.cos(phi) / self.m) * self.dt  # x_vel depends on phi
        A[4, 2] = (-u1 * np.sin(phi) / self.m) * self.dt  # z_vel depends on phi

        return A


    
    def approx_C(self, x):
        # raise NotImplementedError
        x_pos, z_pos, phi, _, _, _ = x
        lx, ly, lz = self.landmark

        # Jacobian of the measurement model (C matrix)
        distance = np.sqrt((lx - x_pos)**2 + ly**2 + (lz - z_pos)**2)
        C = np.zeros((2, 6))
        C[0, 0] = -(lx - x_pos) / distance  # Partial derivative of distance w.r.t. x_pos
        C[0, 1] = -(lz - z_pos) / distance  # Partial derivative of distance w.r.t. z_pos
        C[1, 2] = 1  # Partial derivative of bearing w.r.t. phi

        return C
    




    def approx_B(self, x, u):
        """Jacobian of the dynamics model with respect to u."""
        phi = x[2]  # Pitch angle

        B = np.zeros((6, 2))
        B[3, 0] = (-np.sin(phi)) / self.m * self.dt  # vx depends on f1
        B[3, 1] = (-np.sin(phi)) / self.m * self.dt  # vx depends on f2
        B[4, 0] = (np.cos(phi)) / self.m * self.dt  # vz depends on f1
        B[4, 1] = (np.cos(phi)) / self.m * self.dt  # vz depends on f2
        B[5, 0] = (self.L / (2 * self.J)) * self.dt  # omega depends on f1
        B[5, 1] = (-self.L / (2 * self.J)) * self.dt  # omega depends on f2

        return B


    

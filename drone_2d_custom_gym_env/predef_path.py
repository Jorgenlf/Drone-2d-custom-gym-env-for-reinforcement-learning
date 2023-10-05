import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import fminbound

from typing import Tuple

class QPMI2D():
    def __init__(self, waypoints):
        self.waypoints = waypoints
        self.wp_idx = 0
        self.us = self._calculate_us()
        self.length = self.us[-1]
        self.calculate_quadratic_params()
    
    def _calculate_us(self):
        '''
        Calculate the distance along the path from the beginning of the path to each waypoint
        '''
        diff = np.diff(self.waypoints, axis=0)
        seg_lengths = np.cumsum(np.sqrt(np.sum(diff**2, axis=1)))
        return np.array([0,*seg_lengths[:]])
    
    def calculate_quadratic_params(self):
        '''
        Calculate the quadratic parameters for each segment of the path
        '''
        self.x_params = []
        self.y_params = []
        for n in range(1, len(self.waypoints)-1):
            wp_prev = self.waypoints[n-1]
            wp_n = self.waypoints[n]
            wp_next = self.waypoints[n+1]
            
            u_prev = self.us[n-1]
            u_n = self.us[n]
            u_next = self.us[n+1]

            U_n = np.vstack([np.hstack([u_prev**2, u_prev, 1]),
                           np.hstack([u_n**2, u_n, 1]),
                           np.hstack([u_next**2, u_next, 1])])
            x_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[0], wp_n[0], wp_next[0]]))
            y_params_n = np.linalg.inv(U_n).dot(np.array([wp_prev[1], wp_n[1], wp_next[1]]))

            self.x_params.append(x_params_n)
            self.y_params.append(y_params_n)


    def get_u_index(self, u):
        ''' 
        Get the index of the segment of the path that the parameter u is in
        '''
        n = 0
        while n < len(self.us) - 1:
            if u <= self.us[n+1]:
                break
            else:
                n += 1
        return n


    def calculate_ur(self, u):
        '''
        Calculate the membership function for the segment of the path that the parameter u is in.
        μr,m and μf,m are increasing and decreasing membership functions with value spans between zero and one.
        They represent the transition from one polynomial to another.
        '''
        n = self.get_u_index(u)
        try:
            ur = (u - self.us[n]) / (self.us[n+1] - self.us[n])
        except IndexError:
            ur = (u - self.us[n]) / (self.us[n+1] - self.us[n])
        return ur


    def calculate_uf(self, u):
        '''
        Calculate the membership function for the segment of the path that the parameter u is in
        '''
        n = self.get_u_index(u)
        uf = (self.us[n+1]-u)/(self.us[n+1] - self.us[n])
        return uf        


    def __call__(self, u):
        '''
        Calculate the x and y coordinates of the path at parameter u
        '''
        if u >= self.us[0] and u <= self.us[1]: # first stretch
            ax = self.x_params[0][0]
            ay = self.y_params[0][0]
            bx = self.x_params[0][1]
            by = self.y_params[0][1]
            cx = self.x_params[0][2]
            cy = self.y_params[0][2]
            
            x = ax*u**2 + bx*u + cx
            y = ay*u**2 + by*u + cy

        elif u >= self.us[-2]-0.001 and u <= self.us[-1]: # last stretch
            ax = self.x_params[-1][0]
            ay = self.y_params[-1][0]
            bx = self.x_params[-1][1]
            by = self.y_params[-1][1]
            cx = self.x_params[-1][2]
            cy = self.y_params[-1][2]
            
            x = ax*u**2 + bx*u + cx
            y = ay*u**2 + by*u + cy

        else: # else we are in the intermediate waypoints and we use membership functions to calc polynomials
            n = self.get_u_index(u)
            ur = self.calculate_ur(u)
            uf = self.calculate_uf(u)
            
            ax1 = self.x_params[n-1][0]
            ay1 = self.y_params[n-1][0]
            bx1 = self.x_params[n-1][1]
            by1 = self.y_params[n-1][1]
            cx1 = self.x_params[n-1][2]
            cy1 = self.y_params[n-1][2]
            
            x1 = ax1*u**2 + bx1*u + cx1
            y1 = ay1*u**2 + by1*u + cy1

            ax2 = self.x_params[n][0]
            ay2 = self.y_params[n][0]
            bx2 = self.x_params[n][1]
            by2 = self.y_params[n][1]
            cx2 = self.x_params[n][2]
            cy2 = self.y_params[n][2]
            
            x2 = ax2*u**2 + bx2*u + cx2
            y2 = ay2*u**2 + by2*u + cy2

            x = ur*x2 + uf*x1
            y = ur*y2 + uf*y1

        return np.array([x, y])


    def calculate_gradient(self, u):
        '''
        Calculate the gradient of the path at parameter u
        '''
        if u >= self.us[0] and u <= self.us[1]: # first stretch
            ax = self.x_params[0][0]
            ay = self.y_params[0][0]
            bx = self.x_params[0][1]
            by = self.y_params[0][1]
            
            dx = ax*u*2 + bx
            dy = ay*u*2 + by
        elif u >= self.us[-2]: # last stretch
            ax = self.x_params[-1][0]
            ay = self.y_params[-1][0]
            bx = self.x_params[-1][1]
            by = self.y_params[-1][1]
            
            dx = ax*u*2 + bx
            dy = ay*u*2 + by
        else: # else we are in the intermediate waypoints and we use membership functions to calc polynomials
            n = self.get_u_index(u)
            ur = self.calculate_ur(u)
            uf = self.calculate_uf(u)
            
            ax1 = self.x_params[n-1][0]
            ay1 = self.y_params[n-1][0]
            bx1 = self.x_params[n-1][1]
            by1 = self.y_params[n-1][1]
            
            dx1 = ax1*u*2 + bx1
            dy1 = ay1*u*2 + by1

            ax2 = self.x_params[n][0]
            ay2 = self.y_params[n][0]
            bx2 = self.x_params[n][1]
            by2 = self.y_params[n][1]
            
            dx2 = ax2*u*2 + bx2
            dy2 = ay2*u*2 + by2

            dx = ur*dx2 + uf*dx1
            dy = ur*dy2 + uf*dy1
        return np.array([dx, dy])
    

    def calculate_vectors(self, u: float) -> Tuple[np.array, np.array]:
        """
        Calculate path describing vectors at point u.

        Parameters:
        ----------
        u : float
            Distance along the path from the beginning of the path

        Returns:
        -------
        t_hat : np.array
            Unit tangent vector
        n_hat : np.array
            Unit normal vector - perpendicular to the tangent vector
        """

        dp = self.calculate_gradient(u)

        t_hat = dp / np.linalg.norm(dp)
        n_hat = np.array([-t_hat[1], t_hat[0]])  # Perpendicular to the tangent vector

        return t_hat, n_hat


    def get_direction_angles(self, u):
        '''
        Calculate the azimuth angle of the path at parameter u
        '''
        dx, dy = self.calculate_gradient(u)[:]
        azimuth = np.arctan2(dy, dx)
        return azimuth


    def get_closest_u(self, position, wp_idx, margin=10.0):
        '''
        Calculate the parameter u of the path that is closest to the given position
        '''
        x1 = self.us[wp_idx] - margin
        x2 = self.us[wp_idx+1] + margin if wp_idx < len(self.us) - 2 else self.length
        output = fminbound(lambda u: np.linalg.norm(self(u) - position), 
                        full_output=0, x1=x1, x2=x2, xtol=1e-6, maxfun=500)
        return output


    def get_closest_position(self, position, wp_idx):
        '''
        Calculate the position on the path that is closest to the given position
        '''
        return self(self.get_closest_u(position, wp_idx))


    def get_endpoint(self):
        '''
        Calculate the endpoint of the path
        '''
        return self(self.length)


    def plot_path(self, wps_on=True):
        '''
        Plot the path
        '''
        u = np.linspace(self.us[0], self.us[-1], 10000)
        quadratic_path = []
        for du in u:
            quadratic_path.append(self(du))
            self.get_direction_angles(du)
        quadratic_path = np.array(quadratic_path)
        plt.plot(quadratic_path[:, 0], quadratic_path[:, 1], color="#3388BB", label="Path")
        if wps_on:
            for i, wp in enumerate(self.waypoints):
                plt.scatter(*wp, color="#EE6666", label="Waypoints" if i == 1 else None)
        
        plt.xlabel(xlabel="X [m]", fontsize=14)
        plt.ylabel(ylabel="Y [m]", fontsize=14)
        plt.legend(fontsize=14)

        return plt


def generate_random_waypoints_2d(nwaypoints, scen):
    waypoints = [np.array([0, 0])]

    if scen == '2d':
        for i in range(nwaypoints - 1):
            distance = 50
            azimuth = np.random.uniform(-np.pi / 4, np.pi / 4)
            x = waypoints[i][0] + distance * np.cos(azimuth)
            y = waypoints[i][1] + distance * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)

    return np.array(waypoints)


if __name__ == "__main__":
    # wps = np.array([np.array([0, 0]), np.array([20, 10]), np.array([50, 20]), np.array([80, 20]), np.array([90, 50]), np.array([80, 80]), np.array([50, 80]), np.array([20, 60]), np.array([20, 40]), np.array([0, 0])])
    wps = generate_random_waypoints_2d(10, scen='2d')
    path = QPMI2D(wps)

    point = path(20)
    azi = path.get_direction_angles(20)
    vec_x = point[0] + 20 * np.cos(azi)
    vec_y = point[1] + 20 * np.sin(azi)

    plt.figure()
    ax = path.plot_path()
    plt.plot(wps[:, 0], wps[:, 1], linestyle="dashed", color="#33bb5c")
    plt.scatter(*point, label="Current Position", color="b")
    plt.quiver(*point, vec_x - point[0], vec_y - point[1], angles='xy', scale_units='xy', scale=1, color="g", label="Tangent Vector")
    plt.legend(fontsize=14)
    plt.xlabel(xlabel="X [m]", fontsize=14)
    plt.ylabel(ylabel="Y [m]", fontsize=14)
    plt.rc('lines', linewidth=3)
    plt.show()

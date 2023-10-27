import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def ssa(angle):
    return ((angle + np.pi) % (2*np.pi)) - np.pi

def R_w_b(theta):
    return np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])

def translate(x, y):
    return np.array([[1, 0, x],
                     [0, 1, y],
                     [0, 0, 1]])

def homogeneous_transform(x, y, theta):
    return np.array([[np.cos(theta), -np.sin(theta), x],
                     [np.sin(theta), np.cos(theta), y],
                     [0, 0, 1]])


#Testing
if __name__ == '__main__':
    point_w = np.array([5, 3])
    body__origin_w = np.array([2, 2.5])
    body__angle = -np.pi/4

    point_b = np.matmul(R_w_b(body__angle), point_w - body__origin_w)

    print(point_b)

    #Angle between body and point_b
    angle_b_to_point_in_b = ssa(np.arctan2(point_b[1], point_b[0]))

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(point_w[0], point_w[1], 'ro', label='Point W')
    ax.plot(body__origin_w[0], body__origin_w[1], 'bo', label='Body Origin W')

    # Plot vector between body origin and point
    ax.plot([body__origin_w[0], point_w[0]], [body__origin_w[1], point_w[1]], 'r', label='Vector between Body Origin and Point')

    # Plot body frame using the body origin and body_angle
    body_end_x = body__origin_w[0] + 0.5 * np.cos(body__angle)
    body_end_y = body__origin_w[1] + 0.5 * np.sin(body__angle)
    ax.plot([body__origin_w[0]-0.5 * np.cos(body__angle), body_end_x], [body__origin_w[1]-0.5 * np.sin(body__angle), body_end_y], 'b', linewidth=3, label='Body Frame')

    # Plot angle between body and point in body frame as an arc
    arc = patches.Arc((body__origin_w[0], body__origin_w[1]), 1, 1, theta1=body__angle*180/np.pi, theta2=(-body__angle+angle_b_to_point_in_b)*180/np.pi, color='g', label='Angle between body and point in body frame')
    ax.add_patch(arc)

    plt.grid()
    plt.axis('equal')
    plt.legend()
    plt.show()




from obstacles import*
from predef_path import*

def generate_scen_obstacles(n, scen, space, path:QPMI2D,obs_size):
    obstacles = []
    color = (188, 72, 72)
    if scen == 'perpendicular':
        x=1
    elif scen == 'parallel':
        path_lenght = path.length
        for i in range(2,n+2):
            u_obs = i*path_lenght/(n+1)
            x,y = path.__call__(u_obs)
            obstacles.append(obs)
            obs = Circle(x,y,obs_size,color,space)    
            obstacles.append(obs)
    return obstacles


def generate_scen_waypoints_2d(nwaypoints, distance, scen, screen_x = None,screen_y = None):
    waypoints = []
    if scen == 'perpendicular' or scen == 'parallel' or scen == 'corridor' or scen == 'impossible':
        x1 = 100
        y1 = screen_y/2
        waypoints = [np.array([x1, y1])]
        for i in range(nwaypoints - 1):
            azimuth = 0
            x = waypoints[i][0] + distance * np.cos(azimuth)
            y = waypoints[i][1] + distance * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
    elif scen == 'S_parallel' or scen == 'S_corridor':
        x1 = 100
        y1 = screen_y/2
        waypoints = [np.array([x1, y1])]
        for i in range(nwaypoints - 1):
            if i % 2 == 0:
                azimuth = -np.pi/6
            else:
                azimuth = np.pi/6
            x = waypoints[i][0] + distance * np.cos(azimuth)
            y = waypoints[i][1] + distance * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
    elif scen == 'large':
        x1 = 100
        y1 = screen_y/2
        waypoints = [np.array([x1, y1])]
        obs_rad = screen_x/5
        margin = 50
        distance = screen_x/10
        circle_to_follow_radius = obs_rad + margin
        #second waypoint is distance pixels to the right of the first wp
        waypoints.append(np.array([x1+distance,y1]))
        #third to n-2 waypoints are on the circle to follow
        #start at 0 degrees and go counterclockwise
        #start at i=1 because we already have two waypoints end before n-2 because we want to end with two waypoints
        for i in range(1,nwaypoints-1):
            #Start at 270 degrees and go counterclockwise make a half cirlce that ends at same y as start
            azimuth = np.pi/2 - (i-1)*np.pi/(nwaypoints-3)
            x = waypoints[i][0] + circle_to_follow_radius * np.cos(azimuth)
            y = waypoints[i][1] + circle_to_follow_radius * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
        #Two last waypoints are distance pixels to the right of the last wp on the cirlce to follow
        prev_x = waypoints[-1][0]
        prev_y = waypoints[-1][1]
        waypoints.append(np.array([prev_x+distance,prev_y]))
        # prev_x = waypoints[-1][0]
        # prev_y = waypoints[-1][1]
        # waypoints.append(np.array([prev_x+distance,prev_y]))


    return np.array(waypoints)

def create_test_scenario(space,scen:str,screen_x:int,screen_y:int):
    '''Creates test scenarios for the drone to navigate through. Possible scenarios are:
    - perpendicular: obstacles are perpendicular to the path
    - parallel: obstacles are parallel on the path
    - corridor: obstacles are on both sides of the path
    - S_parallel: obstacles are parallel on the path, but in an S shape
    - S_corridor: obstacles are on both sides of the path, but in an S shape
    - impossible: obstacles are surrounding the endgoal
    - large: one large obstacle that the path (and drone) must go around'''
    path = None
    obstacles = []

    if scen == 'perpendicular':
        wps = generate_scen_waypoints_2d(10,100,'perpendicular',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(10,'perpendicular',space,path)
    elif scen == 'parallel':
        wps = generate_scen_waypoints_2d(10,100,'parallel',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(10,'parallel',space,path)
    elif scen == 'corridor':
        wps = generate_scen_waypoints_2d(10,100,'corridor',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(10,'corridor',space,path)
    elif scen == 'S_parallel':
        wps = generate_scen_waypoints_2d(10,100,'S_parallel',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(10,'S_parallel',space,path)
    elif scen == 'S_corridor': 
        wps = generate_scen_waypoints_2d(10,100,'S_corridor',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(10,'S_corridor',space,path)
    elif scen == 'impossible':
        wps = generate_scen_waypoints_2d(10,100,'impossible',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(10,'impossible',space,path)
    elif scen == 'large':
        wps = generate_scen_waypoints_2d(10,100,'large',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(10,'large',space,path)
    
    return path,obstacles

if __name__ == "__main__":
    screen_x = 1000
    screen_y = 1000 
    wps = generate_scen_waypoints_2d(10,100,'large',screen_x,screen_y)
    test_path = QPMI2D(wps)
    test_path.mpl_plot_path()
    plt.show()

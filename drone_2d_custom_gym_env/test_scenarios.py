from obstacles import*
from predef_path import*

def generate_scen_obstacles(n, scen, space, path:QPMI2D,obs_size,screen_x = None,screen_y = None):
    obstacles = []
    color = (188, 72, 72)
    if scen == 'perpendicular':
        half_path_lenght = path.length/2
        path_angle = path.get_direction_angle(half_path_lenght)
        u_obs = half_path_lenght
        x,y = path.__call__(u_obs)
        obs_on_path_pos = np.array([x,y])
        start = n*obs_size - obs_size
        for i in range(0,n):
            obs_pos = obs_on_path_pos + (start - i*obs_size*2)*np.array([np.cos(path_angle-np.pi/2),np.sin(path_angle-np.pi/2)])
            obs = Circle(obs_pos[0],obs_pos[1],obs_size,color,space)  
            obstacles.append(obs)

    elif scen == 'parallel':
        path_lenght = path.length
        space_occupied = n*obs_size*2
        offset = (path_lenght - space_occupied)/2
        for i in range(1,n+1):
            u_obs = offset + i*obs_size*2
            x,y = path.__call__(u_obs)
            obs = Circle(x,y,obs_size,color,space)    
            obstacles.append(obs)
    elif scen == 'S_parallel':
        path_lenght = path.length
        space_occupied = n*obs_size*2
        offset = (path_lenght - space_occupied)/2
        for i in range(1,n+1):
            u_obs = offset + i*obs_size*2
            x,y = path.__call__(u_obs)
            obs = Circle(x,y,obs_size,color,space)    
            obstacles.append(obs)

    elif scen == 'corridor':
        #Assume the path given is to be populated with n obstacles along the whole path
        #Must then scale the obstacle size to fit the path
        n=10
        path_lenght = path.length
        obs_size = path_lenght/(n*2)
        for i in range(1,n+1):
            u_obs = i*obs_size*2
            x,y = path.__call__(u_obs)
            obs = Circle(x,y,obs_size,color,space)    
            obstacles.append(obs)
    elif scen == 'S_corridor':
        path_lenght = path.length
        obs_size = path_lenght/(n*2)
        for i in range(1,n+1):
            u_obs = i*obs_size*2
            x,y = path.__call__(u_obs)
            obs = Circle(x,y,obs_size,color,space)    
            obstacles.append(obs)

    elif scen == 'impossible':
        path_lenght = path.length
        end_goal_circle_radius = 100
        circumference = 2*np.pi*end_goal_circle_radius
        obs_size = circumference/(n*2)
        path_angle = path.get_direction_angle(path_lenght)
        u_obs = path_lenght
        x,y = path.__call__(u_obs)
        obs_on_path_pos = np.array([x,y])
        #Must iterate over the angle of the circle to place the obstacles
        pi_update = 2*np.pi/n
        for i in range(1,n+1):
            obs_pos = obs_on_path_pos + end_goal_circle_radius*np.array([np.cos(path_angle-i*pi_update),np.sin(path_angle-i*pi_update)])
            obs = Circle(obs_pos[0],obs_pos[1],obs_size,color,space)  
            obstacles.append(obs)
    
    elif scen == 'large':
        x = screen_x/2
        y = screen_y/2
        obs = Circle(x,y,obs_size,color,space)
        obstacles.append(obs)

    return obstacles


def generate_scen_waypoints_2d(nwaypoints, distance, scen, screen_x = None,screen_y = None, offset = 0):
    waypoints = []
    if scen == 'perpendicular' or scen == 'parallel' or scen == 'impossible':
        x1 = screen_x/10
        y1 = screen_y/2
        waypoints = [np.array([x1, y1])]
        for i in range(nwaypoints - 1):
            azimuth = 0
            x = waypoints[i][0] + distance * np.cos(azimuth)
            y = waypoints[i][1] + distance * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
    elif scen == 'S_parallel':
        x1 = screen_x/10
        y1 = screen_y/2
        waypoints = [np.array([x1, y1])]
        phase = np.pi/4
        for i in range(nwaypoints - 1):
            if i % 2 == 0:
                azimuth = -phase
            else:
                azimuth = phase
            x = waypoints[i][0] + distance * np.cos(azimuth)
            y = waypoints[i][1] + distance * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
    elif scen == 'corridor':
        x1 = screen_x/20
        y1 = screen_y/2 + offset
        waypoints = [np.array([x1, y1])]
        for i in range(nwaypoints - 1):
            azimuth = 0
            x = waypoints[i][0] + distance * np.cos(azimuth)
            y = waypoints[i][1] + distance * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
    elif scen == 'S_corridor':
        x1 = screen_x/10
        y1 = screen_y/2 + offset
        waypoints = [np.array([x1, y1])]
        phase = np.pi/4
        for i in range(nwaypoints - 1):
            if i % 2 == 0:
                azimuth = -phase
            else:
                azimuth = phase
            x = waypoints[i][0] + distance * np.cos(azimuth)
            y = waypoints[i][1] + distance * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
    elif scen == 'large':
        x1 = screen_x/10
        y1 = screen_y/2
        waypoints = [np.array([x1, y1])]
        obs_rad = screen_x/5
        margin = 50
        distance = screen_x/10
        circle_to_follow_radius = obs_rad + margin
        waypoints.append(np.array([x1+distance,y1]))
        for i in range(1,nwaypoints-1):
            azimuth = np.pi/2 - (i-1)*np.pi/(nwaypoints-3)
            x = waypoints[i][0] + circle_to_follow_radius * np.cos(azimuth)
            y = waypoints[i][1] + circle_to_follow_radius * np.sin(azimuth)
            wp = np.array([x, y])
            waypoints.append(wp)
        prev_x = waypoints[-1][0]
        prev_y = waypoints[-1][1]
        waypoints.append(np.array([prev_x+distance,prev_y]))


    return np.array(waypoints)

def create_test_scenario(space,scen:str,
                         screen_x:int,
                         screen_y:int, 
                         offset:int = 0,
                         obs_size:int = 30,
                         n_wps:int = 10,
                         n_obs:int = 6):
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
    wps = np.array([])
    if scen == 'perpendicular':
        n_obs = 6
        obs_size = 20
        wps = generate_scen_waypoints_2d(n_wps,100,'perpendicular',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(n_obs,'perpendicular',space,path,obs_size)
    elif scen == 'parallel':
        wps = generate_scen_waypoints_2d(n_wps,100,'parallel',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(n_obs,'parallel',space,path,obs_size)
    elif scen == 'corridor':
        offset = 200
        wps = generate_scen_waypoints_2d(n_wps,100,'corridor',screen_x,screen_y)
        pluss_offset_wps = generate_scen_waypoints_2d(n_wps,100,'corridor',screen_x,screen_y,offset)
        minus_offset_wps = generate_scen_waypoints_2d(n_wps,100,'corridor',screen_x,screen_y,-offset)

        path = QPMI2D(wps)
        po_path = QPMI2D(pluss_offset_wps)
        mo_path = QPMI2D(minus_offset_wps)

        obstacles = generate_scen_obstacles(n_obs,scen,space,po_path,obs_size)
        obstacles2 = generate_scen_obstacles(n_obs,scen,space,mo_path,obs_size)
        obstacles.extend(obstacles2)
    elif scen == 'S_parallel':
        segment_length = 300
        n_wps = 6
        n_obs = 20
        obs_size = 15
        wps = generate_scen_waypoints_2d(n_wps,segment_length,scen,screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(n_obs,scen,space,path,obs_size)
    elif scen == 'S_corridor': 
        offset = 100
        segment_length = 200
        n_wps = 7
        n_obs = 30
        wps = generate_scen_waypoints_2d(n_wps,segment_length,scen,screen_x,screen_y)
        pluss_offset_wps = generate_scen_waypoints_2d(n_wps,segment_length,scen,screen_x,screen_y,offset)
        minus_offset_wps = generate_scen_waypoints_2d(n_wps,segment_length,scen,screen_x,screen_y,-offset)

        path = QPMI2D(wps)
        po_path = QPMI2D(pluss_offset_wps)
        mo_path = QPMI2D(minus_offset_wps)

        obstacles = generate_scen_obstacles(n_obs,scen,space,po_path,obs_size=None)
        obstacles2 = generate_scen_obstacles(n_obs,scen,space,mo_path,obs_size=None)
        obstacles.extend(obstacles2)
    elif scen == 'impossible':
        n_obs = 20
        wps = generate_scen_waypoints_2d(n_wps,100,'impossible',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(n_obs,'impossible',space,path,obs_size)
    elif scen == 'large':
        n_obs = 1
        obs_size = screen_x/5
        wps = generate_scen_waypoints_2d(n_wps,100,'large',screen_x,screen_y)
        path = QPMI2D(wps)
        obstacles = generate_scen_obstacles(n_obs,'large',space,path,obs_size,screen_x,screen_y)
    
    return wps,path,obstacles

if __name__ == "__main__":
    screen_x = 1000
    screen_y = 1000 
    

    #Corridor
    wps = generate_scen_waypoints_2d(10,100,'corridor',screen_x,screen_y)
    test_path = QPMI2D(wps)
    po_wps = generate_scen_waypoints_2d(10,100,'corridor',screen_x,screen_y,100)
    po_path = QPMI2D(po_wps)
    mo_wps = generate_scen_waypoints_2d(10,100,'corridor',screen_x,screen_y,-100)
    mo_path = QPMI2D(mo_wps)
    po_path.mpl_plot_path()
    mo_path.mpl_plot_path()
    test_path.mpl_plot_path()
    plt.show()

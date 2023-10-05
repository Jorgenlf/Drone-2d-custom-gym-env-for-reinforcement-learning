import numpy as np

#My implementation of the Quadratic Polynomial Interpolation (QPMI) path-smoothing method 
# for generating a continuous 2D path from a set of waypoints

class PredefPath():
    def __init__(self, init_w_point, n_waypoints:int,d,chi_m_range:tuple, nu_m_range:tuple):

        WPs = np.zeros((n_waypoints,2))
        WPs[0,:] = init_w_point

        for i in range (1,n_waypoints-1):
            chi_m = np.random.uniform(chi_m_range) 
            nu_m = np.random.uniform(nu_m_range)  

            xm_prev = WPs[i-1,0]
            ym_prev = WPs[i-1,1]

            xm = xm_prev + d * np.cos(chi_m) * np.cos(nu_m)
            ym = ym_prev + d * np.sin(chi_m) * np.cos(nu_m)

            np.append(WPs,[xm,ym],axis=0)

        self.WPs = WPs


pp = PredefPath([0,0],4,2,(-np.pi,np.pi),(-np.pi/2,np.pi/2))     
print(pp.WPs)

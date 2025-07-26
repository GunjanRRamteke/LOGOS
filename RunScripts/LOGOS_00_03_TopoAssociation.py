import numpy as np
from ase.io import read, write, Trajectory
import math

def plane_stuffs(V,rP,  pts):

    """
    V  = plane normal
    rP = any point on plane 
    d = constant

    """

    d = np.dot(V, rP)
    a, b, c = V
    proj_pts = []

    for D in pts:
        proj_pts.append(D - ((a*D[0] + b*D[1] + c*D[2] + d) / (a**2 + b**2 + c**2)) * np.array([a, b, c]))

    return(proj_pts)



def rotation(ideal, rotor, cn1, cn2):   

    def direction_check(V1, V2, theta, AR):

        V1 = V1 / np.linalg.norm(V1)
        V2 = V2 / np.linalg.norm(V2)
        
        dev_1 = abs(np.sum(V1 - (rodrigues([V2], theta, AR))))
        dev_2 = abs(np.sum(V1 - (rodrigues([V2],-theta, AR))))

        if (dev_1 > dev_2):
            theta = -theta

        return(theta)



    """    rotation_1     """

    theta = angle_between_two_vectors(ideal[cn1], rotor[cn1])
    AR = axis_of_rotation(ideal[cn1], rotor[cn1], theta)
    theta = direction_check(ideal[cn1].copy(), rotor[cn1].copy(), theta, AR)
    N_C_1 = rodrigues(rotor, theta, AR)
    

    """    rotation_2     """


    AR =  N_C_1[cn1]/np.linalg.norm(N_C_1[cn1])
    rP = (N_C_1[cn1]/np.linalg.norm(N_C_1[cn1])) * 2.5

    theta = angle_between_three_points(plane_stuffs(AR, rP, (N_C_1[cn2], rP,  ideal[cn2])))
    theta = direction_check(ideal[cn2].copy(), N_C_1[cn2].copy(), theta, AR)

    N_C_2 = rodrigues(N_C_1.copy(), theta, AR)
    N_C_2[cn2] = N_C_2[cn2]/np.linalg.norm(N_C_2[cn2])


    return(N_C_2)


def angle_between_three_points(pts):

    pts = pts - pts[1]
    V1 = pts[0]
    V2 = pts[2]
    return (angle_between_two_vectors(V1, V2))


def angle_between_two_vectors(Rv, Ov):

    Ov = Ov/ np.linalg.norm(Ov)
    Rv = Rv/ np.linalg.norm(Rv)

    num = np.dot(Ov,Rv) / (np.linalg.norm(Ov) * np.linalg.norm(Rv))
    theta = np.arccos(num)

    return(theta)


def axis_of_rotation(Ov, Rv, alpha):
    return(np.cross(Ov,Rv) / (np.linalg.norm(Ov) * np.linalg.norm(Rv) * np.sin(alpha)))


def rodrigues(rotor, theta, AR):

    v_rot = np.zeros([len(rotor),3])

    for i in range(0, len(rotor)):
        
        v = rotor[i]
        term1 = v * np.cos(theta)
        term2 = np.cross(AR, v) * np.sin(theta)
        dot_p = (np.dot(AR, v)) 
        term3 = AR * (np.dot(v, AR)) * (1-np.cos(theta))
        v_rot[i]  = term1 + term2 + term3

    return(v_rot)



def superimpose(target, cn0=0, cn1=1, cn2=2):

    N_C_pack = []
    ideal = target[0]
    i_fac = ideal[cn0]
    ideal = ideal - i_fac

    for i in range(1, len(target)):

        rotor = target[i] - target[i][cn0]
        N_C = rotation(ideal, rotor, cn1, cn2)
        N_C_pack = N_C+i_fac
   
    return(N_C_pack)



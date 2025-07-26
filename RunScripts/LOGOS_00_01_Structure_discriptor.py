import numpy as np
import time
from ase.io import read, write, Trajectory
import os


'''
input is atomic cordinates

'''

def radial_symmetry_function(clu1, clu2, r_c):

    eta = 0.01
    r_s = 0
    G1 = np.zeros([len(clu1),2])
    sym = []
    for i in range(len(clu1)):
        for j in range(len(clu2)):
            r_ij = np.linalg.norm(clu1[i]-clu2[j])
            if (r_ij > r_c):
                f_c = 0
            else:
                f_c = 0.5 * (np.cos((np.pi * r_ij) /r_c) + 1)
            G1[i,0] = G1[i,0] + (np.exp (-eta *(r_ij - r_s) **2) * (f_c))
        G1[i,1] = i
    return(G1[i,0])


class StructureDiscriptor():

    def __init__(sf, clu_1, clu_2, rc):

        sf.clu_1  =  clu_1
        sf.clu_2  =  clu_2
        sf.rc     =  rc

    def radial_symmetry_function_G_s(sf):

        """
        G3 is a COS Function And So Is Complicated and to Be Used With Extreme Care

        """
        r_s, sym, k_p = 0, [], 1

        eta_r = np.linspace(3, 0.01, 10)

        G2 = np.zeros([len(sf.clu_1), len(sf.clu_2)])
        G3 = np.zeros([len(sf.clu_1), len(sf.clu_2)])

        r_ij_M = np.zeros([len(sf.clu_1), len(sf.clu_2)])

        for i in range(len(sf.clu_1)):
            for j in range(len(sf.clu_2)):
                r_ij_M[i,j] = np.linalg.norm(sf.clu_1[i]-sf.clu_2[j])

        for eta in eta_r:
            for i in range(len(sf.clu_1)):
                for j in range(len(sf.clu_2)):
                    r_ij = r_ij_M[i,j]
                    if (r_ij > sf.rc):
                        f_c = 0
                    else:
                        f_c = 0.5 * (np.cos((np.pi * r_ij) /sf.rc) + 1)

                    G2[i,j] = G2[i,j] + np.exp (-eta *(r_ij - r_s) **2) * (f_c)
                    G3[i,j] = G3[i,j] + np.cos (-k_p *(r_ij)) * (f_c)

        return(G2)


    def angular_symmetry_function(sf):

        zeta = 0.1
        eta = 0.01
        rs = 0
        lamda = 1
        R_ij   = np.zeros([3])
        R_ik   = np.zeros([3])
        fc     = np.zeros([3])
        r_dist = np.zeros([3])
        G4     = np.zeros([len(sf.clu_1)])
        G5     = np.zeros([len(sf.clu_1)])
        
        for i in range(len(sf.clu_1)):
            G4_list = []
            G5_list = []
            for j in range(len(sf.clu_2)):
                for k in range(len(sf.clu_2)):
                    if (j != k):

                        R_ij = sf.clu_1[i] - sf.clu_2[j]
                        R_ik = sf.clu_1[i] - sf.clu_2[k]
                        r_dist[0] = np.linalg.norm(sf.clu_1[i] - sf.clu_2[j])
                        r_dist[1] = np.linalg.norm(sf.clu_1[i] - sf.clu_2[k])
                        r_dist[2] = np.linalg.norm(sf.clu_2[j] - sf.clu_2[k])

                        dot = 0
                        for l in range(3):
                            dot = dot + (R_ij[l] * R_ik[l])

                        theta = np.arccos(np.around((dot/(r_dist[0] * r_dist[1])),4))

                        for l in range(3):
                            if (r_dist[l] <= sf.rc):
                               fc[l]  = 0.5 * (np.cos((np.pi * r_dist[l])/sf.rc) + 1)
                            else:
                               fc[l] = 0.0

                        G4_list.append(((1 + (lamda * np.cos(theta))) ** zeta) * np.exp (-eta * ((r_dist[0]**2) + (r_dist[1]**2) + (r_dist[2]**2))) * (fc[0]) * (fc[1]) * (fc[2]))
                        G5_list.append(((1 + (lamda * np.cos(theta))) ** zeta) * np.exp (-eta * ((r_dist[0]**2) + (r_dist[1]**2) )) * (fc[0]) * (fc[1]))

            G4[i] = (2 ** (1-zeta)) * (sum(G4_list))
            G5[i] = (2 ** (1-zeta)) * (sum(G5_list))

        return(G4)



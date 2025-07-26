import numpy as np
import time
from ase.io import read, write, Trajectory
import os


'''
input is atomic cordinates

'''


def r_fit(clu_1, clu_2):

    m = 3
    r_ij = [] #np.zeros([len(clu_1), len(clu_2)])

    for i in range(len(clu_1)):
        for j in range(len(clu_2)):
#            r_ij[i,j] = np.linalg.norm(clu_1[i]-clu_2[j])
            r_ij.append(np.linalg.norm(clu_1[i]-clu_2[j]))


    """ 
    Srt_rg, Lng_rg = [], []
    for i in range(0, len(clu_1), m):
        for j in range(i, i+m):
            for k in range(i+m, len(clu_2)):
                if (r_ij[j,k] < 6.0):
                    Srt_rg.append(r_ij[j,k])
                else:
                    Lng_rg.append(r_ij[j,k])

    """ 
#    return(min(Srt_rg), min(Lng_rg))
    return(min(r_ij)) #, max(Lng_rg))
#    return(sum(Srt_rg)/len(Srt_rg), sum(Lng_rg)/len(Lng_rg))


def radial_symmetry_function_ETA(clu, step):

    m = 3
    r_ij = np.zeros([len(clu), len(clu)])
    for i in range(len(clu)):
        for j in range(len(clu)):
            r_ij[i,j] = np.linalg.norm(clu[i]-clu[j])

    dist = []
    for i in range(0, len(clu), m):
        for j in range(i, i+m):
            for k in range(i+m, len(clu)):
                dist.append(r_ij[j,k])

    eta_r = np.linspace(0.1, 0.2, step)
    r_c = 2.2
    r_s = 0
    G = [[0] for i in range(step)]
    sym = []

    for x, eta in enumerate(eta_r):
        for i in range(len(clu)):
            for j in range(len(clu)):
                if (r_ij[i,j] > r_c):
                    f_c = 0
                else:
                    f_c = 0.5 * (np.cos((np.pi * r_ij[i,j]) /r_c) + 1)
                G[x] = G[x] + (np.exp (-eta *(r_ij[i,j] - r_s) **2) * (f_c))

    return(dist,G)




def radial_symmetry_function_Rcut(clu, rc_v):
    m = 3
    r_ij = np.zeros([len(clu), len(clu)])
    for i in range(len(clu)):
        for j in range(len(clu)):
            r_ij[i,j] = np.linalg.norm(clu[i]-clu[j])

    dist = []
    for i in range(0, len(clu), m):
        for j in range(i, i+m):
            for k in range(i+m, len(clu)):
                dist.append(r_ij[j,k])

    r_cut = np.linspace(2.0, 3.4, rc_v)
    eta = 0.001
    r_s = 0

    G = [[0] for i in range(rc_v)]
    sym = []
    for x, r_c in enumerate(r_cut):
        for i in range(len(clu)):
            for j in range(len(clu)):
                if (r_ij[i,j] > r_c):
                    f_c = 0
                else:
                    f_c = 0.5 * (np.cos((np.pi * r_ij[i,j]) /r_c) + 1)
                G[x] = G[x] + (np.exp (-eta *(r_ij[i,j] - r_s) **2) * (f_c))

    return(dist,G)


class StructureDiscriptor():

    def __init__(sf, clu_1, clu_2, rc):

        sf.clu_1  =  clu_1
        sf.clu_2  =  clu_2
        sf.rc     =  rc

    def radial_symmetry_function_G_s(sf, clu_1=False, clu_2=False):

#        (clu_1)
#        print(clu_2)
#        exit()

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



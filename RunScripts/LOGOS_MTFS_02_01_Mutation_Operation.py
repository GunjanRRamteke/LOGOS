import math
import os
import subprocess
import collections
import numpy                            as np
import random                           as rd
import matplotlib.pyplot                as plt
from time import time 

from ase                                import Atoms
from ase.io                             import read, write, Trajectory
from ase.visualize                      import view
from itertools                          import combinations, chain

#from LOGOS_00_00_objects import PICKLE_load, PICKLE_unload
from LOGOS_MTFS_04_01_Potential         import  abinitio_g09
import LOGOS_00_01_Structure_discriptor                            as symF
from LOGOS_00_03_TopoAssociation        import superimpose         as associate
from LOGOS_MTFS_00_Structure_discriptor import r_fit

def axis_of_rotation(Ov, Rv, alpha):
    if alpha == 0.0:
        alpha = 0.0001
    denom = (np.linalg.norm(Ov) * np.linalg.norm(Rv) * np.sin(alpha))
    if denom == 0.0:
        denom = 0.0001
    return(np.cross(Ov,Rv) / denom)

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


class T_Mutation():

    def __init__(sf, LC, E_t, MutMag, mutation_F_coe, matm):

        sf.LC             = LC
        sf.E_t            = E_t
        sf.mutation_F_coe = mutation_F_coe
        sf.matm           = matm
        sf.MutMag         = MutMag

    def FSO_Mutation(sf, MFea1, MFea2, cs, s_pos, TPM, Discr_env, Discr, g_Mon, PFit, cycle, x):

        print('Construction of Topography Interaction Matrix and consequent evaluation')

        F1r, F2r = MFea1[1], MFea2[1]
        F1_ls_r = [TPM[F1r,ii] for ii in range(sf.E_t*(sf.LC+1))]
        Tm_1 = F1_ls_r.index(min(F1_ls_r))
        F2r, F2c = MFea2[1], MFea2[2]
        F2_ls_r = [TPM[F2r,ii] for ii in range(sf.E_t*(sf.LC+1))]
        Tm_2 = F2_ls_r.index(min(F2_ls_r))

        def get_mid_points(coords1, coords2):

            M = sf.MutMag
            points = []
            for i in range(0,M+1):
                P = []
                for j in range(3):
                    P.append(coords1[j] + (i / M) * (coords2[j] - coords1[j]))
                points.append(P)
            return points

        # Mutation Vectors

        a_env_v1 = Discr_env[Tm_1]
        a_nml_v1 = Discr[F1r]
        mdps_V1 = get_mid_points(a_nml_v1, a_env_v1)
        
        a_env_v2 = Discr_env[Tm_2]
        a_nml_v2 = Discr[F2r]
        mdps_V2 = get_mid_points(a_nml_v2, a_env_v2)

        mut_pos_V = np.zeros([len(mdps_V1),3])
        mut_pos_V[0] = [(a + b) / 2 for a, b in zip(mdps_V1[0] , mdps_V2[0])]
        
        O_M_g, O_Discr = g_Mon.copy(), Discr.copy()
        sub_muts = []
        mut_path_Mol = []
        mut_path_Topo = []

        failed = 0
        Local_mutation = []
        SubSyst_mutation = []
        dev_mut_path = []
        dev_Features = []

        dev_mut_path.append(O_M_g)
        mut_path_Mol.append(O_M_g)
        mut_path_Topo.append(O_Discr)

        GLO_Rej_idp = []
        LCO_Rej_idp = []

        GLO_Acc_idp = []
        LCO_Acc_idp = []

        Mutation_Accept, Mutation_Reject, T_Mutation_Accept, T_Mutation_Reject = [], [], [], []
        Mutation_Accept.append(O_M_g)
        Mutation_Reject.append(O_M_g)
        T_Mutation_Accept.append(O_Discr)
        T_Mutation_Reject.append(O_Discr)

        AccFit = [PFit]
        Accepted, Rejected = 1, 0

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        for j in range(1, len(mdps_V1)):

            mut_pos_V[j] = [(a + b) / 2 for a, b in zip(mdps_V1[j] , mdps_V2[j])]
        
            # Orientation And Position Vector
        
            Bm = np.array(mdps_V2[j-1]) - np.array(mut_pos_V[j-1])
            Cm = np.array(mdps_V2[j])   - np.array(mut_pos_V[j])
            dot_product = np.dot(Bm,Cm)

            if dot_product == 0.0:
                dot_product = 0.001
            dist_pro = (np.linalg.norm(Bm) * np.linalg.norm(Cm))
            if dist_pro == 0.0:
                dist_pro = 0.0001
            quant = dot_product / dist_pro
            if quant > 1.00:
                alpha = 0.000001
            else:
                alpha = np.arccos(quant)

            O_M_g_R  = O_M_g   - mdps_V1[j-1]
            O_DiscrR = O_Discr - mdps_V1[j-1]
            ax       = axis_of_rotation(Bm, Cm, alpha)
            r_M_g    = rodrigues(O_M_g_R, alpha, ax)
            r_Discr  = rodrigues(O_DiscrR,alpha, ax)
        
            M_Discr  = (r_Discr + mdps_V1[j-1]) + (mut_pos_V[j] - mut_pos_V[j-1])
            M_g      = (r_M_g   + mdps_V1[j-1]) + (mut_pos_V[j] - mut_pos_V[j-1])
            Fitness_1= r_fit(M_g, s_pos)
            O_M_g    = M_g.copy()
            O_Discr  = M_Discr.copy()

            if Fitness_1 >= sf.mutation_F_coe:
                
                Mutation_Accept.append(O_M_g)
                T_Mutation_Accept.append(O_Discr)
                AccFit.append(Fitness_1)
                Accepted = Accepted + 1
            else:
                Mutation_Reject.append(O_M_g)
                T_Mutation_Reject.append(O_Discr)
                Rejected = Rejected + 1

        for l in Mutation_Accept:
            GLO_Acc_idp.append(Atoms(sf.matm * (cs+1), np.vstack((l, s_pos))))

        return(GLO_Acc_idp, Mutation_Accept, Accepted, AccFit, T_Mutation_Accept)


import math
import os
import subprocess
import collections
import numpy                            as np
import random                           as rd
import matplotlib.pyplot                as plt

from   time                                import time 
from   ase                                 import Atoms
from   ase.io                              import read, write, Trajectory
from   ase.visualize                       import view
from   itertools                           import combinations, chain
                                           
#from   LOGOS_00_00_objects                 import PICKLE_load, PICKLE_unload
from   LOGOS_MTFS_04_01_Potential          import abinitio_g09
from   LOGOS_00_03_TopoAssociation         import superimpose as associate
from   LOGOS_MTFS_02_01_Mutation_Operation import T_Mutation
from   LOGOS_MTFS_00_Structure_discriptor  import r_fit
import LOGOS_00_01_Structure_discriptor                       as symF



#===============================================================================================================


def axis_of_rotation(Ov, Rv, alpha):
    if alpha == 0.0:
        alpha = 0.00001
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


class MTFSO_main():

    def __init__(sf, matm, mpos, Topo, m, MTFSGO_output, opt_prm=False):

        sf.matm = matm
        sf.pos = mpos
        sf.m = m

        sf.Topo = Topo
        sf.T_pos = Topo.get_positions()
        sf.T_atm = Topo.get_chemical_symbols()

        sf.r_vec = mpos[1]-mpos[0]
        sf.r = np.linalg.norm(mpos[0]-mpos[1])

        sf.ESP_V = [0.0317825, 0.0317826, 0.0317825, 0.0317826, 0.0317826, 0.0317826, 0.0317825, 0.0317825, -0.020291, -0.0202909, -0.0109551, -0.0109551, -0.0109551, -0.010955, -0.0109551, -0.0109551, -0.0109551, -0.0109551]
        sf.ESP_V = [1, 1, 1, 1, 1, 1, 1, 1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
        sf.E_t = len(sf.ESP_V)
        sf.E_mono_unit = -188.530879538

        MutMag                   = 15
        Pertb_Mag                = 15
        max_cyc                  = 5
        opt_Per                  = 40
        LocalMutationEnvironment = 5
        FeaCut                   = 6.0
        mutation_F_coe           = 2.6
    
        opt_prm = [MutMag, Pertb_Mag, max_cyc, opt_Per, LocalMutationEnvironment, FeaCut, mutation_F_coe]
        sf.MutMag, sf.Pertb_Mag, sf.max_cyc, sf.opt_Per, sf.LocalMutationEnvironment, sf.FeaCut, sf.mutation_F_coe = opt_prm
        sf.T_Mut  = T_Mutation(sf.LocalMutationEnvironment, sf.E_t, sf.MutMag, sf.mutation_F_coe, sf.matm)
        

    def SegregateMonomers(sf, agg_pos, c_cs):

        idev_units = np.zeros([c_cs, sf.m, 3])
        idev_units_topo = []

        c1 = 0
        for i in range(0, len(agg_pos), sf.m):
            c2 = 0
            for j in  range(i, i+sf.m):
                idev_units[c1,c2]  = agg_pos[j]
                c2 = c2 + 1

            Discr = associate([idev_units[c1], sf.T_pos])
            idev_units_topo.append(Discr[sf.m:])
            c1 = c1 + 1
        
        return(idev_units, idev_units_topo)

    def Perturb(sf, i_mol):

        pert, c1 = [], 0
        for j in range(sf.Pertb_Mag):

            mins = []
            p_per = [rd.uniform(0.05,0.8) for k in range(3)]
            o_per = np.radians(rd.uniform(5,190))
            ARper = [rd.uniform(10,190) for k in range(3)]
            ARper = np.array(ARper/np.linalg.norm(ARper))
            rotar = i_mol-i_mol[0]
            pert.append(rodrigues(rotar , o_per, ARper)+i_mol[0])
            pert.append(pert[c1] + p_per)

        return(pert)


    def FeatureCutoff(sf, i_units, tmol):

        dens = []
        for i in range(tmol):
            env = np.vstack([i_units[j] for j in range(tmol) if i != j])
            dens.append(symF.radial_symmetry_function(i_units[i], env, sf.FeaCut))
        return min(dens), max(dens)


    def IndivFeatureEval(sf, i_units):

        dens = []
        for i in range(sf.cs):
            env = np.vstack([i_units[j] for j in range(sf.cs) if i != j])
            dens.append([r_fit(i_units[i], env), i])

        dens.sort()
        dens.reverse()
        opt_mol = int(np.ceil((sf.opt_Per * sf.cs)/100))
        mutate = [i_units[dens[i][1]] for i in range(opt_mol)]
        mut_F =  [dens[i][0] for i in range(opt_mol)]
        sf.mut_I =  [dens[i][1] for i in range(opt_mol)]
        core   = [i_units[dens[i][1]] for i in range(opt_mol, sf.cs)]
        core_F   = [dens[i][0] for i in range(opt_mol, sf.cs)]
        sf.core_F_I   = [dens[i][1] for i in range(opt_mol, sf.cs)]
        UpT_lm, Lwt_lm = sf.FeatureCutoff(core, len(core))

        return mutate, core, mut_F, core_F, Lwt_lm


    def Run_FS_Optimizer(sf, pos, atm, cs, id_N):

        sf.cs = cs  
        sf.agg_pos = pos
        sf.agg_atm = atm

        idev_units, idev_units_topo  = sf.SegregateMonomers(sf.agg_pos, cs)
        mutate, core, mut_F, core_F, Lwt_lm = sf.IndivFeatureEval(idev_units)
        sf.immut = core.copy()
        core = np.vstack(core)
        sf.FSO_LocalTuner(core.copy(), mutate, mut_F, Lwt_lm, int(len(core)/sf.m), id_N)

        return 


    def FSO_LocalTuner(sf, s_pos, Io_D_pos, mut_F, Lwt_lm, cs, aggregation=False):


        sf.immut = s_pos
        idev_units, idev_units_topo  = sf.SegregateMonomers(s_pos, cs)
        o_D_pos = []
        Ifit_com = []
        Ifit_Gene = []
        see_e = []
        see_p = []

        orr = np.array([-100.0, -100.0])
        for I_e, g_Mon in enumerate(Io_D_pos):
            I_dist_l = []
            for a_i in range(len(idev_units)):
                    mind = 100
                    for k in g_Mon:
                        for j in idev_units[a_i]:
                            dist = np.linalg.norm(k-j)
                            if dist < mind:
                                mind = dist
                    I_dist_l.append([mind, a_i])

            I_dist_l.sort()
            sub_s = [g_Mon]
            sub_P = []
            Ids = 5

            for js in range(Ids):
                a_i = I_dist_l[js][1]
                sub_s.append(idev_units[a_i])
                sub_P.append(idev_units[a_i])

            Ifit_ch   = [Atoms(sf.matm*(Ids+1), np.vstack(sub_s)), Atoms(sf.matm*(Ids), np.vstack(sub_P))]
            Id_ene, _ = abinitio_g09(Ifit_ch.copy(), 'Int_', 'HF')
            Ifit_e    = (Id_ene[0]/Ids+1) - (Id_ene[1]/Ids)
            Ifit_e    = rd.uniform(-3, -5)
            G_com     = -symF.radial_symmetry_function(g_Mon, s_pos, 12)
            P_chr     = np.array([G_com, Ifit_e])
            Odis      = np.linalg.norm(orr-P_chr) 
            Ifit_Gene.append([Odis, I_e])

        Ifit_Gene.sort()

        for i in range(10):
            o_D_pos.append(Io_D_pos[Ifit_Gene[i][1]])

        sf.optimized_g = []
        sf.opt_traj = [[] for i in range(len(o_D_pos))]
        sf.opt_EneP = [[] for i in range(len(o_D_pos))]
        UnFit = [True for i in range(len(o_D_pos))]
        sf.optimized_g = np.zeros([len(o_D_pos),sf.m,3])
        LC = sf.LocalMutationEnvironment
        sf.immut = s_pos.copy()

        for x,g in enumerate(o_D_pos):
            g_Mon = g
            cycle = 1 
            PFit = r_fit(g_Mon, s_pos)

            print('Optimizing :', x)
            print('OPtimization Cycle:', cycle)

            while UnFit[x]:

                dist_l = []
                for a_i in range(len(idev_units)):
                        mind = 100
                        for k in g_Mon:
                            for j in idev_units[a_i]:
                                dist = np.linalg.norm(k-j)
                                if dist < mind:
                                    mind = dist
                        dist_l.append([mind, a_i])

                dist_l.sort()
                Subsyst  = []
                Subsyst_mN, Subsyst_mX = 2,4

                for ds in range(Subsyst_mN, Subsyst_mX):
                    sub_s = []
                    for js in range(ds):
                        a_i = dist_l[js][1]
                        sub_s.append(idev_units[a_i])
                    Subsyst.append(sub_s)

                Discr_env, mols_env = [],[]
                for j in range(0, LC+1):

                    a_i = dist_l[j][1]
                    Discr_env.append(idev_units_topo[a_i])
                    mols_env.append(idev_units[a_i])

                Discr_env = np.vstack(Discr_env)
                mols_env = np.vstack(mols_env)
                Discr = associate([g_Mon, sf.T_pos])[sf.m:]
                LC_ESP = sf.ESP_V * (LC+1)

                TPM = np.zeros([sf.E_t, sf.E_t*(LC+1)])
                TPM_al = []

                for j in range(sf.E_t):
                    for k in range(sf.E_t*(LC+1)):
                        TPM[j,k] = sf.ESP_V[j] * LC_ESP[k] / np.linalg.norm( Discr[j]-Discr_env[k])
                        TPM_al.append([TPM[j,k], j, k])
                TPM_al.sort()
                TPM_al.reverse()

                for j in range(1, len(TPM_al)):
                    if TPM_al[0][1] != TPM_al[j][1]:
                        break

                Tp_atm = ['X'] * ( sf.E_t * (LC+2)) + (sf.matm * (LC+2))
                atm = (sf.matm * (LC+2))
                MFea1, MFea2 = TPM_al[0], TPM_al[j]
                sf.LC = sf.LocalMutationEnvironment
                F1r, F2r = MFea1[1], MFea2[1]
                F1_ls_r = [TPM[F1r,ii] for ii in range(sf.E_t*(sf.LC+1))]
                Tm_1 = F1_ls_r.index(min(F1_ls_r))
                F2r, F2c = MFea2[1], MFea2[2]
                F2_ls_r = [TPM[F2r,ii] for ii in range(sf.E_t*(sf.LC+1))]
                Tm_2 = F2_ls_r.index(min(F2_ls_r))
                a_env_v1 = Discr_env[Tm_1]
                a_nml_v1 = Discr[F1r]
                a_env_v2 = Discr_env[Tm_2]
                a_nml_v2 = Discr[F2r]


                print('Undergoing Topography Mutation')

                GLO_Acc_idp, Mutation_Accept, Accepted, AccFit, T_Mutation_Accept  = sf.T_Mut.FSO_Mutation(TPM_al[0], TPM_al[j], cs, s_pos, TPM, Discr_env, Discr, g_Mon, PFit, cycle, x)
                LCO_Acc_idp, LCO_ene, LCO_ene_Ctm = [], [], []

                print('Extracting subsystem energy')

                for bb,k in enumerate(Subsyst, 2):
                    LCO_Acc_ss = []
                    atm = (sf.matm * (len(k)+1))

                    for l in Mutation_Accept:
                        LCO_Acc_ss.append(Atoms(atm, np.vstack((l,*k))))

                    LCO_ene_sub, L_tim = abinitio_g09(LCO_Acc_ss, 'Lco', 'HF')
                    LCO_ene.append(LCO_ene_sub)
                    LCO_ene_Ctm.append(L_tim)

                GLO_ene, G_tim = LCO_ene[-1], LCO_ene_Ctm[-1]
                LCO_ene_P = [[] for k in range(Accepted)]

                for l in range(len(LCO_ene)):
                    for k in range(Accepted):
                        LCO_ene_P[k].append(LCO_ene[l][k])

                ene_Accp = []
                stuck_in_well = False

                if Accepted > 1:
                    D_ene = GLO_ene.copy()
                    for z in range(len(D_ene)):
                        print('-------------------')
                        print('mutation energy:', D_ene[z])

                    if (min(D_ene) < D_ene[0]):
                        for z in range(1, len(D_ene)):
                            if (D_ene[z] < D_ene[0]):

                                sf.opt_traj[x].append(Mutation_Accept[z])
                                sf.opt_EneP[x].append(D_ene[z])

                                GLO_Acc_idp[z].info['energy'] = D_ene[z]
                                ene_Accp.append(GLO_Acc_idp[z])

                        stb = D_ene.index(min(D_ene))
                        o_D_pos[x] = Mutation_Accept[stb]

                        g_Mon = Mutation_Accept[stb]
                        g_Topo     = T_Mutation_Accept[stb]

                        PFit = AccFit[stb]
                        cycle = cycle + 1

                        if PFit < sf.mutation_F_coe:
                            UnFit[x] = False
                            passed   = False
                        else:
                            UnFit[x] = True
                            passed   = True

                        if cycle >= sf.max_cyc:
                            UnFit[x] = False
                            passed   = True
                        else:
                            passed   = False

                    else:
                        stuck_in_well = True
                else:
                    stuck_in_well = True

                if stuck_in_well:
                    pert = sf.Perturb(g_Mon)
                    pert_M = []
                    p_atm = sf.matm * Subsyst_mX

                    for l in pert:
                        pert_M.append(Atoms(p_atm, np.vstack((l, *Subsyst[-1]))))
                    P_ene, _ = abinitio_g09(pert_M.copy(), 'Pert', 'HF')

                    if (min(P_ene) <= LCO_ene[-1][0]):

                        stb = P_ene.index(min(P_ene))
                        g_Mon = pert[stb]

                        o_D_pos[x] = pert[stb]
                        sf.opt_traj[x].append(g_Mon)
                        sf.opt_EneP[x].append(min(P_ene))
                        
                        g_glo =  Atoms(sf.matm * (cs+1), np.vstack((s_pos, g_Mon)))
                        g_glo.info['energy'] = min(P_ene)
                        ene_Accp.append(g_glo)

                        PFit = r_fit(pert[stb], s_pos)

                        if PFit > sf.mutation_F_coe:
                            UnFit[x] = False
                            passed   = True
                        else:
                            passed   = False

                        if cycle >= sf.max_cyc:
                            UnFit[x] = False
                            passed   = True
                        else:
                            passed   = False
                    else:
                        if cycle >= sf.max_cyc:
                            UnFit[x] = False

                    cycle = cycle + 1
                    Discr  = associate([g_Mon, sf.T_pos])
                    g_Topo = Discr[sf.m:]

            sf.optimized_g[x] = g_Mon
            UnFit[x] = False
            
            if aggregation == False:

                g_Mon = np.array([g_Mon])
                g_Topo = np.array([g_Topo])
                idev_units = np.concatenate((idev_units, g_Mon), axis=0)
                idev_units_topo = np.concatenate((idev_units_topo, g_Topo), axis=0)

        return 


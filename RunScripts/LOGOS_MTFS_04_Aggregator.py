import os
import subprocess
import collections
import numpy                            as np
import random                           as rd
import matplotlib.pyplot                as plt

from ase                                import Atoms
from ase.io                             import read, write, Trajectory
from ase.visualize                      import view
from itertools                          import combinations, chain

from LOGOS_MTFS_04_01_Potential         import  abinitio_g09
from LOGOS_MTFS_02_00_Geometry_Optimizer import MTFSO_main

class Opt_Aggregation(MTFSO_main):

    def __init__(sf, aggregate, apf):

        sf.aggregate = aggregate
        sf.apf       = apf

    def EEval(sf, O_M_g):

        sub_muts = []
        for M_g in O_M_g:

            S_dist_l = []
            for k in range(len(sf.idev_units)):
                    mind_s = 100
                    for xx in M_g:
                        for yy in sf.idev_units[k]:
                            dist = np.linalg.norm(xx-yy)
                            if dist < mind_s:
                                mind_s = dist
                    S_dist_l.append([mind_s, k])
            S_dist_l.sort()

            subsystem = [M_g]
            for k in range(sf.LocalMutationEnvironment):
                xx = S_dist_l[k][1]
                subsystem.append(sf.idev_units[xx])

            # System Extraction And Energy Evaluation

            sub_muts.append(Atoms(sf.matm * (sf.LocalMutationEnvironment+1), np.vstack((subsystem))))
        ene = abinitio_g09(sub_muts, 'N_c_g09', 'HF')

        return(ene)


    def Aggregation(sf, par, cs):

        def padd(elst, i_lst):

            lst = [elst[j] for j in i_lst]
            max_length = max(len(sublist) for sublist in lst)
            pd_lst = [
                sublist + [sublist[-1]] * (max_length - len(sublist)) if len(sublist) < max_length else sublist
                for sublist in lst
            ]
            return(pd_lst)

        def One_Shot_Generation(fitness_ind, lab):

            Engy_pd = padd(sf.trajs_E.copy(), fitness_ind)
            Traj_pd = padd(sf.trajs_M.copy(), fitness_ind) 
            z_engy = list(zip(*Engy_pd))
            z_traj = list(zip(*Traj_pd))
            iso_opt_traj, iso_opt_engy = [], []
            s_atms = sf.matm * (cs)

            for b,j in enumerate(z_traj):

                srt = Atoms(s_atms, np.vstack(j))
                iso_opt_engy.append(sum(z_engy[b]))
                srt.info['energy'] = iso_opt_engy[b]
                iso_opt_traj.append(srt)

            return(iso_opt_traj[-1], iso_opt_engy[-1])

        if sf.aggregate == False:

            sf.trajs_E = [[] for i in range(cs)]
            sf.trajs_M = [[] for i in range(cs)]

            for i in range(len(sf.core_F_I)):
                sf.trajs_E[sf.core_F_I[i]] = [0.00]
                sf.trajs_M[sf.core_F_I[i]] = [sf.immut[i]]

            for i in range(len(sf.mut_I)):
                sf.trajs_E[sf.mut_I[i]] = sf.opt_EneP[i]
                sf.trajs_M[sf.mut_I[i]] = sf.opt_traj[i]

            fitness_ind = [i for i in range(cs)]
            Gsrt, Gene = One_Shot_Generation(fitness_ind, 'FSO_sf.opt_traj')

            return(Gsrt, Gene)

        else:

            sf.idev_units, _ = sf.SegregateMonomers(sf.immut, int(len(sf.immut)/sf.m))
            local_int, LI_en = [], []

            for a, M_g in enumerate(sf.optimized_g):
                S_dist_l = []
                for k in range(len(sf.idev_units)):
                        mind_s = 100
                        for xx in M_g:
                            for yy in sf.idev_units[k]:
                                dist = np.linalg.norm(xx-yy)
                                if dist < mind_s:
                                    mind_s = dist
                        S_dist_l.append([mind_s, k])

                S_dist_l.sort()
                LIsystem = [M_g]
                system = []
                T_ls = []
                sf.subsystem_size = 3

                for k in range(sf.subsystem_size+1):
                    xx = S_dist_l[k][1]
                    LIsystem.append(sf.idev_units[xx])
                    system.append(sf.idev_units[xx])

                T_ls.append(Atoms(sf.matm * (sf.subsystem_size+2), np.vstack((LIsystem))))
                T_ls.append(Atoms(sf.matm * (sf.subsystem_size+1), np.vstack((system))))

                E, _ = abinitio_g09(T_ls, 'N_c_g09', 'MPW1PW91')
                LI_en.append(E[0]-E[1])

            minE, maxE = min(LI_en), max(LI_en)
            prm = 8
            fit_ind = []

            while len(fit_ind) <= 3:
                prm = prm-1
                fitness = [np.exp(-prm*((i-minE)/(maxE-minE))) for i in LI_en]
                fit_ind = [ i for i in range(len(fitness)) if fitness[i] >= 0.5 ]


            print('StepWise Daughter Structure Generation')

            combs = combinations(fit_ind, sf.apf)
            comb_list = [list(comb) for comb  in combs]
            s_atms = sf.matm * (int(len(sf.immut)/sf.m) + sf.apf)
            isomers,gE = [], []

            for a,i in enumerate(comb_list):

                # Optimization Trajectory

                ik, jk = i
                Fitr_ij = []

                for k in (sf.optimized_g[ik]):
                    for l in (sf.optimized_g[jk]):
                        Fitr_ij.append(np.linalg.norm(k-l))

                if min(Fitr_ij) > 2.8:

                    Engy_pd = padd(sf.opt_EneP.copy(), i)
                    Traj_pd = padd(sf.opt_traj.copy(), i)
                    z_engy = list(zip(*Engy_pd))
                    z_traj = list(zip(*Traj_pd))
                    iso_opt_traj, iso_opt_engy = [], []

                    for b,j in enumerate(z_traj):
                        iso_opt_engy.append(sum(z_engy[b]))
                        srt = Atoms(s_atms, np.vstack((sf.immut, *j)))
                        srt.info['energy'] = iso_opt_engy[b]
                        iso_opt_traj.append(srt)

                    # Final Optimized Geometry

                    gE.append(iso_opt_engy[-1])  
                    N_mols = [sf.optimized_g[j] for j in i]
                    isomers.append(Atoms(s_atms, np.vstack((sf.immut, *N_mols))))

            isomers_E = []
            for i in range(len(isomers)):
                isomers[i].info['energy'] = gE[i]
                isomers_E.append(isomers[i])

            f_ind = gE.index(min(gE))


            print('Total Isomers Generated:', len(isomers))
            FIene, _ = abinitio_g09([isomers[f_ind]], 'N_c_g09', 'MPW1PW91')
            print('Best Isomer is Reported with energy:', FIene[0], min(gE), '\n\n')

            return(isomers[f_ind], FIene[0], isomers)





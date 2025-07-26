import sys
import os
import numpy as np

from   ase.io import read
from   ase    import Atoms

from   LOGOS_00_01_Structure_discriptor import StructureDiscriptor as a_env
from   LOGOS_00_02_SpaceTransformation  import CartesianToPolar    as spc


class X_requirements():

    def __init__(sf, m, atom_Env, L_atm, L_pos):

        sf.m        = m
        sf.atom_Env = atom_Env
        sf.env      = sf.atom_Env[0]
        sf.env_i    = sf.atom_Env[2]
        sf.atm_i    = [i for i in range(m)]
        sf.L_atm    = L_atm
        sf.L_pos    = L_pos

    def X_feature_vector_construction(sf, L_clu, C_env):

        dim = 0
        for i in range(sf.env):
            for j in range(i,sf.env):
                dim = dim + 1
        lst = [[L_clu, C_env], [L_clu, L_clu]]
        XFV = []
        for phs in range(len(lst)):
            cls_0 = lst[phs][0]
            cls_1 = lst[phs][1]
            EnvD = a_env(cls_0, cls_1, 8)
            G1 = EnvD.radial_symmetry_function_G_s()
            cs  = int(len(cls_0)/sf.m)
            gcs = int(len(cls_1)/sf.m)
            fv = np.zeros([cs,dim])
            int_E = sf.atm_i
            i_i = 0
            for i in range(0, cs):
                for j in range(0, gcs):
                    if  i != j:
                        c1 = 0
                        int_A = [i_i + l for l in sf.atm_i]
                        for a in range(sf.env):
                            for b in range(a, sf.env):
                                for c in sf.env_i[a]:
                                    for d in sf.env_i[b]:
                                        x = c + (sf.m*i)
                                        y = d + (sf.m*j)
                                        fv[i,c1] = G1[x,y] + fv[i,c1]
                                c1 = c1 + 1
            XFV.append(np.sum(fv,0))

        return(np.array(XFV).flatten())


def Y_Paramter_extraction(Glo_graphs, Glo_graphs_M, Glo_pprt_M):

    TParam = [[] for i in range(len(Glo_graphs))]
    r, theta, phi = [], [], []
    al_Ori, be_Ori, gm_Ori = [], [], []
    ID_LC_clusters, cont = [], []
    for i in range(len(Glo_graphs)):
        Geo_M = sum(Glo_graphs[i])/3
        Glo_g_C = Glo_graphs[i] - Geo_M
        Glo_g_C_M = Glo_graphs_M[i] - Geo_M
        Glo_prp = Glo_pprt_M[i] - Geo_M
        dot_product = np.dot(Glo_g_C[0], Glo_g_C[1])
        alpha = np.arccos(dot_product / (np.linalg.norm(Glo_g_C[0]) * np.linalg.norm(Glo_g_C[1])))
        AR = ax_param(Glo_g_C[0], Glo_g_C[1], alpha)
        Pm = np.array([1.0, 1.0, 1.0])
        alpha = np.arccos(np.dot(AR,Pm) / (np.linalg.norm(AR) * np.linalg.norm(Pm)))
        AR1 = ax_param(AR, Pm , alpha)
        N_or = rt_param(Glo_g_C_M, alpha, AR1)
        N_or_M = rt_param(Glo_prp, alpha, AR1)
        pos_Vec = sum(N_or_M) / 3
        if ((pos_Vec < 0.0).all()):
            AR = np.array([0.0, 0.0, 0.0])
            N_or = rt_param(N_or, np.pi, AR)
            N_or_M = rt_param(N_or_M, np.pi, AR)
            pos_Vec = sum(N_or_M) / 3
        if (pos_Vec > 0.0).any() and (pos_Vec < 0.0).any():
            pass
        if (pos_Vec > 0.0).all():
            TParam[i] = spc(pos_Vec)
            r.append(TParam[i][0])
            theta.append(TParam[i][1])
            phi.append(TParam[i][2])
            ref = 2
            or_vc = N_or_M[ref] - pos_Vec
            proj = np.zeros([3,3])
            ax_vx = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
            angles = [] 
            for a in range(3):
                proj[a] = or_vc.copy()
                proj[a,a] = 0.0
                Av = proj[a]
                Bv = ax_vx[a]
                dot_product = np.dot(Av, Bv)
                angles.append(np.degrees(np.arccos(dot_product / (np.linalg.norm(Av) * np.linalg.norm(Bv)))))
            al_Ori.append(angles[0])
            be_Ori.append(angles[1])
            gm_Ori.append(angles[2])
            ID_LC_clusters.append(i)

    return(r, theta, phi, al_Ori, be_Ori, gm_Ori, ID_LC_clusters)

def ax_param(Ov, Rv, alpha):
    return(np.cross(Ov,Rv) / (np.linalg.norm(Ov) * np.linalg.norm(Rv) * np.sin(alpha)))
def rt_param(rotor, theta, AR):
    v_rot = np.zeros([len(rotor),3])
    for i in range(0, len(rotor)):
        v = rotor[i]
        term1 = v * np.cos(theta)
        term2 = np.cross(AR, v) * np.sin(theta)
        dot_p = (np.dot(AR, v))
        term3 = AR * (np.dot(v, AR)) * (1-np.cos(theta))
        v_rot[i]  = term1 + term2 + term3
    return(v_rot)

def Pattern_Retriver(pos, atm, m, X_ins, dist_rc = 5.5):

    j = int(len(atm)/m)
    cents = np.zeros([j,3])
    d_p, a_p, c1 = [], [], 0
    for k in range(0, len(pos), m):
        d_p.append(pos[k:k+m])
        a_p.append(atm[k:k+m])
        cents[c1] = Atoms(atm[k:k+m], pos[k:k+m]).get_center_of_mass()
        c1 = c1 + 1
    Con_M = np.zeros([len(cents), len(cents)])
    for k in range(len(cents)):
        for l in range(k+1, len(cents)):
            Con_M[k,l] = np.linalg.norm(cents[k]-cents[l])
    data_3s = []
    for k in range(len(cents)):
        cont = []
        for l in range(k+1, len(cents)):
            if Con_M[k,l] < dist_rc:
                cont.append(l)
        for l in cont:
            cont_2 = []
            for o in range(l+1, len(cents)):
                if Con_M[l,o] < dist_rc:
                    cont_2.append(o)
            if (set(cont) & set(cont_2)):
                ls_i = [ k, l, *(set(cont) & set(cont_2))]
                data_3s.append(ls_i)
    NL_C, d_in, data_3s_1 = [], [], []
    Diff_Patterns_C = []
    for a,tr in enumerate(data_3s):
        if len(tr) > 3:
            d_in.append(a)
            Diff_Patterns_C.append(data_3s[a].copy())
            NL_C.append([[tr[el_1], tr[el_2], tr[el_3]] for el_1 in range(0,len(tr)) for el_2 in range(el_1+1,len(tr)) for el_3 in range(el_2+1,len(tr))])
    NL_C = sum(NL_C, [])
    for a in sorted(d_in, reverse=True):
        del data_3s[a]
    data_3s_1 = [*data_3s, *NL_C]
    fc_v      = 3
    pos_3s_M  = np.zeros([len(data_3s_1), fc_v * m, 3])   
    pos_3s    = np.zeros([len(data_3s_1), fc_v, 3])   
    pprt      = np.zeros([len(data_3s_1),3])
    pprt_M    = np.zeros([len(data_3s_1),m, 3])
    ind_s     = np.arange(0,len(cents))
    X_FV      = [ [] for g in range(len(data_3s_1))]
    id_indexes = [ [] for g in range(len(data_3s_1))]

    for d,a in enumerate(data_3s_1):
        pos_3s_M[d] = [pos[g] for b in a for g in range(b * m, b * m + m)]
        pos_3s_Mi   = [    g  for b in a for g in range(b * m, b * m + m)]
        pos_3s[d]   = [cents[b] for b in a]
        Geo_M = [sum(pos_3s[d,:,k])/3 for k in range(3)]
        NL = list(set(a) ^ set(ind_s))
        T_cents = [cents[l] for l in (NL)]
        mn_d = [np.linalg.norm(Geo_M-k) for k in T_cents]
        ind = min(mn_d)
        mn_ind = mn_d.index(ind)
        ml_ind = NL[mn_d.index(ind)]
        pprt[d] = T_cents[mn_ind]
        pprt_M[d] = pos[ml_ind * m : (ml_ind * m) + m]
        id_indexes[d] = [*[g for g in range(ml_ind * m , (ml_ind * m) + m)], *pos_3s_Mi]
        R_pos = np.delete(pos, id_indexes[d], axis=0)
        X_FV[d] = X_ins.X_feature_vector_construction(pos_3s_M[d], R_pos)

    return(X_FV, id_indexes, pos_3s, pos_3s_M, pprt_M.copy())

def FormatData(G_X_FV, G_r, G_th, G_phi, G_al_Ori, G_be_Ori, G_gm_Ori):

    os.system('mkdir LOGOS_01_NN_train')
    f_icsv = open('LOGOS_01_NN_train/LOGOS-NN-Input-Training-Set.csv','w')
    f_rpcsv   = open('LOGOS_01_NN_train/LOGOS-NN-Prop_01_r_Training-Set.csv','w')
    f_thepcsv = open('LOGOS_01_NN_train/LOGOS-NN-Prop_02_the_Training-Set.csv','w')
    f_phipcsv = open('LOGOS_01_NN_train/LOGOS-NN-Prop_03_phi_Training-Set.csv','w')
    f_ag1pcsv = open('LOGOS_01_NN_train/LOGOS-NN-Prop_04_ag1_Training-Set.csv','w')
    f_ag2pcsv = open('LOGOS_01_NN_train/LOGOS-NN-Prop_05_ag2_Training-Set.csv','w')
    f_ag3pcsv = open('LOGOS_01_NN_train/LOGOS-NN-Prop_06_ag3_Training-Set.csv','w')

    for i in range(len(G_X_FV)):

        print(G_r[i],      file = f_rpcsv)
        print(G_th[i],     file = f_thepcsv)
        print(G_phi[i],    file = f_phipcsv)
        print(G_al_Ori[i], file = f_ag1pcsv)
        print(G_be_Ori[i], file = f_ag2pcsv)
        print(G_gm_Ori[i], file = f_ag3pcsv)
        print(*G_X_FV[i], sep = '   , ', file = f_icsv)

    f_icsv.close()
    f_rpcsv.close()  
    f_thepcsv.close()
    f_phipcsv.close()
    f_ag1pcsv.close()
    f_ag2pcsv.close()
    f_ag3pcsv.close()


def TrainingData(L_atm, L_pos, m, lbl, X_rq):

    G_X_FV, G_X_FV_O, G_Y_prp, G_id = [], [], [], []
    G_r, G_th, G_phi, G_r2, G_th2, G_phi2 = [], [], [], [], [], []
    G_al_Ori, G_be_Ori, G_gm_Ori = [], [], []

    for i in range(len(L_atm)):
        pos = L_pos[i]
        atm = L_atm[i]
        X_fv_i, iden, G_graph, G_graphs_M, G_pprt_M = Pattern_Retriver(pos, atm, m, X_rq)
        r, theta, phi, al_Ori, be_Ori, gm_Ori, LC_id = Y_Paramter_extraction(G_graph, G_graphs_M, G_pprt_M)
        if len(LC_id) != 0:
            for j in range(len(LC_id)):
                a = LC_id[j]
                G_id.append([i, iden[a]])
                G_X_FV.append(X_fv_i[a])
                G_r.append(r[j])
                G_th.append(theta[j])
                G_phi.append(phi[j])
                G_X_FV_O.append([*X_fv_i[a], r[j], theta[j], phi[j]])
                G_al_Ori.append(al_Ori[j])
                G_be_Ori.append(be_Ori[j])
                G_gm_Ori.append(gm_Ori[j])
    G_r, G_th, G_phi = np.array(G_r), np.array(G_th), np.array(G_phi)
    G_al_Ori, G_be_Ori, G_gm_Ori = np.array(G_al_Ori), np.array(G_be_Ori), np.array(G_gm_Ori)
    G_X_FV, G_X_FV_O = np.array(G_X_FV), np.array(G_X_FV_O)
    FormatData(G_X_FV, G_r, G_th, G_phi, G_al_Ori, G_be_Ori, G_gm_Ori)
    return

def Retrieve_Input(ipt_F):

    ipt_F = open(ipt_F)
    ipt = ipt_F.readlines()
    lbl = ipt[0].split()[2]
    filename = ipt[1].split()[2]
    m = int(ipt[2].split()[2])
    N_env = int(ipt[3].split()[2])
    ap_env = list(np.array(ipt[4].split()[2:]).astype(np.int_))
    E_lst = N_env, ap_env, [list(np.array(ipt[6+i].split()).astype(np.int_)) for i in range(N_env)]
    return(lbl, filename, m, E_lst)

if __name__ == '__main__':

    lbl, Filename, m, env_D = Retrieve_Input(sys.argv[1])
    T_dat = read(Filename+'@:')
    atm, pos = [], []
    for i in range(len(T_dat)):
        atm.append(T_dat[i].get_chemical_symbols())
        pos.append(T_dat[i].get_positions())
    X_rq = X_requirements(m, env_D, atm, pos)
    TrainingData(atm, pos, m, lbl, X_rq)

    print('''\n     Data Set Is Generated     \n\n All The Files Are Generated in the directory "LOGOS_01_NN_train" \n Proceed The NN Training\n''')


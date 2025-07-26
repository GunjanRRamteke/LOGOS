import  sys
import  os
import  numpy as np
from    ase.io             import  read, write, Trajectory, xyz
from    ase                import  Atoms
from    ase.visualize      import  view
from    LOGOS_02_NN_Model  import  LOGOS_NN_00_00_module_r      as NNpy_r 
from    LOGOS_02_NN_Model  import  LOGOS_NN_00_01_module_th     as NNpy_th
from    LOGOS_02_NN_Model  import  LOGOS_NN_00_02_module_phi    as NNpy_phi
from    LOGOS_02_NN_Model  import  LOGOS_NN_01_04_module_alpha  as NNpy_al
from    LOGOS_02_NN_Model  import  LOGOS_NN_01_04_module_beta   as NNpy_be
from    LOGOS_02_NN_Model  import  LOGOS_NN_01_05_module_gamma  as NNpy_gm
from    LOGOS_00_02_SpaceTransformation import PolarToCartesian as ptc
import  LOGOS_00_01_Structure_discriptor                        as sym_f
import  LOGOS_01_01_TrainingDataGeneration                      as Par
import  LOGOS_MTFS_02_00_Geometry_Optimizer                     as FSO_M
import  LOGOS_MTFS_04_01_Potential                              as otO
from    LOGOS_MTFS_04_Aggregator        import Opt_Aggregation

def reconstruct_Ovec(Eangs):

    ax_r = np.zeros([3])
    ay_vy = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    OV_InV = []
    for ang_m in Eangs:
        rc_Nml, rc_D = [], []
        for a in range(3):
            ax_r[a] = 1.00
            ang = np.radians(ang_m[a])
            pt_pl = rt_param([ay_vy[a]], -ang, ax_r)[0]
            pt_pr  = pt_pl.copy()
            pt_pr[a] = 1.00
            rc_Nml.append(np.cross(pt_pl, pt_pr))
            pt_xy = (pt_pr + pt_pl)/3
            rc_D.append(sum([rc_Nml[a][j] * pt_xy[j] for j in range(3)]))
            ax_r[a] = 0.00
        InV = np.linalg.solve(rc_Nml, rc_D)
        OV_InV.append(InV/np.linalg.norm(InV))
    return(OV_InV)

def Position_Module(Glo_graphs, Glo_graphs_M):

    reCons, alt_pos, alt_pos_2 = [[] for i in range(len(Glo_graphs_M))], [], []
    for i in range(len(Glo_graphs)):
        Geo_M = sum(Glo_graphs[i])/3
        Glo_g_C = Glo_graphs[i] - Geo_M
        Glo_g_C_M = Glo_graphs_M[i] - Geo_M
        dot_product = np.dot(Glo_g_C[0], Glo_g_C[1])
        alpha = np.arccos(dot_product / (np.linalg.norm(Glo_g_C[0]) * np.linalg.norm(Glo_g_C[1])))
        AR = ax_param(Glo_g_C[0], Glo_g_C[1], alpha)
        Pm = np.array([1.0, 1.0, 1.0])
        alpha = np.arccos(np.dot(AR,Pm) / (np.linalg.norm(AR) * np.linalg.norm(Pm)))
        AR = ax_param(AR, Pm , alpha)
        N_or = rt_param(Glo_g_C_M, alpha, AR)
        reCons[i] = [Geo_M, alpha, AR]
        alt_pos.append(N_or)
        alt_pos_2.append(rt_param(Glo_g_C, alpha, AR))
    return(alt_pos, alt_pos_2,  reCons)

def write_xyz_structure(OptStr, PotEnergy, fld2, i, cs):

    xyz_filename = "{}/Structure_{:02d}_{:02d}.xyz".format(fld2,i,cs)
    with open(xyz_filename, 'w') as xyz_file:
        xyz_file.write(f'{len(OptStr)}\n')
        xyz_file.write(f'Done Energy: {PotEnergy:.5f} H  {(PotEnergy-(-188.530879538*cs))*627.51:.3f} kcal/mol \n') 
        for atom, position in zip(OptStr.symbols, OptStr.positions):
            xyz_file.write(f'{atom} {position[0]} {position[1]} {position[2]}\n')

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

def points_picking(pos, pos_vectors, rc = 2.5):

    R_points, R_points_I = [], []
    den = sym_f.StructureDiscriptor(pos_vectors, pos, 6.0)
    G = den.radial_symmetry_function_G_s()
    G_tot = [sum(i) for i in G]

    for j in range(len(pos_vectors)):
        val = True
        for i in range(len(pos)):
            if (np.linalg.norm(pos_vectors[j]- pos[i]) < rc):
                val = False
        if val:
            R_points.append(pos_vectors[j])
            R_points_I.append(int(j/2))
    return(np.array(R_points), R_points_I)


def InitiateClustering(M_file, M_file_TP, env_D, inputfile, DesSize, add, NN_T_W_r, NN_T_W_th, NN_T_W_phi, NN_T_W_al, NN_T_W_be, NN_T_W_gm):

    mol    =  read(M_file)
    mol_Tp =  read(M_file_TP)
    m      =  len(mol)
    cms    =  mol.get_center_of_mass()
    M_atm  =  mol.get_chemical_symbols()
    M_pos  =  mol.get_positions()-cms
    fld2   =  '01-O-Paths'
    tds    =  'TDS'
    os.system(f' mkdir {tds}')
    os.system(f' mkdir {fld2}')


    lbl = 'MTFSO'
    FSO_ins = FSO_M.MTFSO_main(M_atm, M_pos, mol_Tp, m, lbl)

    cms = mol.get_center_of_mass()
    mpos = mol.get_positions()-cms
    rc_2  = 2.6
    rc_1  = 5.5
    ifac  = 1.2

    srt = read(inputfile)
    pos = srt.get_positions()
    atm = srt.get_chemical_symbols()
    cs = int(len(atm)/m)
    c2 = 0
    ad_ls = [add] * int((DesSize-cs)/add)
    cl_i = 1


    if int((DesSize-cs)/add) < (DesSize-cs)/add:
        ad_ls.append(DesSize-(sum(ad_ls)+cs))

#    PotEnergy = otO.abinitio_g09([srt], f'SinglePoint_{cs}','HF' )
    PotEnergy = [-5629.08908332, 0.94893836975098]
    write_xyz_structure(srt, PotEnergy[0], fld2, cl_i, cs)

    while cs < DesSize:

        print('\n\n=============================================================\n')
        print('The Parent aggregate size:', cs, '\n')
        
        print('Identifying locally interacting regoins within the parent structure')

        X_rq = Par.X_requirements(m, env_D, atm, pos)
        X_fv_i, iden, G_graph, G_graphs_M, _ = Par.Pattern_Retriver(pos, atm, m, X_rq, rc_1)

        print('Total number of local sites identified: ',len(X_fv_i))
        print('And subsequent input vector construction')
        print('Neural Network is employed to predict the position and orientation vectors')

        r = NNpy_r.Predict_Pretrained(X_fv_i, 1, NN_T_W_r)    
        th = NNpy_th.Predict_Pretrained(X_fv_i, 1, NN_T_W_th) 
        phi = NNpy_phi.Predict_Pretrained(X_fv_i, 1, NN_T_W_phi)
        predPOS_Vec = [np.array(ptc([r[i]+ifac, th[i],  phi[i]]))  for i in range(len(r))]
        alt_pos,P, reCons = Position_Module(G_graph, G_graphs_M)
        POS_vec = []

        for i in range(len(reCons)):
            Geo_M, alpha, AR = reCons[i]
            Rt = np.vstack((predPOS_Vec[i], -predPOS_Vec[i]))
            POS_vec.append(rt_param(Rt, -alpha, AR) + Geo_M)

        POS_vec, p_v_ind = points_picking(pos, np.vstack(POS_vec), rc_2)
        O_X_fv_i = np.array([[*X_fv_i[i], r[i][0], th[i][0], phi[i][0]] for i in p_v_ind])

        or_al = NNpy_al.Predict_Pretrained(O_X_fv_i, 1, NN_T_W_al)
        or_be = NNpy_al.Predict_Pretrained(O_X_fv_i, 1, NN_T_W_be)
        or_gm = NNpy_al.Predict_Pretrained(O_X_fv_i, 1, NN_T_W_gm)
        Eangs = [[or_al[i], or_be[i], or_gm[i]] for i in range(len(or_al))]
        ORI_Vec = reconstruct_Ovec(Eangs)
        D_mol = []

        for i in range(len(ORI_Vec)):
            dot_product = np.dot(mpos[1], ORI_Vec[i])
            alpha = np.arccos(dot_product / (np.linalg.norm(mpos[1]) * np.linalg.norm(ORI_Vec[i])))
            AR = ax_param(mpos[1], ORI_Vec[i], alpha)
            ompos = (rt_param(mpos, alpha, AR))
            a = p_v_ind[i]
            Geo_M, alpha, AR = reCons[a]
            D_mol.append(rt_param(ompos, -alpha, AR) + POS_vec[i])

        print('A set of ',len(POS_vec), ' plausible sptial arrangements are predicted based on input geometry of parent cluster')

        Vector = Atoms(atm + ['X'] * len(ORI_Vec), np.vstack((pos, POS_vec)))
        Building = True

        print('=============================================================\n')
        print('MTFSO is Activated')

        FSO_ins.FSO_LocalTuner(pos, D_mol[:10], False, False, cs, Building)
        FSO_agg = Opt_Aggregation(Building, ad_ls[c2])

        print('-------------------------------------------------------------\n')
        print('Optimization Terminated')
        FSO_agg.__dict__.update(FSO_ins.__dict__)

        print('=============================================================\n')
        print('Compiling the pool of candidate structures')
        OptStr, PotEnergy, D_pool = FSO_agg.Aggregation(cl_i, cs)

        cs = cs + ad_ls[c2]
        write_xyz_structure(OptStr, PotEnergy, fld2, cl_i, cs)
        write('{}/ValSet_{:02d}_{:02d}.xyz'.format(tds,cl_i,cs), D_pool)
        c2 = c2 + 1
        pos = OptStr.get_positions()
        atm = OptStr.get_chemical_symbols()

        print('\n')
        print( '=============================================================')
        print(f'Generated the daughter cluster of size: cs')
        print( '=============================================================\n\n')
        os.system(f' rm -r  {tds}')


def Retrieve_Input(ipt_F):

    ipt_F = open(ipt_F)
    ipt = ipt_F.readlines()
    N_env = int(ipt[0].split()[2])
    ap_env = list(np.array(ipt[1].split()[2:]).astype(np.int_))
    E_lst = N_env, ap_env, [list(np.array(ipt[3+i].split()).astype(np.int_)) for i in range(N_env)]
    NV = 4 + N_env
    variables = [ipt[i].split()[2] for i in range(NV, NV+3)]
    molecule = [ipt[i].split()[2] for i in range(NV+4, NV+4+2)]
    trained_weights = [ipt[i].split()[2] for i in range(NV+7, NV+7+6)]

    return(*variables, E_lst, molecule, trained_weights)


if __name__ == '__main__':

    print('The model is trained to estimate both the spatial and directional information based on input data')
    inputfile, DesSize, add, E_lst, mol, weights = Retrieve_Input(sys.argv[1])
    DesSize, add = int(DesSize), int(add)

    print('Retrived the contents of input file')
    InitiateClustering(mol[0], mol[1], E_lst, inputfile, DesSize, add, *weights)

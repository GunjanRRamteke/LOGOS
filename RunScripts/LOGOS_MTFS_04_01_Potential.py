import signal
import subprocess
import time
import os
from ase.io import read, write, Trajectory
import numpy as np
from ase import Atoms
from ase.visualize import view
from itertools import permutations
##import NN1_torch as trc
###import NN2_keras as krs
###import NN2_keras_V1 as krs_V1
import math
import torch
import random

from ase.md.langevin import Langevin
from ase.optimize import BFGS
from ase import units
import torchani
from ase.calculators.gaussian import Gaussian, GaussianOptimizer

path = os.getcwd()


def abinitio_g09(struc_O, lbl, func):

    struc = struc_O.copy()

    # Calculates Single Point Energy

#    calc = Gaussian(label=f'TDS/gaussian', #{lbl}_{a}',
    calc = Gaussian(label=f'TDS/{lbl}',
                    xc=func,
                    basis='6-31G(d)',
                    nprocshared='16')
    opt_ene = []
    opt_time= []
    
    for a,srt in enumerate(struc):
        try:
            srt_t = time.time()
            srt.set_calculator(calc)
            ene = srt.get_potential_energy()
            opt_ene.append(ene/27.2113860244)
            opt_time.append(time.time() - srt_t)

        except Exception as e:

            try:
                os.system("grep Done TDS/{}.log | awk '{}print  $5{}' > TDS/grepped_Energy".format(lbl, '{', '}'))
                fene = open('TDS/grepped_Energy')
                ene = fene.read()
                fene.close()
                ene = ene.split()
                opt_ene.append(float(ene[0]))
                opt_time.append(time.time() - srt_t)
                os.system('rm TDS/grepped_Energy')

            except Exception as e:

                opt_ene.append(100)
                opt_time.append(100)
 

    return(opt_ene, opt_time)


def MM_g09(struc, par):


    opt_iso = []
    opt_ene = []
    trj = Trajectory('opt_obs_MM.traj', 'a')
    trj_am = Trajectory('opt_obs_MM_argmin.traj', 'a')

    calc = Gaussian('nosymm',label='TDS/g_MM',
                xc='pm6',
#                nosymm = True,
                mem='10GB',
                nprocshared='8')


    for a,srt in enumerate(struc):

        try:
            srt.set_calculator(calc)
#            srt.run()
            ene = srt.get_potential_energy()
            print(ene, 'ENE ')
            exit()
#            opt_ene.append(ene)

        except Exception as e:
            pass
        

        """
        with open('TDS/g_MM.com', 'w') as f1:
            f1.write(f'%mem=10GB\n%nprocshared=8\n#pm6  opt=(MaxCycles=5) nosymm\n\nLOGOS\n\n0 1\n')
            for i in range(len(atm)):
                print(atm[i], *pos[i], file = f1)
            f1.write('\n')

        try:
           os.system(f'g09  TDS/g_MM.com')
        except Exception as e:
            pass
        """


        print('TDS/g_MM.log@:')
        jd = read('TDS/g_MM.log@:')
        for i in range(len(jd)):
            opt_iso.append([jd[i].get_potential_energy(), jd[i]])
            opt_ene.append(jd[i].get_potential_energy())
            print(jd[i].get_potential_energy())
            trj.write(jd[i])

#    min_I = np.argmin(opt_ene)
#    trj_am.write(opt_iso[min_I][1])
    exit()

    return(opt_iso[min_I])



def ANI(srt):

    calc = ANI(xyz='/path/to/training/data.xyz', ensemble='ani-1ccx', cuda=False)
    srt.set_calculator(calc)
    opt = BFGS(srt)
    opt.run(fmax=0.05)
    write('daughter_optimized.xyz', water)
    final_energy = water.get_potential_energy()
    forces = water.get_forces()

def ANI_T2(struc, par):

    E, ANI_OPT = [], []
    for atoms in struc:

        calculator = torchani.models.ANI1ccx().ase()
        atoms.set_calculator(calculator)

        opt = BFGS(atoms)
        opt.run(fmax=0.1)


        def printenergy(a=atoms):
            """Function to print the potential, kinetic and total energy."""
            epot = a.get_potential_energy() / len(a)
            ekin = a.get_kinetic_energy() / len(a)
            return(epot + ekin)

            print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
                  'Etot = %.3feV' % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin))

        dyn = Langevin(atoms, 1 * units.fs, 300 * units.kB, 0.2)
        dyn.attach(printenergy, interval=1)
        Ene = printenergy()
        dyn.run(10)
        E.append([ene,i])
        ANI_OPT.append(opt)

    E.sort()
    write(f'02_Daughter_Structures/Daughter_isomers_ANIopt-{par}.xyz', ANI_OPT)

    return(E[0][1], ANI_OPT[E[0][1]])

def set_optimizer(struc, par):

    E, ANI_OPT = [], []
    for i,srt in enumerate(struc):
        ene, opt = ANI_T2(srt)
        E.append([ene,i])
        ANI_OPT.append(opt)

    E.sort()
    write(f'02_Daughter_Structures/Daughter_isomers_ANIopt-{par}.xyz', ANI_OPT)

    return(E[0][1], ANI_OPT[E[0][1]])

    exit()

def dftb(atm, pos):

    f1 = open('dftb_in.hsd', 'w')
    f1.write('Geometry = xyzFormat {\n')
    f1.write(f'{len(atm)}\n')
    for i in range(len(atm)):
        print(atm[i], *pos[i], sep = '  ', file = f1)

    f1.write('}\n')

    Driver = 'GeometryOptimization {\n  Optimizer = Rational {}\n  MovedAtoms = 1:-1\n  MaxSteps = 100\n  OutputPrefix = "geom.out"\n  Convergence {GradElem = 1E-4}\n}\n'

    f1.write(Driver)

    a_l = list(set(atm))


    Hamiltonian_1 = 'Hamiltonian = DFTB {\n  Scc = Yes\n  SlaterKosterFiles {\n'
    f1.write(Hamiltonian_1)
    sk_f = []
    for i in a_l:
        for j in a_l:
            print(f'/home/gunjan/2024/MolecularCrystalStructurePrediction/SourceCodes/Sivaz_Project/3ob-3-1/{i}-{j}.skf', file = f1)

    Hamiltonian_2 = '\n }\n  MaxAngularMomentum {\n    C = "p"\n    O = "p"\n    }\n\n}'
    f1.write(Hamiltonian_2)

    pot = open('/home/gunjan/2024/MolecularCrystalStructurePrediction/SourceCodes/Sivaz_Project/tail')
    pot = pot.read()
    f1.write(pot)
    exit()


def MM_g09_OPT(struc, par):


    def pop_SP():

        calc = Gaussian('nosymm',label='TDS/gaussianSP',
                    xc='MPW1PW91',
                    basis='6-31G(d)',
                    mem='10GB',
                    nprocshared='12')
        opt_iso = []
        opt_ene = []

        for a,srt in enumerate(struc):
            try:
                srt.set_calculator(calc)
                ene = srt.get_potential_energy()
                print(ene, 'ENE ')
                opt_ene.append(ene)
                opt_iso.append([ene, srt])
                print(opt_ene)
            except Exception as e:
                pass
        return(opt_ene, opt_iso)


    method = '%mem=10GB\n%nprocshared=12\n#P Mpw1pw91/6-31G(d) opt=maxcycle=15 nosymm  \n\nComm\n\n0 1\n'
#    method = '%mem=10GB\n%nprocshared=18\n#P Mpw1pw91/6-31G(d) nosymm  \n\nComm\n\n0 1\n'
    opt_iso = []
    opt_srt = []
    opt_ene = []
    trj_argmin = Trajectory('Lopt_obs_argmin.traj', 'a')
    trj_all = Trajectory('Lopt_obs_all.traj', 'a')

    job_ran = False

    for a,srt in enumerate(struc):

        f1 = open(f'TDS/gaussianSP_{a}.com','w')
        f1.write(method)
        atm = srt.get_chemical_symbols()
        pos = srt.get_positions()

        for i in range(len(atm)):
            print(atm[i], *pos[i], file = f1)
        f1.write('\n')
        f1.close()

        pros = subprocess.Popen(f'g09  TDS/gaussianSP_{a}.com', shell=True)
        sleep = 3
        time.sleep(sleep)
        os.system(f'grep Grad  TDS/gaussianSP_{a}.log > runner')
        top_pid = pros.pid + 1

        pros.terminate()
        time.sleep(1)

        if pros.poll() != None:
            try:
                os.kill(top_pid, signal.SIGKILL)
            except Exception as e:
                pass

        if os.path.getsize('runner') != 0:  # valid on optimization
#       if os.path.getsize('runner') == 0:  # valid on single point
            job_ran = True
            os.system(f'g09  TDS/gaussianSP_{a}.com')
            opt_O = read(f'TDS/gaussianSP_{a}.log@:')

            for i in opt_O:
                ene = i.get_potential_energy()
                opt_ene.append(ene)
                opt_srt.append(i)
                opt_iso.append([ene, i])
                trj_all.write(i)

        os.system('rm Gau-*  TDS/gaussianSP_*.com ')

#    if job_ran == False:
#        opt_ene, opt_iso = pop_SP(struc, par)

    min_I = np.argmin(opt_ene)
    trj_argmin.write(opt_iso[min_I][1])

    return(opt_iso[min_I])


def orca(struc, par):

    method = ' !M062X  def2/J  opt D3zero\n %basis\n Basis "6-31G(d)"\nend\n\n%geom\nMaxIter 5\nend\n\n*xyz 0 1\n'

    opt_iso = []
    opt_srt = []
    opt_ene = []
    trj_argmin = Trajectory('Lopt_obs_argmin.traj', 'a')
    trj_all = Trajectory('Lopt_obs_all.traj', 'a')

    for a,srt in enumerate(struc):

        f1 = open(f'TDS/orca_{a}.inp','w')
        f1.write(method)
        atm = srt.get_chemical_symbols()
        pos = srt.get_positions()

        for i in range(len(atm)):
            print(atm[i], *pos[i], file = f1)
        f1.write('*\n')
        f1.close()

        os.system(f'orca TDS/orca_{a}.inp > TDS/orca_{a}.out')
        opt_O = read(f'TDS/orca_{a}_trj.xyz@:')

        os.system("grep Coordinates TDS/orca_{}_trj.xyz | awk '{}print  $6{}' > {}/Energy".format(a,'{', '}', path))
        ene = open(f'{path}/Energy')
        ene = ene.read()
        ene = ene.split()
        ene = np.array(ene).astype(float)

        for a,i in enumerate(opt_O):
            opt_ene.append(ene[a])
            opt_srt.append(i)
            opt_iso.append([ene[a], i])
            trj_all.write(i, energy=ene[a])

    min_I = np.argmin(opt_ene)
    trj_argmin.write(opt_iso[min_I][1])
#    os.system('rm TDS/gaussianSP_*.com ')

    return(opt_iso[min_I])


if __name__ == '__main__':

    print('Hello World')

    srt = read('40-Mer.xyz@:')

    for i in range(len(srt)-1, 0, -1):
        E, T = abinitio_g09([srt[i]], f'OptR_1_stp_{i}', 'MPW1PW91')
        print(E, T)



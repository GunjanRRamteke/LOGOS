import numpy as np
from ase.io import read, write, Trajectory
from ase.visualize import view
import pickle
import os
 

def PICKLE_load(lbl, objt_name, objt_cont):
 
    os.system(f'rm  {lbl}.PIC.cb')
    db = {}
    for i in range(len(objt_name)):
        db[objt_name[i]] = objt_cont[i]
 
    dbfl = open(f'{lbl}.PIC.cb','ab')
    pickle.dump(db,dbfl)
    dbfl.close()
 
 
def PICKLE_unload(db_flnm):
 
    dbfl    = open(db_flnm, 'rb')
    db      = pickle.load(dbfl)
    objls   = []
    retrive = []
    dbfl.close()
 
    for obj in db:
        objls.append(obj)
 
    for i in range(len(objls)):
        retrive.append(db[objls[i]])
 
    return(retrive)


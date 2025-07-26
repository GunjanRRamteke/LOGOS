# LOGOS
A Local to Global optimization strategy (LOGOS) is a computational method developed to generate structural isomers of molecular clusters. It employs neural networks (NN) to guide a systematic sampling procedure, which is subsequently refined through local optimization driven by electrostatic potential (ESP) analysis.

To function effectively, LOGOS relies on various external software tools and dependencies. These pre-installed utilities are utilized for data handling, as well as for performing energy calculations and structural optimizations.
*_______________________________________________________________________________________________________________________________________

Required Softwares:

Gaussian()
pytorch()
*_______________________________________________________________________________________________________________________________________

Required Python Enviroment and utilities:

Anaconda3
ASE
pytorch
*_______________________________________________________________________________________________________________________________________

Running LOGOS

LOGOS operates through three distinct steps, each contributing to the systematic generation and optimization of molecular isomers.

Step 1: Data acquisition, generation, and formatting for neural network preprocessing
        This step involves collecting, generating, and structuring the data in a format suitable for input into the neural network, ensuring compatibility and optimal model performance.

Step 2: Training the neural network using PyTorch
        In this phase, the neural network is trained on the preprocessed data using the PyTorch framework, enabling the model to learn patterns and make accurate predictions related to molecular structures.

Step 3: Generation of daughter structures from input parent structures
        Empolying the trained neural network, this step involves predicting new, plausible daughter structures derived from the provided parent molecular clusters. These predictions reflect potential structural isomers informed by the learned patterns.

*_______________________________________________________________________________________________________________________________________

========================================================================================================================================
     STEP 1 --->   Database generation, processing and formating
========================================================================================================================================

     To use LOGOS, the following files must be present in the working directory beforehand

      1.1] An Input file (in the format as follows)
     
          lbl                     = _ (a label | string)
          Filename                = _ (data in .xyz | string)
          m                       = _ (No. of atoms in monomer | integer)
          chemically_distinct_env = _ (No of atom chemically inequivalent atom types | integer)
          atoms_per_env           = _ (the total 1 2 count for each chemically_distinct_env' separated by a space in between | integers)
          indices_per_env         = _ (specify the indices of each 'atoms_per_env' as in an molecule.xyz file)
          _ _                         (indices separated by a space in between | integers)
     
      1.2] A dataset file in 'xyz' format containing molecular structures

      OUTPUT
     
      1.3] A directory is created 'LOGOS_01_NN_train' having a 'csv'
           Inputfile is LOGOS-NN-Input-Training-Set.csv trained against LOGOS_NN_Prop_01_*_Training-Set.csv
     
      1.4] Command line argument:
           >>> python ~/RunScripts/LOGOS_01_01_TrainingDataGeneration.py  01_input.cb

========================================================================================================================================
      STEP 2 --->  NN Training 
========================================================================================================================================

      2.1] Proceed to the next step and perform the training which requires previously generated '.csv' files to be present in the working directory.

      2.2] The scripts to perform training are accessible in the directory '/RunScripts/LOGOS_02_NN_Model/NN-Trainining/*.py'

      2.3] As an Output the best weights are save in the 'tar' files. 

      2.4] Command line argument:
           >>> python ~/RunScripts/LOGOS_02_NN_Model/NN-Trainining/*.py'

========================================================================================================================================
      STEP 2 --->  Prediction based cluster building
========================================================================================================================================

      3.1] The Input in given in '02_input.cb' file formated as below

	chemically_distinct_env = _ (No of atom chemically inequivalent atom types | integer)
	atoms_per_env           = _ (the total 1 2 count for each chemically_distinct_env' separated by a space in between | integers)
	indices_per_env         = _ (specify the indices of each 'atoms_per_env' as in an molecule.xyz file)
	0                           (indices separated by a space in between | integers)
                                    (blank line)
	Filename                = _ (filename of parant structure | string)
	Desired_Size            = _ desired cluster size | integer)
	PGSymmetry              = _ 0 (if no symmetry) / 1 (if it is symmetric)
	addition                = _ no. of monomer units to be added in each step | integer)
                                    (blank line)
	indices_per_env         = _ (specify the indices of each 'atoms_per_env' as in an molecule.xyz file)
	0                           (indices separated by a space in between | integers)
                                    (blank line)
	No_of_parent_structures = _ (No. of parant structure | integer)
	Desired_Size            = _ desired cluster size | integer)
	PGSymmetry              = _ 0 (if no symmetry) / 1 (if it is symmetric)
	addition                = _ no. of monomer units to be added in each step | integer)
                                    (blank line)
	cords                   = _ a monomer coordinate file
	topography              = _ a molecular topography coordinate file
                                    (blank line)
	NN_potential_r          =   _.pth.tar | character)
	NN_potential_t          =   _.pth.tar
	NN_potential_phi        =   _.pth.tar
	NN_potential_delta      =   _.pth.tar
	NN_potential_epsilon    =   _.pth.tar
	NN_potential_omega      =   _.pth.tar
     
      3.2] OUTPUT
          generates Daughter structures in 01-O-Paths
  
      3.3] Command line:
        >>> python ~/RunScripts/LOGOS_03_01_DaughterStructureGeneration.py 02_input.cb

========================================================================================================================================

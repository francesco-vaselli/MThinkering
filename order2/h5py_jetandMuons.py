import uproot
import pandas as pd
import h5py 
import numpy as np


if __name__=='__main__':
    f = 1 
    tree = uproot.open(f"MJets{f}.root:MJets", num_workers=20)
    vars_to_save = tree.keys()
    print(vars_to_save)

    # define pandas df for fast manipulation 
    df = tree.arrays(library="pd").reset_index(drop=True).astype('float32').dropna()
    print(df)
    
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]	
    # apply cuts
    df= df[df.MJet_massRatio < 7]
    df = df[df.MJet_massRatio > 0]
    df = df[df.MJet_ptRatio < 3]
    df = df[df.MJet_ptRatio > 0]
    maskB = (df["MJet_btagDeepB"]<0) 
    maskC = (df["MJet_btagDeepB"]<0) 
    maskCSV = (df["MJet_btagDeepB"]<0) 
    maskQGL = (df["MJet_qgl"]<0) 
    df.loc[maskB, "MJet_btagDeepB"] = -0.1
    df.loc[maskC, "MJet_btagCSVV2"] = -0.1
    df.loc[maskCSV, "MJet_btagDeepC"] = -0.1
    df.loc[maskQGL, "MJet_qgl"] = -0.1
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]	 
    print(df)
        
    # open hdf5 file for saving
    f = h5py.File(f'jets_and_muons{f}.hdf5','w')

    dset = f.create_dataset("data", data=df.values)#, dtype='f4')

    f.close()

import uproot
import pandas as pd
import h5py 
import numpy as np


if __name__=='__main__':
	f = 7 
	tree = uproot.open(f"MMuons{f}.root:MMuons", num_workers=20)
	vars_to_save = tree.keys()
	print(vars_to_save)

	# define pandas df for fast manipulation 
	df = tree.arrays(library="pd").reset_index(drop=True).astype('float32').dropna()
	print(df)
	
	df = df.drop(["MGenPart_statusFlags2", "MGenPart_statusFlags12", "MGenPart_statusFlags14"], axis=1)
	dr = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]	
	# apply cuts
	df = df[df.MMuon_dxy < 1.5]
	df = df[df.MMuon_dxy > -1.5]
	df = df[df.MMuon_dxyErr < 0.02]
	#	df = df[df.MMuon_dxybs < 1]
	# df = df[df.MMuon_dxybs > -1]
	df = df[df.MMuon_dz < 2]
	df = df[df.MMuon_dz > -2]
	df = df[df.MMuon_dzErr < 2]
	df = df[df.MMuon_ip3d < 1]
	df = df[df.MMuon_jetPtRelv2 < 200]
	df = df[df.MMuon_jetRelIso < 100]
	df = df[df.MMuon_pfRelIso03_all < 100]
	df = df[df.MMuon_pfRelIso03_chg < 40]
	df = df[df.MMuon_pfRelIso04_all < 70]
	df = df[df.MMuon_ptErr < 20]
	df = df[df.MMuon_sip3d < 1000]
	print(df)
		
	# open hdf5 file for saving
	f = h5py.File(f'muons{f}.hdf5','w')

	dset = f.create_dataset("data", data=df.values)#, dtype='f4')

	f.close()
	

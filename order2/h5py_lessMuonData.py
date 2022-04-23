import uproot
import pandas as pd
import h5py 
import numpy as np


if __name__=='__main__':
	f = 1 
	tree = uproot.open(f"MMuons{f}.root:MMuons", num_workers=20)
	vars_to_save = tree.keys()
	print(vars_to_save)

	# define pandas df for fast manipulation 
	df = tree.arrays(library="pd").reset_index(drop=True).astype('float32').dropna()
	print(df)
	
	df = df.drop(["MGenPart_statusFlags2", "MGenPart_statusFlags12", "MGenPart_statusFlags14",
			"Pileup_gpudensity", "Pileup_nPU", "Pileup_pudensity", "Pileup_sumEOOT",
			"Pileup_sumLOOT", "MMuon_isGlobal", "MMuon_isPFcand", "MMuon_isTracker",
			"MMuon_pfRelIso03_all", "MMuon_pfRelIso03_chg", "MMuon_softMvaId"], axis=1)
	df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]	
	# apply cuts
	df = df[df.MMuon_dxy < 1.5]
	df = df[df.MMuon_dxy > -1.5]
	df = df[df.MMuon_dxyErr < 0.02]
	df = df[df.MMuon_dz < 2]
	df = df[df.MMuon_dz > -2]
	df = df[df.MMuon_dzErr < 2]
	df = df[df.MMuon_ip3d < 1]
	df = df[df.MMuon_jetPtRelv2 < 200]
	df = df[df.MMuon_jetRelIso < 100]
	df = df[df.MMuon_pfRelIso04_all < 70]
	df = df[df.MMuon_ptErr < 20]
	df = df[df.MMuon_sip3d < 1000]
	# apply transforms
	df["MGenMuon_pt"] = df["MGenMuon_pt"].apply(lambda x: np.log(x))
	df["MClosestJet_pt"] = df["MClosestJet_pt"].apply(lambda x: np.log1p(x))
	df["MClosestJet_mass"] = df["MClosestJet_mass"].apply(lambda x: np.log1p(x))
	df["MMuon_etaMinusGen"] = df["MMuon_etaMinusGen"].apply(lambda x: np.arctan(x*100))
	df["MMuon_phiMinusGen"] = df["MMuon_phiMinusGen"].apply(lambda x: np.arctan(x*80))
	df["MMuon_ptRatio"] = df["MMuon_ptRatio"].apply(lambda x: np.arctan((x-1)*10))
	df["MMuon_dxy"] = df["MMuon_dxy"].apply(lambda x: np.arctan(x*150))
	df["MMuon_dxyErr"] = df["MMuon_dxyErr"].apply(lambda x: np.log1p(x))
	df["MMuon_dz"] = df["MMuon_dz"].apply(lambda x: np.arctan(x*50))
	df["MMuon_dzErr"] = df["MMuon_dzErr"].apply(lambda x: np.log(x+0.001))
	df["MMuon_ip3d"] = df["MMuon_ip3d"].apply(lambda x: np.log(x+0.001))
	df["MMuon_jetPtRelv2"] = df["MMuon_jetPtRelv2"].apply(lambda x: np.log(x+0.001))
	arr = df["MMuon_jetPtRelv2"].values
	arr[arr<=-4] = np.random.normal(loc=-6.9, scale=1, size=arr[arr<=-4].shape)
	df["MMuon_jetPtRelv2"] = arr
	df["MMuon_jetRelIso"] = df["MMuon_jetRelIso"].apply(lambda x: np.log(x+0.08))
	df["MMuon_pfRelIso04_all"] = df["MMuon_pfRelIso04_all"].apply(lambda x: np.log(x+0.00001))
	arr = df["MMuon_pfRelIso04_all"].values
	arr[arr<=-7.5] = np.random.normal(loc=-11.51, scale=1, size=arr[arr<=-7.5].shape)
	df["MMuon_pfRelIso04_all"] = arr
	df["MMuon_ptErr"] = df["MMuon_ptErr"].apply(lambda x: np.log(x+0.001))
	df["MMuon_sip3d"] = df["MMuon_sip3d"].apply(lambda x: np.log1p(x))
	df['MMuon_mediumId'] = df['MMuon_mediumId'].apply(lambda x: x + 0.1*np.random.normal())
	df['MMuon_softId'] = df['MMuon_softId'].apply(lambda x: x + 0.1*np.random.normal())
	df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
	print(df)
		
	# open hdf5 file for saving
	f = h5py.File(f'muons{f}.hdf5','w')

	dset = f.create_dataset("data", data=df.values, dtype='f4')

	f.close()
	

import h5py
import numpy as np
import matplotlib.pyplot as plt 
from numpy import mean, sqrt, square
import pandas as pd

if __name__=='__main__':
	f = h5py.File('data/muons1.hdf5', 'r')
	df = f['data']
	data = (df[:, 25:47])
	names = ["MMuon_etaMinusGen", "MMuon_phiMinusGen", "MMuon_ptRatio", "MMuon_dxy", "MMuon_dxyErr", "MMuon_dz", "MMuon_dzErr", 
				"MMuon_ip3d", "MMuon_isGlobal", "MMuon_isPFcand", "MMuon_isTracker", "MMuon_jetPtRelv2","MMuon_jetRelIso", "MMuon_mediumId", 
				"MMuon_pfRelIso03_all", "MMuon_pfRelIso03_chg", "MMuon_pfRelIso04_all", "MMuon_ptErr", "MMuon_sip3d", "MMuon_softId", 
				"MMuon_softMva", "MMuon_softMvaId"]
	df = pd.DataFrame(data=data, columns=names)
	df['MMuon_etaMinusGen'] = df['MMuon_etaMinusGen'].apply(lambda x: x*100)
	df['MMuon_phiMinusGen'] = df['MMuon_phiMinusGen'].apply(lambda x: x*100)
	df['MMuon_dxy'] = df['MMuon_dxy'].apply(lambda x: x*10)
	df['MMuon_dxyErr'] = df['MMuon_dxyErr'].apply(lambda x: np.log(x)+6.5)
	df['MMuon_dz'] = df['MMuon_dz'].apply(lambda x: x*10)
	df['MMuon_dzErr'] = df['MMuon_dzErr'].apply(lambda x: np.log(x)+5.5)
	df['MMuon_ip3d'] = df['MMuon_ip3d'].apply(lambda x: np.log(x)+5)
	df['MMuon_jetPtRelv2'] = df['MMuon_jetPtRelv2'].apply(lambda x: np.log1p(x))
	df['MMuon_jetRelIso'] = df['MMuon_jetRelIso'].apply(lambda x: np.log1p(x))
	df['MMuon_pfRelIso03_all'] = df['MMuon_pfRelIso03_all'].apply(lambda x: np.log1p(x))
	df['MMuon_pfRelIso03_chg'] = df['MMuon_pfRelIso03_chg'].apply(lambda x: np.log1p(x))
	df['MMuon_pfRelIso04_all'] = df['MMuon_pfRelIso04_all'].apply(lambda x: np.log1p(x))
	df['MMuon_ptErr'] = df['MMuon_ptErr'].apply(lambda x: np.log(x)+1.5)
	df['MMuon_sip3d'] = df['MMuon_sip3d'].apply(lambda x: np.log(x))
	

	curr = 20
	cdata = df.iloc[:, curr].values
	plt.hist(cdata, bins=100)
	plt.title(f"{names[curr]}")
	print(sqrt(mean(square(cdata))))
	plt.show()
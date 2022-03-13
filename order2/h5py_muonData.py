import uproot
import pandas as pd
import h5py 


if __name__=='__main__':
	tree = uproot.open("MMuons.root:MMuons", num_workers=20)
	#  tree = f["Events"]
	# filter_keys = ["nGenJet", "GenJet_*", "Jet*"]
	vars_to_save = tree.keys()
	print(vars_to_save)
	'''
	df = tree.arrays(filter_name=["GenJet_eta"], library="pd")
	print(df)
	'''
	# actually a tuple df
	df = tree.arrays(library="pd").reset_index(drop=True).astype('float32').dropna()
	#q = pandas.DataFrame(data=t.arrays(filter_name=filter_keys))
	print(df)

	f = h5py.File('muons.hdf5','w')

	dset = f.create_dataset("data", data=df.values)

	f.close()
	

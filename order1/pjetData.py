import uproot
import pandas as pd

if __name__=='__main__':
	tree = uproot.open("MJets.root:MJets", num_workers=20)
	#  tree = f["Events"]
	# filter_keys = ["nGenJet", "GenJet_*", "Jet*"]
	vars_to_save = tree.keys()
	print(vars_to_save)
	'''
	df = tree.arrays(filter_name=["GenJet_eta"], library="pd")
	print(df)
	'''
	# actually a tuple df
	df = tree.arrays(library="pd").reset_index(drop=True)
	#q = pandas.DataFrame(data=t.arrays(filter_name=filter_keys))
	print(df)
	
	df.to_hdf('MjetData.h5', key='df', mode='w')
	

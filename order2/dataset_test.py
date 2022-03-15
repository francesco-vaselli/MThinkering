import torch
from torch.utils.data import Dataset
import h5py
import numpy as np


class H5Dataset(Dataset):
	"""Pytorch Dataset for reading input data from hdf5 files on disk
	Datasets are lazy loaded as suggested by: https://vict0rs.ch/2021/06/15/pytorch-h5/
	Expects hdf5 filese containing a "data" Dataset, which in turn contains correctly processed data
	(there is no preprocessing here), and returns two separate tensor for each instance
	x: target variables (expects a 25:-1 ordering on each row)
	y: conditioning variables (expects a 0:25 ordering on each row)

	Args:
		Dataset (Pytorch Dataset): Pytorch Dataset class
	"""
	def __init__(self, h5_paths, limit=-1):
		"""Initialize the class, set indeces across datasets and define lazy loading

		Args:
			h5_paths (strings): paths to the various hdf5 files to include in the final Dataset
			limit (int, optional): optionally limit dataset length to specified values, if negative 
				returns the full length as inferred from files. Defaults to -1.
		"""		
		self.limit = limit
		self.h5_paths = h5_paths
		self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]

		a_strides = []
		idx_strides = []
		for a, archive in enumerate(self.archives):

			a_length = len(archive["data"])
			a_array = np.full(fill_value=a, shape=a_length)
			idx_array = np.arange(a_length)

			a_strides.append(a_array)
			idx_strides.append(idx_array)

		archive_full = np.concatenate(a_strides)
		idx = np.concatenate(idx_strides) 

		self.indices = dict(enumerate(np.column_stack((archive_full, idx))))
	
		self._archives = None

	@property
	def archives(self):
		if self._archives is None: # lazy loading here!
			self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
		return self._archives

	def __getitem__(self, index):
		a, i = self.indices[index]
		archive = self.archives[a]
		y = archive["data"][i, 0:25]
		x = archive["data"][i, 25:-1]
		y = torch.from_numpy(y)
		x = torch.from_numpy(x)

		return a, x, y

	def __len__(self):
		if self.limit > 0:
			return min([len(self.indices), self.limit])
		return len(self.indices)


if __name__=='__main__':

	f = h5py.File('data/muons.hdf5', 'r')
	df = f['data']
	data = torch.from_numpy(df[1, 0:25])

	dataset = H5Dataset(["data/muons.hdf5", "data/muons_(copy).hdf5"])

	loader = torch.utils.data.DataLoader(H5Dataset(["data/muons.hdf5", "data/muons_(copy).hdf5"]), batch_size= 1, num_workers=4)
	batch = next(iter(loader))
	for batch_idx, (x, y) in enumerate(loader):
		print(batch_idx, torch.mean(x), torch.mean(y))


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
        max_events = int(5e9)
        self.limit = max_events if limit==-1 else int(limit)
        self.h5_paths = h5_paths
        self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]

        self.strides = []
        for archive in self.archives:
            with archive as f:
                self.strides.append(len(f['data']))

        self.len_in_files = self.strides[1:]
        self.strides = np.cumsum(self.strides)
        self._archives = None

    @property
    def archives(self):
        if self._archives is None: # lazy loading here!
            self._archives = [h5py.File(h5_path, "r") for h5_path in self.h5_paths]
        return self._archives

    def __getitem__(self, index):
        file_idx = np.searchsorted(self.strides, index, side='right')
        idx_in_file = index - self.strides[max(0, file_idx-1)] 
        y = self.archives[file_idx]["data"][idx_in_file, 0:25]
        x = self.archives[file_idx]["data"][idx_in_file, 25:-1]
        y = torch.from_numpy(y)
        x = torch.from_numpy(x)

        return file_idx, idx_in_file, x, y

    def __len__(self):
        #return self.strides[-1] #this will process all files
        if self.limit <= self.strides[-1]:
            return self.limit
        else:
            return self.strides[-1]


if __name__=='__main__':

    f = h5py.File('data/muons.hdf5', 'r')
    df = f['data']
    data = torch.from_numpy(df[1, 0:25])

    dataset = H5Dataset(["data/muons1.hdf5", "data/muons1.hdf5"])
    print(len(dataset))
    print(dataset.__getitem__(int(1e6)))
    print(file_idx, idx_in_file, x, y)

    loader = torch.utils.data.DataLoader(H5Dataset(["data/muons.hdf5", "data/muons_(copy).hdf5"]), batch_size= 1, num_workers=4)
    batch = next(iter(loader))
    for x, y in test_loader:
        print(torch.mean(x), torch.mean(y))


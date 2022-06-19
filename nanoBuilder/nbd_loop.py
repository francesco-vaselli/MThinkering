import ROOT
import uproot
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
import nflows
import time
import os
import awkward as ak
import mynflow
import nbd_func

if __name__=='__main__':

    root = 'root_path_of_nanos'
    new_root = 'potentially_new_root_path_for_synth'
    files_paths = [os.path.join(d, f) for d in os.listdir(root) for f in os.listdir(os.path.join(root, d))]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	

    jet_flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_jets_plus_muons_@epoch_180.pt")
    muon_flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_muons_final_@epoch_430.pt")

    for path in tqdm(files_paths):
        path_str = str(path)
        nbd_func.nbd(jet_flow, muon_flow, root, path_str, new_root)

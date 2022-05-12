import ROOT
import uproot
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import nflows
import mynflow
import time
import awkward as ak


class GenDS(Dataset):
	def __init__(self, df, cond_vars):
	
		y= df.loc[:, cond_vars].values
		self.y_train=torch.tensor(y,dtype=torch.float32)#.to(device)

	def __len__(self):
		return len(self.y_train)
  
	def __getitem__(self,idx):
		return self.y_train[idx]



ROOT.gInterpreter.Declare('''
auto closest_muon_dr(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> distances;
	distances.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		distances.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
			}
		}
		if (closest < 0.4){
			distances[i] = closest;
		}
	}
	return distances;
}


auto closest_muon_pt(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & ptm) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> pts;
	pts.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		pts.emplace_back(0.0);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				pts[i] = ptm[j];
			}
		}
	}
	return pts;
}


auto closest_muon_deta(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> detas;
	detas.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		detas.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				detas[i] = deta;
			}
		}
	}
	return detas;
}


auto closest_muon_dphi(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		dphis.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				dphis[i] = dphi;
			}
		}
	}
	return dphis;
}

auto second_muon_dr(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> distances;
	distances.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		distances.emplace_back(0.5);
		float closest = 0.4;
		float second_closest = 0.5;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				closest = dr;
			}
		}
		if (second_closest < 0.4){
			distances[i] = second_closest;
		}
	}
	return distances;
}


auto second_muon_pt(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & ptm) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> pts;
	pts.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		pts.emplace_back(0.0);
		float closest = 0.4;
		float second_closest = 0.5;
		float closest_pt = 0.0;
		float second_pt = 0.0;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				second_pt = closest_pt;
				closest = dr;
				closest_pt = ptm[j];
			}
		if (second_closest < 0.4){
			pts[i] = second_pt;
		}
		}
	}
	return pts;
}


auto second_muon_deta(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> detas;
	detas.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		detas.emplace_back(0.5);
		float closest = 0.4;
		float second_closest = 0.5;
		float closest_deta = 0.0;
		float second_deta = 0.0;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				second_deta = closest_deta;
				closest = dr;
				closest_deta = deta;
			}
		if (second_closest < 0.4){
			detas[i] = second_deta;
		}
		}
	}
	return detas;
}


auto second_muon_dphi(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etaj.size();
	auto size_inner = etam.size();
	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		dphis.emplace_back(0.5);
		float closest = 0.4;
		float second_closest = 0.5;
		float closest_dphi = 0.0;
		float second_dphi = 0.0;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etaj[i]-etam[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phij[i]-phim[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				second_closest = closest;
				second_dphi = closest_dphi;
				closest = dr;
				closest_dphi = dphi;
			}
		if (second_closest < 0.4){
			dphis[i] = second_dphi;
		}
		}
	}
	return dphis;
}

auto DeltaPhi(ROOT::VecOps::RVec<float> &Phi1, ROOT::VecOps::RVec<float> &Phi2) {
	auto size = Phi1.size();
   	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size);
	for (size_t i = 0; i < size; i++) {
		Double_t dphi = TVector2::Phi_mpi_pi(Phi1[i]-Phi2[i]);
		dphis.emplace_back(dphi);
	}
	return dphis;
	}
auto closest_jet_dr(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> distances;
	distances.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		distances.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
			}
		}
		if (closest < 0.4){
			distances[i] = closest;
		}
	}
	return distances;
}
auto closest_jet_mass(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & massj) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> masses;
	masses.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		masses.emplace_back(0.0);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				masses[i] = massj[j];
			}
		}
	}
	return masses;
}
auto closest_jet_pt(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim, ROOT::VecOps::RVec<float> & ptj) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> pts;
	pts.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		pts.emplace_back(0.0);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				pts[i] = ptj[j];
			}
		}
	}
	return pts;
}
auto closest_jet_deta(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> detas;
	detas.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		detas.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				detas[i] = deta;
			}
		}
	}
	return detas;
}
auto closest_jet_dphi(ROOT::VecOps::RVec<float> & etaj, ROOT::VecOps::RVec<float> & phij, ROOT::VecOps::RVec<float> & etam, 
						ROOT::VecOps::RVec<float> & phim) {
	
	auto size_outer = etam.size();
	auto size_inner = etaj.size();
	ROOT::VecOps::RVec<float> dphis;
	dphis.reserve(size_outer);
	for (size_t i = 0; i < size_outer; i++) {
		dphis.emplace_back(0.5);
		float closest = 0.4;
		for (size_t j = 0; j < size_inner; j++) {
			Double_t deta = etam[i]-etaj[j];
	  		Double_t dphi = TVector2::Phi_mpi_pi(phim[i]-phij[j]);
	  		float dr = TMath::Sqrt( deta*deta+dphi*dphi );
			if (dr < closest) {
				closest = dr;
				dphis[i] = dphi;
			}
		}
	}
	return dphis;
}
auto BitwiseDecoder(ROOT::VecOps::RVec<int> &ints, int &bit) {
	auto size = ints.size();
   	ROOT::VecOps::RVec<float> bits;
	bits.reserve(size);
	int num = pow(2, (bit - 1));
	for (size_t i = 0; i < size; i++) {
		Double_t bAND = ints[i] & num;
		if (bAND == num) {
			bits.emplace_back(1);
		}
		else {bits.emplace_back(0);}
	}
	return bits;
	}

auto muons_per_event(ROOT::VecOps::RVec<int> &MGM){
	int size = MGM.size();
	return size;
	}


void gens(std::string x){
		ROOT::EnableImplicitMT();
		ROOT::RDataFrame d("Events", x);
	// create first mask
	auto d_def = d.Define("MuonMask", "Muon_genPartIdx >=0").Define("MatchedGenMuons", "Muon_genPartIdx[MuonMask]")
		//	.Define("JetMask","Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet").Define("MatchedGenJets","Jet_genJetIdx[JetMask]")
			.Define("MuonMaskJ", "GenPart_pdgId == 13 | GenPart_pdgId == -13") ;
	
	auto d_matched = d_def
				//.Define("MGenJet_eta", "Take(GenJet_eta,MatchedGenJets)")
				//.Define("MGenJet_mass", "Take(GenJet_mass,MatchedGenJets)")
				//.Define("MGenJet_phi", "Take(GenJet_phi,MatchedGenJets)")
				//.Define("MGenJet_pt", "Take(GenJet_pt,MatchedGenJets)")
				//.Define("MGenJet_partonFlavour", "Take(GenJet_partonFlavour,MatchedGenJets)")
				//.Define("MGenJet_hadronFlavour", "Take(GenJet_hadronFlavour,MatchedGenJets)")
				//.Alias("MGenJet_eta", "GenJet_eta")
				.Define("MMuon_pt", "GenPart_pt[MuonMaskJ]")
				.Define("MMuon_eta", "GenPart_eta[MuonMaskJ]")
				.Define("MMuon_phi", "GenPart_phi[MuonMaskJ]")
				.Define("MClosestMuon_dr", closest_muon_dr, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_deta", closest_muon_deta, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_dphi", closest_muon_dphi, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_pt", closest_muon_pt, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MSecondClosestMuon_dr", second_muon_dr, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_deta", second_muon_deta, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_dphi", second_muon_dphi, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_pt", second_muon_pt, {"GenJet_eta", "GenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MGenMuon_eta", "GenPart_eta[MuonMaskJ]")
				.Define("MGenMuon_phi", "GenPart_phi[MuonMaskJ]")
				.Define("MGenMuon_pt", "GenPart_pt[MuonMaskJ]")
				//.Define("nGenMuons", "Take(nGenPart, MuonMaskJ)")
				.Define("MGenPart_statusFlags","GenPart_statusFlags[MuonMaskJ]")
				.Define("MGenPart_statusFlags1", [](ROOT::VecOps::RVec<int> &ints){ int bit = 1; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags2", [](ROOT::VecOps::RVec<int> &ints){ int bit = 2; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags3", [](ROOT::VecOps::RVec<int> &ints){ int bit = 3; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags4", [](ROOT::VecOps::RVec<int> &ints){ int bit = 4; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags5", [](ROOT::VecOps::RVec<int> &ints){ int bit = 5; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags6", [](ROOT::VecOps::RVec<int> &ints){ int bit = 6; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags7", [](ROOT::VecOps::RVec<int> &ints){ int bit = 7; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags8", [](ROOT::VecOps::RVec<int> &ints){ int bit = 8; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags9", [](ROOT::VecOps::RVec<int> &ints){ int bit = 9; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags10", [](ROOT::VecOps::RVec<int> &ints){ int bit = 10; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags11", [](ROOT::VecOps::RVec<int> &ints){ int bit = 11; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags12", [](ROOT::VecOps::RVec<int> &ints){ int bit = 12; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags13", [](ROOT::VecOps::RVec<int> &ints){ int bit = 13; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MGenPart_statusFlags14", [](ROOT::VecOps::RVec<int> &ints){ int bit = 14; return BitwiseDecoder(ints, bit); }, {"MGenPart_statusFlags"})
				.Define("MClosestJet_dr", closest_jet_dr, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_deta", closest_jet_deta, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_dphi", closest_jet_dphi, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_pt", closest_jet_pt, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi", "GenJet_pt"})
				.Define("MClosestJet_mass", closest_jet_mass, {"GenJet_eta", "GenJet_phi","MGenMuon_eta", "MGenMuon_phi", "GenJet_mass"});
				
	vector<string> col_to_save = 
		{"nGenJet", "MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour","MGenMuon_eta" , "MGenMuon_phi", "MGenMuon_pt", "MGenPart_statusFlags1", "MGenPart_statusFlags2", "MGenPart_statusFlags3", "MGenPart_statusFlags4",	"MGenPart_statusFlags5", "MGenPart_statusFlags6", "MGenPart_statusFlags7", "MGenPart_statusFlags8", "MGenPart_statusFlags9", "MGenPart_statusFlags10", "MGenPart_statusFlags11",	"MGenPart_statusFlags12", "MGenPart_statusFlags13", "MGenPart_statusFlags14", "MClosestJet_dr", "MClosestJet_deta", "MClosestJet_dphi", "MClosestJet_pt", "MClosestJet_mass",	"Pileup_gpudensity", "Pileup_nPU", "Pileup_nTrueInt", "Pileup_pudensity", "Pileup_sumEOOT", "Pileup_sumLOOT", "event", "run"};	
	
	//d_matched.Snapshot("GensJ", "testGensJ.root", col_to_save);
	d_matched.Snapshot("Gens", "testGens.root", col_to_save);
}
''')

if __name__=='__main__':
	
	# select nano aod, process and save intermmediate filesto disk
	s = "../nanoaods/0088F3A1-0457-AB4D-836B-AC3022A0E34F.root"
	ROOT.gens(s)
	print('done')
	
	muon_cond = ["MGenMuon_eta", "MGenMuon_phi", "MGenMuon_pt", "MGenPart_statusFlags1", "MGenPart_statusFlags3", "MGenPart_statusFlags4",
			"MGenPart_statusFlags5", "MGenPart_statusFlags6", "MGenPart_statusFlags7", "MGenPart_statusFlags8", "MGenPart_statusFlags9", "MGenPart_statusFlags10", "MGenPart_statusFlags11", "MGenPart_statusFlags13", "MClosestJet_dr", "MClosestJet_deta", "MClosestJet_dphi", "MClosestJet_pt", "MClosestJet_mass","Pileup_gpudensity", "Pileup_nPU", "Pileup_nTrueInt", "Pileup_pudensity", "Pileup_sumEOOT", "Pileup_sumLOOT"]
	jet_cond = ["MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour",]
	# read processed files for jets and save event structure
	tree = uproot.open("testGens.root:Gens", num_workers=20)

	df = tree.arrays(jet_cond, library="pd").astype('float32').dropna()
	print(df)
	df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
	# print((df['nGenMuons'] == 0).any())
	# in short: this is a better way of keeping track of substructure with multiindex
	# for jets is the ONLY way as genjet is not accurate (different from our mask?)
	events_structure_jets = df.reset_index(level=1).index.value_counts().sort_index().values
	print(events_structure_jets)
	print(len(events_structure_jets))
	print(sum(events_structure_jets))
	
	# reset dataframe index fro 1to1 gen
	df.reset_index(drop=True)
	
	dfm = tree.arrays(muon_cond, library="pd").astype('float32').dropna()
	dfm = dfm[~dfm.isin([np.nan, np.inf, -np.inf]).any(1)]
	dfm["MGenMuon_pt"] = dfm["MGenMuon_pt"].apply(lambda x: np.log(x))
	dfm["MClosestJet_pt"] = dfm["MClosestJet_pt"].apply(lambda x: np.log1p(x))
	dfm["MClosestJet_mass"] = dfm["MClosestJet_mass"].apply(lambda x: np.log1p(x))
	dfm["Pileup_sumEOOT"] = dfm["Pileup_sumEOOT"].apply(lambda x: np.log(x))
	dfm["Pileup_sumLOOT"] = dfm["Pileup_sumLOOT"].apply(lambda x: np.log1p(x))
	dfm = dfm[~dfm.isin([np.nan, np.inf, -np.inf]).any(1)]
	print(dfm)
	muons_ev_index = np.unique( dfm.index.get_level_values(0).values)
	print(muons_ev_index)
	events_structure_muons = dfm.reset_index(level=1).index.value_counts().sort_index().values
	print(len(events_structure_muons))
	print(sum(events_structure_muons))
	#dfg = dfg.loc[df.index.get_level_values(0)]

	# reset dataframe index fro 1to1 gen
	dfm.reset_index(drop=True)
	
		
	dfe = tree.arrays(['event', 'run'], library="pd").astype('float32').dropna()
	print(dfe)
	dfe = dfe[~dfe.isin([np.nan, np.inf, -np.inf]).any(1)]
	# print((df['nGenMuons'] == 0).any())
	# in short: this is a better way of keeping track of substructure with multiindex
	# for jets is the ONLY way as genjet is not accurate (different from our mask?)
	events_structure = dfe.values
	print(events_structure.shape, events_structure.shape)
	
	zeros = np.zeros(len(dfe), dtype=int)
	print(len(muons_ev_index), len(events_structure_muons))
	np.put(zeros, muons_ev_index, events_structure_muons, mode='rise')
	events_structure_muons = zeros
	print(events_structure_muons.shape, events_structure_muons)
	print(sum(events_structure_muons))
	
	jet_dataset = GenDS(df, jet_cond)
	muon_dataset = GenDS(dfm, muon_cond)
	
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
	batch_size = 10000
	jet_loader = DataLoader(jet_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,  num_workers=20)
	flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_jets_plus_muons_@epoch_180.pt")
	tot_sample = []
	leftover_sample = []
	times = []

	for batch_idx, y in enumerate(jet_loader):

		if device is not None:
			y = y.float().to(device, non_blocking=True)

    # Compute log prob
		# print(y.shape)
		if len(y) == batch_size:
			start = time.time()
			sample = flow.sample(1, context=y)
			taken = time.time() - start
			print(f'Done {batch_size} data in {taken}s')
			times.append(taken)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    # print(sample.shape)
			tot_sample.append(sample)
		
		else:
			leftover_shape = len(y)
			sample = flow.sample(1, context=y)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    # print(sample.shape)
			leftover_sample.append(sample)



	print(np.mean(times))
	tot_sample = np.array(tot_sample)
	tot_sample = np.reshape(tot_sample, ((len(jet_loader)-1)*batch_size, 17))
	leftover_sample = np.array(leftover_sample)
	leftover_sample = np.reshape(leftover_sample, (leftover_shape, 17))
	totalj = np.concatenate((tot_sample, leftover_sample), axis=0)
	print(totalj.shape)
	
	
	muon_loader = DataLoader(muon_dataset, batch_size=batch_size, shuffle=False, pin_memory=True,  num_workers=20)
	flow, _, _, _, trh, tsh = mynflow.load_model("../nflows/saves", "model_only_muons_@epoch_300.pt")
	tot_sample = []
	leftover_sample = []
	times = []
	print('now generating muons')
	for batch_idx, y in enumerate(muon_loader):

		if device is not None:
			y = y.float().to(device, non_blocking=True)

    # Compute log prob
		# print(y.shape)
		if len(y) == batch_size:
			start = time.time()
			sample = flow.sample(1, context=y)
			taken = time.time() - start
			print(f'Done {batch_size} data in {taken}s')
			times.append(taken)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    # print(sample.shape)
			tot_sample.append(sample)
		
		else:
			leftover_shape = len(y)
			sample = flow.sample(1, context=y)
			sample = sample.detach().cpu().numpy()
			sample = np.squeeze(sample, axis=1)
	    # print(sample.shape)
			leftover_sample.append(sample)



	print(np.mean(times))
	tot_sample = np.array(tot_sample)
	tot_sample = np.reshape(tot_sample, ((len(muon_loader)-1)*batch_size, 22))
	leftover_sample = np.array(leftover_sample)
	leftover_sample = np.reshape(leftover_sample, (leftover_shape, 22))
	totalm = np.concatenate((tot_sample, leftover_sample), axis=0)
	
	# np.save('npsave', total)	
	jet_names = ["area", "btagCMVA", "btagCSVV2", "btagDeepB", "btagDeepC", "btagDeepFlavB", "btagDeepFlavC",
        	"etaMinusGen", "bRegCorr", "massRatio", "nConstituents", "phiMinusGen", "ptRatio", "puIdDisc", "qgl",
        	"muEF", "nMuons"]
	to_ttreej = dict(zip(jet_names, totalj.T))
	to_ttreej = ak.unflatten(ak.Array(to_ttreej), events_structure_jets)


	muon_names = ["etaMinusGen", "phiMinusGen", "ptRatio", "dxy", "dxyErr", "dxybs", "dz", "dzErr", "ip3d", "isGlobal", "isPFcand","isTracker", "jetPtRelv2","jetRelIso", "mediumId", "pfRelIso03_all", "pfRelIso03_chg", "pfRelIso04_all", "ptErr","sip3d", "softId", "softMva", "softMvaId"]

	to_ttreem = dict(zip(muon_names, totalm.T))
	to_ttreem = ak.Array(to_ttreem)
	to_ttreem = ak.unflatten(to_ttreem, events_structure_muons)
	
	to_ttreee = dict(zip(['event', 'run'], events_structure.T))	
	to_ttreee = ak.Array(to_ttreee)	
	
	# final = ak.concatenate([to_ttreej, to_ttreem, to_ttreee])
	with uproot.recreate("output.root") as file:
		file["tree"] = {'Jets': to_ttreej, 'Muons': to_ttreem, 'event': to_ttreee.event, 'run': to_ttreee.run}

import ROOT
import uproot

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
			.Define("JetMask","Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet").Define("MatchedGenJets","Jet_genJetIdx[JetMask]")
			.Define("MuonMaskJ", "GenPart_pdgId == 13 | GenPart_pdgId == -13") ;
	
	auto d_matched = d_def.Define("MGenJet_eta", "Take(GenJet_eta,MatchedGenJets)")
				.Define("MGenJet_mass", "Take(GenJet_mass,MatchedGenJets)")
				.Define("MGenJet_phi", "Take(GenJet_phi,MatchedGenJets)")
				.Define("MGenJet_pt", "Take(GenJet_pt,MatchedGenJets)")
				.Define("MGenJet_partonFlavour", "Take(GenJet_partonFlavour,MatchedGenJets)")
				.Define("MGenJet_hadronFlavour", "Take(GenJet_hadronFlavour,MatchedGenJets)")
				.Define("MMuon_pt", "GenPart_pt[MuonMaskJ]")
				.Define("MMuon_eta", "GenPart_eta[MuonMaskJ]")
				.Define("MMuon_phi", "GenPart_phi[MuonMaskJ]")
				.Define("MClosestMuon_dr", closest_muon_dr, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_deta", closest_muon_deta, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_dphi", closest_muon_dphi, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_pt", closest_muon_pt, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MSecondClosestMuon_dr", second_muon_dr, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_deta", second_muon_deta, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_dphi", second_muon_dphi, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_pt", second_muon_pt, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MGenMuon_eta", "Take(GenPart_eta,MatchedGenMuons)")
				.Define("MGenMuon_phi", "Take(GenPart_phi,MatchedGenMuons)")
				.Define("MGenMuon_pt", "Take(GenPart_pt,MatchedGenMuons)")
				.Define("nGenMuons", muons_per_event, {"MatchedGenMuons"})
				.Define("MGenPart_statusFlags","Take(GenPart_statusFlags,MatchedGenMuons)")
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
		{"MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi", "MGenJet_eta", "MGenJet_mass", "MGenJet_phi", "MGenJet_pt", "MGenJet_partonFlavour", "MGenJet_hadronFlavour"};


vector<string> cols = { "MGenMuon_eta" , "MGenMuon_phi", "MGenMuon_pt", "MGenPart_statusFlags1", "MGenPart_statusFlags2", "MGenPart_statusFlags3", "MGenPart_statusFlags4",	"MGenPart_statusFlags5", "MGenPart_statusFlags6", "MGenPart_statusFlags7", "MGenPart_statusFlags8", "MGenPart_statusFlags9", "MGenPart_statusFlags10", "MGenPart_statusFlags11",	"MGenPart_statusFlags12", "MGenPart_statusFlags13", "MGenPart_statusFlags14", "MClosestJet_dr", "MClosestJet_deta", "MClosestJet_dphi", "MClosestJet_pt", "MClosestJet_mass",	"Pileup_gpudensity", "Pileup_nPU", "Pileup_nTrueInt", "Pileup_pudensity", "Pileup_sumEOOT", "Pileup_sumLOOT", "nGenMuons"};	
	
	d_matched.Snapshot("GensJ", "testGensJ.root", col_to_save);
	d_matched.Snapshot("GensM", "testGensM.root", cols);
	d_matched.Snapshot("GenJet", "nGenJet.root", "nGenJet");
	d_matched.Snapshot("GenMuon", "nGenMuon.root", "nGenMuons");
}
''')

s = "../nanoaods/0088F3A1-0457-AB4D-836B-AC3022A0E34F.root"
ROOT.gens(s)
tree = uproot.open("testGensJ.root:GensJ", num_workers=20)
vars_to_save = tree.keys()
print(vars_to_save)

df = tree.arrays(library="pd").astype('float32').dropna()
print(df)
# print((df['nGenMuons'] == 0).any())
# in short: this is a better way of keeping track of substructure with multiindex
# fro jets is the ONLY way as genjet is not accurate (different from our mask?)
df1 = df.reset_index(level=1).index.value_counts().sort_index()
print(df1)
print(sum(df1.values))
tree1 = uproot.open("nGenJet.root:GenJet", num_workers=20)
vars_to_save = tree1.keys()
print(vars_to_save)

dfg= tree1.arrays(library="pd").astype('float32')
print(dfg)
#dfg = dfg.loc[df.index.get_level_values(0)]
# dfg= dfg[dfg.nGenMuons != 0]
print(dfg)
print(sum(dfg.values))

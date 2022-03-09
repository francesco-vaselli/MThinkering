
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


void only_muons(){
        ROOT::EnableImplicitMT();
        ROOT::RDataFrame d("Events","../data/0088F3A1-0457-AB4D-836B-AC3022A0E34F.root");

	// create first mask
	auto d_def = d.Define("JetMask","Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet").Define("MatchedGenJets","Jet_genJetIdx[JetMask]")
					.Define("MuonMask", "Muon_genPartIdx >=0").Define("MatchedGenMuons", "Muon_genPartIdx[MuonMask]") ;
	
	auto colType1 = d_def.GetColumnType("JetMask");
	// Print column type
	std::cout << "Column JetMask" << " has type " << colType1 << std::endl;

	auto colType2 = d_def.GetColumnType("MatchedGenJets");
	// Print column type
	std::cout << "Column MjG " << " has type " << colType2 << std::endl;

	std::vector<std::string> gen_vars = { "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour" };
	auto d_matched = d_def.Define("MGenJet_eta", "Take(GenJet_eta,MatchedGenJets)")
                .Define("MGenJet_mass", "Take(GenJet_mass,MatchedGenJets)")
				.Define("MGenJet_phi", "Take(GenJet_phi,MatchedGenJets)")
				.Define("MGenJet_pt", "Take(GenJet_pt,MatchedGenJets)")
                .Define("MGenMuon_eta", "Take(GenPart_eta,MatchedGenMuons)")
				.Define("MGenMuon_phi", "Take(GenPart_phi,MatchedGenMuons)")
				.Define("MGenMuon_pt", "Take(GenPart_pt,MatchedGenMuons)")
                .Define("MClosestJet_dr", closest_jet_dr, {"MGenJet_eta", "MGenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_deta", closest_jet_deta, {"MGenJet_eta", "MGenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_dphi", closest_jet_dphi, {"MGenJet_eta", "MGenJet_phi","MGenMuon_eta", "MGenMuon_phi"})
				.Define("MClosestJet_pt", closest_jet_pt, {"MGenJet_eta", "MGenJet_phi","MGenMuon_eta", "MGenMuon_phi", "MGenJet_pt"})
                .Define("MClosestJet_mass", closest_jet_mass, {"MGenJet_eta", "MGenJet_phi","MGenMuon_eta", "MGenMuon_phi", "MGenJet_mass"})
				.Define("MMuon_etaMinusGen", "Muon_eta[MuonMask]-MGenMuon_eta")
                .Define("MMuon_filteredphi", "Muon_phi[MuonMask]")
				.Define("MMuon_phiMinusGen", DeltaPhi, {"MMuon_filteredphi", "MGenMuon_phi"})
                .Define("MMuon_ptRatio", "Muon_pt[MuonMask]/MGenMuon_pt");


	//d_matched.Foreach([](ROOT::VecOps::RVec<float> k, ROOT::VecOps::RVec<float> i){ std::cout<<k<<i<<std::endl;}, {"MGenJet_pt","MJet_pt"});
	auto v2 = d_matched.GetColumnNames();
	// for (auto &&colName : v2) std::cout <<"\""<< colName<<"\", ";
	vector<string> col_to_save = 
		{"MGenMuon_eta", "MGenMuon_phi", "MGenMuon_pt", "MClosestJet_dr", "MClosestJet_deta", "MClosestJet_dphi", "MClosestJet_pt", "MClosestJet_mass",
            "MMuon_etaMinusGen", "MMuon_phiMinusGen", "MMuon_ptRatio"};

	d_matched.Snapshot("MMuons", "MMuons.root", col_to_save);

}


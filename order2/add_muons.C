#define M_PI 3.14159265358979323846

// curr not working as intended (extra dims)
auto DeltaPhi(ROOT::VecOps::RVec<float> &Phi1, ROOT::VecOps::RVec<float> &Phi2) {
	ROOT::VecOps::RVec<float> dphi = Phi1 - Phi2;
	auto thisSize = dphi.size();
   	ROOT::VecOps::RVec<float> diff;
	diff.reserve(thisSize);
        for (auto &&val : dphi) {
		if ( val > M_PI ) {
        		val -= 2.0*M_PI;
			diff.emplace_back(val);
        	} else if ( val <= -M_PI ) {
               		val += 2.0*M_PI;
			diff.emplace_back(val);
        	}
	}
        return diff;
      }


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


void add_muons(){
        ROOT::EnableImplicitMT();
        ROOT::RDataFrame d("Events","../data/0088F3A1-0457-AB4D-836B-AC3022A0E34F.root");

	// create first mask
	auto d_def = d.Define("JetMask","Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet").Define("MatchedGenJets","Jet_genJetIdx[JetMask]")
					.Define("MuonMask", "GenPart_pdgId == 13 | GenPart_pdgId == -13") ;
	
	auto colType1 = d_def.GetColumnType("JetMask");
	// Print column type
	std::cout << "Column JetMask" << " has type " << colType1 << std::endl;

	auto colType2 = d_def.GetColumnType("MatchedGenJets");
	// Print column type
	std::cout << "Column MjG " << " has type " << colType2 << std::endl;
	
	//auto d2 = d_def.Display({"JetMask", "MatchedGenJets"});
	// Printing the short representations, the event loop will run
	// std::cout<<d2->Print();
	// d_def.Foreach([](ROOT::VecOps::RVec<int> k, ROOT::VecOps::RVec<int> i, ROOT::VecOps::RVec<int> j){ std::cout<<k<<i<<j<<std::endl;}, {"Jet_genJetIdx","JetMask", "MatchedGenJets"});
	// d_def.Foreach([](ROOT::VecOps::RVec<int> j){ operator<<(std::cout,j);}, {"MatchedGenJets"});
	std::vector<std::string> gen_vars = { "GenJet_eta", "GenJet_mass", "GenJet_phi", "GenJet_pt", "GenJet_partonFlavour", "GenJet_hadronFlavour" };
	auto d_matched = d_def.Define("MGenJet_eta", "Take(GenJet_eta,MatchedGenJets)")
				.Define("MGenJet_mass", "Take(GenJet_mass,MatchedGenJets)")
				.Define("MGenJet_phi", "Take(GenJet_phi,MatchedGenJets)")
				.Define("MGenJet_pt", "Take(GenJet_pt,MatchedGenJets)")
				.Define("MGenJet_partonFlavour", "Take(GenJet_partonFlavour,MatchedGenJets)")
				.Define("MGenJet_hadronFlavour", "Take(GenJet_hadronFlavour,MatchedGenJets)")
				.Define("MMuon_pt", "GenPart_pt[MuonMask]")
				.Define("MMuon_eta", "GenPart_eta[MuonMask]")
				.Define("MMuon_phi", "GenPart_phi[MuonMask]")
				.Define("MClosestMuon_dr", closest_muon_dr, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_deta", closest_muon_deta, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_dphi", closest_muon_dphi, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MClosestMuon_pt", closest_muon_pt, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MSecondClosestMuon_dr", second_muon_dr, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_deta", second_muon_deta, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_dphi", second_muon_dphi, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi"})
				.Define("MSecondClosestMuon_pt", second_muon_pt, {"MGenJet_eta", "MGenJet_phi","MMuon_eta", "MMuon_phi", "MMuon_pt"})
				.Define("MJet_area", "Jet_area[JetMask]")
				.Define("MJet_bRegCorr", "Jet_bRegCorr[JetMask]")
				.Define("MJet_bRegRes", "Jet_bRegRes[JetMask]")
				.Define("MJet_btagCMVA", "Jet_btagCMVA[JetMask]")
				.Define("MJet_btagCSVV2", "Jet_btagCSVV2[JetMask]")
				.Define("MJet_btagDeepB", "Jet_btagDeepB[JetMask]")
				.Define("MJet_btagDeepC", "Jet_btagDeepC[JetMask]")
				.Define("MJet_btagDeepCvB", "Jet_btagDeepCvB[JetMask]")
				.Define("MJet_btagDeepCvL", "Jet_btagDeepCvL[JetMask]")
				.Define("MJet_btagDeepFlavB", "Jet_btagDeepFlavB[JetMask]")
				.Define("MJet_btagDeepFlavC", "Jet_btagDeepFlavC[JetMask]")
				.Define("MJet_btagDeepFlavCvB", "Jet_btagDeepFlavCvB[JetMask]")
				.Define("MJet_btagDeepFlavCvL", "Jet_btagDeepFlavCvL[JetMask]")
				.Define("MJet_btagDeepFlavQG", "Jet_btagDeepFlavQG[JetMask]")			
				.Define("MJet_cRegCorr", "Jet_cRegCorr[JetMask]")			
				.Define("MJet_cRegRes", "Jet_cRegRes[JetMask]")			
				.Define("MJet_chEmEF", "Jet_chEmEF[JetMask]")			
				.Define("MJet_chFPV0EF", "Jet_chFPV0EF[JetMask]")			
				.Define("MJet_chFPV1EF", "Jet_chFPV1EF[JetMask]")			
				.Define("MJet_chFPV2EF", "Jet_chFPV2EF[JetMask]")			
				.Define("MJet_chFPV3EF", "Jet_chFPV3EF[JetMask]")			
				.Define("MJet_chHEF", "Jet_chHEF[JetMask]")			
				.Define("MJet_cleanmask", "Jet_cleanmask[JetMask]")			
				.Define("MJet_etaMinusGen", "Jet_eta[JetMask]-MGenJet_eta")
				.Define("MJet_hfsigmaEtaEta", "Jet_hfsigmaEtaEta[JetMask]")		
				.Define("MJet_hfsigmaPhiPhi", "Jet_hfsigmaPhiPhi[JetMask]")			
				.Define("MJet_hadronFlavour", "Jet_hadronFlavour[JetMask]")		
				.Define("MJet_jetId", "Jet_jetId[JetMask]")
				.Define("MJet_mass", "Jet_mass[JetMask]")
				.Define("MJet_massRatio", "Jet_mass[JetMask]/MGenJet_mass")
				.Define("MJet_muEF", "Jet_muEF[JetMask]")
				.Define("MJet_muonSubtrFactor", "Jet_muonSubtrFactor[JetMask]")
				.Define("MJet_nConstituents", "Jet_nConstituents[JetMask]")
				.Define("MJet_nElectrons", "Jet_nElectrons[JetMask]")
				.Define("MJet_nMuons", "Jet_nMuons[JetMask]")
				.Define("MJet_neEmEF", "Jet_neEmEF[JetMask]")
				.Define("MJet_neHEF", "Jet_neHEF[JetMask]")
				.Define("MJet_partonFlavour", "Jet_partonFlavour[JetMask]")
				.Define("MJet_phiMinusGen", "Jet_phi[JetMask]-MGenJet_phi")
				//.Define("MJet_phiMinusGen", DeltaPhi,{"MJet_phi", "MGenJet_phi"})
				.Define("MJet_ptRatio", "Jet_pt[JetMask]/MGenJet_pt")
				.Define("MJet_puId", "Jet_puId[JetMask]")
				.Define("MJet_hfadjacentEtaStripsSize", "Jet_hfadjacentEtaStripsSize[JetMask]")
				.Define("MJet_hfcentralEtaStripSize", "Jet_hfcentralEtaStripSize[JetMask]")
				.Define("MJet_puIdDisc", "Jet_puIdDisc[JetMask]")
				.Define("MJet_qgl", "Jet_qgl[JetMask]")
				.Define("MJet_rawFactor", "Jet_rawFactor[JetMask]");


	//d_matched.Foreach([](ROOT::VecOps::RVec<float> k, ROOT::VecOps::RVec<float> i){ std::cout<<k<<i<<std::endl;}, {"MGenJet_pt","MJet_pt"});
	auto v2 = d_matched.GetColumnNames();
	// for (auto &&colName : v2) std::cout <<"\""<< colName<<"\", ";
	vector<string> col_to_save = 
		{"MGenJet_eta", "MClosestMuon_dr", "MClosestMuon_pt", "MClosestMuon_deta", "MClosestMuon_dphi", "MSecondClosestMuon_dr", "MSecondClosestMuon_pt",
			"MSecondClosestMuon_deta", "MSecondClosestMuon_dphi"};
		
		// "MGenJet_mass", "MGenJet_phi", "MGenJet_pt", "MGenJet_partonFlavour", "MGenJet_hadronFlavour", "MJet_area", "MJet_bRegCorr", "MJet_bRegRes",		"MJet_btagCMVA", "MJet_btagCSVV2", "MJet_btagDeepB", "MJet_btagDeepC", "MJet_btagDeepCvB", "MJet_btagDeepCvL","MJet_btagDeepFlavB", "MJet_btagDeepFlavC", "MJet_btagDeepFlavCvB", "MJet_btagDeepFlavCvL", "MJet_btagDeepFlavQG", "MJet_cRegCorr", "MJet_cRegRes", 
		// 	"MJet_chEmEF", "MJet_chFPV0EF", "MJet_chFPV1EF", "MJet_chFPV2EF", "MJet_chFPV3EF", "MJet_chHEF", "MJet_cleanmask", "MJet_etaMinusGen", "MJet_hfsigmaEtaEta", "MJet_hfsigmaPhiPhi", "MJet_hadronFlavour","MJet_hfadjacentEtaStripsSize","MJet_hfcentralEtaStripSize",
		//  	"MJet_jetId", "MJet_mass", "MJet_muEF", "MJet_muonSubtrFactor", "MJet_nConstituents", "MJet_nElectrons", "MJet_nMuons", "MJet_neEmEF", "MJet_neHEF",
		// 	"MJet_massRatio", "MJet_partonFlavour", "MJet_phiMinusGen", "MJet_ptRatio", "MJet_puId", "MJet_puIdDisc", "MJet_qgl", "MJet_rawFactor"};

	d_matched.Snapshot("MJets", "MJets.root", col_to_save);

}


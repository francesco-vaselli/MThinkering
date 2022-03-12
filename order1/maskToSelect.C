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


void maskToSelect(){
		ROOT::EnableImplicitMT();
		ROOT::RDataFrame d("Events","0088F3A1-0457-AB4D-836B-AC3022A0E34F.root");

	// create first mask
	auto d_def = d.Define("JetMask","Jet_genJetIdx >=0  && Jet_genJetIdx < nGenJet").Define("MatchedGenJets","Jet_genJetIdx[JetMask]") ;
	
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
				.Define("MJet_phifiltered", "Jet_phi[JetMask]")
				.Define("MJet_phiMinusGen", DeltaPhi,{"MJet_phifiltered", "MGenJet_phi"})
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
		{"MGenJet_eta", "MGenJet_mass", "MGenJet_phi", "MGenJet_pt", "MGenJet_partonFlavour", "MGenJet_hadronFlavour", "MJet_area", "MJet_bRegCorr", "MJet_bRegRes",		"MJet_btagCMVA", "MJet_btagCSVV2", "MJet_btagDeepB", "MJet_btagDeepC", "MJet_btagDeepCvB", "MJet_btagDeepCvL","MJet_btagDeepFlavB", "MJet_btagDeepFlavC", "MJet_btagDeepFlavCvB", "MJet_btagDeepFlavCvL", "MJet_btagDeepFlavQG", "MJet_cRegCorr", "MJet_cRegRes", 
		"MJet_chEmEF", "MJet_chFPV0EF", "MJet_chFPV1EF", "MJet_chFPV2EF", "MJet_chFPV3EF", "MJet_chHEF", "MJet_cleanmask", "MJet_etaMinusGen", "MJet_hfsigmaEtaEta", "MJet_hfsigmaPhiPhi", "MJet_hadronFlavour","MJet_hfadjacentEtaStripsSize","MJet_hfcentralEtaStripSize",
		 "MJet_jetId", "MJet_mass", "MJet_muEF", "MJet_muonSubtrFactor", "MJet_nConstituents", "MJet_nElectrons", "MJet_nMuons", "MJet_neEmEF", "MJet_neHEF",
"MJet_massRatio",		 "MJet_partonFlavour", "MJet_phiMinusGen", "MJet_ptRatio", "MJet_puId", "MJet_puIdDisc", "MJet_qgl", "MJet_rawFactor"};

	d_matched.Snapshot("MJets", "MJets.root", col_to_save);

}


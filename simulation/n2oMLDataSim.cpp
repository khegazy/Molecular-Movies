#include "n2oMLDataSim.h"
#include <ctime>

int main(int argc, char* argv[]) {


  PLOTclass plt;

  int Nsimulations = 1;
  int NtimeSteps = 11;
  int bins = 457; //256;
  int Nbessels = 3;
  std::string pdfFile = "NULL";
  std::string fileName = "n2oMLData";
  std::string outputDir = "./output/testing/";

  bool makeStatic = false;
  bool plotVerbose = false;




  /////////////////////////////////////
  /////  Setting Input Variables  /////
  /////////////////////////////////////

  int index = 0;

  if (argc > 1) {
    Nsimulations = atoi(argv[1]);
  }
  if (argc > 2) {
    for (int iarg=2; iarg<argc; iarg+=2) {
      if (strcmp(argv[iarg],"-Ofile")==0) {string str(argv[iarg+1]); fileName=str;}
      else if (strcmp(argv[iarg],"-PDF")==0) {string str(argv[iarg+1]); pdfFile=str;}
      else if (strcmp(argv[iarg],"-Odir")==0) {string str(argv[iarg+1]); outputDir=str;}
      else if (strcmp(argv[iarg],"-Index")==0) {index = atoi(argv[iarg+1]);}
      else {
        cerr<<"ERROR!!! Option "<<argv[iarg]<<" does not exist!"<<endl;
        exit(0);
      }
    }
  }
  std::string outFilePrefix = outputDir + "/" + fileName;

  double Iebeam = 5;
  double screenDist = 4;
  double elEnergy = 3.7e6;
  double qMax = 14;

//  if (tools::fileExists(outFilePrefix + "atmDiffractionPattern_" + outFileSuffix + ".dat")) {
//    cerr << "\n\nINFO: The output files for \"" << outFileSuffix << "\" already exists!!!\n\n";
//    exit(0);
//  }

  // Finding the seed for the random number generator
  double seed = (int)(std::time(0))%500000 + 1000*(index + 1); //clock();
  for (int ir=0; ir<200; ir++) {
    TRandom3* seedRand = new TRandom3(seed);
    seed = 1e7*seedRand->Uniform();
    delete seedRand;
  }
  N2OMCclass n2oMC(seed*(index + 1));
  //N2OMCclass n2oMC(seed, "/reg/neh/home/khegazy/simulations/n2o/rotation/output/alignment/job39_357950.000000.root", "thetaPhiPDF");

  n2oMC.verbose = 0;
  n2oMC.Nmols = 1;
  n2oMC.NmolAtoms = 3;
  n2oMC.atomTypes.push_back(N);
  n2oMC.atomTypes.push_back(O);


  int NradDiffBins = bins/2 + bins%2;
  int ib, ir, it;
  double tStep, w1, w2, w3, wx, wy;
  double itm, c1, c2, c3;
  double scale = 1e19;
  std::string runID;
  std::vector<double> n1, n2, n3;
  std::vector< std::vector<double> > curLineOuts;
  std::vector< std::vector<double> > lineOuts(NtimeSteps);
  std::map<std::string, double> inpVals;
  inpVals["z1"] = 0; inpVals["z2"] = 0; inpVals["z3"] = 0;
  inpVals["x"] = 0; inpVals["y"] = 0;
  for (it=0; it<NtimeSteps; it++) {
    lineOuts[it].resize(NradDiffBins);
  }


  // Calculate scattering amplitudes
  n2oMC.makeMolEnsemble(inpVals);
  std::map<ATOMS, std::vector<double> > scatAmpInterp;
  DIFFRACTIONclass diffP(&n2oMC, qMax, Iebeam, screenDist, elEnergy, bins,
      "/reg/neh/home5/khegazy/simulations/scatteringAmplitudes/3.7MeV/");
  for (uint ia=0; ia<n2oMC.atomTypes.size(); ia++) {
    scatAmpInterp[n2oMC.atomTypes[ia]].resize(NradDiffBins, 0);
    int iang = 0;
    for (int iq=0; iq<NradDiffBins; iq++) {
      double ang = 2*asin((((iq + 0.5*(1 - NradDiffBins%2))/NradDiffBins)*qMax
                      *diffP.lambda/angs_to_au)/(4*PI))*180/PI;
      scatAmpInterp[n2oMC.atomTypes[ia]][iq]  = 1;
              //= diffP.interpScatAmp(n2oMC.atomTypes[ia], ang, iang);
    }
  }


  // Calculate X matrix for fitting bessels
  float irShift = 0;
  int qShift = 0;
  int Nradii = (int)(NradDiffBins/2); // Do not include r=0, done in loop
  float delta = 2*PI/NradDiffBins;
  if (bins%2) {
    irShift = 1;
    qShift = 1;
  }
  else {
    irShift = 0.5;
    qShift = 0.5;
  }


  std::vector<Eigen::MatrixXd> Xfits(Nbessels);
  std::vector<Eigen::MatrixXd> XNorms(Nbessels);
  Eigen::MatrixXd X0(NradDiffBins, Nradii);
  Eigen::MatrixXd X(NradDiffBins - bins%2, Nradii);      // Drop first entry (0.0) to invert 
  Eigen::VectorXd Y0(NradDiffBins);
  Eigen::VectorXd Y(NradDiffBins - bins%2);
  Eigen::VectorXd weights(Nradii);

  // If NradDiffBins are odd then do ir=0 for j=0 only
  //    j>0 will give vector of 0s
  for (int ir=0; ir<Nradii; ir++) {
    for (uint iq=0; iq<X0.rows(); iq++) {
      X0(iq,ir) = 
            boost::math::cyl_bessel_j(0, 
                (double)(delta*(ir+irShift)*(iq+(1-qShift))));
            // /(scatAmpInterp[N][iq]*scatAmpInterp[O][iq]);
    }
  }
  Xfits[0] = X0;

  for (int ib=1; ib<Nbessels; ib++) {
    for (int ir=0; ir<Nradii; ir++) {
      for (int iq=0; iq<X.rows(); iq++) {
        X(iq,ir) = boost::math::cyl_bessel_j(ib, 
             (double)(delta*(ir+irShift)*(iq + qShift)))
              /(scatAmpInterp[N][iq]*scatAmpInterp[O][iq]);
      }
    }
    Xfits[ib] = X;
  }

  Eigen::MatrixXd inp;
  for (ib=0; ib<(int)Xfits.size(); ib++) {
    inp = Xfits[ib].transpose()*Xfits[ib];
    XNorms[ib] = tools::SVDinvert(inp);  
  }

  std::vector< std::vector<float> > coeffs(NtimeSteps);
  std::vector< std::vector<float> > positions(NtimeSteps);
  for (it=0; it<NtimeSteps; it++) {
    positions[it].resize(3*n2oMC.NmolAtoms, 0);
    coeffs[it].resize(1+Nbessels*Nradii, 0);
  }

  ////////////////////////////////
  /////  Running Simulation  /////
  ////////////////////////////////

  int tShift = 1;
  if (makeStatic) {
    tShift = 0;
    Nsimulations = 1;
    NtimeSteps = 1;
  }

  for (int ism=0; ism<Nsimulations; ism++) {

    runID = to_string(std::time(0) - 1499887450) 
            + to_string(n2oMC.samplePDF("uniform"));

    tStep = 0.1;
    w1 = 2*PI/(6 + 4*(0.5 - n2oMC.samplePDF("uniform")));
    w2 = 2*PI/(6 + 4*(0.5 - n2oMC.samplePDF("uniform")));
    w3 = 2*PI/(6 + 4*(0.5 - n2oMC.samplePDF("uniform")));
    wy = 2*PI/(6.5 + 4*(0.5 - n2oMC.samplePDF("uniform")));
    wx = 2*PI/(5.5 + 4*(0.5 - n2oMC.samplePDF("uniform")));

    //////  Simulate Time Dynamics  /////
    for (it=0; it<NtimeSteps; it++) {

      // Dynamics are set by changing coefficients of normal modes
      // if itm=0 then all coefficients should be 0
      itm = (it + tShift)*tStep;
      n1 = n1fxn(itm, 0);
      n2 = n2fxn(itm, 0);
      n3 = n3fxn(itm, 0);

      c1 = 1.5*sin(w1*itm);
      c2 = 1.5*sin(w2*itm);
      c3 = 1.5*sin(w3*itm);
      inpVals["z1"] = c1*n1[0] + c2*n2[0] + c3*n3[0];
      inpVals["z2"] = c1*n1[1] + c2*n2[1] + c3*n3[1];
      inpVals["z3"] = c1*n1[2] + c2*n2[2] + c3*n3[2];

      inpVals["y"] = sin(wy*it);
      inpVals["x"] = sin(wx*it);

      //cout<<"it: "<<it<<endl;
      //cout<<inpVals["z1"]<<"  "<<inpVals["z1"]<<"  "<<inpVals["z1"]<<endl;
      //cout<<inpVals["x"]<<"  "<<inpVals["y"]<<endl;
      //cout<<endl;

      n2oMC.reset();
      n2oMC.makeMolEnsemble(inpVals);
    

      DIFFRACTIONclass diffP(&n2oMC, qMax, Iebeam, screenDist, elEnergy, bins,
          "/reg/neh/home5/khegazy/simulations/scatteringAmplitudes/3.7MeV/");

      curLineOuts = diffP.lineOut_uniform();

      if (bins%2) {
        Y0(0) = curLineOuts[1][0];
        lineOuts[it][0] = curLineOuts[1][0];
        qShift = 1;
      }
      else {
        qShift = 0;
      }

      for (uint i=qShift; i<curLineOuts[1].size(); i++) {
        Y0(i) = curLineOuts[1][i];
        Y(i-qShift) = curLineOuts[1][i];
        lineOuts[it][i] = curLineOuts[1][i];
      }

      // Isotropy projection
      coeffs[it][0] = Y0.sum()*scale;

      // Projection onto bessel Functions
      for (ib=0; ib<Nbessels; ib++) {
        if (ib == 0) {
          weights = XNorms[0]*Xfits[0].transpose()*Y0;
        }
        else {
          weights = XNorms[ib]*Xfits[ib].transpose()*Y;
        }

        // Scale weights
        weights *= scale; //1e34;

        for (ir=0; ir<Nradii; ir++) {
          coeffs[it][ib*Nradii+ir+1] = weights(ir);
        }

        if (plotVerbose) {
          std::vector<double> plot(Nradii);
          for (ir=0; ir<Nradii; ir++) {
            plot[ir] = weights(ir);
          }
          plt.print1d(plot, "test" + to_string(ib) + "_" + to_string(itm));
        }
      }

      for (int ia=0; ia<n2oMC.NmolAtoms; ia++) {
        for (int ip=0; ip<3; ip++) {
          positions[it][ia*3+ip] = n2oMC.molecules->atoms[ia]->location(ip);
        }
      }

      /*
      FILE* otpDp = fopen((outFilePrefix + "diffractionPattern_" 
            + outFileSuffix + ".dat").c_str(), "wb");
      //FILE* otpAp = fopen((outFilePrefix + "atmDiffractionPattern_" 
      //      + outFileSuffix + ".dat").c_str(), "wb");
      FILE* otpMp = fopen((outFilePrefix + "molDiffractionPattern_" 
            + outFileSuffix + ".dat").c_str(), "wb");
      //FILE* otpSp = fopen((outFilePrefix + "sPattern_" 
      //      + outFileSuffix + ".dat").c_str(), "wb");
    
      fwrite(&curLineOuts[0][0], sizeof(double), curLineOuts[0].size(), otpDp);
      fwrite(&curLineOuts[1][0], sizeof(double), curLineOuts[1].size(), otpMp);

      fclose(otpDp);
      //fclose(otpAp);
      fclose(otpMp);
      //fclose(otpSp);
*/
    }

    std::string outFileName;
    // Name of Bessel data file
    if (makeStatic) {
      outFileName = "output/" + fileName 
          + "_Static_BESSELPROJ_Nbessels-" 
          + to_string(Nbessels) + "_iso-1_Nradii-"
          + to_string(Nradii) + ".dat";
    }
    else {
      outFileName = outFilePrefix + "_" + runID + "_BESSELPROJ_Ntime-"
          + to_string(NtimeSteps) + "_Nbessels-"
          + to_string(Nbessels) + "_iso-1_Nradii-"
          + to_string(Nradii);
      if (index) {
        outFileName += "_Job-" + to_string(index);
      }
      outFileName += ".dat";
    }
    // Bessel data file
    FILE* outBesselFile = fopen(outFileName.c_str(), "wb");

    // name of position data file
    if (makeStatic) {
      outFileName = "output/" + fileName 
          + "_Static_ATOMPOS_Natoms-" + to_string(n2oMC.NmolAtoms) 
          + "_Pos-3_Nradii-" + to_string(Nradii) + ".dat";
    }
    else {
      outFileName = outFilePrefix + "_" + runID + "_ATOMPOS_Ntime-"
          + to_string(NtimeSteps) + "_Natoms-"
          + to_string(n2oMC.NmolAtoms) 
          + "_Pos-3_Nradii" + to_string(Nradii);
      if (index) {
        outFileName += "_Job-" + to_string(index);
      }
      outFileName += ".dat";
    }
    // Position data file
    FILE* outAtomPosFile = fopen(outFileName.c_str(), "wb");
 
    // Write data files
    for (it=0; it<NtimeSteps; it++) {
      fwrite(&coeffs[it][0], sizeof(float), coeffs[it].size(), outBesselFile);
      fwrite(&positions[it][0], sizeof(float), positions[it].size(), outAtomPosFile);
      fwrite(&lineOuts[it][0], sizeof(float), lineOuts[it].size(), outAtomPosFile);
    }
    fclose(outBesselFile);
    fclose(outAtomPosFile);

  }

  return 1;
}

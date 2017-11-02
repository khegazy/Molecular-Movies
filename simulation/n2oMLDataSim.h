#include "/reg/neh/home/khegazy/baseScripts/atomClass.h"
#include "/reg/neh/home/khegazy/baseScripts/moleculeClass.h"
#include "/reg/neh/home/khegazy/baseScripts/molEnsembleMC.h"
#include "/reg/neh/home/khegazy/baseScripts/diffractionClass.h"
#include "/reg/neh/home/khegazy/baseScripts/plotClass.h"
#include "/reg/neh/home/khegazy/baseScripts/imageProcessing.h"
#include "/reg/neh/home/khegazy/baseScripts/saveClass.h"


class N2OMCclass : public MOLENSEMBLEMCclass {

    public:
        N2OMCclass(long int seed, string pdfPath, string pdfNames) : MOLENSEMBLEMCclass(seed, pdfPath, pdfNames) {}
        N2OMCclass(long int seed) : MOLENSEMBLEMCclass(seed) {}

        void buildMolecule(MOLECULEclass &molecule, 
            std::map<std::string, double> inpVals) {

	  TVector3 position;
          position.SetXYZ(3*inpVals["x"], 2*inpVals["y"], 1.1257 + inpVals["z1"]);
          molecule.addAtom(new ATOMclass("N1", N, 2*angs_to_au*position));

	  position.SetXYZ(-6*inpVals["x"], -4*inpVals["y"], inpVals["z2"]);
          molecule.addAtom(new ATOMclass("N2", N, 2*angs_to_au*position));

	  position.SetXYZ(3*inpVals["x"], 2*inpVals["y"], -1.1863 + inpVals["z3"]);
          molecule.addAtom(new ATOMclass("O1", O, 2*angs_to_au*position));
	}
};


std::vector<double> n1fxn(double tm, double phi) {

  double tmDep = cos(sqrt(2)*sqrt(1 - 1/sqrt(2))*tm + phi);
  std::vector<double> results(3);
  results[0] = tmDep/2;
  results[1] = tmDep/sqrt(2);
  results[2] = tmDep/2;

  return results;
}

std::vector<double> n2fxn(double tm, double phi) {

  double tmDep = cos(sqrt(2)*tm + phi);
  std::vector<double> results(3);
  results[0] = -tmDep/sqrt(2);
  results[1] = 0;
  results[2] = tmDep/sqrt(2);
  
  return results;
}

std::vector<double> n3fxn(double tm, double phi) {

  double tmDep = cos(sqrt(2)*sqrt(1 + 1/sqrt(2))*tm + phi);
  std::vector<double> results(3);
  results[0] = tmDep/2;
  results[1] = tmDep/sqrt(2);
  results[2] = tmDep/2;

  return results;
}

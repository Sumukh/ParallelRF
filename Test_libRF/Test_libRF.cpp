/*  This file is part of libRandomForest - http://www.alexander-schwing.de/
 *
 *  libRandomForest is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  libRandomForest is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with libRandomForest.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Copyright (C) 2009-2012  Alexander Schwing  [aschwing _at_ inf _dot_ ethz _dot_ ch]
 */

#include <iostream>
 #include <string.h>
#include <cmath>
#include <algorithm>
#include <sys/time.h>
#include <getopt.h>
#include <omp.h>

#include "../libRF/FeaturesTable.h"
#include "../libRF/ClassifierRF.h"


double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(int argc, char** argv) {
	std::cout << "Test..." << std::endl;

	std::string fp;
#ifdef USE_ON_WINDOWS
	fp = "..\\Dataset\\Ionosphere";
#else
	fp = "Dataset/Mnist_full";
#endif
	
	size_t numTrees = 16;
	int numThreads = 16 ; // numThreads to use for openMP 
	int c;
	std::string datasetName = "Mnist_full";
	std::string validation = "FiveFold";
	/* Read options of command line */
	while((c = getopt(argc, argv, "n:t:d:v:"))!=-1)
	  {
	    switch(c)
	      {
	      case 'n':
		numTrees = (size_t) atoi(optarg);
		break;
	      case 't':
		numThreads = atoi(optarg);
		break;
	      case 'd':
		datasetName = optarg;
#ifdef USE_ON_WINDOWS
		fp = "..\\Dataset\\" + datasetName;
#else
		fp = "Dataset/" + datasetName;
#endif
		break;
	      case 'v':
		if (std::string(optarg).compare("LeaveOneOut") == 0) {
		  validation = optarg;
		}
		break;
	      }
	  }
	// if(argc > 2 && strncmp(argv[1], "-n",2)==0)
	// {
	// 	rp.numTrees = atoi(argv[2]);
	// }

  	std::cout << "NumTrees..." << numTrees << std::endl;
	std::cout << "NumThreads..." << numThreads << std::endl;
	std::cout << "Dataset..." << datasetName << std::endl;
	std::cout << "Validation..." << validation << std::endl;
	FeaturesTable* ft = new FeaturesTable(fp);
	//ClassifierRF* RF1 = new ClassifierRF(numTrees, ft);
	size_t NumSamples = ft->NumSamples();
	size_t NumClasses = ft->NumClasses();
	//std::cout << NumClasses << std::endl;

	// Should add option to pass in the num threads as parameter.
	// omp_set_num_threads(16);

	double t0;
	size_t error = 0;
	if (validation.compare("FiveFold") == 0) {
	  // Select 20% of data for test dataset.
	  // Hacky, but deterministic which is good for testing.
	  std::vector<size_t> test;
	  for(size_t i=0; i<NumSamples; i+=5) {
	    test.push_back(i);
	  }

	  // Create random forest model.
	  ClassifierRF* RF = new ClassifierRF(numTrees, numThreads, ft);

	  // Remove test dataset so it is not used in training.
	  ft->RemoveSampleWithID(test);

	  // Time training (including cross validation) of random forest model.
	  t0 = timestamp();
	
	  // Learn (train) random forest model.
	  RF->Learn();

	  // Return test data to model for use in cross-validation (I think).
	  ft->ResetRemovedIDs();

	  omp_set_num_threads(numThreads);
	  // Compute error of model on test data.
	  // Tried to use OpenMP to parallelize the classification but ended up being over a second slower with 16 threads than without OpenMP.
	  // #pragma omp parallel for reduction(+:error)
	  for (size_t i=0; i<NumSamples; i++) {
	    std::vector<size_t> tmp(1,i);
	    double* distri = new double[NumClasses];
	    std::fill(distri, distri+NumClasses, 0.0);
	    RF->Classify(i, distri, NumClasses);
	    std::vector<size_t> trueCls;
	    ft->GetTrueClass(&trueCls, tmp);
	    // Compare predicted class of test data from model to actual class.
	    if (size_t(std::max_element(distri, distri + NumClasses) - distri)!=trueCls[0]) {
	      ++error;
	    }
	    delete[] distri;
	  }
	  t0 = timestamp() - t0;
	} else if (validation.compare("LeaveOneOut") == 0) {
	    t0 = timestamp();
	    // #pragma omp parallel for
	    for(size_t k=0;k<NumSamples;++k) {
	      ClassifierRF* RF = new ClassifierRF(numTrees,numThreads, ft);
	      // std::cout << k << "/" << NumSamples << std::endl;
	      std::vector<size_t> tmp(1,k);
	      ft->RemoveSampleWithID(tmp);
	      RF->Learn();
	      ft->ResetRemovedIDs();
	      //std::vector<double> distri(NumClasses,0.0);
	      double* distri = new double[NumClasses];
	      std::fill(distri, distri+NumClasses, 0.0);
	      RF->Classify(k,distri,NumClasses);
	      std::vector<size_t> trueCls;
	      ft->GetTrueClass(&trueCls, tmp);
	      RF->ClearCLF(); 
	      // if(size_t(std::max_element(distri.begin(), distri.end())-distri.begin())!=trueCls[0]) {
	      // 	#pragma omp critical
	      // 	++error;
	      // }
	      if(size_t(std::max_element(distri, distri + NumClasses) - distri)!=trueCls[0]) { 
		//#pragma omp critical
		++error;
	      }
	      delete [] distri;
	    }
	    t0 = timestamp() - t0;
	  }

	  std::cout << "" << t0 << " seconds elapsed" << std::endl;

	  std::cout << "Error: " << double(error)/NumSamples << std::endl;

	  return 0;
}

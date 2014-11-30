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
	fp = "Dataset/Ionosphere";
#endif
	
	size_t numTrees = 10;
	int numThreads = 16 ; // numThreads to use for openMP 
	int c;
	std::string datasetName = "Ionosphere";
	/* Read options of command line */
	while((c = getopt(argc, argv, "n:t:d:"))!=-1)
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
			}
	    }
	// if(argc > 2 && strncmp(argv[1], "-n",2)==0)
	// {
	// 	rp.numTrees = atoi(argv[2]);
	// }

  	std::cout << "NumTrees..." << numTrees << std::endl;
    std::cout << "NumThreads..." << numThreads << std::endl;
    std::cout << "Dataset..." << datasetName << std::endl;
    FeaturesTable* ft = new FeaturesTable(fp);
	//ClassifierRF* RF1 = new ClassifierRF(numTrees, ft);
	size_t NumSamples = ft->NumSamples();
	size_t NumClasses = ft->NumClasses();

	// Should add option to pass in the num threads as parameter.
	// omp_set_num_threads(16);

	size_t error = 0;
	double t0 = timestamp();
	// #pragma omp parallel for
	for(size_t k=0;k<NumSamples;++k) {
		ClassifierRF* RF = new ClassifierRF(numTrees, ft);
		// std::cout << k << "/" << NumSamples << std::endl;
		std::vector<size_t> tmp(1,k);
		ft->RemoveSampleWithID(tmp);
		RF->Learn();
		ft->ResetRemovedIDs();
		std::vector<double> distri(NumClasses,0.0);
		RF->Classify(k,distri);
		std::vector<size_t> trueCls;
		ft->GetTrueClass(&trueCls, tmp);
		RF->ClearCLF();
		if(size_t(std::max_element(distri.begin(), distri.end())-distri.begin())!=trueCls[0]) {
			#pragma omp critical
			++error;
		}
	}
	t0 = timestamp() - t0;
	std::cout << "" << t0 << " seconds elapsed" << std::endl;

	std::cout << "Error: " << double(error)/NumSamples << std::endl;

	return 0;
}
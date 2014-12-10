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
#include <cmath>
#include <algorithm>
#include <sys/time.h>

#include <omp.h>

#include "../libRF/FeaturesTable.h"
#include "../libRF/ClassifierRF.h"

typedef double NUM_TYPE;
typedef FeaturesTable<NUM_TYPE> feature_type;
typedef ClassifierRF<NUM_TYPE> classifier_type;

double timestamp()
{
  struct timeval tv;
  gettimeofday (&tv, 0);
  return tv.tv_sec + 1e-6*tv.tv_usec;
}

int main(int argc, char** argv) {
	// std::cout << "Test..." << std::endl;

	feature_type::SpecialFeatureParams fp;
#ifdef USE_ON_WINDOWS
	fp.Folder = "..\\Dataset\\Ionosphere";
#else
	fp.Folder = "Dataset/Mnist_full";
#endif

	classifier_type::SpecialParams rp;
	rp.numTrees = 32;
	// int numThreads = atoi(argv[1]);

	Classifier<NUM_TYPE,classifier_type,feature_type> RF(&rp, &fp);

	size_t NumSamples = RF.NumSamples();
	size_t NumClasses = RF.NumClasses();

	// Should add option to pass in the num threads as parameter.
	// omp_set_num_threads(16);

	std::vector<size_t> tmp;
	for(size_t i=0;i<NumSamples;i+=5) {
		// Classifier<NUM_TYPE,classifier_type,feature_type> RF(&rp, &fp);
		tmp.push_back(i);
	}
	// RF.RemoveSampleWithID(tmp);

	std::cout << "Threads,Seconds,Error,Actual Threads" << std::endl;

	for (int t=16; t>=15; t-=5) {
		RF.RemoveSampleWithID(tmp);
		size_t error = 0;
		double t0 = timestamp();
		// #pragma omp parallel for
		// for(size_t k=0;k<NumSamples;++k) {
		// Classifier<NUM_TYPE,classifier_type,feature_type> RF(&rp, &fp);
		// std::cout << k << "/" << NumSamples << std::endl;
		int actual_threads = RF.Learn(t);
		RF.ResetRemovedIDs();
		for (size_t j=0; j<NumSamples; j+= 5) {
			std::vector<double> distri(NumClasses,0.0);
			std::vector<size_t> tmp2(1,j);
			RF.Classify(j,distri);
			std::vector<size_t> trueCls;
			RF.GetTrueClass(&trueCls, tmp2);
			// RF.ClearCLF();
			if(size_t(std::max_element(distri.begin(), distri.end())-distri.begin())!=trueCls[0]) {
				// #pragma omp critical
				++error;
			}
		}
		// }
		t0 = timestamp() - t0;
		RF.ClearCLF();

		// std::cout << "Threads,Seconds,Error,Actual_threads" << std::endl;
		std::cout << t << "," << t0 << "," << double(error)/(NumSamples/5) << ',' << actual_threads << std::endl;
		std::cout << error << std::endl;
		// std::cout << "Error: " << double(error)/(NumSamples/5) << std::endl;
	}
	return 0;
}
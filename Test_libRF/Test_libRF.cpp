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

#include "../libRF/FeaturesTable.h"
#include "../libRF/ClassifierRF.h"

typedef double NUM_TYPE;
typedef FeaturesTable<NUM_TYPE> feature_type;
typedef ClassifierRF<NUM_TYPE> classifier_type;

int main(int argc, char** argv) {
	std::cout << "Test..." << std::endl;

	feature_type::SpecialFeatureParams fp;
#ifdef USE_ON_WINDOWS
	fp.Folder = "..\\Dataset\\Ionosphere";
#else
	fp.Folder = "Dataset/Ionosphere";
#endif

	classifier_type::SpecialParams rp;
	rp.numTrees = 10;

	Classifier<NUM_TYPE,classifier_type,feature_type> RF(&rp, &fp);

	size_t NumSamples = RF.NumSamples();
	size_t NumClasses = RF.NumClasses();

	size_t error = 0;
	for(size_t k=0;k<NumSamples;++k) {
		std::cout << k << "/" << NumSamples << std::endl;
		std::vector<size_t> tmp(1,k);
		RF.RemoveSampleWithID(tmp);
		RF.Learn();
		RF.ResetRemovedIDs();
		std::vector<double> distri(NumClasses,0.0);
		RF.Classify(k,distri);
		std::vector<size_t> trueCls;
		RF.GetTrueClass(&trueCls, tmp);
		RF.ClearCLF();
		if(size_t(std::max_element(distri.begin(), distri.end())-distri.begin())!=trueCls[0]) {
			++error;
		}
	}

	std::cout << "Error: " << double(error)/NumSamples << std::endl;

	return 0;
}
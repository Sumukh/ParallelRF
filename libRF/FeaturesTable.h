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

#ifndef __FEATURESTABLE_H__
#define __FEATURESTABLE_H__

#include <string>
#include <map>
 #include <vector>

//#include "Features.h"

class FeaturesTable{
public:
	std::string Folder;
private:
	std::vector<std::vector<double>*> FlatData;

	std::vector<size_t> ClassDistribution;
	std::vector<size_t> CumSamplesPerClass;

	std::vector<size_t> ValidClassDistribution;
	std::vector<size_t> ValidCumSamplesPerClass;

	bool removed;
	std::map<size_t,size_t> ValidDataIDXToLine;

	int LoadDataSet();
	void TraverseDirectory(const std::string& path, std::string& pattern, bool subdirectories, std::vector<std::string>& fileNames);
	size_t convertStr(std::vector<double>& L, std::string& seq, std::string& _1cdelim, bool _removews );
public:
	FeaturesTable() : Folder(NULL){};
	FeaturesTable(std::string fp);
	~FeaturesTable();

	int ClearFeat();

	size_t NumSamples();
	size_t NumFeatures();
	size_t NumClasses();
	const std::vector<size_t>* GetClassDistribution();
	int GetClassDistribution(double* dist, std::vector<size_t>* cls, std::vector<size_t>& dataIdx);
	int GetTrueClass(std::vector<size_t> *cls, std::vector<size_t> &dataIdx);
	double FeatureResponse(size_t dataIdx, size_t featureId);

	int RemoveSampleWithID(std::vector<size_t>& ids);
	int ResetRemovedIDs();
};

#endif
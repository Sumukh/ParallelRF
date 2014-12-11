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

#include "FeaturesTable.h"

#include <iostream>
#include <fstream>
#include <string>
#include <assert.h>
#include <stdlib.h>

#ifdef USE_ON_WINDOWS
#include <io.h>

void FeaturesTable::TraverseDirectory(const std::string& path, std::string& pattern, bool subdirectories, std::vector<std::string>& fileNames) {
	struct _finddatai64_t data;
	std::string fname = path + "\\" + pattern;
	// start the finder -- on error _findfirsti64() will return -1, otherwise if no
	// error it returns a handle greater than -1.
	intptr_t h = _findfirsti64(fname.c_str(),&data);
	if(h >= 0) {
		do {
			if( (data.attrib & _A_SUBDIR) ) {
				if( subdirectories && strcmp(data.name,".") != 0 && strcmp(data.name,"..") != 0) {
					fname = path + "\\" + data.name;
					TraverseDirectory(fname,pattern,true, fileNames);
				}
			} else {
				fileNames.push_back(path + "\\" + data.name);
			}
		} while( _findnexti64(h,&data) == 0);

		_findclose(h);
	}
}
#else
#include <dirent.h>
#include <fnmatch.h>

void FeaturesTable::TraverseDirectory(const std::string& path, std::string& pattern, bool subdirectories, std::vector<std::string>& fileNames) {
	DIR *dir, *tstdp;
    struct dirent *dp;

    //open the directory
    if((dir  = opendir(path.c_str())) == NULL)
    {
		std::cout << "Error opening " << path << std::endl;
        return;
    }

    while ((dp = readdir(dir)) != NULL)
    {
        tstdp=opendir(dp->d_name);
		
		if(tstdp) {
			closedir(tstdp);
			if(subdirectories) {
				//TraverseDirectory(
			}
		} else {
			if(fnmatch(pattern.c_str(), dp->d_name, 0)==0) {
				std::string tmp(path);
				tmp.append("/").append(dp->d_name);
				fileNames.push_back(tmp);
				//std::cout << fileNames.back() << std::endl;
			}
		}
    }

    closedir(dir);
    return;

}
#endif

FeaturesTable::FeaturesTable(std::string f) {
	Folder = f;
	LoadDataSet();
	removed = false;
}


FeaturesTable::~FeaturesTable() {
	ClearFeat();
	delete [] ClassDistribution;
}


size_t FeaturesTable::NumFeatures() {
	return FlatData[0]->size();
}

size_t FeaturesTable::NumSamples() {
	return ValidDataIDXToLine.size();
}


size_t FeaturesTable::NumClasses() {
	return ClassDistributionSize;
}

int FeaturesTable::ClearFeat() {
	for(typename std::vector<std::vector<double>*>::iterator f=FlatData.begin(), f_e=FlatData.end();f!=f_e;++f) {
		(*f)->clear();
		delete *f;
	}
	FlatData.clear();
	return 0;
}

double FeaturesTable::FeatureResponse(size_t dataIdx, size_t featureId) {
	if(removed) {
		size_t tmp = ValidDataIDXToLine[dataIdx];
		return FlatData[tmp]->at(featureId);
	} else {
		return FlatData[dataIdx]->at(featureId);
	}
}


int FeaturesTable::GetClassDistribution(double* dist, std::vector<size_t>* cls, std::vector<size_t>& dataIdx) {
	size_t curPos = 0;
	if(cls!=NULL) {
		cls->assign(dataIdx.size(), 0);
	}
	for(size_t k=1;k<ValidCumSamplesPerClass.size();++k) {
		size_t numSamples = 0;
		while(curPos<dataIdx.size() && dataIdx[curPos]<ValidCumSamplesPerClass[k]) {
			++numSamples;
			if(cls!=NULL) {
				cls->at(curPos) = k-1;
			}
			++curPos;
		}
		if(dist!=NULL) {
			dist[k-1] = double(numSamples);
		}
	}
	return 0;
}

size_t FeaturesTable::GetClassDistributionSize()
{
	return ClassDistributionSize;
}
size_t FeaturesTable::GetValidClassDistributionSize()
{
	return ValidClassDistributionSize;
}

int FeaturesTable::LoadDataSet() {
	std::vector<std::string> fNames;
	std::string pattern("Class*");
	TraverseDirectory(Folder, pattern, false, fNames);

	size_t numFeatures = size_t(-1);
	CumSamplesPerClass.assign(fNames.size()+1,0);
	//ClassDistribution.assign(fNames.size(), 0);
	ClassDistributionSize = fNames.size();
	ClassDistribution = new size_t[ClassDistributionSize];
	std::fill(ClassDistribution, ClassDistribution + ClassDistributionSize, 0);
	std::string delimStr = "\t";
	for(size_t k=0;k<fNames.size();++k) {
		// std::cout << fNames[k] << std::endl;
		std::ifstream ifs(fNames[k].c_str(), std::ios_base::in);
		std::string line;
		while(!ifs.eof()) {
			line.clear();
			std::getline(ifs, line, '\n');
			std::vector<double>* L = new std::vector<double>;
			size_t tmpNumFeatures = convertStr(*L,line, delimStr, false);
			if(numFeatures==size_t(-1)) {
				numFeatures = tmpNumFeatures;
			} else {
				if(numFeatures!=tmpNumFeatures) {
					continue;
				}
			}
			size_t numEl = FlatData.size();
			FlatData.push_back(L);
			ValidDataIDXToLine.insert(std::make_pair<size_t,size_t>(numEl,numEl));
		}
		ifs.close();
		CumSamplesPerClass[k+1] = FlatData.size();
		ClassDistribution[k] = CumSamplesPerClass[k+1]-CumSamplesPerClass[k];
	}
	ValidClassDistribution = ClassDistribution;
	ValidClassDistributionSize = ClassDistributionSize;
	ValidCumSamplesPerClass = CumSamplesPerClass;
	return 0;
}


const size_t* FeaturesTable::GetClassDistribution() {
	return ValidClassDistribution;
}

int FeaturesTable::RemoveSampleWithID(std::vector<size_t>& ids) {
	if(removed) {
		return -1;
	}
	double* removeDist = new double[NumClasses()];
	GetClassDistribution(removeDist,NULL,ids);

	for(size_t k=0;k<ValidClassDistributionSize;++k) {
		ValidClassDistribution[k] -= size_t(removeDist[k]);
	}
	double cumsum = 0;
	for(size_t k=0;k<ValidCumSamplesPerClass.size();++k) {
		ValidCumSamplesPerClass[k] -= size_t(cumsum);
		cumsum += removeDist[k];
	}
	delete [] removeDist;

	for(std::vector<size_t>::iterator v=ids.begin(),v_e=ids.end();v!=v_e;++v) {
		ValidDataIDXToLine.erase(ValidDataIDXToLine[*v]);
	}
	size_t cnt = 0;
	std::map<size_t,size_t> NewIDXMap;
	for(std::map<size_t,size_t>::iterator v=ValidDataIDXToLine.begin(),v_e=ValidDataIDXToLine.end();v!=v_e;++v) {
		NewIDXMap.insert(std::make_pair<size_t,size_t>(cnt++,v->second));
	}
	ValidDataIDXToLine.clear();
	ValidDataIDXToLine = NewIDXMap;
	removed = true;
	return 0;
}

int FeaturesTable::ResetRemovedIDs() {
	std::map<size_t,size_t> NewIDXMap;
	for(size_t k=0;k<FlatData.size();++k) {
		NewIDXMap.insert(std::make_pair<size_t,size_t>(k,k));
	}
	ValidDataIDXToLine.clear();
	ValidDataIDXToLine = NewIDXMap;

	ValidClassDistribution = ClassDistribution;
	ValidCumSamplesPerClass = CumSamplesPerClass;

	removed = false;
	return 0;
}


int FeaturesTable::GetTrueClass(std::vector<size_t> *cls, std::vector<size_t> &dataIdx) {
	return GetClassDistribution(NULL,cls,dataIdx);
}


size_t FeaturesTable::convertStr(std::vector<double>& L, std::string& seq, std::string& _1cdelim, bool _removews ) {
    typedef std::string::size_type ST;
    std::string delims = _1cdelim;
    std::string STR;
    if(delims.empty()) delims = "\n\r";
    if(_removews) delims += " ";

    ST pos=0, LEN = seq.size();
    while(pos < LEN ){
        STR=""; // Init/clear the STR token buffer
        // remove any delimiters including optional (white)spaces
        while( (delims.find(seq[pos]) != std::string::npos) && (pos < LEN) ) ++pos;
        // leave if @eos
        if(pos==LEN) return L.size();
        // Save token data
        while( (delims.find(seq[pos]) == std::string::npos) && (pos < LEN) ) STR += seq[pos++];
        // put valid STR buffer into the supplied list
        //std::cout << "[" << STR << "]";
        if( ! STR.empty() ) L.push_back(atof(STR.c_str()));
    }
    return L.size();
}

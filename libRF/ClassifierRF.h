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

#include <vector>
#include "FeaturesTable.h"

class RFNode;
class ClassifierRF{
public:
	size_t numTrees;
	FeaturesTable* feat;	
	
private:
	RFNode** RFHeadNodes;
	double randBetween(double From, double To, size_t resolution);
	int WeightedSampling(const std::vector<size_t>* SamplesPerClass, std::vector<std::vector<double> >& DataWeights, std::vector<size_t>& oobIdx, std::vector<size_t>& ibIdx, std::vector<size_t>& ibRep);
	int ConstructTree(RFNode* head, std::vector<size_t>& dataIdx, std::vector<size_t>& cls, std::vector<double>& wAttr, size_t AttributesToSample);
	bool stoppingCriteria(RFNode* node);
	int whichAttributes(std::vector<double>& wAttr, size_t AttributesToSample, std::vector<int>& selAttr);
	int ImpuritySplit(std::vector<size_t>& dataIdx, std::vector<size_t>& cls, size_t featureId, double* splitVal, double* bestEstimation);
	double GiniImpurity(size_t weight, size_t* noClassAttrVal, size_t valIdx, size_t noClasses);
	double GiniImpurityGain(double priorImp, size_t weight, size_t* noAttrVal, size_t* noClassAttrVal, size_t noClasses);

	int ClassifyTree(RFNode* node, size_t dataIdx, std::vector<double>& distri);
	int ClearNode(RFNode* v);

	inline double sqr(double x);
public:
	ClassifierRF() : RFHeadNodes(NULL){}; 
	ClassifierRF(size_t num, FeaturesTable* feat);
	~ClassifierRF();
	
	int Learn();
	int Classify(size_t dataIdx, std::vector<double>& distri);
	int ClearCLF();
};

class RFNode{
public:
		RFNode* NodeSmaller;
		RFNode* NodeLarger;
		size_t featID;
		double splitVal;
		double* dist;
		RFNode() : NodeSmaller(NULL), NodeLarger(NULL), featID(size_t(-1)), splitVal(0.0), dist(NULL) {};
};
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

#ifndef __CLASSIFIERRF_H__
#define __CLASSIFIERRF_H__

#include <vector>

#include "Classifier.h"

template <class T>
class ClassifierRF : public ClassifierGeneral<T> {
public:
	struct SpecialParams : ClassifierGeneral<T>::parameter_type {
		size_t numTrees;
	};
private:
	SpecialParams* params;

	struct RFNode {
		RFNode* NodeSmaller;
		RFNode* NodeLarger;
		size_t featID;
		T splitVal;
		T* dist;
		RFNode() : NodeSmaller(NULL), NodeLarger(NULL), featID(size_t(-1)), splitVal(T(0.0)), dist(NULL) {};
	};
	std::vector<struct RFNode*> RFHeadNodes;

	double randBetween(double From, double To, size_t resolution);
	int WeightedSampling(const std::vector<size_t>* SamplesPerClass, std::vector<std::vector<double> >& DataWeights, std::vector<size_t>& oobIdx, std::vector<size_t>& ibIdx, std::vector<size_t>& ibRep);
	int ConstructTree(struct RFNode* head, std::vector<size_t>& dataIdx, std::vector<size_t>& cls, std::vector<double>& wAttr, size_t AttributesToSample);
	bool stoppingCriteria(struct RFNode* node);
	int whichAttributes(std::vector<double>& wAttr, size_t AttributesToSample, std::vector<int>& selAttr);
	int ImpuritySplit(std::vector<size_t>& dataIdx, std::vector<size_t>& cls, size_t featureId, T* splitVal, double* bestEstimation);
	double GiniImpurity(size_t weight, size_t* noClassAttrVal, size_t valIdx, size_t noClasses);
	double GiniImpurityGain(double priorImp, size_t weight, size_t* noAttrVal, size_t* noClassAttrVal, size_t noClasses);

	int ClassifyTree(struct RFNode* node, size_t dataIdx, std::vector<T>& distri);
	int ClearNode(struct RFNode* v);

	inline double sqr(double x);
public:
	ClassifierRF(typename ClassifierGeneral<T>::parameter_type* rp);
	~ClassifierRF();
	
	int Learn();
	int Classify(size_t dataIdx, std::vector<T>& distri);
	int ClearCLF();
};

#endif
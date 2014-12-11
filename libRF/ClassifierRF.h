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
#include <xmmintrin.h>

class RFNode;
class ClassifierRF{
public:
	size_t numTrees;
	size_t numThreads;	
	FeaturesTable* feat;	
private:
	RFNode** RFHeadNodes;
	double randBetween(double From, double To, size_t resolution);
	int WeightedSampling(const std::vector<size_t>* SamplesPerClass, std::vector<std::vector<double> >& DataWeights, std::vector<size_t>& oobIdx, std::vector<size_t>& ibIdx, std::vector<size_t>& ibRep);
	int ConstructTree(RFNode* head, std::vector<size_t>& dataIdx, std::vector<size_t>& cls, double* wAttr, size_t wAttrSize, size_t AttributesToSample);
	bool stoppingCriteria(RFNode* node);
	int whichAttributes(double* wAttr, size_t wAttrSize, size_t AttributesToSample, std::vector<int>& selAttr);
	int ImpuritySplit(std::vector<size_t>& dataIdx, std::vector<size_t>& cls, size_t featureId, double* splitVal, double* bestEstimation);
	double GiniImpurity(size_t weight, size_t* noClassAttrVal, size_t valIdx, size_t noClasses);
	double GiniImpurityGain(double priorImp, size_t weight, size_t* noAttrVal, size_t* noClassAttrVal, size_t noClasses);

	int ClassifyTree(RFNode* node, size_t dataIdx, double* distri, size_t distri_size);
	int ClearNode(RFNode* v);

	__m128 scan_SSE(__m128 x);
	float pass1_SSE(float *a, float *s, const int n);
	void pass2_SSE(float *s, __m128 offset, const int n);
	void scan_omp_SSEp2_SSEp1_chunk(float a[], float s[], int n);

	inline double sqr(double x);
public:
	ClassifierRF() : RFHeadNodes(NULL){}; 
	ClassifierRF(size_t num, size_t numT, FeaturesTable* feat);
	~ClassifierRF();
	
	int Learn();
	int Classify(size_t dataIdx, double* distri, size_t distri_size);
	int ClearCLF();
};

class RFNode{
public:
		size_t featID;
		double splitVal;
		RFNode* NodeSmaller;
		RFNode* NodeLarger;
		double* dist;
		RFNode() :featID(size_t(-1)), splitVal(0.0),NodeSmaller(NULL), NodeLarger(NULL), dist(NULL) {};
};
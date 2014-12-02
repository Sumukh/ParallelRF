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
#include <numeric>
#include <functional>
#include <map>
#include <limits>

#include <string.h>
// #include <omp.h>

#include "ClassifierRF.h"
#include "Features.h"

#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif

#define LOG2(x) std::log(x)/0.693147180559945
#define MY_MAX(a,b) (((a)>=(b)) ? (a) : (b))

template <class T>
ClassifierRF<T>::ClassifierRF(typename ClassifierGeneral<T>::parameter_type* clfp) {
	params = static_cast<SpecialParams*>(clfp);
}

template <class T>
ClassifierRF<T>::~ClassifierRF() {
	if(ClassifierGeneral<T>::feat!=NULL) {
		ClassifierGeneral<T>::feat->ClearFeat();
	}
	ClearCLF();
}

template <class T>
int ClassifierRF<T>::ClearCLF() {
	for(typename std::vector<struct RFNode*>::iterator v=RFHeadNodes.begin(),v_e=RFHeadNodes.end();v!=v_e;++v) {
		if(*v!=NULL) {
			ClearNode(*v);
			delete *v;
			*v = NULL;
		}
	}
	return 0;
}

template <class T>
int ClassifierRF<T>::ClearNode(struct RFNode* v) {
	if(v->dist!=NULL) {
		delete [] v->dist;
		v->dist = NULL;
	}
	if(v->NodeSmaller!=NULL) {
		ClearNode(v->NodeSmaller);
		delete v->NodeSmaller;
		v->NodeSmaller = NULL;
	}
	if(v->NodeLarger!=NULL) {
		ClearNode(v->NodeLarger);
		delete v->NodeLarger;
		v->NodeLarger = NULL;
	}
	return 0;
}

template <class T>
double ClassifierRF<T>::randBetween(double From, double To, size_t resolution) {
	double r;
	if(resolution<RAND_MAX)
		r = (double)rand()/RAND_MAX;
	else {
		//how many bins do we need
		size_t numBins = size_t((double)resolution/RAND_MAX)+1;

		//first we randomly select bin uniformly
		size_t bin = size_t((double)rand()/RAND_MAX*numBins);

		if(bin>=numBins)
			bin = numBins-1;
		
		//then we draw corresponding sample
		r = ((double)RAND_MAX*bin + (double)rand())/(RAND_MAX*numBins);
	}
	return From + r * (To-From);
}

template <class T>
int ClassifierRF<T>::Learn() {
	//std::cout << "Learning started..." << std::endl;


	size_t numAttr = ClassifierGeneral<T>::feat->NumFeatures();
	size_t AttributesToSample = MY_MAX(1, size_t(std::ceil(std::sqrt(double(numAttr)))));
	//AttributesToSample = MY_MAX(1, 1+size_t(LOG2(double(numAttr))));

	//uniform distribution over attributes
	std::vector<double> wAttr(numAttr, 1.0/numAttr);
	std::partial_sum(wAttr.begin(), wAttr.end(), wAttr.begin(), std::plus<double>());
	wAttr.back() = 1.01;


	const std::vector<size_t>* dist = ClassifierGeneral<T>::feat->GetClassDistribution();

	RFHeadNodes.assign(params->numTrees, NULL);



	srand(1);

	// omp_set_num_threads(16);

	// #pragma omp parallel for
	for(size_t k=0;k<params->numTrees;++k) {
		//set class uniform data weights
		std::vector<std::vector<double> > DataWeights(dist->size(), std::vector<double>());
		for(size_t m=0;m<dist->size();++m) {
			DataWeights[m].assign(dist->at(m), 1.0/dist->at(m));
		}


		//sample data
		std::vector<size_t> oobIdx, ibIdx, ibRep;
		WeightedSampling(dist, DataWeights, oobIdx, ibIdx, ibRep);


		//initialize tree
		RFHeadNodes[k] = new struct RFNode;
		RFHeadNodes[k]->dist = new T[dist->size()];
		std::vector<size_t> cls;
		ClassifierGeneral<T>::feat->GetClassDistribution(RFHeadNodes[k]->dist, &cls, ibIdx);



		ConstructTree(RFHeadNodes[k], ibIdx, cls, wAttr, AttributesToSample);



		cls.clear();
		ClassifierGeneral<T>::feat->GetClassDistribution(NULL, &cls, oobIdx);
		std::vector<T> distri;


		size_t error = 0;
		for(size_t m=0;m<oobIdx.size();++m) {
			distri.assign(dist->size(), T(0.0));
			ClassifyTree(RFHeadNodes[k], oobIdx[m], distri);
			size_t predCls = std::max_element(distri.begin(), distri.end())-distri.begin();
			if(predCls!=cls[m]) {
				++error;
			}
		}

		std::cout << "Performance Tree " << k << ": " << T(error)/oobIdx.size() << std::endl;
	}

	//std::cout << "Learning finished..." << std::endl;
	return 0;
}

template <class T>
int ClassifierRF<T>::ClassifyTree(struct RFNode* node, size_t dataIdx, std::vector<T>& distri) {
	while(node->NodeLarger!=NULL && node->NodeSmaller!=NULL) {
		if(ClassifierGeneral<T>::feat->FeatureResponse(dataIdx, node->featID)<=node->splitVal) {
			node = node->NodeSmaller;
		} else {
			node = node->NodeLarger;
		}
	}
	T* d = node->dist;
	for(typename std::vector<T>::iterator p=distri.begin(),p_e=distri.end();p!=p_e;++p, ++d) {
		*p += *d;
	}
	return 0;
}

template <class T>
int ClassifierRF<T>::Classify(size_t dataIdx, std::vector<T>& distri) {
	for(typename std::vector<struct RFNode*>::iterator node=RFHeadNodes.begin(),node_e=RFHeadNodes.end();node!=node_e;++node) {
		ClassifyTree(*node, dataIdx, distri);
	}
	return 0;
}

template <class T>
int ClassifierRF<T>::ConstructTree(struct RFNode* head, std::vector<size_t>& dataIdx, std::vector<size_t>& cls, std::vector<double>& wAttr, size_t AttributesToSample) {
	if(stoppingCriteria(head)) {
		cls.clear();
		dataIdx.clear();
		return 0;
	}

	size_t numCls = ClassifierGeneral<T>::feat->NumClasses();



	int maxTries = 20;
	while(maxTries>0) {



		std::vector<int> selAttr(wAttr.size(), 0);
		whichAttributes(wAttr, AttributesToSample, selAttr);


		T BestSplitVal = 0;
		double BestEstimation = -std::numeric_limits<double>::max();
		size_t bestAttr = size_t(-1);


		for(size_t k=0;k<selAttr.size();k++) {
			T splitVal;
			double est;
			if(selAttr[k]!=0) {
				ImpuritySplit(dataIdx, cls, k, &splitVal, &est);
				if(est>BestEstimation) {
					BestEstimation = est;
					BestSplitVal = splitVal;
					bestAttr = k;
				}
			}
		}


		std::cout << "Best estimation:" << BestEstimation << std::endl;
		if(BestEstimation<=0) {
			if(maxTries==0) {
				std::cout << "We cannot do better anymore..." << std::endl;
				return -1;
			} else {
				--maxTries;
			}
		} else {
			head->featID = bestAttr;
			head->splitVal = BestSplitVal;
			delete [] head->dist;
			head->dist = NULL;
			maxTries = 0;
		}
	}


	//split data
	std::vector<size_t> dataIdxSmaller,dataIdxLarger,clsSmaller,clsLarger;
	T* distriSmaller = new T[numCls];
	std::fill(distriSmaller, distriSmaller+numCls, T(0.0));
	T* distriLarger = new T[numCls];
	std::fill(distriLarger, distriLarger+numCls, T(0.0));

	for(size_t k=0;k<dataIdx.size();++k) {
		if(ClassifierGeneral<T>::feat->FeatureResponse(dataIdx[k],head->featID)<=head->splitVal) {
			dataIdxSmaller.push_back(dataIdx[k]);
			++distriSmaller[cls[k]];
			clsSmaller.push_back(cls[k]);
		} else {
			dataIdxLarger.push_back(dataIdx[k]);
			++distriLarger[cls[k]];
			clsLarger.push_back(cls[k]);
		}
	}



	//clear
	cls.clear();
	dataIdx.clear();



	//new nodes
	if(dataIdxSmaller.size()>0 && dataIdxLarger.size()>0) {
		head->NodeSmaller = new struct RFNode;
		head->NodeSmaller->dist = distriSmaller;
		head->NodeLarger = new struct RFNode;
		head->NodeLarger->dist = distriLarger;
		ConstructTree(head->NodeSmaller, dataIdxSmaller, clsSmaller, wAttr, AttributesToSample);
		ConstructTree(head->NodeLarger, dataIdxLarger, clsLarger, wAttr, AttributesToSample);
	} else {
		delete [] distriSmaller;
		delete [] distriLarger;
		clsSmaller.clear();
		clsLarger.clear();
		dataIdxSmaller.clear();
		dataIdxLarger.clear();
		std::cout << "ERROR." << std::endl;
	}

	return 0;
}

template <class T>
int ClassifierRF<T>::ImpuritySplit(std::vector<size_t>& dataIdx, std::vector<size_t>& cls, size_t featureId, T* splitVal, double* bestEstimation) {
	size_t numCls = ClassifierGeneral<T>::feat->NumClasses();
	std::multimap<T, size_t> split_points;
	size_t *tab = new size_t[2*numCls];
	memset(tab,0,2*numCls*sizeof(size_t));
	for(size_t k=0;k<dataIdx.size();k++) {
		split_points.insert(std::make_pair<T, size_t>(ClassifierGeneral<T>::feat->FeatureResponse(dataIdx[k],featureId),k));

		size_t position = cls[k];
		++tab[2*position+1];			//store everything on the right hand side
	}

	//compute prior impurity from the right side
	double priorImp = GiniImpurity(split_points.size(), tab, 1, numCls);

	//loop through all possible split points
	size_t cnt = 1;						//counts how many variables are on the left side
	size_t numLR[2];

	//noAttrVal[0] = 0;
	//noAttrVal[1] = split_points.size();
	//double test = ImpurityGain(priorImp, split_points.size(), noAttrVal, noClassesAttrVal, numCls);		//yields zero!!!

	*bestEstimation = -std::numeric_limits<double>::max();
	typename std::multimap<T, size_t>::iterator lastDifferent = split_points.begin();
	*splitVal = -std::numeric_limits<T>::max();

	//shift points to left
	size_t position = cls[lastDifferent->second];
	--tab[2*position+1];		//remove variable from right hand side
	++tab[2*position];			//add variable to left hand side

	typename std::multimap<T, size_t>::iterator iter=lastDifferent;
	++iter;
	for(;iter!=split_points.end();++iter, ++cnt) {
		if(lastDifferent->first!=iter->first) {
			numLR[0] = cnt;
			numLR[1] = split_points.size()-cnt;
			double est = GiniImpurityGain(priorImp, split_points.size(), numLR, tab, numCls);
			if (est > *bestEstimation) {
				*bestEstimation = est;
    			*splitVal = (iter->first + lastDifferent->first)/T(2.0) ;
    		}
    		lastDifferent = iter;
		}

		//shift points to left
		position = cls[iter->second];
		--tab[2*position+1];
		++tab[2*position];
	}

	delete [] tab;
	return 0;
}

template <class T>
double ClassifierRF<T>::GiniImpurity(size_t weight, size_t* tab, size_t valIdx, size_t numClasses) {
	double gi = 0.0;
    for(size_t classIdx=0;classIdx<numClasses;++classIdx)
		gi += sqr(double(tab[2*classIdx+valIdx])/weight);
    return  gi;
}

template <class T>
double ClassifierRF<T>::GiniImpurityGain(double priorImp, size_t weight, size_t* numLR, size_t* tab, size_t numClasses) {
	double tempP, gini=0.0;
    for(int valIdx=0;valIdx<2;++valIdx) {		//loop over left and right side
	   tempP = double(numLR[valIdx])/weight;
	   if (numLR[valIdx]>0)
         gini += tempP * GiniImpurity(numLR[valIdx], tab, valIdx, numClasses);
    }
    return (gini - priorImp);
}

template <class T>
inline double ClassifierRF<T>::sqr(double x) {
	return x*x;
}

template <class T>
int ClassifierRF<T>::whichAttributes(std::vector<double>& wAttr, size_t AttributesToSample, std::vector<int>& selAttr) {
	double rndNum;
	size_t i=0, totalNumAttr = wAttr.size(), j; 
	while(i < AttributesToSample) {
		rndNum = randBetween(0.0, 1.0, totalNumAttr);
		for(j=0;j < totalNumAttr; ++j) {
			if(rndNum <= wAttr[j])
				break;
		}
		if (selAttr[j]==0) {
			selAttr[j] = 1;
			++i;
		}
	}
	return 0;
}

template <class T>
bool ClassifierRF<T>::stoppingCriteria(struct RFNode* node) {
	size_t numCls = ClassifierGeneral<T>::feat->NumClasses();
	T sum = std::accumulate(node->dist, node->dist+numCls, T(0.0));

	if(sum<5)
		return true;

	if(*std::max_element(node->dist, node->dist+numCls)/sum >= 0.99)
		return true;

	return false;
}

template <class T>
int ClassifierRF<T>::WeightedSampling(const std::vector<size_t>* SamplesPerClass, std::vector<std::vector<double> >& DataWeights, std::vector<size_t>& oobIdx, std::vector<size_t>& ibIdx, std::vector<size_t>& ibRep) {
	std::vector<std::vector<double> > sortedWeights;

	size_t NumClasses = SamplesPerClass->size();
	sortedWeights.resize(NumClasses);

	for(size_t k=0;k<NumClasses;++k) {
		size_t validSamples = 0;
		size_t numSampleReq = SamplesPerClass->at(k);
		while(validSamples++<numSampleReq) {
			sortedWeights[k].push_back( randBetween(0, 1, numSampleReq) );
		}
	}

	for(size_t k=0;k<NumClasses;++k) {
		std::sort(sortedWeights[k].begin(), sortedWeights[k].end());
	}

	size_t idCounter = 0;
	size_t repCounter = 0;
	for(size_t k=0;k<NumClasses;++k) {
		size_t curPos = 0;
		double ClassCDF = 0;
		bool inBag = false;
		for(size_t m=0;m<DataWeights[k].size();++m) {
			ClassCDF += DataWeights[k][m];
			while(curPos<sortedWeights[k].size() && sortedWeights[k][curPos]<ClassCDF) {
				++curPos;
				ibIdx.push_back(idCounter);
				ibRep.push_back(repCounter);
				inBag = true;
			}
			if(!inBag) {
				oobIdx.push_back(idCounter);
			} else {
				++repCounter;
			}
			++idCounter;
			inBag = false;
		}
	}
	return 0;
}

template class ClassifierRF<double>;
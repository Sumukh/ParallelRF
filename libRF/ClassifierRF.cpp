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
#include <omp.h>

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


ClassifierRF::ClassifierRF(size_t num, size_t numT, FeaturesTable* f) {
	numTrees = num;
	numThreads = numT;
	feat = f;
	RFHeadNodes = new RFNode*[numTrees];
	omp_set_num_threads(numThreads);	
}


ClassifierRF::~ClassifierRF() {
	if(feat!=NULL) {
		feat->ClearFeat();
	}
	ClearCLF();
}

int ClassifierRF::ClearCLF() {
	// for(typename std::vector<struct RFNode*>::iterator v=RFHeadNodes.begin(),v_e=RFHeadNodes.end();v!=v_e;++v) {
	// 	if(*v!=NULL) {
	// 		ClearNode(*v);
	// 		delete *v;
	// 		*v = NULL;
	// 	}
	// }

	for(size_t i = 0; i< numTrees; i++) {
		RFNode *v = RFHeadNodes[i];
		if(v!=NULL) {
			ClearNode(v);
			delete v;
			v = NULL;
		}

	}
	delete RFHeadNodes;
	return 0;
}

int ClassifierRF::ClearNode(RFNode* v) {
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

double ClassifierRF::randBetween(double From, double To, size_t resolution) {
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


int ClassifierRF::Learn() {
	//std::cout << "Learning started..." << std::endl;

	size_t numAttr = feat->NumFeatures();
	size_t AttributesToSample = MY_MAX(1, size_t(std::ceil(std::sqrt(double(numAttr)))));
	//AttributesToSample = MY_MAX(1, 1+size_t(LOG2(double(numAttr))));


	double wAttr[numAttr];
	std::fill_n(wAttr, numAttr, 1.0/numAttr);
	
	//std::partial_sum(wAttr, wAttr + numAttr, wAttr, std::plus<double>());
	// #pragma omp parallel for
	// for(size_t i=0; i < numAttr; i++)
	// {
	// 	wAttr[i] = 1.0/numAttr;
	// }
	std::partial_sum(wAttr, wAttr+numAttr, wAttr); // <-- SSE ?
	wAttr[numAttr-1] = 1.01;

	const std::vector<size_t>* dist = feat->GetClassDistribution();
	srand(1);
	#pragma omp parallel for schedule(dynamic)
	for(size_t k=0;k<numTrees;++k) {
		//set class uniform data weights
		std::vector<std::vector<double> > DataWeights(dist->size(), std::vector<double>());
		//std::cout << "Dist size " << dist->size() << std::endl;
		for(size_t m=0;m<dist->size();++m) {
			DataWeights[m].assign(dist->at(m), 1.0/dist->at(m));
			// std::cout << "size used in assign " << dist->at(m) << std::endl;
		}

		//sample data
		std::vector<size_t> oobIdx, ibIdx, ibRep;
		WeightedSampling(dist, DataWeights, oobIdx, ibIdx, ibRep);

		//initialize tree
		RFHeadNodes[k] = new RFNode();
		RFHeadNodes[k]->dist = new double[dist->size()];
		std::vector<size_t> cls;
		feat->GetClassDistribution(RFHeadNodes[k]->dist, &cls, ibIdx);

		ConstructTree(RFHeadNodes[k], ibIdx, cls, wAttr, numAttr, AttributesToSample);

		cls.clear();
		feat->GetClassDistribution(NULL, &cls, oobIdx);
		size_t error = 0;
		for(size_t m=0;m<oobIdx.size();++m) {
			double* distri = new double[dist->size()];
			std::fill_n(distri, dist->size(), 0.0); 
			ClassifyTree(RFHeadNodes[k], oobIdx[m], distri,dist->size());
			size_t predCls = std::max_element(distri, distri + dist->size())-distri; 
			if(predCls!=cls[m]) {
				++error;
			}
			delete [] distri;
		}

		//std::cout << "Performance Tree " << k << ": " << T(error)/oobIdx.size() << std::endl;
	}

	//std::cout << "Learning finished..." << std::endl;
	return 0;
}


int ClassifierRF::ClassifyTree(RFNode* node, size_t dataIdx, double* distri, size_t distri_size) {
	while(node->NodeLarger!=NULL && node->NodeSmaller!=NULL) {
		if(feat->FeatureResponse(dataIdx, node->featID)<=node->splitVal) {
			node = node->NodeSmaller;
		} else {
			node = node->NodeLarger;
		}
	}
	double* d = node->dist;
	// for(typename std::vector<double>::iterator p=distri.begin(),p_e=distri.end();p!=p_e;++p, ++d) {
	// 	*p += *d;
	// }
	double* p = distri;
	for (size_t i = 0; i < distri_size ; i++) // <-- SSE no omp (order matters)
	{
		*p += *d;
		++p;
		++d;
	}
	return 0;
}

int ClassifierRF::Classify(size_t dataIdx,double* distri, size_t distri_size) {
	// for(typename std::vector<struct RFNode*>::iterator node=RFHeadNodes.begin(),node_e=RFHeadNodes.end();node!=node_e;++node) {
	// 	ClassifyTree(*node, dataIdx, distri);
	// }

	for(size_t i = 0; i < numTrees; i++)
	{
		ClassifyTree(RFHeadNodes[i], dataIdx, distri, distri_size);
	}
	return 0;
}

int ClassifierRF::ConstructTree(RFNode* head, std::vector<size_t>& dataIdx, std::vector<size_t>& cls, double* wAttr,size_t wAttrSize, size_t AttributesToSample) {
	if(stoppingCriteria(head)) {
		cls.clear();
		dataIdx.clear();
		return 0;
	}

	size_t numCls = feat->NumClasses();

	int maxTries = 10;
	while(maxTries>0) {
		std::vector<int> selAttr(wAttrSize, 0);
		whichAttributes(wAttr, wAttrSize, AttributesToSample, selAttr);

		double BestSplitVal = 0;
		double BestEstimation = -std::numeric_limits<double>::max();
		size_t bestAttr = size_t(-1);
		
    //Just trying this out TODO
     #pragma omp parallel for
    for(size_t k=0;k<selAttr.size();k++) {
			double splitVal;
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
	double* distriSmaller = new double[numCls];
	std::fill(distriSmaller, distriSmaller+numCls, 0.0);
	//std::cout << "numCls " <<numCls << std::endl;
	double* distriLarger = new double[numCls];
	std::fill(distriLarger, distriLarger+numCls, 0.0);
  
  //TODO
   #pragma omp parallel for 
  for(size_t k=0;k<dataIdx.size();++k) {
		if(feat->FeatureResponse(dataIdx[k],head->featID)<=head->splitVal) {
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
		head->NodeSmaller = new RFNode();
		head->NodeSmaller->dist = distriSmaller;
		head->NodeLarger = new RFNode();
		head->NodeLarger->dist = distriLarger;
		ConstructTree(head->NodeSmaller, dataIdxSmaller, clsSmaller, wAttr, wAttrSize, AttributesToSample);
		ConstructTree(head->NodeLarger, dataIdxLarger, clsLarger, wAttr, wAttrSize, AttributesToSample);
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


int ClassifierRF::ImpuritySplit(std::vector<size_t>& dataIdx, std::vector<size_t>& cls, size_t featureId, double* splitVal, double* bestEstimation) {
	size_t numCls = feat->NumClasses();
	std::multimap<double, size_t> split_points;
	size_t *tab = new size_t[2*numCls];
	memset(tab,0,2*numCls*sizeof(size_t));
	for(size_t k=0;k<dataIdx.size();k++) {
		split_points.insert(std::make_pair<double, size_t>(feat->FeatureResponse(dataIdx[k],featureId),k));

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
	typename std::multimap<double, size_t>::iterator lastDifferent = split_points.begin();
	*splitVal = -std::numeric_limits<double>::max();

	//shift points to left
	size_t position = cls[lastDifferent->second];
	--tab[2*position+1];		//remove variable from right hand side
	++tab[2*position];			//add variable to left hand side

	typename std::multimap<double, size_t>::iterator iter=lastDifferent;
	++iter;
	for(;iter!=split_points.end();++iter, ++cnt) {
		if(lastDifferent->first!=iter->first) {
			numLR[0] = cnt;
			numLR[1] = split_points.size()-cnt;
			double est = GiniImpurityGain(priorImp, split_points.size(), numLR, tab, numCls);
			if (est > *bestEstimation) {
				*bestEstimation = est;
    			*splitVal = (iter->first + lastDifferent->first)/2.0 ;
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


double ClassifierRF::GiniImpurity(size_t weight, size_t* tab, size_t valIdx, size_t numClasses) {
	double gi = 0.0;
    for(size_t classIdx=0;classIdx<numClasses;++classIdx)
		gi += sqr(double(tab[2*classIdx+valIdx])/weight);
    return  gi;
}


double ClassifierRF::GiniImpurityGain(double priorImp, size_t weight, size_t* numLR, size_t* tab, size_t numClasses) {
	double tempP, gini=0.0;
    for(int valIdx=0;valIdx<2;++valIdx) {		//loop over left and right side
	   tempP = double(numLR[valIdx])/weight;
	   if (numLR[valIdx]>0)
         gini += tempP * GiniImpurity(numLR[valIdx], tab, valIdx, numClasses);
    }
    return (gini - priorImp);
}


inline double ClassifierRF::sqr(double x) {
	return x*x;
}

int ClassifierRF::whichAttributes(double* wAttr, size_t wAttrSize, size_t AttributesToSample, std::vector<int>& selAttr) {
	double rndNum;
	size_t i=0, totalNumAttr = wAttrSize, j; 
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


bool ClassifierRF::stoppingCriteria(RFNode* node) {
	size_t numCls = feat->NumClasses();
	double sum = std::accumulate(node->dist, node->dist+numCls, 0.0);

	if(sum<5)
		return true;

	if(*std::max_element(node->dist, node->dist+numCls)/sum >= 0.99)
		return true;

	return false;
}


int ClassifierRF::WeightedSampling(const std::vector<size_t>* SamplesPerClass, std::vector<std::vector<double> >& DataWeights, std::vector<size_t>& oobIdx, std::vector<size_t>& ibIdx, std::vector<size_t>& ibRep) {
	std::vector<std::vector<double> > sortedWeights;
	size_t NumClasses = SamplesPerClass->size();
	sortedWeights.resize(NumClasses);
	//std::cout << "NumClasses " << NumClasses << std::endl;
	for(size_t k=0;k<NumClasses;++k) {
		size_t validSamples = 0;
		size_t numSampleReq = SamplesPerClass->at(k);
		//std::cout << "numSampleReq " << numSampleReq << std::endl;
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

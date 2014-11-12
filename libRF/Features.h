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

#ifndef __FEATURES_H__
#define __FEATURES_H__

#include <vector>

template <class T>
class FeaturesGeneral {
public:
	struct parameter_type {
		virtual ~parameter_type() {;}
	};
public:
	virtual int ClearFeat() = 0;
	virtual size_t NumSamples() = 0;
	virtual size_t NumFeatures() = 0;
	virtual size_t NumClasses() = 0;
	virtual const std::vector<size_t>* GetClassDistribution() = 0;
	virtual int GetClassDistribution(T* dist, std::vector<size_t>* cls, std::vector<size_t>& dataIdx) = 0;
	virtual int GetTrueClass(std::vector<size_t> *cls, std::vector<size_t> &dataIdx) = 0;
	virtual T FeatureResponse(size_t dataIdx, size_t featureId) = 0;

	virtual int RemoveSampleWithID(std::vector<size_t>& ids) = 0;
	virtual int ResetRemovedIDs() = 0;
};

#endif
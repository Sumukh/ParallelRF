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

#ifndef __CLASSIFIER_H__
#define __CLASSIFIER_H__

#include <vector>

#include <cstddef>

#include "Features.h"

template <class T>
class ClassifierGeneral {
public:
	struct parameter_type {
		virtual ~parameter_type() {;}
	};
	FeaturesGeneral<T>* feat;
public:
	virtual int Learn() = 0;
	virtual int Classify(size_t dataIdx, std::vector<T>& distri) = 0;
	virtual int ClearCLF() = 0;
};

template <class T, class cl, class f>
class Classifier : public cl, public f {
public:
	Classifier(struct ClassifierGeneral<T>::parameter_type* clp, struct FeaturesGeneral<T>::parameter_type* fp) : cl(clp), f(fp) { ClassifierGeneral<T>::feat = this;}
};

#endif
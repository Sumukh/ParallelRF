------------------------------------------------------
lib Random Forest (libRF)
------------------------------------------------------
Test Status:
[![Build Status](https://magnum.travis-ci.com/Sumukh/ParallelRF.svg?token=NdwRmsyfoFUF1zEjsQkw&branch=master)](https://magnum.travis-ci.com/Sumukh/ParallelRF)

Content:
1. Installing
2. Executing
3. References

------------------------------------------------------
1. Installing
1.1 Dependencies

None except for STL.

1.2 Compiling libRandomForest
1.2.1 Linux/Mac

cd to the directory containing the libRF Makefile. Then type

$ make

which should result in an executable named prog.

1.2.2 Windows
For convenience we provide a Microsoft Visual Studio 2008 Solution file. Open the Visual Studio Solution and compile as usual.

------------------------------------------------------
2. Executing

The command

$ ./prog

performs leave one out on the Ionosphere data set [2,3]. 


------------------------------------------------------
3. References

[1] L. Breiman; Random Forests. Machine Learning, 45(1), 5-32, 2001
[2] V.G. Sigillito, S.P. Wing, L.V. Hutton, and K.B. Baker; Classification of radar returns from the ionosphere using neural networks. Johns Hopkins APL Technical Digest, 10, 262-266, 1989
[3] http://archive.ics.uci.edu/ml/datasets/Ionosphere

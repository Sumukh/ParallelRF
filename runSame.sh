  
#!bin/bash
echo "trees;threads;time;error" > outputSame.csv
for t in {1..32}
    do
       ./prog -n 10 -t 16 -v FiveFold -p 1 >> outputSame.csv          
    done
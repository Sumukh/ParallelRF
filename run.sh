  
#!bin/bash
echo "trees;threads;time;error" > output.csv
for t in 32 48 50 58 64 68 72 78 82 86
        do
                for i in 1 2 4 8 16 32 64 128
                do
                        echo $(./prog -n $i -t $t -v FiveFold  -p 1) >> output.csv
                done
        done

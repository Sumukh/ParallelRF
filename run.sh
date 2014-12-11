  
#!bin/bash
echo "trees;threads;time;error" > output.csv
for t in 1 2 4 8 12 16 32 49 64 128
        do
                for i in 16
                do
                        echo $(./prog -n $i -t $t -v FiveFold  -p 1) >> output.csv
                done
        done

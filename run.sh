  
#!bin/bash
"trees;threads;time;error" > output.csv
for t in {1..16}
        do
                for i in 1 2 4 8 16 32 64 128
                do
                        ./prog -n $i -t $t -v FiveFold -p 1 >> output.csv
                done
        done
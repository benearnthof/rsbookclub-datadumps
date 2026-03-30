"""
All kinds of utils
"""

import csv

with open("stats_unreviewed.csv","r",encoding="utf-8") as f:
    reader = csv.reader(f,delimiter=",")
    out =[x for x in reader]

out = out[1::]
# now we're interested in sorting this by the length of label strings
# intuitively the shorter the string the higher the likelihood of false positives

for x in out:
    x.insert(0,len(x[0]))

out.sort()

temp =[x[1:] for x in sorted(out,key=lambda x: (x[0],-int(x[2])))]

for i in range(0,50):
    print(temp[i])
# we can use label density as a proxy for the likelihood of any one label being a
# false positive. 
# in the example of "o",it turned out to be just a single document for example
# another label like "V" is present in numerous separate threads, (and also one
# of the most popular novels on the sub) which indicates it is not a false positive.


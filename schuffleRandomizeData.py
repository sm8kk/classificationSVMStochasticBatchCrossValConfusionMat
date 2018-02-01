from __future__ import division
import sys
import random
# Open the file
f1 = open(str(sys.argv[1]), 'r')
lines = f1.readlines()
f1.close()
randData=[]
header=True
for l in lines:
    if(header == True):
        header=False
	print l[:-1]
        continue;
    randData.append(l)

random.shuffle(randData)
#print randData

for i in xrange(0,len(randData)):
    print randData[i][:-1]

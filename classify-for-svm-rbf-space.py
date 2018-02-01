from __future__ import division
import sys
#import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
from sklearn import svm
import math
from sklearn.metrics import confusion_matrix
# Open the file
f1 = open(str(sys.argv[1]), 'r')
f2 = open(str(sys.argv[2]), 'w')
lines = f1.readlines()
f1.close()

DEBUG = False
DEBUG1 = False
header = True
data=[]

#Classes are chosen from the rate distribution data as seen in cdf
# (t1, t2): (0, 150) 62 pts
t1=0
t2=150
C1=0
# (t2, t3): (150, 300) 60 pts
t3=300
C2=1
# (t3, t4): (300, 400) 55 pts
t4=400
C3=2
# (t4, t5): (400, 500) 25 pts
t5=500
C4=3
#(t5 and above) 25pts
C5=4

classes=[C1,C2,C3,C4,C5]

#after analysis of the relationship between the feature values and the class
#we see that sendCpuUtil, recvCpuUtil, and recvProcNum are not good feature values.

header=True
for l in lines:
    if(header == True):
	header=False
        continue;
    val=l.split(",")
    sendDiskUtil = float(val[2])
    sendProcNum = float(val[3])
    recvDiskUtil = float(val[6])
    btlNck = float(val[9])

    rate = float(val[12])
    if(rate < t2):
        C=C1
    elif(rate < t3 and rate > t2):
        C=C2
    elif(rate < t4 and rate > t3):
        C=C3
    elif(rate < t5 and rate > t4):
        C=C4
    elif(rate > t5):
        C=C5

    f = [sendDiskUtil,sendProcNum,recvDiskUtil,btlNck,C]
    data.append(f)

if (DEBUG == True):
    print "The strong transfer features and classes of data:"
    for i in xrange(0, len(data)):
        print data[i]


SNDDSK=0
SNDPROC=1
RCVDSK=2
BTLNCK=3
CLASS=4

#Break the data into 90% training and 10% test and perform 1: 10 cross validation
testPnct=10
runs=int(100/testPnct)
dataPts=len(data)
testPts=int(testPnct*dataPts/100)


def trainAndTestData(data,trainData,testData,testDataInd,testPts):
    dataPts = len(data)
    testData.extend(data[testDataInd:(testDataInd+testPts)])
    if(testDataInd > 0):
        trainData.extend(data[0:testDataInd])
    if((testDataInd+testPts) < dataPts):
        trainData.extend(data[(testDataInd+testPts):dataPts])
    return

def printTrainTestData(trainData,testData):
    print "Training data:"
    for i in xrange(0, len(trainData)):
        print trainData[i]

    print "Testing data:"
    for i in xrange(0, len(testData)):
        print testData[i]
    return

def featuresTarget(dat,features,target):
    for i in xrange(0, len(dat)):
        features.append(dat[i][:-1]) 
        target.append(dat[i][CLASS]) 
    return  

op = "C,Gamma,DiagSum\n"
f2.write(op)

#Vary the hyper-parameters of the classifier to locate the values for which
#the confusion matrix has highest classification accuracy
#For SVM with "rbf" kernel the hyper-parameters are C and gamma
C_val = [math.pow(10,x) for x in range(-2, 10)] #10^-2 to 10^10
Gamma_val = [math.pow(10,x) for x in range(-9, 3)] #10^-9 to 10^3

for a in xrange(0,len(C_val)):
#for a in xrange(0,2):
    for b in xrange(0,len(Gamma_val)):
#    for b in xrange(0,1):

	confProbRuns = []
	#print "Runs: " + str(runs)
	for i in range(0,runs):
    	    trainData = []
    	    testData = []
    	    testDataInd = i*testPts
    	    trainAndTestData(data,trainData,testData,testDataInd,testPts)
    	    trainDataFeatures = []
    	    trainDataTarget = []
    	    testDataFeatures = []
    	    testDataTarget = []

    	    if (DEBUG == True):
        	print "For run : " + str(i) + " Test index: " + str(testDataInd)
        	printTrainTestData(trainData,testData)

    	    featuresTarget(trainData,trainDataFeatures,trainDataTarget)
    	    featuresTarget(testData,testDataFeatures,testDataTarget)
    
    	    if (DEBUG == True):
        	print trainDataFeatures
        	print trainDataTarget
        	print "**************"
        	print testDataFeatures
        	print testDataTarget
    
    	    #Now use the SVM classifier for each train and test set
    	    clf = svm.SVC(C=C_val[a], cache_size=1000, kernel='rbf',gamma=Gamma_val[b], max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)
    	    classify=clf.fit(trainDataFeatures, trainDataTarget)
    	    if (DEBUG1 == True):
        	print classify


    	    #print "Predict from test features:"
    	    testLen = len(testDataFeatures)
    	    testPredict = []
    	    for j in xrange(testLen):
        	if (DEBUG == True):
                    print "Test features: "
            	    print testDataFeatures[j]
            	    print "Actual class:"
            	    print testDataTarget[j]
            	    print "Prediction:"
        	pred = clf.predict(testDataFeatures[j])
        	#print str(pred)
        	#create the Confusion Matrix
		#y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
		#y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
		#confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
 		#        predicted
		#       ant, bird, cat
		#T (ant) [2, 0, 0]
		#r (bird)[0, 0, 1]
		#u (cat) [1, 0, 2]
		#e
        	testPredict.append(int(pred))
    	    cm=confusion_matrix(testDataTarget, testPredict, labels=classes)
    	    confProbRuns.append(cm)

    	#Find the probability of confusion matrices

	z=np.matrix(confProbRuns[0])
	zp=np.matlib.zeros((len(classes), len(classes)))

	for i in xrange(1,len(confProbRuns)):
    	    z = z + np.matrix(confProbRuns[i])

	print "Run with C: " + str(C_val[a]) + ", Gamma: " + str(Gamma_val[b]) 
	print "sum confProbRuns:"
	print z

	for i in xrange(0, np.shape(z)[0]):
    	    for j in xrange(0, np.shape(z)[1]): 
        	zp[i,j] = z[i,j]/(float(numpy.sum(z, axis=1)[i])) #sum along the rows
	
	diagSum = 0
	for i in xrange(0, np.shape(zp)[0]):
    	    diagSum = diagSum + float(zp[i,i])

	print "Confusion probability:"        
	print zp

	print "Sum of diagonal elements:"
	print diagSum
        op = str(C_val[a]) + "," + str(Gamma_val[b]) + "," + str(diagSum) + "\n"
        f2.write(op)

f2.close()










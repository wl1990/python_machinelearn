from numpy import *
from os import listdir
import operator
import matplotlib
import matplotlib.pyplot as plt

def createDataSet():
	group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels=['A','A','B','B']
	return group,labels

def classify0(inX,dataSet,labels,k):
	dataSetSize=dataSet.shape[0]
	diffMat=tile(inX,(dataSetSize,1))-dataSet
	sqDiffMat=diffMat**2
	sqDistances=sqDiffMat.sum(axis=1)
	distances=sqDistances**0.5
	sortedDistIndicies=distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel=labels[sortedDistIndicies[i]]
		classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
	sortedClassCount=sorted(classCount.items(),key=lambda x:x[1],reverse=True)
	return sortedClassCount[0][0]
group,labels=createDataSet()
#print(classify0([0,0],group,labels,3))

def file2matrix(filename):
	fr=open(filename)
	numberOfLines=len(fr.readlines())
	returnMat=zeros((numberOfLines,3))
	classLabelVector=[]
	fr=open(filename)
	index=0
	for line in fr.readlines():
		line=line.strip()
		listFromLine=line.split('\t')
		returnMat[index,:]=listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index+=1
	return returnMat,classLabelVector
#datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')

def matplot(datingDataMat,datingLabels):
	fig=plt.figure()
	ax=fig.add_subplot(111)
	ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15.0*array(datingLabels),15.0*array(datingLabels))
	plt.show()
#matplot(datingDataMat,datingLabels)

def autoNorm(dataSet):
	minVals=dataSet.min(0)
	maxVals=dataSet.max(0)
	ranges=maxVals-minVals
	normDataSet=zeros(shape(dataSet))
	m=dataSet.shape[0]
	normDataSet=dataSet-tile(minVals,(m,1))
	normDataSet=normDataSet/tile(ranges,(m,1))
	return normDataSet,ranges,minVals
#normMat,ranges,minVals=autoNorm(datingDataMat)

def datingClassTest():
	hoRatio=0.1
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	m=normMat.shape[0]
	numTestVecs=int(m*hoRatio)
	errorCount=0.0
	for i in range(numTestVecs):
		classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print("the classifier came back with:%d,the real answer is :%d" %(classifierResult,datingLabels[i]))
		if(classifierResult!=datingLabels[i]):errorCount+=1.0
		print("the total error rate is: %f" % (errorCount/float(numTestVecs)))
#datingClassTest()

def classifyPerson():
	resultList=['not at all','in small doses','in large doses']
	percentTats=float(input("percentage of time spent playing video games?"))
	ffMiles=float(input("frequent flier miles earned per year?"))
	iceCream=float(input("liters of ice cream consumed per year?"))
	datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals=autoNorm(datingDataMat)
	inArr=array([ffMiles,percentTats,iceCream])
	print(inArr)
	print(minVals)
	print(ranges)
	print((inArr-minVals)/ranges)
	classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print("you will probably like this person:%s" % resultList[classifierResult-1])
#classifyPerson()

def img2vector(filename):
	returnVect=zeros((1,1024))
	fr=open(filename)
	for i in range(32):
		lineStr=fr.readline()
		for j in range(32):
			returnVect[0,32*i+j]=int(lineStr[j])
	return returnVect
#testVector=img2vector('testDigits/0_13.txt')
#print(testVector[0,0:31])
#print(testVector[0,32:63])

def handwritingClasstest():
	hwLabels=[]
	trainingFileList=listdir('trainingDigits')
	m=len(trainingFileList)
	print("m:%s" % m)
	trainingMat=zeros((m,1024))
	for i in range(m):
		fileNameStr=trainingFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		hwLabels.append(classNumStr)
		trainingMat[i,:]=img2vector('trainingDigits/%s' % fileNameStr)
	testFileList=listdir('testDigits')
	errorCount=0.0
	mTest=len(testFileList)
	print("test m:%s" % mTest)
	for i in range(mTest):
		fileNameStr=testFileList[i]
		fileStr=fileNameStr.split('.')[0]
		classNumStr=int(fileStr.split('_')[0])
		vectorUnderTest=img2vector('testDigits/%s' % fileNameStr)
		classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
		print("the classifier come back with:%d,the real answer is:%d" % (classifierResult,classNumStr))
		if(classifierResult!=classNumStr):
			errorCount+=1.0
	print("the total number of errors is:%f" % errorCount)
	print("the total error rate is:%f" % (errorCount/float(mTest)))
handwritingClasstest()

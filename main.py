# -*- coding: utf-8 -*-
'''
data importion
'''
import numpy as np  # for matrix calculation
import pandas as pd

# load the CSV file as a numpy matrix
#separate the data with " "(blank,\t)

dataset = pd.read_csv('/home/parker/watermelonData/watermelon_3.csv', delimiter=",")
Attributes=dataset.columns
print(Attributes)
m,n=np.shape(dataset)
dataset=np.matrix(dataset)
for i in range(m):
    if dataset[i,n-1]=="是":dataset[i,n-1]=1
    else:dataset[i,n-1]=0
    dataset[i,7]=round(dataset[i,7],3)
    dataset[i, 8] = round(dataset[i, 8], 3)

attributeSet=[]
for i in range(n):
    curSet=set()
    for j in range(m):
        curSet.add(dataset[j,i])
    attributeSet.append(curSet)

print(dataset)

#precision for float numbers
EPS=0.000001

#D is a ID list with values in [0,16], A is a list with the size of 10,
# if attribute i is used,the position i is -1,else is a positive number
#the density and honey ratio should be a positive float value or -1
D=np.arange(0,m,1)
A=np.ones(n)
A[0]=A[n-1]=-1
print(A)
print(D)



def treeGenerate(D,A,title):
    node=Node(title)
    if isSameY(D):#p74 condition(1),samples are in the same cluster
        node.v=dataset[D[0], n - 1]
        return node
    if isBlankA(A) or isSameAinD(D,A):#condition(2),A==NULL or all the D have the same attribute selected
        node.v=mostCommonY(D)
        return node
    #choose the best attribute
    entropy=0
    floatV=0
    p=0
    for i in range(len(A)):
        if(A[i]>0):
            curEntropy,divideV=gain(D,i)
            if curEntropy>entropy:
                p=i
                entropy=curEntropy
                floatV=divideV
    if isSameValue(-1000,floatV,EPS):#not a float devide
        node.v=Attributes[p]+"=?"
        curSet=attributeSet[p]
        for i in curSet:
            Dv=[]
            for j in range(len(D)):
                if dataset[D[j],p]==i:
                    Dv.append(D[j])
            if Dv==[]:#condition(3)
                nextNode = Node(i)
                nextNode.v=mostCommonY(D)
                node.children.append(nextNode)
                #book said we should return here, but I think we should continue
            else:
                newA=A[:]
                newA[p]=-1
                node.children.append(treeGenerate(Dv,newA,i))
    else:#is a float devide,the floatV is the boundary
        Dleft=[]
        Dright=[]
        node.v=Attributes[p]+"<="+str(floatV)+"?"
        for i in range(len(D)):
            if dataset[D[i],p]<=floatV:Dleft.append(D[i])
            else: Dright.append(D[i])
        node.children.append(treeGenerate(Dleft,A[:],"yes"))
        node.children.append(treeGenerate(Dright,A[:],"no"))
    return node

class Node(object):
    def __init__(self,title):
        self.title=title
        self.v=1
        self.children=[]
        self.deep=0#for plot
        self.ID=-1#for plot


def isSameY(D):
    curY = dataset[D[0], n - 1]
    for i in range(1, len(D)):
        if dataset[D[i],n-1]!=curY:
            return False
    return True

def isBlankA(A):
    for i in range(n):
        if A[i]>0:return False
    return True

def isSameAinD(D,A):
    for i in range(n):
        if A[i]>0:
            for j in range(1,len(D)):
                if not isSameValue(dataset[D[0],i],dataset[D[j],i],EPS):
                    return False
    return True
def isSameValue(v1,v2,EPS):
    if type(v1)==type(dataset[0,8]):
        return abs(v1-v2)<EPS
    else: return v1==v2

def mostCommonY(D):
    res=dataset[D[0],n-1]#1 or 0
    maxC = 1
    count={}
    count[res]=1
    for i in range(1,len(D)):
        curV=dataset[D[i],n-1]
        count[curV]+=1
        if count[curV]>maxC:
            maxC=count[curV]
            res=curV
    return res


import math
def entropyD(D):#P75-formula 4.1
    types=[]
    count={}
    for i in range(len(D)):
        curY=dataset[D[i],n-1]
        if curY not in count:
            count[curY]=1
            types.append(curY)
        else:
            count[curY]+=1
    ans=0
    total=len(D)
    for i in range(len(types)):
        ans-=count[types[i]]/total*math.log2(count[types[i]]/total)
    return ans

def gain(D,p):#P75-formula 4.2
    if type(dataset[0,p])==type(dataset[0,8]):
        res,divideV=gainFloat(D,p)
    else:
        types=[]
        count={}
        for i in range(len(D)):
            a=dataset[D[i],p]
            if a not in count:
                count[a]=[D[i]]
                types.append(a)
            else:
                count[a].append(D[i])
        res=entropyD(D)
        total=len(D)
        for i in range(len(types)):
            res-=len(count[types[i]])/total*entropyD(count[types[i]])
        divideV=-1000
    return res,divideV

def gainFloat(D,p):#P84-formula 4.8
    a=[]
    for i in range(len(D)):
        a.append(dataset[D[i],p])
    a.sort()
    T=[]
    for i in range(len(a)-1):
        T.append((a[i]+a[i+1])/2)
    res=entropyD(D)
    ans=0
    divideV=T[0]
    for i in range(len(T)):
        left=[]
        right=[]
        for j in range(len(D)):
            if (dataset[D[j], p] <=T[i]):
                left.append(D[j])
            else:right.append(D[j])
        temp=res-entropyD(left)-entropyD(right)
        if temp>ans:
            divideV=T[i]
            ans=temp
    return ans,divideV

myDecisionTreeRoot=treeGenerate(D,A,"root")



#plot the tree

def countLeaf(root,deep):
    root.deep=deep
    res=0
    if root.v==1 or root.v==0:
        res+=1
        return res,deep
    curdeep=deep
    for i in root.children:
        a,b=countLeaf(i,deep+1)
        res+=a
        if b>curdeep:curdeep=b
    return res,curdeep
cnt,deep=countLeaf(myDecisionTreeRoot,0)
def giveLeafID(root,ID):
    if root.v==1 or root.v==0:
        root.ID=ID
        #print(root.title,ID,root.deep)
        ID+=1
        return ID
    for i in root.children:
        ID=giveLeafID(i,ID)
    return ID
giveLeafID(myDecisionTreeRoot,0)

import matplotlib.pyplot as plt
decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    plt.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,
                                textcoords='axes fraction',va="center",ha="center",bbox=nodeType,arrowprops=arrow_args)
fig=plt.figure(1,facecolor='white')

import matplotlib as  mpl
mpl.rcParams[u'font.sans-serif'] = ['simhei']
mpl.rcParams['axes.unicode_minus'] = False


def dfsPlot(root):
    if root.ID==-1:
        childrenPx=[]
        meanPx=0
        for i in root.children:
            cur=dfsPlot(i)
            meanPx+=cur
            childrenPx.append(cur)
        meanPx=meanPx/len(root.children)
        c=0
        for i in root.children:
            nodetype=leafNode
            if i.ID<0:nodetype=decisionNode
            plotNode(i.v,(childrenPx[c],0.9-i.deep*0.8/deep),(meanPx,0.9-root.deep*0.8/deep),nodetype)
            plt.text((childrenPx[c]+meanPx)/2,(0.9-i.deep*0.8/deep+0.9-root.deep*0.8/deep)/2,i.title)
            c+=1
        return meanPx
    else:
        return 0.1+root.ID*0.8/(cnt-1)
rootX=dfsPlot(myDecisionTreeRoot)
plotNode(myDecisionTreeRoot.v,(rootX,0.9),(rootX,0.9),decisionNode)
plt.show()
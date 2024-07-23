import numpy as np
import pandas as pd
from sklearn.preprocessing import scale, StandardScaler
from sklearn import tree,svm
from sklearn.model_selection import train_test_split, cross_val_score
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from collections import Counter, defaultdict
import random
import math
from sklearn.metrics import confusion_matrix, classification_report,balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from time import time
from sklearn.metrics.cluster import normalized_mutual_info_score
import random
from sklearn import metrics 
from sklearn.cluster import KMeans
from collections import OrderedDict
import statistics
import operator
from sklearn.model_selection import StratifiedKFold
import copy

import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore")

# Load the datasets at your local drive
# colon; labels are 0 or 1
df = pd.read_excel ('D:\Colon.xlsx', header=None)
df.iloc[:,df.shape[1]-1].replace({'Normal':1, 'Tumor':2},inplace=True)

# CNS; labels are 1 or 2
df = pd.read_excel ('D:\CNS.xlsx', header=None)
df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# Leukemia-2c; labels are 1 or 2
df = pd.read_excel ('D:\Leukemia.xlsx', header=None)

#SMK
df = pd.read_csv ('D:\SMK.csv', header=None)
df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# GLI
df = pd.read_csv ('D:\GLI.csv', header=None)
df.iloc[:,df.shape[1]-1].replace({0:2},inplace=True)

# covid-3c
df = pd.read_csv ('D:\Covid.csv', header=None)
df.iloc[:,df.shape[1]-1].replace({'no virus':1, 'other virus':2, 'SC2':3},inplace=True)

#Leukemia-3c
df = pd.read_excel ('D:\Leukemia_3c.xlsx', header=None)

#MLL-3c
df = pd.read_excel ('D:\MLL.xlsx', header=None)

#SRBCT-4c
df = pd.read_excel ('D:\SRBCT.xlsx', header=None)



X=df.iloc[:,0:df.shape[1]-1]
# X=pd.DataFrame(scale(X))
y=df.iloc[:,df.shape[1]-1]

##### Calculating  quantity of each label in a
labels=np.unique(y)
a = {}

c=1
for i in range (len(labels)):
    # dynamically create key
    key = c
    # calculate value
    value = sum(y==labels[i])
    a[key] = value 
    
    c +=1

############# Mutual Congestion (MC) ##########################################
def MC(arr,dff):
   
       # if labels start with 1, find the location of the first place!='1'
       if newdf.iloc[0,dff.shape[1]-1]==1:
         first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(1).idxmax())
#      co=1
         co=first
         for j in range(first,newdf.shape[0]):
      
           if newdf.iloc[j,newdf.shape[1]-1]==1:
             co=co+1
           if co==a[1]:
             last=j
             break
         alpha=(last-first)/(dff.shape[0])
       else:
           first=newdf.index.get_loc(newdf[newdf.shape[1]-1].ne(2).idxmax())
#        co=1
           co=first
           for j in range(first,newdf.shape[0]):
      
             if newdf.iloc[j,newdf.shape[1]-1]==2:
               co=co+1
             if co==a[2]:
               last=j
               break
           alpha=(last-first)/(df.shape[0])
       return alpha
###############################################################################
##################### Extended Mutual Congestion (EMC)#########################
def EMC(arr):
    unique_elements = np.unique(arr)
    w_values = []
    s1=0
    s2=0
    for current_element in unique_elements:
        indices = np.where(arr == current_element)[0]
        arr_element = arr[indices[0]:indices[-1] + 1]
        consecutive_appearances_beginning=np.argmax(arr_element != arr_element[0])
        consecutive_appearances_end= np.argmax(arr_element[::-1] != arr_element[-1])
        n_other = len(arr_element) - (consecutive_appearances_beginning + consecutive_appearances_end)
        n_current = consecutive_appearances_beginning + consecutive_appearances_end
        s1=s1+n_other
        s2=s2+n_other+n_current
       
    return (s1/s2)
###############################################################################
###################Classification Error Impurity (CEI)#########################
def Classification_Error_Impurity(arr):
    unique_elements = np.unique(arr)
    w_values = []

    for current_element in unique_elements:
        # Skip if the current element is not present in the array
        if current_element not in arr:
            continue

        # Find indices of the current element in the array
        indices = np.where(arr == current_element)[0]

        # Create subarray for the current element
        arr_element = arr[indices[0]:indices[-1] + 1]

        # Count the occurrences of other elements in the subarray
        n_other = np.count_nonzero(arr_element != current_element)

        # Count the occurrences of the current element in the subarray
        n_current = np.count_nonzero(arr_element == current_element)

        # Avoid division by zero
        if n_current == 0:
            w_values.append(float('inf'))  # or any other suitable value
        else:
            w = n_other / (n_current+n_other)
            w_values.append(w)

    return 1-min(w_values)


################################################################

t0=time()
from operator import itemgetter
import itertools

alpha=np.zeros(df.shape[1]-1)


for w in range(df.shape[1]-1):
    print(w) 
    newdf=df.sort_values(w) 
    alpha[w]=Classification_Error_Impurity(np.array(newdf.iloc[:,newdf.shape[1]-1]))  
    # alpha[w]=MC(np.array(newdf.iloc[:,newdf.shape[1]-1]),df)
    # alpha[w]=EMC(np.array(newdf.iloc[:,newdf.shape[1]-1])) 


    
################################################################    
# MPR, CEI 
# alpha=alphaKamyar
limit=50
zz=(alpha).argsort()[::-1]
Xn=zz[:int(limit)]
Xn=X[Xn]

#DMC,EMC,MC;SLI,SLI-gama
limit=50
zz=(alpha).argsort()
top_CEI=zz[:int(limit)]
Xn=X[Xn]


# np.unique(np.concatenate((top_CEI, top_fisher, top_mutual)))

############ END COMPREHENSIVE


###########  Metrics Evaluation using 5-fold stratified cross validation before applying CMF-AGAwER
s = np.zeros(5)
precision = np.zeros(5)
recall = np.zeros(5)
f1 = np.zeros(5)
mcc = np.zeros(5)
balanced_acc = np.zeros(5)
 
 # Xn.reset_index(drop=True, inplace=True)  # Resetting the index of Xn
skf = StratifiedKFold(n_splits=5)
 
for i, (train_index, test_index) in enumerate(skf.split(X, y)):
     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
     y_train, y_test = y[train_index], y[test_index]

     dectree = tree.DecisionTreeClassifier(random_state=42)
     dectree.fit(X_train, y_train)
     s[i] = dectree.score(X_test, y_test)
     y_pred = dectree.predict(X_test)
     precision[i] = metrics.precision_score(y_test, y_pred, average='micro')
     recall[i] = metrics.recall_score(y_test, y_pred, average='micro')
     f1[i] = metrics.f1_score(y_test, y_pred, average='micro')
     mcc[i] = metrics.matthews_corrcoef(y_test, y_pred) 
     balanced_acc[i] = balanced_accuracy_score(y_test, y_pred)
print('mcc =   ', round(np.mean(mcc),2))
print('acc =   ', round(np.mean(s),2))
print('precision =   ', round(np.mean(precision),2))
print('recall =   ', round(np.mean(recall),2))
print('fscore =   ', round(np.mean(f1),2))
print('balanced_acc =   ', round(np.mean(balanced_acc),2))


###########  Metrics Evaluation using  stratified train-test split before applying CMF-AGAwER
precision=np.zeros(100)
recall=np.zeros(100)
f1=np.zeros(100)
s=np.zeros(100)
mcc=np.zeros(100)
balanced_acc=np.zeros(100)
# # Xn=df.iloc[:,0:50]
for i in range(100):
  X_train, X_test, y_train, y_test = train_test_split(X,y, stratify=y, test_size=0.2)

  dectree = tree.DecisionTreeClassifier()
  dectree.fit(X_train,y_train)
  s[i]=dectree.score(X_test,y_test)
  
  
  cm=confusion_matrix(y_test,dectree.predict(X_test))
  precision[i] = metrics.precision_score(y_test, dectree.predict(X_test))
  recall[i] = metrics.recall_score(y_test, dectree.predict(X_test))
  f1[i] = metrics.f1_score(y_test, dectree.predict(X_test))
  mcc[i] = metrics.matthews_corrcoef(y_test, dectree.predict(X_test))
  balanced_acc[i] = balanced_accuracy_score(y_test, dectree.predict(X_test))

 
print('acc   ',  round(np.mean(s),2))
print('balanced_acc   ',  np.mean(balanced_acc))
print('pre   ',np.mean(precision))
print('rec   ',np.mean(recall)) 
print('fscore   ',np.mean(f1) )
print('mcc   ',np.mean(mcc))





##############################  GENETIC ALGORITHM #######################
#########################################################################
import copy
def fitness(Xn, yn):
    global NFE
    s=np.zeros(10)
    precision=np.zeros(10)
    recall=np.zeros(10)
    f1=np.zeros(10)
    mcc=np.zeros(10)
    balanced_acc=np.zeros(10)
    for i in range(10):
      X_train, X_test, y_train, y_test = train_test_split(Xn,yn, stratify=y, test_size=0.2)

      dectree = tree.DecisionTreeClassifier(random_state=42)
      dectree.fit(X_train,y_train)
      s[i]=dectree.score(X_test,y_test)
      y_pred = dectree.predict(X_test)
      precision[i] = metrics.precision_score(y_test, y_pred, average='micro')
      recall[i] = metrics.recall_score(y_test, y_pred, average='micro')
      f1[i] = metrics.f1_score(y_test, y_pred, average='micro')
      mcc[i] = metrics.matthews_corrcoef(y_test, dectree.predict(X_test)) 
      balanced_acc[i] = balanced_accuracy_score(y_test, dectree.predict(X_test))
    print(round(np.mean(s),2))
      
        
     
    NFE=NFE+1  

    return round(np.mean(s),2), round(np.mean(precision),2), round(np.mean(recall),2), round(np.mean(f1),2), round(np.mean(mcc),2)




from sklearn.model_selection import StratifiedKFold

def fitness(Xn, yn):
    global NFE
    s = np.zeros(5)
    precision = np.zeros(5)
    recall = np.zeros(5)
    f1 = np.zeros(5)
    mcc = np.zeros(5)
    balanced_acc = np.zeros(5)
    
    # Xn.reset_index(drop=True, inplace=True)  # Resetting the index of Xn
    skf = StratifiedKFold(n_splits=5)
    
    for i, (train_index, test_index) in enumerate(skf.split(Xn, yn)):
        X_train, X_test = Xn.iloc[train_index], Xn.iloc[test_index]
        y_train, y_test = yn[train_index], yn[test_index]

        dectree = tree.DecisionTreeClassifier(random_state=42)
        dectree.fit(X_train, y_train)
        s[i] = dectree.score(X_test, y_test)
        y_pred = dectree.predict(X_test)
        precision[i] = metrics.precision_score(y_test, y_pred)
        recall[i] = metrics.recall_score(y_test, y_pred)
        f1[i] = metrics.f1_score(y_test, y_pred)
        mcc[i] = metrics.matthews_corrcoef(y_test, y_pred) 
        balanced_acc[i] = balanced_accuracy_score(y_test, y_pred)

    NFE += 1  

    return round(np.mean(s), 2), round(np.mean(precision), 2), round(np.mean(recall), 2), round(np.mean(f1), 2), round(np.mean(mcc), 2)



def SinglePointCrossover(x1,x2):
    import random
    import numpy as np
    nVar=len(x1)
    C=random.randint(1,nVar-1)
    y1=(x1[0:C]).tolist() + (x2[C:]).tolist()
    y2=(x2[0:C]).tolist() + (x1[C:]).tolist()
    return y1,y2


def VariableSinglePointCrossover(x1,x2):
    import random
    import numpy as np
    nVar1=len(x1)
    nVar2=len(x2)
    if nVar1 == 1 and  nVar2 == 1:
        y1=x1;  y2=x2;
    elif   nVar1 == 1 and  nVar2 > 1: 
        C2=random.randint(1,nVar2-1)
        y1=(x1)+ (x2[C2:])
        y2=(x2[0:C2]) + (x1)
        y1=list(OrderedDict.fromkeys(y1))
        y2=list(OrderedDict.fromkeys(y2))
    elif  nVar2 == 1 and  nVar1 > 1: 
        C1=random.randint(1,nVar1-1)
        y1=(x2)+ (x1[C1:])
        y2=(x1[0:C1]) + (x2)
        y1=list(OrderedDict.fromkeys(y1))
        y2=list(OrderedDict.fromkeys(y2))
    else: 
       C1=random.randint(1,nVar1-1)
       C2=random.randint(1,nVar2-1)

       y1=(x1[0:C1]) + (x2[C2:])
       y2=(x2[0:C2]) + (x1[C1:])
       y1=list(OrderedDict.fromkeys(y1))
       y2=list(OrderedDict.fromkeys(y2))
    return y1,y2


def Mutate(x,seq):
    import random
    import numpy as np
    random_number = random.choice(seq)   
    nVar=len(x)
    J=random.randint(0,nVar-1)
    y=copy.deepcopy(x)
    y[J]=random_number
    y=list(OrderedDict.fromkeys(y))
    return y




def RouletteWheelSelection(P):
    r=random.uniform(0,1)
    c=np.cumsum(P)
    i=np.where(r<np.array(c))[0][0]
    return i


def find_individual_with_highest_fit(pope):
    max_fit_individual = None
    max_fit_value = float('-inf')

    for individual in pope:
        if individual.fit is not None and individual.fit > max_fit_value:
            max_fit_individual = individual
            max_fit_value = individual.fit

    return max_fit_individual


def tournament_selection(cluster_features, num_tournaments=10):
    best_solution = None
    best_fitness = float('-inf')  # Initialize with negative infinity
    current_fitness=empty_individual.repeat(num_tournaments)
    for ii in range(num_tournaments):
        # Select random clusters
        selected_clusters = random.sample(list(cluster_features.keys()), random.randint(2, len(cluster_features)))
        
        # Select a random member from each selected cluster
        selected_features = []
        for label in selected_clusters:
            feature = cluster_features[label]
            selected_feature = random.choice(feature)
            selected_features.append(selected_feature)
        
        # Calculate fitness of the current solution
        current_fitness[ii].position=X[selected_features]
        current_fitness[ii].fit,current_fitness[ii].precision, current_fitness[ii].recall, current_fitness[ii].fmeasure,current_fitness[ii].mcc = fitness(X[selected_features],y)
        
        # Update best solution if the current solution has higher fitness
        if current_fitness[ii].fit > best_fitness:
            best_solution = selected_features
            best_fitness = current_fitness[ii].fit
    
    return best_solution



def tournament_selection2(B, num_tournaments=3):
    best_solution = None
    best_fitness = float('-inf')  # Initialize with negative infinity
    current_fitness=empty_individual.repeat(num_tournaments)
    for ii in range(num_tournaments):
        # # Select random clusters
        # selected_clusters = random.sample(list(cluster_features.keys()), random.randint(2, len(cluster_features)))
        
        # # Select a random member from each selected cluster
        # selected_features = []
        # for label in selected_clusters:
        #     feature = cluster_features[label]
        #     selected_feature = random.choice(feature)
        #     selected_features.append(selected_feature)
        
        # Calculate fitness of the current solution
        current_fitness[ii].position=X[B[ii]]
        current_fitness[ii].List=B[ii]
        current_fitness[ii].fit,current_fitness[ii].precision, current_fitness[ii].recall, current_fitness[ii].fmeasure,current_fitness[ii].mcc = fitness(X[B[ii]],y)
        
        # Update best solution if the current solution has higher fitness
        if current_fitness[ii].fit > best_fitness:
            best_solution = current_fitness[ii].List
            best_fitness = current_fitness[ii].fit
            best_precision=current_fitness[ii].precision
            best_recall=current_fitness[ii].recall
            best_F=current_fitness[ii].fmeasure
            best_mcc=current_fitness[ii].mcc
    return best_solution,best_fitness, best_precision,best_recall, best_F, best_mcc
#####################modified version of the Hausdorff distance#####################

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def closest_vector_distance(vectors_set, target_vector):
    distances = [euclidean_distance(target_vector, vec) for vec in vectors_set]
    return min(distances)

def average_minimum_distance(S1, S2):
# Convert each set to a list of numpy arrays for vector operations
    S1_vectors = [np.array(vec) for vec in S1]
    S2_vectors = [np.array(vec) for vec in S2]

# Calculate the minimum distance from each vector in S1 to the closest vector in S2
    distances_from_S1 = [closest_vector_distance(S2_vectors, vec) for vec in S1_vectors]

# Calculate the minimum distance from each vector in S2 to the closest vector in S1
    distances_from_S2 = [closest_vector_distance(S1_vectors, vec) for vec in S2_vectors]

# Calculate the average of these minimum distances
    average_distance_S1 = np.mean(distances_from_S1)
    average_distance_S2 = np.mean(distances_from_S2)

# The final distance is the average of the averages from S1 to S2 and S2 to S1
    final_distance = (average_distance_S1 + average_distance_S2) / 2
    return final_distance

####################################################################################
#####################Calculate the pairwise distance of solutions in Pop############ 

def calculate_distance_matrix(population):
    L = len(population)
    distance_matrix = np.zeros((L, L))
    for i in range(L):
        for j in range(i+1, L):  # Only need to calculate half of the matrix due to symmetry
            position_i = (population[i].position)  # Convert position to NumPy array
            position_j = (population[j].position)  # Convert position to NumPy array
            distance_matrix[i, j] = average_minimum_distance(position_i, position_j)
            distance_matrix[j, i] = distance_matrix[i, j]  # Symmetrically fill the other half
    return distance_matrix

def find_max_distance(distance_matrix):
    max_distance = np.max(distance_matrix)
    max_indices = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
    return max_distance, max_indices



search_space=features


## random selection from entire dataset
# search_space =  list(range(X.shape[1])) # Total number of features in your dataset
# 


# distance=[]
indexes=[]
externalIt=[]
counter=0

indexes2=[]
externalIt2=[]
counter2=0

matching_index=[]

# Best_REP_fit=0
pcc=0.9
pmm=0.4
Adaptive=0
sp=12
nPop=10
nVar=np.random.randint(1, 21)
pc=0.9;
nc=2*round(pc*nPop/2)
pm=0.4;
nm=round(pm*nPop)
NFE=0
MaxIt=100
MutRate=np.zeros(MaxIt)
MutRate[0]=pm

CrossRate=np.zeros(MaxIt)
CrossRate[0]=pc
BestPosition=0
from ypstruct import struct
empty_individual=struct(position=None,  List=None, fit=None, precision=None, recall=None, fmeasure=None, mcc=None)
pop=empty_individual.repeat(nPop)

Fits=np.zeros(nPop)
it=0

unique_features = set() 
for i in range (nPop):
       pop[i].List= np.random.choice(search_space, size=np.random.randint(1, 11), replace=False)
       pop[i].position=X[pop[i].List]
       pop[i].fit,pop[i].precision, pop[i].recall, pop[i].fmeasure,pop[i].mcc= fitness(pop[i].position,y)  
       Fits[i]=pop[i].fit
           
       unique_features.update(pop[i].List)  
       

P=np.zeros(nPop)     
for j in range (nPop):
     P[j]=Fits[j]/sum(Fits)        


#############  Generate initial Best_Rep_list
q=list(set(search_space) - set(unique_features))
 # Convert the list to a numpy array for compatibility with scikit-learn and reshape it to a 2D array
features_array = np.array(list(set(search_space) - set(unique_features))  ).reshape(-1, 1)
 # Initialize and fit the KMeans model
kmeans = KMeans(n_clusters=int(np.sqrt(len(q))))
kmeans.fit(features_array)

 # Get the cluster labels for each sample
cluster_labels = kmeans.labels_
 
 
 # Initialize a dictionary to store features for each cluster
cluster_features = {}

 # Iterate over the cluster labels and populate the cluster_features dictionary
for i, label in enumerate(cluster_labels):
     if label not in cluster_features:
         cluster_features[label] = []
     cluster_features[label].append(q[i])

count=0
Best=[]
 # # Randomly select one feature from each cluster
 # selected_features = []
 # for label, f in cluster_features.items():
 #     selected_feature = np.random.choice(f)
 #     selected_features.append(selected_feature)

avg_distance=0
# radius=1
distance_matrix=calculate_distance_matrix(pop)
radius=find_max_distance(distance_matrix)[0]/2
while (count<10):
   distance=[]
   avg_distance=0
   selected_clusters=random.sample(list(cluster_features.keys()), random.randint(1, len(cluster_features)))
# Select a random member from each selected cluster
   selected_features = []
   for label in selected_clusters:
       feature = cluster_features[label]
       selected_feature = random.choice(feature)
       selected_features.append(selected_feature)


######## Calculate distance of X[Best_REP_list]   from Pop

   

   for i in range (nPop):
        distance.append(average_minimum_distance(X[selected_features], pop[i].position))
   avg_distance=np.mean(distance)
   if avg_distance>radius:
       Best.append(selected_features)
       count=count+1
Best_REP_list,Best_REP_fit,Pr,Re,Fm,Mcc=tournament_selection2(Best,10)       
# Best_REP_list=selected_features






# Best_REP_fit, B, C, D, E=fitness(X[Best_REP_list],y)
ind_highest_fit=struct(position=X[Best_REP_list],  List=Best_REP_list, fit=Best_REP_fit)

##### Sort population
pop = pop+[ind_highest_fit]    
# import operator
pop=sorted(pop,key=operator.attrgetter('fit'), reverse=True)
    
# for i in range (nPop):
#    print(pop[i].position, "  ",pop[i].List, "  ",pop[i].fit)
pop=pop[0:nPop]   
   
   
### store best solutions in each iteration
BestSol=pop[0]
BestFits=np.zeros(MaxIt)
Best_External_fits=np.zeros(MaxIt)
BestFits[it]=BestSol.fit
Best_External_fits[it]=Best_REP_fit
BestAcc=BestSol.fit
##store worst fit
WorstFit=pop[nPop-1].fit


### array to hold best values in all iterations

print("Best fit in Rep =  ",Best_REP_fit )
print("Best list in Rep =  ",Best_REP_list  )
for individual in range(len(pop)):
# Check if ind_highest_fit.List is in the individual's List
   if np.array_equal(Best_REP_list, pop[individual].List):
       appears = True
       matching_index.append(it)
       print("$$$$$$$$$$$ Best of Repository SEEN in Pop $$$$$$$$$$$$$$$$$",      it)

     # Break out of the loop since the list appears only once
       break
#### array to hold NFEs
nfe=np.zeros(MaxIt)   
nfe[it]=NFE
print("Iteration ", str(it) ,": Best fit = ",  BestAcc,  "NFE =  ", nfe[it])

### Main Loop
import random
import math
# for it in range (MaxIt):
it=it+1
Tag=1
TagCheck=20
AdaptCheck=6
ATag=1     
while (it<MaxIt and  Tag!=TagCheck):
        
    
    popc1=empty_individual.repeat(int(nc/2))
    popc2=empty_individual.repeat(int(nc/2))
    Xover=list(zip(popc1,popc2))
    for k in range (int(nc/2)):
             
             
             # Select First Parent
             i1=RouletteWheelSelection(P)
             # i1=random.randint(0,nPop-1)
             if not isinstance(pop[i1].List, list):
                p1=(pop[i1].List).tolist()
             else:
                p1=(pop[i1].List)    
             # Select Second Parent
             i2=RouletteWheelSelection(P)
             # i2=random.randint(0,nPop-1)
             if not isinstance(pop[i2].List, list):
                p2=(pop[i2].List).tolist()
             else:
                p2=(pop[i2].List)
             # p2=(pop[i2].List).tolist()
             #Apply Crossover
             Xover[k][0].List,Xover[k][1].List=np.array(VariableSinglePointCrossover(p1,p2))
             Xover[k][0].position=X[Xover[k][0].List]
             Xover[k][1].position=X[Xover[k][1].List]
             unique_features.update(X[Xover[k][0].List])  
             unique_features.update(X[Xover[k][1].List])               
             #Evaluate Offspring
             Xover[k][0].fit,Xover[k][0].precision, Xover[k][0].recall, Xover[k][0].fmeasure,Xover[k][0].mcc=fitness(Xover[k][0].position,y)
             Xover[k][1].fit,Xover[k][1].precision, Xover[k][1].recall, Xover[k][1].fmeasure,Xover[k][1].mcc=fitness(Xover[k][1].position,y)
             
    popc=empty_individual.repeat(nc)
    i=0
    for s in range (len(Xover)):
        for j in range(2):
             popc[i]=Xover[s][j]
             i=i+1
    # Mutation
    popm=empty_individual.repeat(nm)    
    for k in range(nm):
       # Select Parent
         i=random.randint(0,nPop-1)
         p=pop[i].List
         available_numbers = list(set(search_space) - set(p) )
         
         popm[k].List=Mutate(p,available_numbers)
         unique_features.update(popm[k].List)  
         popm[k].position=X[popm[k].List]
         popm[k].fit, popm[k].precision, popm[k].recall, popm[k].fmeasure,popm[k].mcc=fitness(popm[k].position,y)
   
       
            
               
    # Generate external repository feature pope
         

    external_feature_repository = list(set(search_space) - set(unique_features))       

    # Convert the list to a numpy array for compatibility with scikit-learn and reshape it to a 2D array
    features_array = np.array(external_feature_repository).reshape(-1, 1)
    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=int(np.sqrt(len(external_feature_repository))))
    kmeans.fit(features_array)

    # Get the cluster labels for each sample
    cluster_labels = kmeans.labels_
    
    
    # Initialize a dictionary to store features for each cluster
    cluster_features = {}

    # Iterate over the cluster labels and populate the cluster_features dictionary
    for i, label in enumerate(cluster_labels):
        if label not in cluster_features:
            cluster_features[label] = []
        cluster_features[label].append(external_feature_repository[i])


   
    
    count=0
    Best=[]
    

    avg_distance=0
    # radius=1
    distance_matrix=calculate_distance_matrix(pop)
    radius=find_max_distance(distance_matrix)[0]/2
    while (count<10):
       distance=[]
       avg_distance=0
       selected_clusters=random.sample(list(cluster_features.keys()), random.randint(1, len(cluster_features)))

    # Select a random member from each selected cluster
       selected_features = []
       for label in selected_clusters:
           feature = cluster_features[label]
           selected_feature = random.choice(feature)
           selected_features.append(selected_feature)


    ######## Calculate distance of X[Best_REP_list]   from Pop

       

       for i in range (nPop):
            distance.append(average_minimum_distance(X[selected_features], pop[i].position))
       avg_distance=np.mean(distance)
       if avg_distance>radius:
           Best.append(selected_features)
           count=count+1
    A1,A2,Pr,Re,Fm,Mcc=tournament_selection2(Best,10)       
    pope=empty_individual.repeat(3)  
    pope[0].List=A1
    pope[0].position=X[pope[0].List]; pope[0].precision=Pr; pope[0].recall=Re; pope[0].fmeasure=Fm; pope[0].mcc=Mcc
    # pope[0].fit, pope[0].precision, pope[0].recall, pope[0].fmeasure,pope[0].mcc=fitness(pope[0].position,y)
    pope[0].fit=A2
    
    pope[1].List,pope[2].List=np.array(VariableSinglePointCrossover(pope[0].List,Best_REP_list))
    pope[1].position=X[pope[1].List]
    pope[1].fit,pope[1].precision, pope[1].recall, pope[1].fmeasure,pope[1].mcc=fitness(pope[1].position,y)
    
    pope[2].position=X[pope[2].List]
    pope[2].fit,pope[2].precision, pope[2].recall, pope[2].fmeasure,pope[2].mcc=fitness(pope[2].position,y)

    ind_highest_fit2 = find_individual_with_highest_fit(pope)
    # ind_highest_fit=list(OrderedDict.fromkeys(ind_highest_fit))
    if ind_highest_fit2.fit>Best_REP_fit:
       Best_REP_fit=ind_highest_fit2.fit 
       Best_REP_list=ind_highest_fit2.List
       Best_REP_list=list(OrderedDict.fromkeys(Best_REP_list))
       ind_highest_fit=ind_highest_fit2
    Best_External_fits[it]=Best_REP_fit
   
       # Rep=ind_highest_fit
    print("********************** ")

    print("Best fit in Rep =  ",Best_REP_fit )
    print("Best list in Rep =  ",Best_REP_list  )

     # merge population        
    pop= pop+popc+popm+[ind_highest_fit]
    # pop= pop+popc+popm
    pop=sorted(pop,key=operator.attrgetter('fit'), reverse=True)
    
      #truncate
    pop=pop[0:nPop]
    
    
    
    appears = False
    appears2= False
    
# Iterate through each individual in the population
    for individual in range(len(pop)):
    # Check if ind_highest_fit.List is in the individual's List
       if np.array_equal(Best_REP_list, pop[individual].List):
           appears = True
           matching_index.append(it)
           print("$$$$$$$$$$$ Best of Repository SEEN in Pop $$$$$$$$$$$$$$$$$",      it)

         # Break out of the loop since the list appears only once
           break
    
    unique_features = set() 
    for i in range (nPop):
        unique_features.update(pop[i].List)  




    for i in range(nPop):
        # print("fit:  " , pop[i].fit, "  size:",  len(pop[i].List) )
        print((pop[i].List),(pop[i].fit))
    
    
    
    
    for j in range (nPop):
       Fits[i]=pop[i].fit
       
    for j in range (nPop):
        P[j]=Fits[j]/sum(Fits)        

    # store best solution ever found
    BestSol=pop[0]
    BestFits[it]=BestSol.fit
    if BestSol.fit > BestAcc:
       BestAcc=BestSol.fit
       BestList=BestSol.List
       BestPosition=BestSol.position
       BestPre=BestSol.precision
       BestRec=BestSol.recall
       BestFmeasure=BestSol.fmeasure
       BestMCC=BestSol.mcc
       # pcc=0.9;
       # pmm=0.4;
       nc=2*round(pcc*nPop/2)
       nm=round(pmm*nPop) 
       MutRate[it]=pmm
       CrossRate[it]=pcc
    ### store NFE
    
    nfe[it]=NFE
    
    
    if (BestFits[it]==BestFits[it-1]):
         Tag=Tag+1 
         ATag=ATag+1
         MutRate[it]=pmm
         CrossRate[it]=pcc
    else:
        Tag=1
        ATag=1
        pcc=0.9;
        pmm=0.4;
    if ATag==6:
      ATag=1
      pcc=pcc-0.3;
      pmm=pmm+0.2;
               
      nm=round(pmm*nPop)
      nc=2*round(pcc*nPop/2)
      Adaptive=Adaptive+1
      MutRate[it]=pmm
      CrossRate[it]=pcc

          
    print("Iteration ", str(it) ,": Best fit = ", BestAcc,  "NFE = ", nfe[it], "mutation rate = ", pmm, "cross-over rate = ", pcc)
    # print("sum  P is  ",sum(P))
    print("********************** ")

    it=it+1   
    if    BestAcc==1:
        break
print ('PRE  ', BestPre, "REC  ",BestRec, "BestFmeasure  ", BestFmeasure, "BestMCC  ", BestMCC)


###############SHAP


import shap
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Assuming X is your DataFrame with 62 instances and 2000 features
# Replace this with your actual DataFrame

X_numpy = X.values  # Convert DataFrame to NumPy array

# Initialize your classification model (replace this with your actual model)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_numpy, y)
# Initialize the SHAP explainer with your model and dataset
explainer = shap.Explainer(clf, X)

# Calculate SHAP values for all instances
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type='bar', max_display=50)


##### For Colon
columns_to_select = [ 128, 2049, 1954,  741,  245, 1002, 2045, 1318, 2145,  173, 565, 1707, 544, 1195, 975,254, 1953, 168, 1193, 1388, 379, 553, 337, 1054, 566, 1262, 1600, 1661, 2143, 1644, 508, 950, 2302, 1737, 1941, 416,1, 1485, 1206, 1576, 713, 1300, 1605, 152, 970, 0, 1676, 66, 1421, 521 ]
Xn = X.iloc[:, columns_to_select]
# Xn=X[1896,1670]
s=np.zeros(100)

for i in range(100):
  X_train, X_test, y_train, y_test = train_test_split(Xn,y, stratify=y, test_size=0.2)

  dectree = tree.DecisionTreeClassifier()
  dectree.fit(X_train,y_train)
  s[i]=dectree.score(X_test,y_test)
      
 
print('acc   ',  np.mean(s))
################ RF

from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Fit the model
rf_classifier.fit(X, y)

# Get feature importances
feature_importances = rf_classifier.feature_importances_

limit = 10
for i in range(5): # Get indices of top 10 features
     top10_indices = feature_importances.argsort()[-limit:][::-1]  # Select top 10 indices

# print("Top 10 feature indices:", top10_indices)

     Xn=X[top10_indices]
     s=np.zeros(100)

     for i in range(100):
           X_train, X_test, y_train, y_test = train_test_split(Xn,y, stratify=y, test_size=0.2)

           dectree = tree.DecisionTreeClassifier()
           dectree.fit(X_train,y_train)
           s[i]=dectree.score(X_test,y_test)
      
     limit=limit+10
print('acc   ',  round(np.mean(s),2))

##############  Correlation

# Compute the correlation matrix between features and target
correlation_matrix = pd.concat([X, pd.DataFrame({'Target': y})], axis=1).corr()

# Get absolute correlations with the target variable
correlation_with_target = abs(correlation_matrix['Target']).drop('Target')  # Exclude the target itself

# Get indices of top features based on correlation
limit = 10
for i in range(5): 
    top_indices = correlation_with_target.argsort()[-limit:][::-1]
    top_feature_names = X.columns[top_indices]

    Xn=X[top_feature_names]
    s=np.zeros(100)

    for i in range(100):
         X_train, X_test, y_train, y_test = train_test_split(Xn,y, stratify=y, test_size=0.2)

         dectree = tree.DecisionTreeClassifier()
         dectree.fit(X_train,y_train)
         s[i]=dectree.score(X_test,y_test)
      
    limit=limit+10
print('acc   ',  round(np.mean(s),2))


############################### FISHER Ratio  Random train-test split
class_means = {}
class_variances = {}

for class_label in np.unique(y):
    class_data = X[y == class_label]
    class_means[class_label] = np.mean(class_data, axis=0)
    class_variances[class_label] = np.var(class_data, axis=0)

# Compute within-class and between-class variances
within_class_variances = np.sum(list(class_variances.values()), axis=0)
overall_mean = np.mean(X, axis=0)
between_class_variances = np.sum([len(class_data) * np.square(class_means[class_label] - overall_mean)
                                 for class_label, class_data in class_means.items()], axis=0)

# Compute Fisher ratios
fisher_ratios = between_class_variances / within_class_variances

# Get indices of top features based on Fisher ratios
limit = 10
for i in range(5): 
    top_indices = fisher_ratios.argsort()[-limit:][::-1]
    top_feature_names = X.columns[top_indices]

    Xn=X[top_feature_names]
    s=np.zeros(100)

    for i in range(100):
       X_train, X_test, y_train, y_test = train_test_split(Xn,y, stratify=y, test_size=0.2)

       dectree = tree.DecisionTreeClassifier()
       dectree.fit(X_train,y_train)
       s[i]=dectree.score(X_test,y_test)
      
    limit=limit+10
print('acc   ',  round(np.mean(s),2))




####FISHER RATIO with Cross Validation

class_means = {}
class_variances = {}

for class_label in np.unique(y):
    class_data = X[y == class_label]
    class_means[class_label] = np.mean(class_data, axis=0)
    class_variances[class_label] = np.var(class_data, axis=0)

# Compute within-class and between-class variances
within_class_variances = np.sum(list(class_variances.values()), axis=0)
overall_mean = np.mean(X, axis=0)
between_class_variances = np.sum([len(class_data) * np.square(class_means[class_label] - overall_mean)
                                 for class_label, class_data in class_means.items()], axis=0)

# Compute Fisher ratios
fisher_ratios = between_class_variances / within_class_variances

# Get indices of top features based on Fisher ratios
M=[]
limit = 10
for i in range(5): 
    top_indices = fisher_ratios.argsort()[-limit:][::-1]
    top_feature_names = X.columns[top_indices]

    Xn=X[top_feature_names]
    skf = StratifiedKFold(n_splits=5)
    s = np.zeros(5)
    

    for i, (train_index, test_index) in enumerate(skf.split(Xn, y)):
     X_train, X_test = Xn.iloc[train_index], Xn.iloc[test_index]
     y_train, y_test = y[train_index], y[test_index]

     dectree = tree.DecisionTreeClassifier(random_state=42)
     dectree.fit(X_train, y_train)
     s[i] = dectree.score(X_test, y_test)
     y_pred = dectree.predict(X_test)
    
    M.append(round(np.mean(s),2)) 
    limit=limit+10
   

print(round(np.mean(M),2))

############################################
###### Mutual Information############
from sklearn.feature_selection import mutual_info_classif

# Assuming X is your feature matrix and y is the target variable for classification
# Calculate mutual information for each feature with the target variable
mutual_info = mutual_info_classif(X, y, random_state=42)

# Get indices of top features based on mutual information
limit = 10
for i in range(5):
    top_indices = (-mutual_info).argsort()[:limit]  # Using negative values for descending order
    top_feature_names = X.columns[top_indices]

    Xn=X[top_feature_names]
    s=np.zeros(100)

    for i in range(100):
      X_train, X_test, y_train, y_test = train_test_split(Xn,y, stratify=y, test_size=0.2)

      dectree = tree.DecisionTreeClassifier()
      dectree.fit(X_train,y_train)
      s[i]=dectree.score(X_test,y_test)
      
    limit=limit+10
print('acc   ',  round(np.mean(s),2))


##### Mutual with  cross validation
mutual_info = mutual_info_classif(X, y, random_state=42)


M=[]
limit=10
for j in range(5):
   top_indices = (-mutual_info).argsort()[:limit]  # Using negative values for descending order
   top_feature_names = X.columns[top_indices]

   Xn=X[top_feature_names]
  
# Xn.reset_index(drop=True, inplace=True)  # Resetting the index of Xn
   skf = StratifiedKFold(n_splits=5)
   s = np.zeros(5)

   for i, (train_index, test_index) in enumerate(skf.split(Xn, y)):
    X_train, X_test = Xn.iloc[train_index], Xn.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]

    dectree = tree.DecisionTreeClassifier(random_state=42)
    dectree.fit(X_train, y_train)
    s[i] = dectree.score(X_test, y_test)
    y_pred = dectree.predict(X_test)
   
   M.append(round(np.mean(s),2)) 
   limit=limit+10
  

print(round(np.mean(M),2))
#####################ReliefF
from skrebate import ReliefF
from sklearn.model_selection import train_test_split
from sklearn import tree
import numpy as np
import pandas as pd
X_array = X.values
X_array = X_array.astype(float)
limit = 10
t0=time()
relieff_selector = ReliefF(n_features_to_select=50, n_neighbors=100)
X_relief = relieff_selector.fit_transform(X_array, y)
t1=time()
print(round(t1-t0))
for i in range(5):
  s = np.zeros(100)
  for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X_relief[:,0:limit], y, stratify=y, test_size=0.2)

    dectree = tree.DecisionTreeClassifier()
    dectree.fit(X_train, y_train)
    s[i] = dectree.score(X_test, y_test)
  limit=limit+10
print('acc   ', round(np.mean(s), 2))









##############  average prediction accuracy based on top N features (N = 10, 20, 30, 40, 50) using 5 times train test split

zC=(alpha).argsort()[::-1]
M=[]
limit=10
# zz=(alpha).argsort()

for i in range(5): 
  Xn=zC[:int(limit)]
  Xn=X[Xn] 

  G=np.zeros(3)  
  for j in range (3):  
    s=np.zeros(100)
    for ii in range(100):
      X_train, X_test, y_train, y_test = train_test_split(Xn,y, stratify=y, test_size=0.2)
      dectree = tree.DecisionTreeClassifier()
      dectree.fit(X_train,y_train)
      s[ii]=dectree.score(X_test,y_test)
    G[j]=round(np.mean(s),2)  
      
  M.append(round(np.mean(G),2))    
  limit=limit+10
  
print(M)    
print('acc   ',  round(np.mean(M),2))



##############  average prediction accuracy based on top N features (N = 10, 20, 30, 40, 50) using stratified 5 fold cross validation

zC=(alpha).argsort()[::-1]
yn=y
s = np.zeros(5)


M=[]
limit=10
# zz=(alpha).argsort()

for j in range(5):
  Xn=zC[:int(limit)]
  Xn=X[Xn] 
# Xn.reset_index(drop=True, inplace=True)  # Resetting the index of Xn
  skf = StratifiedKFold(n_splits=5)

  for i, (train_index, test_index) in enumerate(skf.split(Xn, yn)):
    X_train, X_test = Xn.iloc[train_index], Xn.iloc[test_index]
    y_train, y_test = yn[train_index], yn[test_index]

    dectree = tree.DecisionTreeClassifier(random_state=42)
    dectree.fit(X_train, y_train)
    s[i] = dectree.score(X_test, y_test)
    y_pred = dectree.predict(X_test)
   
  M.append(round(np.mean(s),2)) 
  limit=limit+10
  

print(round(np.mean(M),2))








### average prediction accuracy of SHAP based on top N features (N = 10, 20, 30, 40, 50)  using stratified 5 fold cross validation
M=[]
limit=10
# zz=(alpha).argsort()
for i in range(10):
  Xn=X[name[0:limit]]
  G=np.zeros(5)  
  for j in range (5):  
    s=np.zeros(100)
    # Xn=df.iloc[:,0:50]
    for ii in range(100):
      X_train, X_test, y_train, y_test = train_test_split(Xn,y, stratify=y, test_size=0.2)

      dectree = tree.DecisionTreeClassifier()
      dectree.fit(X_train,y_train)
      s[ii]=dectree.score(X_test,y_test)
    G[j]=round(np.mean(s),2)  
      
  M.append(round(np.mean(G),2))    
  limit=limit+10
  
print(M)    
print('acc   ',  round(np.mean(M),2))


name=np.concatenate((MLL, MLL50))







######The impact of the external repository on GA. The red circles represent the best solutions computed by the external repository. The horizontal dotted line indicates the prediction accuracy before applying CMF-AGAwER: Figure 6 in the paper
GAFits=BestFits[BestFits != 0]
ExternalFits=Best_External_fits[Best_External_fits != 0]
plt.plot(GAFits, label='AGAwER')
for index in matching_index:
    plt.scatter(index, GAFits[index], marker='o', color='r')
plt.plot(ExternalFits, label='External Repository')
plt.axhline(y=0.83, color='b', linestyle='--')  # Adding horizontal line at y-axis = 0.57
plt.legend(loc='lower right')
plt.title('CNS')
plt.ylabel('Accuracy')
plt.xlabel('Iteration')
plt.ylim(0.55)
plt.grid(True)



######The impact of adaptive crossover and mutation rates on GA. The red triangle indicates a setting of rates with Pc = 0 and Pm = 1: Figure 7 in the paper

last_non_zero_index = len(MutRate) - 1
while last_non_zero_index >= 0 and MutRate[last_non_zero_index] == 0:
    last_non_zero_index -= 1

# Trim the array to remove trailing zeros
MutRate = MutRate[:last_non_zero_index + 1]
CrossRate = CrossRate[:len(MutRate)]

M = MutRate.copy()
M[np.where(M == 1.2)] = 1
M[np.where(M == 1.4)] = 1  


fits=GAFits
change_indices = [i for i in range(1, len(MutRate)) if MutRate[i] != MutRate[i - 1]]

# Plot Fits
plt.plot(fits, color='dimgrey', label='DMC-Kmeans-GA')
# Uncomment the line below if fitsW is available
# plt.plot(fitsW, color='black', label='GA')

# Highlight specified iterations
for iteration in change_indices:
    marker = 'o'  # default marker
    if abs(CrossRate[iteration]) <1e-10:
        marker = 'v'  # Change to triangle if CrossRate is 0
        size = 70  # Set the size of the triangle
    else:
        size = 40  # Default size for circles

    color = 'red' if marker == 'v' else 'blue'  # Set color to grey for triangles, blue for circles
    plt.scatter(iteration, fits[iteration], color=color, marker=marker, s=size, label=f'Iteration {iteration}', zorder=5)

plt.title('GA')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.legend(loc='lower right')
plt.show()



# m=np.array(matching_index)
# GAFits=np.array(GAFits)
# ExternalFits=np.array(ExternalFits)








# Ce=(alphaCE).argsort()
# Gi=(alphaG).argsort()

# len(set(Ce[0:int(50)]) & set(z[0:int(50)]))
# for current_element in range(2):
#     print(current_element+1)
    
    
 
    
    
 
    
 
    



############# Calculate the unique features in concatenation of top 50 features of CEI,MI, and FR

# limit=50
# zz=(alpha).argsort()
# top_CEI=zz[:int(limit)]

len(np.unique(np.concatenate((top_CEI, top_fisher, top_mutual))))
features=np.unique(np.concatenate((top_CEI, top_fisher, top_mutual)))

Xn=X[features]


############################ Precision, Recall, Fscore , and MCC: Figure 5 in the paper
#################    Colon    
Be = [0.54, 0.70, 0.60, 0.37]
Af = [0.91, 0.92, 0.91, 0.86]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('Colon')
plt.show()

#################    CNS    
Be = [0.43, 0.61, 0.50, 0.32]
Af = [0.86, 0.87, 0.89, 0.82]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('CNS')
plt.show()

#################    GLI    
Be = [0.74, 0.69, 0.70, 0.57]
Af = [0.92, 0.95, 0.93, 0.90]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('GLI')
plt.show()

#################    SMK    
Be = [0.59, 0.49, 0.48, 0.15]
Af = [0.80, 0.76, 0.76, 0.56]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('SMK')
plt.show()

#################    Leukemia-Binary    
Be = [0.9, 0.94, 0.92, 0.76]
Af = [0.98, 1, 0.99, 0.97]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('Leukemia-Binary')
plt.show()




#################    Leukemia-Muliticlass    
Be = [0.85, 0.85, 0.85, 0.78]
Af = [0.99, 1, 0.99, 0.97]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('Leukemia-Multiclass')
plt.show()


#################    Covid-19    
Be = [0.6, 0.6, 0.6, 0.37]
Af = [0.77, 0.77, 0.77, 0.64]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('Covid-19')
plt.show()


#################    MLL  
Be = [0.79, 0.79, 0.79, 0.7]
Af = [1, 1, 1, 1]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('MLL')
plt.show()

#################    SRBCT  
Be = [0.83, 0.83, 0.83, 0.78]
Af = [0.96, 0.96, 0.96, 0.94]

barWidth = 0.3

# Set position of bar on X axis
br1 = np.arange(len(Be))
br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
 
# Make the plot
plt.bar(br1, Be, color ='grey', width = barWidth,
         label ='Before applying CMF-AGAwER')

plt.bar(br2, Af, color ='r', width = barWidth,
         label ='After applying CMF-AGAwER')
 
# Adding Xticks
plt.ylabel('Percentage')
plt.xticks([r + barWidth for r in range(len(Be))],
        ['Precision', 'Recall', 'Fscore', 'MCC'])
 
plt.legend(loc='lower center', ncol=2, bbox_to_anchor=[0.5, -0.2])
plt.title('SRBCT')
plt.show()
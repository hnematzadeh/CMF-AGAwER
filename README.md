# CMF-AGAwER

This is the implementation of a paper entitled Feature selection using classification error impurity and adaptive genetic algorithm with external repository. The proposed feature selection method (CMF-AGAwER) is a hybrid (ensemble + wrapper). The ensemble section combines the top 50 features ranked by Classification Error Impurity (CEI), a newly proposed frequency-based method, with the top 50 features ranked by Mutual Information (MI) and the top 50 features ranked by Fisher Ratio (FR). The wrapper section utilizes the Adaptive Genetic Algorithm with External Repository (AGAwER). AGAwER's primary contribution lies in its external repository, coupled with an adaptive strategy that adjusts crossover and mutation rates to enhance the exploration capability of genetic algorithms (GA).

Classification Error Impurity (CEI) is a frequency-based filter ranking method, which belongs to a series of frequency-based rankers:

1- [Mutual Congestion (MC)](https://www.sciencedirect.com/science/article/pii/S0888754318304245). Publication year: 2019

2- [Sorted Label Interference (SLI)](https://www.sciencedirect.com/science/article/pii/S0306437921000259#!). Publication year: 2021

3- [Sorted Label Interference-gamma (SLI-gamma)](https://link.springer.com/article/10.1007/s11227-022-04650-w). Publication year: 2022

4- [Extended Mutual Congestion (EMC)](https://https://www.sciencedirect.com/science/article/pii/S1568494622007487#!). Publication year: 2022

5- [Maximum Pattern Recognition (MPR)](https://www.sciencedirect.com/science/article/pii/S0957417424003865). Publication year: 2024

6- [Distance-based Mutual Congestion (DMC)](https://arxiv.org/abs/2407.15611). Publication year: 2024 (in preprint)

7- Classification Error Impurity (CEI). Publication year: Under Review

##################### Instruction #########################

After loading the corresponding dataset from your local drive:

1- Run lines 63-79 to calculate the summation of samples per label

2- Run lines 130-158 for Classification Error Impurity (CEI) function


Repeat steps 3 and 4 for CEI, MI, and FR. Then, concatenate the corresponding lists of sorted features using **features = np.unique(np.concatenate((top_CEI, top_FR, top_MI)))**. For MI and FR, ensure you use **alpha.argsort()** instead of **alpha.argsort()[::-1]**. Alternatively, you can select the corresponding **features.npy** file for each dataset directly from the 'The top 50 features...' folder, eliminating the need to repeat steps 3 and 4. Later in Step 7, you can create the search space of AGAwER using this **features.npy**.

     3- Run lines 163-175 to calculate the alpha.npy which assigns a weight to each feature based on CEI

     4- Run lines 182-185 for sorting the features of the dataset based on the corresponding alpha

5- Run lines 200-228 for metrics evaluation using 5-fold stratified cross validation before applying CMF-AGAwER

6- Run lines 268-509 for the corresponding functions of AGAwER

7- Run line 513 and create the search space of AGAwER using the corresponding **features.npy**

8- Run lines 522-925 for the main body of AGAwER

After line 928, you will find code that calculates the average prediction accuracy using existing filters in the paper, such as SHAP, Random Forest, ReliefF, etc., based on the top N features (where N = 10, 20, 30, 40, 50). Additionally, the code includes functionality for generating illustrations and figures for the paper.

Cite this article



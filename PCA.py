import pandas as pd 
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt 

from sklearn.datasets import load_breast_cancer
cancer=load_breast_cancer()
# print(cancer.keys())


df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
# print(df.head())

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(df)
scaled_data= scalar.transform(df)
# print(scaled_data)

from sklearn.decomposition import PCA
pca=PCA(n_components=4)
pca.fit(scaled_data)
x_pca=pca.transform(scaled_data)
# print(x_pca.shape)

#INORDER TO KNOW THE DIMENSION OF DATA BEFORE PCA
# print(scaled_data.shape)

#  INORDER TO FIND OUT THE VARIANCE
print(pca.explained_variance_ratio_)

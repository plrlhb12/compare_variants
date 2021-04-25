import os
import sys
import argparse 
import math
import time
import h5py
import joblib
import subprocess
import numpy as np
import pandas as pd
import scipy
import statsmodels.api as sm


# cd ./Desktop/scratch/N23_update_June2020/

# Basic regression
data_df = pd.read_hdf("june23rd2020_munged.dataForML.h5", key = "dataForML")
this_formula = "PHENO ~ rs113736300 + rs2406426 + rs17580794"
res = sm.formula.glm(formula=this_formula, family=sm.families.Binomial(), data=data_df).fit() 
res.summary()

# Check for interactions

## Convert the zeros to preserve dynamic range.
data_df['rs113736300_scaled'] = data_df['rs113736300'] + 1
data_df['rs2406426_scaled'] = data_df['rs2406426'] + 1
data_df['rs17580794_scaled'] = data_df['rs17580794'] + 1

## Make interaction terms
data_df['rs113736300_rs2406426'] = data_df['rs113736300_scaled']*data_df['rs2406426_scaled']
data_df['rs113736300_rs17580794'] = data_df['rs113736300_scaled']*data_df['rs17580794_scaled']
data_df['rs2406426_rs17580794'] = data_df['rs2406426_scaled']*data_df['rs17580794_scaled']


int1 = "PHENO ~ rs113736300_rs2406426 + rs113736300 + rs2406426 + rs17580794"
int2 = "PHENO ~ rs113736300_rs17580794 + rs113736300 + rs2406426 + rs17580794"
int3 = "PHENO ~ rs2406426_rs17580794 + rs113736300 + rs2406426 + rs17580794"

## Test the interactions
res1 = sm.formula.glm(formula=int1, family=sm.families.Binomial(), data=data_df).fit() 
res1.summary()
res2 = sm.formula.glm(formula=int2, family=sm.families.Binomial(), data=data_df).fit() 
res2.summary()
res3 = sm.formula.glm(formula=int3, family=sm.families.Binomial(), data=data_df).fit() 
res3.summary()

data_df['PHENO'].describe()
data_df['rs113736300'].describe()/2
data_df['rs2406426'].describe()/2
data_df['rs17580794'].describe()/2

# Now adjust for Ashkenazi ancestry.

## Load in the ancestry data.
ancestry_temp_df = pd.read_csv("output_ancestry_genetic_ancestry_all_pca_plus.csv")
ancestry_temp_df.head()
ancestry_reduced = ancestry_temp_df[['IID','InfAJ']]

## Merge with data
merged_df = data_df.merge(ancestry_reduced, left_on='ID', right_on='IID', how='inner')

## Adjust the model for AJ
this_formula = "PHENO ~ rs113736300 + rs2406426 + rs17580794 + InfAJ"
res = sm.formula.glm(formula=this_formula, family=sm.families.Binomial(), data=merged_df).fit() 
res.summary()

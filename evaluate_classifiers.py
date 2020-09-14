import argparse
import numpy as np
import os
import pandas as pd
import re
import time

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import IterativeImputer, SimpleImputer
from statistics import mean


# Command arguments

arguments = argparse.ArgumentParser()

arguments.add_argument( "-c", "--classifiers", default = 'MultiLayerPerceptron,' +
	'AdaBoost,KNN,RandomForest,DecisionTree,NaiveBayes,LDA',
	help = "List of Classifiers separated by ',' without spaces." )

arguments.add_argument( '-i', '--input', default = 'partitions.csv',
	help = 'CSV file to read. The last columns must be the partition columns, ' +
	'named "target i", beign i the number of partition.' )

arguments.add_argument( "-k", "--kfold", default = 10, help = "Value of K-Fold.")

args = vars( arguments.parse_args() )


# aux functions

def get_estimator( est_type ):
	estimator = None

	if( est_type == 'DecisionTree' ):
		estimator = DecisionTreeClassifier()
	elif( est_type == 'NaiveBayes' ):
		estimator = GaussianNB()
	elif( est_type == 'LDA' ):
		estimator = LinearDiscriminantAnalysis()
	elif( est_type == 'RandomForest' ):
		estimator = RandomForestClassifier()
	elif( est_type == 'KNN' ):
		estimator = KNeighborsClassifier( 5 )
	elif( est_type == 'AdaBoost' ):
		estimator = AdaBoostClassifier()
	elif( est_type == 'MultiLayerPerceptron' ):
		estimator = MLPClassifier()

	return estimator


# Start of the program


program_time = time.time()

kFold = int( args[ 'kfold' ] )
cv = StratifiedKFold( n_splits = kFold )
classifiersList = args[ 'classifiers' ].split( ',' )

# Put header on the output file

fileOut = "evaluation_"+ str( kFold )+ "_fold.txt"
with open( fileOut, "a+" ) as file:
	file.write( 'Partition_File,' + str( args[ 'classifiers' ] ) + ',Maximum\n' )

# Reading and separating dataset on features and partitions

df = pd.read_csv( args[ 'input' ], na_values = np.nan )

regex = re.compile( "^target" )
partitions = list( filter( regex.match, df.columns.values ) )
p_index = len( df.columns ) - len( partitions )

df_features = df[ df.columns[ :p_index ] ].apply( pd.to_numeric )
df_partitions = df[ df.columns[ p_index: ] ].astype( str )

# Perform K-Fold CV for each  partition column

for i, partition in enumerate( partitions ):
	results_row = partition + ","

	X = df_features
	y = df_partitions[ partition ]

	partition_auc = []
	print( 'Analyzing partition: %s' % partition )
	exec_time = time.time()

	for estimator_type in classifiersList:
		print( 'Classifier: %s ' % estimator_type )
		classifier = get_estimator( estimator_type )

		if( classifier ):
			folds_auc = []
			print( 'Starting K-Fold Cross Validation...' )

			# On selected partition, performing K-Fold CV using the estimator

			for i, ( train, test ) in enumerate( cv.split( X, y ) ):
				fold_time = time.time()
				print( 'Training CV-%d...' % i )

				# missing values  imputation

				imputer = SimpleImputer( missing_values = np.nan, strategy = 'mean' )
				imputer.fit( X.iloc[ train ] )
				X_train = imputer.transform( X.iloc[ train ] )
				X_test = imputer.transform( X.iloc[ test ] )

				predictions = classifier.fit( X_train, y.iloc[ train ] ).predict_proba( X_test )
				auc = roc_auc_score( y.iloc[ test ], predictions, multi_class = 'ovr' )

				print( 'Cross Validation: %d AUC: %3.2f Time: %5.3fs' % ( i, auc, time.time() - fold_time ) )
				folds_auc.append( auc )

			estimator_auc = mean( folds_auc )
			print( 'Partition: %s Classifier: %s Mean AUC: %5.3f\n' % ( partition, estimator_type, estimator_auc ) )

			partition_auc.append( estimator_auc )
			results_row += str( round( estimator_auc, 5 ) ) + ','

		else:
			print( '%s is not available. This estimator will be ignored.' % estimator_type )

	print()
	print( 'Maximum AUC among Classifiers: %5.3f' % max( partition_auc ) )
	print( 'Partition\'s total execution time: %6.3fs' % ( time.time() - exec_time ) )
	print()

	results_row += str( round( max( partition_auc ), 5 ) ) + '\n'

	with open( fileOut, 'a+') as file:
		file.write( results_row )

print( '\nProgram\'s total execution time: %6.3fs' % ( time.time() - program_time ) )

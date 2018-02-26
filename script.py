import re
import numpy as np
from scipy import linalg as LA
import cvxopt,os
from cvxopt import solvers
import arff
from svmutil import *
from sympy import Matrix as symMat
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn import decomposition,grid_search

# KERNELS
LINEAR = 0
GAUSSIAN = 1
POLY = 2
TANH = 4
# xi=2
kernel_type = LINEAR
QP_PROGRESS_OUTPUT=0 # Show QP progress output

def sampleCO():
	Q = 2*cvxopt.matrix([ [2, .5], [.5, 1] ])
	p = cvxopt.matrix([(1.0+1.0j), (1.0+2.0j)])
	G = cvxopt.matrix([[-1.0,0.0],[0.0,-1.0]])
	h = cvxopt.matrix([0.0,0.0])
	A = cvxopt.matrix([1.0, 1.0], (1,2))
	b = cvxopt.matrix(1.0)
	sol=cvxopt.solvers.qp(Q, p, G, h, A, b)
	print sol

def k(x, y):
	if kernel_type == LINEAR:
		return np.inner(x, y)
	elif kernel_type == GAUSSIAN:
		exponent = -np.sqrt(la.norm(x-y) ** 2 / (2 * sigma ** 2))
		return np.exp(exponent)
	elif kernel_type == POLY:
		return (offset + np.dot(x, y)) ** dimension
	elif kernel_type == TANH:
		return np.inner(x, y)

def kernel_matrix(X):
    n_samples, n_features = X.shape
    K = np.zeros((n_samples, n_samples))
    for i, x_i in enumerate(X):
        for j, x_j in enumerate(X):
            K[i, j] = k(x_i, x_j)
    return K

def cross_kernel_Matrix(Z,X):
	n_samples, n_features = X.shape
	m_samples, m_features = Z.shape
	K = np.zeros((m_samples, n_samples))
	for i, z_i in enumerate(Z):
		for j, x_j in enumerate(X):
			K[i, j] = k(z_i, x_j)
	return K

def extrapolate(Z,X,eigensystem):
	a = cross_kernel_Matrix(Z,X)*eigensystem[0]             											
	b = eigensystem[1].I
	return a*b

def solveQP(X,PhiZ,KZ,xi):
	f,n=PhiZ.shape
	# print "f,n: ",f,n
	Q=np.multiply((PhiZ.T*PhiZ),(PhiZ.T*PhiZ))
	r=np.diag(np.array(PhiZ.T*KZ*PhiZ))
	I=np.diag(np.ones(n))
	Ibar=np.diag(np.ones(n-1),k=1)
	C=np.matrix(I-xi*Ibar)

	G1=cvxopt.matrix(np.diag(np.ones(n) * -1))
	h1=cvxopt.matrix(np.zeros(n))

	G2=cvxopt.matrix(C*-1)
	h2=cvxopt.matrix(np.zeros(n))

	G = cvxopt.matrix(np.vstack((G1, G2)))
	h = cvxopt.matrix(np.vstack((h1, h2)))

	P = cvxopt.matrix(Q)
	q = cvxopt.matrix(-2 * r)

	solvers.options['show_progress'] = QP_PROGRESS_OUTPUT

	solution = solvers.qp(P, q, G, h)
	return np.ravel(solution['x'])	# returns lambda

def domain_invariant_kernel(new_eigenvector,eigensystem,relaxed_eigenvalue):
	elem00 = new_eigenvector*relaxed_eigenvalue
	elem00 = np.array(elem00*(new_eigenvector.T))
	elem01 = np.array((new_eigenvector*relaxed_eigenvalue)*(eigensystem[0].T))
	elem10 = np.array((eigensystem[0]*relaxed_eigenvalue)*(new_eigenvector.T))
	elem11 = np.array((eigensystem[0]*relaxed_eigenvalue)*(eigensystem[0].T))
	elem1=np.hstack((elem00,elem01))
	elem2=np.hstack((elem10,elem11))

	K = np.vstack((elem1,elem2))
	return K

def readReuters(choice):

	# 3 choices:
		# 1: OrgsPeople
		# 2: OrgsPlaces
		# 3: PeoplePlaces			# DEFAULT

	if choice == 1:
		target_data = arff.load(open('reuters_processed/OrgsPeople.tar.arff', 'rb'))['data']
		source_data = arff.load(open('reuters_processed/OrgsPeople.src.arff', 'rb'))['data']
	elif choice ==2: 
		target_data = arff.load(open('reuters_processed/OrgsPlaces.tar.arff', 'rb'))['data']
		source_data = arff.load(open('reuters_processed/OrgsPlaces.src.arff', 'rb'))['data']
	else:
		arget_data = arff.load(open('reuters_processed/PeoplePlaces.tar.arff', 'rb'))['data']
		source_data = arff.load(open('reuters_processed/PeoplePlaces.src.arff', 'rb'))['data']

	print"TARGET ARFF FILE LOADED"
	target_Y=[int(y[-1]) for y in target_data]
	target_X=[x[0:-1] for x in target_data]

	print"READING SOURCE ARFF"

	print"SOURCE ARFF FILE LOADED"
	source_Y=[int(y[-1]) for y in source_data]
	source_X=[x[0:-1] for x in source_data]

	return source_X,target_X,source_Y,target_Y

def readNewsgroupsFromARFF(choice):		#VERY INEFFICIENT: HANGS
	
		# 5 choices:
		# 1: cs
		# 2: ct
		# 3: rs			
		# 4: rt
		# 5: st 	# DEFAULT

	if choice ==1:
		target_data = arff.load(open('newsgroups_processed/cs_inDomain.arff', 'rb'))['data']
		source_data = arff.load(open('newsgroups_processed/cs_outDomain.arff', 'rb'))['data']
	elif choice==2:
		target_data = arff.load(open('newsgroups_processed/ct_inDomain.arff', 'rb'))['data']
		source_data = arff.load(open('newsgroups_processed/ct_outDomain.arff', 'rb'))['data']
	elif choice==2:
		target_data = arff.load(open('newsgroups_processed/rs_inDomain.arff', 'rb'))['data']
		source_data = arff.load(open('newsgroups_processed/rs_outDomain.arff', 'rb'))['data']
	elif choice==2:
		target_data = arff.load(open('newsgroups_processed/rt_inDomain.arff', 'rb'))['data']
		source_data = arff.load(open('newsgroups_processed/rt_outDomain.arff', 'rb'))['data']
	else:
		target_data = arff.load(open('newsgroups_processed/st_inDomain.arff', 'rb'))['data']
		source_data = arff.load(open('newsgroups_processed/st_outDomain.arff', 'rb'))['data']

	print"TARGET ARFF FILE LOADED"
	target_Y=[int(y[-1]) for y in target_data]
	target_X=[x[0:-1] for x in target_data]

	print"READING SOURCE ARFF"

	print"SOURCE ARFF FILE LOADED"
	source_Y=[int(y[-1]) for y in source_data]
	source_X=[x[0:-1] for x in source_data]

	return source_X,target_X,source_Y,target_Y

def readMNIST(subset=1000):

	IF_PCA=1
	pca_components=25

	source_dataset=open('ule_text/ule_devel.data').readlines()
	source_data=[x.split() for x in source_dataset]
	for i in range(len(source_data)):
		for j in range(len(source_data[i])):
			source_data[i][j]=int(source_data[i][j])

	source_X=source_data
 	source_Y=[]
	source_datasetY=open('ule_text/ule_devel.label').readlines()
	source_dataY=[x.split() for x in source_datasetY]
	for i in range(len(source_dataY)):
		for j in range(len(source_dataY[i])):
			if source_dataY[i][j]=='1':
				source_Y.append(j)	
	
	target_dataset=open('ule_text/ule_final.data').readlines()
	target_data=[x.split() for x in target_dataset]
	for i in range(len(target_data)):
		for j in range(len(target_data[i])):
			target_data[i][j]=int(target_data[i][j])

	target_X=target_data
 	target_Y=[]
	target_datasetY=open('ule_text/ule_final.label').readlines()
	target_dataY=[x.split() for x in target_datasetY]
	for i in range(len(target_dataY)):
		for j in range(len(target_dataY[i])):
			if target_dataY[i][j]=='1':
				target_Y.append(j)

	#SUBSETTING
	source_X=source_X[:subset]
	source_Y=source_Y[:subset]

	#PCA
	if IF_PCA:
		pca = decomposition.PCA(n_components=pca_components)
		pca.fit(source_
		source_X = pca.transform(source_X)

		pca = decomposition.PCA(n_components=pca_components)
		pca.fit(target_X)
		target_X = pca.transform(target_X)

	return source_X,target_X,source_Y,target_Y

def readNewsgroups(choice):	
	# 6 choices availabe: 1-6, DEFAULT:6

	y = open('20news-bydate/train.label','r')
	a = y.read().split('\n')
	x = open('20news-bydate/train.data','r')
	b = x.read().split('\n')
	for i in range(len(a)):
		a[i] = int(a[i])

	P = [2,3,4,5]
	Q = [8,9,10,11]
	R = [12,13,14,15]
	S = [17,18,19,20]

	if choice==1:
		source_one = [2,3]
		source_two = [8,10]
		target_one = [4,5]
		target_two = [10,11]
	elif choice==2:	
		source_one = [2,3]
		source_two = [12,13]
		target_one = [4,5]
		target_two = [14,15]
	elif choice==3:	
		source_one = [2,3]
		source_two = [17,18]
		target_one = [4,5]
		target_two = [19,20]
	elif choice==4:	
		source_one = [8,9]
		source_two = [12,13]
		target_one = [10,11]
		target_two = [14,15]
	elif choice==5:	
		source_one = [8,9]
		source_two = [17,18]
		target_one = [10,11]
		target_two = [19,20]
	else:									# DEFAULT
		source_one = [12,13]
		source_two = [17,18]
		target_one = [14,15]
		target_two = [19,20]

	inp = [0 for i in range(61188)]
	source = []
	source_Y=[]
	target = []
	target_Y=[]
	old = 1

	for i in range(len(b)):
		b[i] = b[i].split(" ")
		doc = int(b[i][0])
		if doc != old :
			if a[old] == source_one[0]:
				source.append(inp)
				source_Y.append(0)
				inp = [0 for i in range(61188)]
			elif a[old] == source_one[1]:
				source.append(inp)
				source_Y.append(0)
				inp = [0 for i in range(61188)]
			elif a[old] == source_two[0]:
				source.append(inp)
				source_Y.append(1)
				inp = [0 for i in range(61188)]
			elif a[old] == source_two[1]:
				source.append(inp)
				source_Y.append(1)
				inp = [0 for i in range(61188)]
		if doc in source_one or source_two :
			for j in range(3):
				b[i][j] = int(b[i][j])
			inp[b[i][1]-1] = b[i][2]
		old = doc

	y = open('20news-bydate/matlab/test.label','r')
	a = y.read().split('\n')
	for i in range(len(a)):
		a[i] = int(a[i])
	x = open('20news-bydate/matlab/test.data','r')
	b = x.read().split('\n')

	old = 1
	out = [0 for i in range(61188)]
	for i in range(len(b)):
		b[i] = b[i].split(" ")
		doc = int(b[i][0])
		if doc != old :
			if a[old] == target_one[0]:
				target.append(out)
				target_Y.append(0)
				out = [0 for i in range(61188)]
			elif a[old] == target_one[1]:
				target.append(out)
				target_Y.append(0)
				out = [0 for i in range(61188)]
			elif a[old] == target_two[0]:
				target.append(out)
				target_Y.append(0)
				out = [0 for i in range(61188)]
			elif a[old] == target_two[1]:
				target.append(out)
				target_Y.append(1)
				out = [0 for i in range(61188)]

		if doc in target_one or target_two :
			for j in range(3):
				b[i][j] = int(b[i][j])
			out[b[i][1]-1] = b[i][2]
		old = doc	
	return source,target,source_Y,target_Y

def main():
	print "READING DATA.."
	reuters_choice=1 		# 3 CHOICES
	newsgroups_choice=1		# 6 CHOICES
	MNIST_subset_size=1000

	source_X,target_X,source_Y,target_Y=readReuters(reuters_choice)				# ON THE REUTERS DATASET
	# source_X,target_X,source_Y,target_Y=readNewsgroups(newsgroups_choice)		# ON THE NEWSGROUPS DATASET
	# source_X,target_X,source_Y,target_Y=readMNIST(MNIST_subset_size)			# ON THE MNIST DATASET

	print "DATA READ."
	Z=np.array(source_X)
	X=np.array(target_X)

	print "DIMENTION OF X", X.shape
	print "DIMENTION OF Z",Z.shape

	print "COMPUTING KERNEL MATRICES KZ AND KX"
	KZ=kernel_matrix(Z)
	KX=kernel_matrix(X)

	print "DIMENTION OF KX",len(KX)
	print "DIMENTION OF KZ",len(KZ)

	print "COMPUTING EIGENVECTORS OF KX"
	e_valsKX_temp, e_vecsKX = np.linalg.eigh(KX)
	select=min(500,len(e_valsKX_temp))
	e_valsKX_temp=e_valsKX_temp[-1:-select-1:-1]
	e_vecsKX=e_vecsKX.T[-1:-select-1:-1]
	# e_valsKX=[float("%.5f" % e) for e in e_valsKX_temp]				# IF REDUCING TO FLOATING POINT
	e_vecsKX=np.matrix(e_vecsKX)
	e_vecsKX=e_vecsKX.T
	print "DIMENSION OF EIGENVECTOR OF KX:",e_vecsKX.shape
	PhiX=e_vecsKX
	e_valsKX_diag=np.matrix(np.diag(e_valsKX))
	eigensystem=(PhiX,e_valsKX_diag)
	print "EXTRAPOLATING(COMPUTING PHIZ_BAR)"
	PhiZ_bar=extrapolate(Z,X,eigensystem)
	print "DIMENSTION OF (EXTRAPOLATED) PHIZ_bar:",PhiZ_bar.shape

	#TO TUNE xi, this needs to be done repeatedly
	# xi=range(1,6)			# TUNE
	xi=[2]					# NOT TUNE
	i=1
	c=10	#WHEN NOT TUNING
	tune=0
	max_score=0.0
	best_c=0
	best_xi=0
	for x in xi:
		print "ITERATION: ",i
		i+=1
		print "SOLVING QP TO GET LAMBDA (SPECTRAL KERNEL)"
		Lambda_vector=solveQP(X,PhiZ_bar,KZ,x)
		Lambda_matrix=np.matrix(np.diag(Lambda_vector))

		# print "COMPUTING DOMAIN INVARIANT KERNEL"
		# KA_bar=domain_invariant_kernel(PhiZ_bar,eigensystem,Lambda_matrix)	#DOMAIN INVARIANT KERNEL MATRIX(NO NOEED TO CALCULATE THIS)

		print "COMPUTING KZ_BAR TO TRAIN SVM"
		KZ_bar=(PhiZ_bar*Lambda_matrix)*PhiZ_bar.T 	#TRAIN SVM ON THIS

		print "COMPUTING KXZ_BAR TO TEST SVM"
		KXZ_bar=(PhiX*Lambda_matrix)*PhiZ_bar.T 	#TEST ON THIS

		KZ_bar=KZ_bar.tolist()

		# m = svm_train(source_Y, [list(row) for row in KZ_bar], '-c 10')
		# p_label, p_acc, p_val = svm_predict(target_Y,[list(row) for row in KXZ_bar], m)
		
		if not tune:
			print "SIMPLE SVM USING SCIKIT, C =",c
			svc = SVC(kernel='linear',C=c)
			svc.fit(source_X, source_Y)
			y_pred = svc.predict(target_X)
			print 'ACCURACY SCORE OF SIMPLE SVM: %0.3f' % accuracy_score(target_Y, y_pred)

			print "DOMAIN INVARIANT SVM USING SCIKIT, C =",c
			svc = SVC(kernel='precomputed',C=c)
			svc.fit(KZ_bar, source_Y)
			y_pred = svc.predict(KXZ_bar)
			print 'ACCURACY SCORE OF DOMAIN INVARIANT SVM: %0.3f' % accuracy_score(target_Y, y_pred)

		else:
			print "GRID SEARCH"
			KZ_bar=np.matrix(KZ_bar)
			# xi = np.logspace(-6, -1, 10)
			svc = SVC(kernel='precomputed')
			tuned_parameters = [
		                    {'C': [1, 10, 100, 1000]}]                
			clf = grid_search.GridSearchCV(estimator=svc,param_grid=tuned_parameters)
			clf.fit(KZ_bar,source_Y)
			best_params= clf.best_params_
			print "BEST PARAM FOR THE ITERATION: ",best_params

			for params, mean_score, scores in clf.grid_scores_:
				print("%0.3f (+/-%0.03f) for %r"
				% (mean_score, scores.std() * 2, params))
	        	if mean_score>max_score:
	        		max_score=mean_score
	        		best_c=best_params['C']
	        		best_xi=x

	if tune:
		print "BEST PARAMETERS..."
		print "MAX SCORE =",max_score,";BEST C =",best_c,";BEST XI =",best_xi

		print "SOLVING QP TO GET LAMBDA (SPECTRAL KERNEL)"
		Lambda_vector=solveQP(X,PhiZ_bar,KZ,x)
		Lambda_matrix=np.matrix(np.diag(Lambda_vector))

		# print "COMPUTING DOMAIN INVARIANT KERNEL"
		# KA_bar=domain_invariant_kernel(PhiZ_bar,eigensystem,Lambda_matrix)	#DOMAIN INVARIANT KERNEL MATRIX (NO NEED TO CALCULATE)

		print "COMPUTING KZ_BAR TO TRAIN SVM"
		KZ_bar=(PhiZ_bar*Lambda_matrix)*PhiZ_bar.T 	#TRAIN SVM ON THIS

		print "COMPUTING KXZ_BAR TO TEST SVM"
		KXZ_bar=(PhiX*Lambda_matrix)*PhiZ_bar.T 	#TEST ON THIS

		KZ_bar=KZ_bar.tolist()

		print "SIMPLE SVM USING SCIKIT, C =",best_c
		svc = SVC(kernel='linear',C=c)
		svc.fit(source_X, source_Y)
		y_pred = svc.predict(target_X)
		print 'ACCURACY SCORE OF SIMPLE SVM: %0.3f' % accuracy_score(target_Y, y_pred)

		print "DOMAIN INVARIANT SVM USING SCIKIT, C =",c
		svc = SVC(kernel='precomputed',C=c)
		svc.fit(KZ_bar, source_Y)
		y_pred = svc.predict(KXZ_bar)
		print 'ACCURACY SCORE OF DOMAIN INVARIANT SVM: %0.3f' % accuracy_score(target_Y, y_pred)

main()

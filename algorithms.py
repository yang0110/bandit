import numpy as np
from scipy.linalg import sqrtm
import math
import time
import datetime
from util_functions import vectorize, matrixize
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from sklearn import cluster

class LinUCBUserStruct:
	def __init__(self, featureDimension, userID, lambda_, RankoneInverse):
		self.userID = userID
		self.A = lambda_*np.identity(n = featureDimension)# correlation matrix
		self.b = np.zeros(featureDimension)# bias vector
		self.AInv = np.linalg.inv(self.A)# inverse correlation matrix
		self.UserTheta = np.zeros(featureDimension)# user paramaters
		self.RankoneInverse = RankoneInverse


	def updateParameters(self, articlePicked, click):
		featureVector = articlePicked.featureVector
		self.A += np.outer(featureVector, featureVector)# update the correlation matrix
		self.b += featureVector*click# click is the payoff
		if self.RankoneInverse:
			temp = np.dot(self.AInv, featureVector)
			self.AInv = self.AInv - (np.outer(temp,temp))/(1.0+np.dot(np.transpose(featureVector),temp))
		else:
			self.AInv = np.linalg.inv(self.A)# update the inverse corrleation matrix
		self.UserTheta = np.dot(self.AInv, self.b)# update the user parameters
		
	def getTheta(self):
		return self.UserTheta
	
	def getA(self):
		return self.A

	def getProb(self, alpha, users, article):# calculate the UCB
		featureVector = article.featureVector
		mean = np.dot(self.getTheta(), featureVector)
		var = np.sqrt(np.dot(np.dot(featureVector, np.linalg.inv(self.getA())), featureVector))
		pta = mean + alpha * var
		return pta


class LinUCBAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, RankoneInverse = False):  # n is number of users
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, i, lambda_ ,RankoneInverse)) 
			# i si the user id, lambda_ is the tune parameter

		self.dimension = dimension
		self.alpha = alpha# the number of Standard deviation used to find UCB

		self.CanEstimateCoUserPreference = False
		self.CanEstimateUserPreference = True
		self.CanEstimateW = False

	def decide(self, pool_articles, userID):
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:# pick article with highest Prob
			x_pta = self.users[userID].getProb(self.alpha, self.users, x)
			if maxPTA < x_pta:
				articlePicked = x
				maxPTA = x_pta

		return articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click)
		
	def getCoTheta(self, userID):
		return self.users[userID].UserTheta

	def getTheta(self, userID):
		return self.users[userID].UserTheta

class LinUCB_SelectUserAlgorithm:
	def __init__(self, dimension, alpha, lambda_, n, RankoneInverse = False):  # n is number of users
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(LinUCBUserStruct(dimension, i, lambda_ , RankoneInverse)) 

		self.dimension = dimension
		self.alpha = alpha

	def decide(self, pool_articles, AllUsers):
		maxPTA = float('-inf')
		articlePicked = None
		userPicked = None
		AllUsers = list(np.random.permutation(AllUsers)) 
		for x in pool_articles:
			for user in AllUsers:
				x_pta = self.users[user.id].getProb(self.alpha, self.users, x)
				# pick article with highest Prob
				if maxPTA < x_pta:
					articlePicked = x
					userPicked = user
					maxPTA = x_pta

		return userPicked, articlePicked

	def updateParameters(self, articlePicked, click, userID):
		self.users[userID].updateParameters(articlePicked, click)
		
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta


####################################################
#############CLUB algorithm 
####################################################
class CLUBUserStruct(LinUCBUserStruct):
	def __init__(self,featureDimension, lambda_, userID):
		LinUCBUserStruct.__init__(self,featureDimension = featureDimension, userID = userID,lambda_= lambda_,RankoneInverse=False)
		self.reward = 0
		self.CA = self.A
		self.Cb = self.b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv, self.Cb)
		self.I = lambda_*np.identity(n = featureDimension)	
		self.counter = 0
		self.CBPrime = 0
		self.d = featureDimension

	def updateParameters(self, articlePicked_FeatureVector, click,alpha_2):
		#LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
		#alpha_2 = 1
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
		self.counter+=1
		self.CBPrime = alpha_2*np.sqrt(float(1+math.log10(1+self.counter))/float(1+self.counter))

	def updateParametersofClusters(self,clusters,userID,Graph,users):
		self.CA = self.I
		self.Cb = np.zeros(self.d)
		#print type(clusters)

		for i in range(len(clusters)):
			if clusters[i] == clusters[userID]:
				self.CA += float(Graph[userID,i])*(users[i].A - self.I)
				self.Cb += float(Graph[userID,i])*users[i].b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv,self.Cb)

	def getProb(self, alpha, article_FeatureVector,time):
		mean = np.dot(self.CTheta, article_FeatureVector)## reward
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
		pta = mean +  alpha * var*np.sqrt(math.log10(time+1))
		return pta

class CLUBAlgorithm(LinUCBAlgorithm):
	def __init__(self,dimension, alpha,lambda_, n, alpha_2, cluster_init="Erdos-Renyi", RankoneInverse=False):
		self.time = 0
		#N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
		self.users = []
		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(CLUBUserStruct(dimension,lambda_, i)) 

		self.dimension = dimension
		self.alpha = alpha
		self.alpha_2 = alpha_2
		if cluster_init=="Erdos-Renyi":
			p = 3*math.log(n)/n
			self.Graph = np.random.choice([0, 1], size=(n,n), p=[1-p, p])
			self.clusters = []
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)
		else:
			self.Graph = np.ones([n,n]) 
			self.clusters = []
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)

		self.CanEstimateCoUserPreference = False
		self.CanEstimateUserPreference = True
		self.CanEstimateW = False
			
	def decide(self, pool_articles, userID):

		self.users[userID].updateParametersofClusters(self.clusters, userID, self.Graph, self.users)
		maxPTA = float('-inf')
		articlePicked = None


		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, x.featureVector, self.time)


			if x_pta>maxPTA:
				articlePicked = x.id
				featureVectorPicked = x.featureVector
				picked = x
				maxPTA = x_pta
			else: 
				maxPTA=maxPTA

		self.time +=1

		return picked
	def getTheta(self, userID):
		return self.users[userID].UserTheta.T

	def getCoTheta(self, userID):
		return self.users.CoTheta.T[userID]

	def getW(self, userID):
		#print self.USERS.W
		return self.users.W.T[userID]

	def updateParameters(self, featureVector, click,userID):
		self.users[userID].updateParameters(featureVector, click, self.alpha_2)

	def updateGraphClusters(self,userID, binaryRatio):
		
		n = len(self.users)
		for j in range(n):
			ratio = float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2))/float(self.users[userID].CBPrime + self.users[j].CBPrime)
			#print float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2)),'R', ratio
			if ratio > 1:
				ratio = 0
			elif binaryRatio == 'True':
				ratio = 1
			elif binaryRatio == 'False':
				ratio = 1.0/math.exp(ratio)
			#print 'ratio',ratio
			self.Graph[userID][j] = ratio
			self.Graph[j][userID] = self.Graph[userID][j]

			# if j==n-1:
			# 	print 'self.Graph2',self.Graph

		N_components, component_list = connected_components(csr_matrix(self.Graph))


		self.clusters = component_list

		return N_components, self.Graph
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta

###############
######SCLUB Algorithm
################
class SCLUBUserStruct(LinUCBUserStruct):
	def __init__(self,featureDimension, lambda_, userID):
		LinUCBUserStruct.__init__(self,featureDimension = featureDimension, userID = userID,lambda_= lambda_,RankoneInverse=False)
		self.reward = 0
		self.CA = self.A
		self.Cb = self.b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv, self.Cb)
		self.I = lambda_*np.identity(n = featureDimension)	
		self.counter = 0
		self.CBPrime = 0
		self.d = featureDimension

	def updateParameters(self, articlePicked_FeatureVector, click,alpha_2):
		#LinUCBUserStruct.updateParameters(self, articlePicked_FeatureVector, click)
		#alpha_2 = 1
		self.A += np.outer(articlePicked_FeatureVector,articlePicked_FeatureVector)
		self.b += articlePicked_FeatureVector*click
		self.AInv = np.linalg.inv(self.A)
		self.UserTheta = np.dot(self.AInv, self.b)
		self.counter+=1
		self.CBPrime = alpha_2*np.sqrt(float(1+math.log10(1+self.counter))/float(1+self.counter))

	def updateParametersofClusters(self,clusters,userID,Graph,users):
		self.CA = self.I
		self.Cb = np.zeros(self.d)
		#print type(clusters)

		for i in range(len(clusters)):
			if clusters[i] == clusters[userID]:
				self.CA += float(Graph[userID,i])*(users[i].A - self.I)
				self.Cb += float(Graph[userID,i])*users[i].b
		self.CAInv = np.linalg.inv(self.CA)
		self.CTheta = np.dot(self.CAInv,self.Cb)

	def getProb(self, alpha, article_FeatureVector,time):
		mean = np.dot(self.CTheta, article_FeatureVector)## reward
		var = np.sqrt(np.dot(np.dot(article_FeatureVector, self.CAInv),  article_FeatureVector))
		pta = mean +  alpha * var*np.sqrt(math.log10(time+1))
		return pta

class SCLUBAlgorithm(LinUCBAlgorithm):
	def __init__(self,dimension,alpha,lambda_,n,alpha_2, cluster_init="Erdos-Renyi", RankoneInverse=False):
		self.time = 0
		#N_LinUCBAlgorithm.__init__(dimension = dimension, alpha=alpha,lambda_ = lambda_,n=n)
		self.users = []

		#algorithm have n users, each user has a user structure
		for i in range(n):
			self.users.append(SCLUBUserStruct(dimension,lambda_, i)) 

		self.dimension = dimension
		self.alpha = alpha
		self.alpha_2 = alpha_2
		if (cluster_init=="Erdos-Renyi"):
			p = 3*math.log(n)/n
			self.Graph = np.random.choice([0, 1], size=(n,n), p=[1-p, p])
			self.clusters = []
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)
		else:
			self.Graph = np.ones([n,n]) 
			# print 'self.Graph', self.Graph
			self.clusters = []
			g = csr_matrix(self.Graph)
			N_components, components = connected_components(g)

		self.CanEstimateCoUserPreference = False
		self.CanEstimateUserPreference = True
		self.CanEstimateW = False
			
	def decide(self,pool_articles,userID):
		self.users[userID].updateParametersofClusters(self.clusters, userID, self.Graph, self.users)
		maxPTA = float('-inf')
		articlePicked = None

		for x in pool_articles:
			x_pta = self.users[userID].getProb(self.alpha, x.featureVector,self.time)
			# print 'x_pta', x_pta
			# pick article with highest Prob
			if maxPTA < x_pta:
				articlePicked = x.id
				featureVectorPicked = x.featureVector
				picked = x
				maxPTA = x_pta
		self.time +=1

		return picked


	def getTheta(self, userID):
		return self.users[userID].UserTheta.T

	def getCoTheta(self, userID):
		return self.users.CoTheta.T[userID]

	def getW(self, userID):
		#print self.USERS.W
		return self.users.W.T[userID]

	def updateParameters(self, featureVector, click,userID):
		self.users[userID].updateParameters(featureVector, click, self.alpha_2)

	def updateGraphClusters(self,userID, binaryRatio):
		n = len(self.users)
		for j in range(n):
			ratio = float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2))/float(self.users[userID].CBPrime + self.users[j].CBPrime)
			#print float(np.linalg.norm(self.users[userID].UserTheta - self.users[j].UserTheta,2)),'R', ratio
			if ratio > 1:
				ratio = 0
			elif binaryRatio == 'True':
				ratio = 1
			elif binaryRatio == 'False':
				ratio = 1.0/math.exp(ratio)
			#print 'ratio',ratio
			self.Graph[userID][j] = ratio
			self.Graph[j][userID] = self.Graph[userID][j]

		# print 'self.Graph', self.Graph

		spectral_clustering=cluster.SpectralClustering(n_clusters=5, eigen_solver='amg', affinity="precomputed")

		component_list=spectral_clustering.fit_predict(csr_matrix(self.Graph))
		N_components=max(component_list)+1
	

		self.clusters = component_list
		return N_components, self.Graph
	def getLearntParameters(self, userID):
		return self.users[userID].UserTheta



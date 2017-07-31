import numpy as np
from operator import itemgetter      #for easiness in sorting and finding max and stuff
from matplotlib.pylab import *
import matplotlib
matplotlib.use('Agg')
from random import sample, choice
from scipy.sparse import csgraph 
import os

# local address to save simulated users, simulated articles, and results
from conf import sim_files_folder, result_folder, save_address
from util_functions import *
# algorithms 
from items import *
from users import *
from algorithms import *
####
from scipy.linalg import sqrtm
import math
import argparse
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
import networkx as nx 

class simulateOnlineData():
	def __init__(self, dimension, iterations, articles, users, 
					batchSize = 1000,
					noise = lambda : 0,
					matrixNoise = lambda:0,
					type_ = 'UniformTheta', 
					signature = '', 
					poolArticleSize = 10, 
					noiseLevel = 0, matrixNoiseLevel =0,
					epsilon = 1, Gepsilon = 1, sparseLevel=0):

		self.simulation_signature = signature
		self.type = type_

		self.dimension = dimension
		self.iterations = iterations
		self.noise = noise
		self.matrixNoise = matrixNoise

		self.articles = articles 
		self.users = users

		self.poolArticleSize = poolArticleSize
		self.batchSize = batchSize
		
		self.W = self.initializeW(sparseLevel,epsilon)

		W = self.W.copy()
		self.NoisyW = self.initializeNoisyW(W)
		self.GW = self.initializeGW(W,Gepsilon)
		NoisyW = self.NoisyW.copy()
		self.GNoisyW = self.initializeGW(NoisyW,Gepsilon)
		self.noiseLevel = noiseLevel
		self.matrixNoiseLevel = matrixNoiseLevel

		self.sparseLevel = sparseLevel
	def constructAdjMatrix(self):
		n = len(self.users)	

		G = np.zeros(shape = (n, n))
		for ui in self.users:
			sSim = 0
			for uj in self.users:
				sim = np.dot(ui.theta, uj.theta)
 				if ui.id == uj.id:
 					sim *= 1.0
				G[ui.id][uj.id] = sim
				sSim += sim
				
			G[ui.id] /= sSim
			'''
			for i in range(n):
				print '%.3f' % G[ui.id][i],
			print ''
			'''
		#G = 1.0/n*np.ones(shape = (n, n))
		G = np.identity(n)
		return G
    
    # top m users
	def constructSparseMatrix(self, m):
		n = len(self.users)	

		G = np.zeros(shape = (n, n))
		for ui in self.users:
			sSim = 0
			for uj in self.users:
				sim = np.dot(ui.theta, uj.theta)
 				if ui.id == uj.id:
 					sim *= 1.0
				G[ui.id][uj.id] = sim
				sSim += sim		
			G[ui.id] /= sSim
		for ui in self.users:
			similarity = sorted(G[ui.id], reverse=True)
			threshold = similarity[m]				
			
			# trim the graph
			for i in range(n):
				if G[ui.id][i] <= threshold:
					G[ui.id][i] = 0;
			G[ui.id] /= sum(G[ui.id])

			'''
			for i in range(n):
				print '%.3f' % G[ui.id][i],
			print ''
			'''
		return G

		

	# create user connectivity graph
	def initializeW(self, sparseLevel, epsilon):	
		n = len(self.users)	
		if sparseLevel >=n or sparseLevel<=0:
 			W = self.constructAdjMatrix()
 		else:
 			W = self.constructSparseMatrix(sparseLevel)   # sparse matrix top m users 
		return W.T

	def initializeNoisyW(self,W):
		NoisyW = W.copy()
		#print 'NoisyW', NoisyW
		for i in range(W.shape[0]):
			for j in range(W.shape[1]):
				NoisyW[i][j] = W[i][j] + self.matrixNoise()
				if NoisyW[i][j] < 0:
					NoisyW[i][j] = 0
			NoisyW[i] /= sum(NoisyW[i]) 
		#NoisyW = np.random.random((W.shape[0], W.shape[1]))  #test random ini
		# print 'NoisyW.T', NoisyW.T

		return NoisyW.T

	def initializeGW(self,G, Gepsilon):
		n = len(self.users)	
 		L = csgraph.laplacian(G, normed = False)
 		I = np.identity(n = G.shape[0]) 
 		GW = I + Gepsilon*L  # W is a double stochastic matrix
		return GW.T

	def getW(self):
		return self.W
	def getNoisyW(self):
		return self.NoisyW
	def getGW(self):
		return self.GW
	def getGNoisyW(self):
		return self.GNoisyW

	def getTheta(self):
		Theta = np.zeros(shape = (self.dimension, len(self.users)))
		for i in range(len(self.users)):
			Theta.T[i] = self.users[i].theta
		return Theta
	def generateUserFeature(self,W):
		svd = TruncatedSVD(n_components=5)
		result = svd.fit(W).transform(W)
		return result

	def batchRecord(self, iter_):
		pass
		# print "Iteration %d"%iter_, "Pool", len(self.articlePool)," Elapsed time", datetime.datetime.now() - self.startTime

	def regulateArticlePool(self):
		# Randomly generate articles
		self.articlePool = sample(self.articles, self.poolArticleSize)   
		# generate articles 

	def CoTheta(self):
		for ui in self.users:
			ui.CoTheta = np.zeros(self.dimension)
			# print 'ui.CoTheta', ui.CoTheta
			for uj in self.users:
				# print 'self.w', self.W[uj.id][ui.id]
				# print 'uj.theta', uj.theta
				ui.CoTheta += self.W[uj.id][ui.id] * np.asarray(uj.theta)

			# print 'Users', ui.id, 'CoTheta', ui.CoTheta

	def getReward(self, user, Article):
		reward = np.dot(user.CoTheta, Article.featureVector)
		# print Article.id, reward
		return reward

	def GetOptimalReward(self, user, articlePool):		
		maxReward = sys.float_info.min
		optimalArticle = None
		for x in articlePool:	 
			reward = self.getReward(user, x)
			if reward > maxReward:
				maxReward = reward
				optimalArticle = x
		return maxReward, optimalArticle
	
	def getL2Diff(self, x, y):
		return np.linalg.norm(x-y) # L2 norm

	def runAlgorithms(self, algorithms):
		# get cotheta for each user
		self.startTime = datetime.datetime.now()
		timeRun = datetime.datetime.now().strftime('_%m_%d') 
		timeRun_Save = datetime.datetime.now().strftime('_%m_%d_%H_%M') 

		filenameWriteRegret = os.path.join(save_address, 'AccRegret' + timeRun_Save + '.csv')
		filenameWritePara = os.path.join(save_address, 'ParameterEstimation' + timeRun_Save + '.csv')
		for alg_name, alg in algorithms.items():
			fileSig = 'New_' +str(alg_name) + '_UserNum'+ str(len(self.users)) + '_Sparsity' + str(self.sparseLevel) +'_Noise'+str(self.noiseLevel)+ '_matrixNoise'+str(self.matrixNoiseLevel)
		filenameWriteResult = os.path.join(save_address, fileSig + timeRun + '.csv')

		self.CoTheta()
		self.startTime = datetime.datetime.now()

		tim_ = []
		BatchAverageRegret = {}
		AccRegret = {}
		ThetaDiffList = {}
		CoThetaDiffList = {}
		WDiffList = {}
		
		ThetaDiffList_user = {}
		CoThetaDiffList_user = {}
		WDiffList_user = {}

		n_cluster={}
		n_edges={}

		
		# Initialization
		for alg_name, alg in algorithms.items():
			BatchAverageRegret[alg_name] = []
			AccRegret[alg_name] = {}

			n_cluster[alg_name]=[] # the number of clusters
			n_edges[alg_name]=[]

	


			if alg.CanEstimateCoUserPreference:
				CoThetaDiffList[alg_name] = []
			if alg.CanEstimateUserPreference:
				ThetaDiffList[alg_name] = []
			if alg.CanEstimateW:
				WDiffList[alg_name] = []


			for i in range(len(self.users)):
				AccRegret[alg_name][i] = []


		
		userSize = len(self.users)
		
		# with open(filenameWriteRegret, 'a+') as f:
		# 	f.write('Time(Iteration)')
		# 	f.write(',' + ','.join( [str(alg_name) for alg_name in algorithms.iterkeys()]))
		# 	f.write('\n')
		# with open(filenameWritePara, 'a+') as f:
		# 	f.write('Time(Iteration)')
		# 	f.write(',' + ','.join( [str(alg_name)+'CoTheta' for alg_name in algorithms.iterkeys()]))
		# 	f.write(','+ ','.join([str(alg_name)+'Theta' for alg_name in ThetaDiffList.iterkeys()]))
		# 	f.write(','+ ','.join([str(alg_name)+'W' for alg_name in WDiffList.iterkeys()]))
		# 	f.write('\n')
		
		
		# Loop begin
		for iter_ in range(self.iterations):
			# prepare to record theta estimation error
			for alg_name, alg in algorithms.items():
				if alg.CanEstimateCoUserPreference:
					CoThetaDiffList_user[alg_name] = []
				if alg.CanEstimateUserPreference:
					ThetaDiffList_user[alg_name] = []
				if alg.CanEstimateW:
					WDiffList_user[alg_name] = []
			#self.regulateArticlePool() # select random articles	
			u=choice(self.users)

			self.regulateArticlePool() # select random articles

			noise = self.noise()
			#get optimal reward for user x at time t
			temp, optimalA = self.GetOptimalReward(u, self.articlePool)
			OptimalReward = temp + noise
						
			for alg_name, alg in algorithms.items():
				pickedArticle = alg.decide(self.articlePool, u.id)
				reward = self.getReward(u, pickedArticle)  + noise
				if alg_name =='CLUB':
					alg.updateParameters(pickedArticle.featureVector, reward, u.id)
					club_n_components, club_graph= alg.updateGraphClusters(u.id,'True')
					nx_club_graph=nx.Graph(club_graph)
					club_n_components=nx.number_connected_components(nx_club_graph)
					club_n_edges=nx.number_of_edges(nx_club_graph)
					print('iter__:', iter_)
					print('club_n_edges:', club_n_edges)
					print('club_n_components:', club_n_components)
					if iter_==0 or iter_==self.iterations/2 or iter_==self.iterations-1:
						# nx.draw_networkx(nx_club_graph, pos=nx.spring_layout(nx_club_graph), with_labels=False, node_size=35)
						nx.draw_random(nx_club_graph, with_labels=True)
						plt.draw()
						plt.title('club_graph_n=%s'%(iter_))
						plt.savefig('club_graph_n=%s'%(iter_)+str(timeRun_Save)+'.png')

						plt.show()

					n_cluster[alg_name].append(club_n_components)
					n_edges[alg_name].append(club_n_edges)

					# print 'CLUB_n_component', n_components
				elif alg_name =='SCLUB':
					alg.updateParameters(pickedArticle.featureVector, reward, u.id)
					sclub_n_components, sclub_graph= alg.updateGraphClusters(u.id,'True')
					nx_sclub_graph=nx.Graph(sclub_graph)
					# sclub_n_components=nx.number_connected_components(nx_sclub_graph)
					sclub_n_edges=nx.number_of_edges(nx_sclub_graph)
					print('sclub_n_components', sclub_n_components)
					print('sclub_n_edges:', sclub_n_edges)
					if iter_==0 or iter_==self.iterations/2 or iter_==self.iterations-1:
						# nx.draw_networkx(nx_sclub_graph, pos=nx.spring_layout(nx_sclub_graph), with_labels=True, node_size=35)
						nx.draw_random(nx_sclub_graph)
						plt.draw()
						plt.title('sclub_graph_n=%s'%(iter_))
						plt.savefig('sclub_graph_n=%s'%(iter_)+str(timeRun_Save)+'.png')
						plt.show()

					n_cluster[alg_name].append(sclub_n_components)
					n_edges[alg_name].append(sclub_n_edges)

					# print 'SCLUB_n_component', n_components

				else:
					alg.updateParameters(pickedArticle, reward, u.id)

				regret = OptimalReward - reward	

				AccRegret[alg_name][u.id].append(regret)


				# every algorithm will estimate co-theta
				if  alg.CanEstimateCoUserPreference:
					CoThetaDiffList_user[alg_name] += [self.getL2Diff(u.CoTheta, alg.getCoTheta(u.id))]
				if alg.CanEstimateUserPreference:
					ThetaDiffList_user[alg_name] += [self.getL2Diff(u.theta, alg.getTheta(u.id))]
				if alg.CanEstimateW:
					WDiffList_user[alg_name] +=  [self.getL2Diff(self.W.T, alg.getW(u.id))]
					#WDiffList_user[alg_name] +=  [self.getL2Diff(self.W.T[u.id], alg.getW(u.id))]
					'''
					print 'w',self.W
					print  'get', alg.getW(u.id)
					print self.getL2Diff(self.W.T, alg.getW(u.id)), WDiffList_user[alg_name]	
					'''	

			for alg_name, alg in algorithms.items():
				if alg.CanEstimateCoUserPreference:
					CoThetaDiffList[alg_name] += [sum(CoThetaDiffList_user[alg_name])/userSize]
				if alg.CanEstimateUserPreference:
					ThetaDiffList[alg_name] += [sum(ThetaDiffList_user[alg_name])/userSize]
				if alg.CanEstimateW:
					WDiffList[alg_name] += [sum(WDiffList_user[alg_name])/userSize]
					#WDiffList[alg_name] += [WDiffList_user[alg_name]]
				
			if iter_%self.batchSize == 0:
				self.batchRecord(iter_)
				tim_.append(iter_)
				for alg_name in algorithms.iterkeys():
					TotalAccRegret = sum(sum (u) for u in AccRegret[alg_name].itervalues())
					BatchAverageRegret[alg_name].append(TotalAccRegret)
					

				with open(filenameWriteRegret, 'a+') as f:
					f.write(str(iter_))
					f.write(',' + ','.join([str(BatchAverageRegret[alg_name][-1]) for alg_name in algorithms.iterkeys()]))
					f.write('\n')
				

		# print 'c_cluster_length', len(n_cluster['CLUB'])
		# print 'n_cluster', max(n_cluster['CLUB'])
		# print 'n_cluster', n_cluster['CLUB']

		f, axa = plt.subplots(3, sharex=True)
		for alg_name, alg in algorithms.items():
			axa[0].plot(tim_, BatchAverageRegret[alg_name],label = alg_name)
			# with open(filenameWriteResult, 'a+') as f:
			# 	f.write(str(alg_name)+ ','+ str( BatchAverageRegret[alg_name][-1]))
			# 	f.write('\n')

		axa[0].legend(loc='lower right',prop={'size':9})
		axa[0].set_xlabel("Iteration")
		axa[0].set_ylabel("Regret")
		axa[0].set_title("Accumulated Regret")

		for alg_name, alg in algorithms.items():
			axa[1].plot(n_cluster[alg_name],label = alg_name)

		axa[1].legend(loc='lower right',prop={'size':12})
		axa[1].set_xlabel("Iteration", fontsize=15)
		axa[1].set_ylabel("No. of Clusters", fontsize=15)
		# if max(n_cluster['CLUB'])>max(n_cluster['SCLUB']):
		axa[1].set_ylim([0,max(n_cluster['CLUB'])+1])
		# else: 
		# 	axa[1].set_ylim([0, max(n_cluster['SCLUB'])+1])
		for alg_name, alg in algorithms.items():
			axa[2].plot(n_edges[alg_name],label = alg_name)

		axa[2].legend(loc='lower right',prop={'size':12})
		axa[2].set_xlabel("Iteration", fontsize=15)
		axa[2].set_ylabel("No. of Edges", fontsize=15)
		# if max(n_cluster['CLUB'])>max(n_cluster['SCLUB']):
		axa[2].set_ylim([0,max(n_edges['CLUB'])+1])


		# time = range(self.iterations)
		
		# for alg_name, alg in algorithms.items():
		# 	if alg.CanEstimateCoUserPreference:
		# 		axa[2].plot(time, CoThetaDiffList[alg_name], label = alg_name + '_CoTheta')
		# 	#plt.lines[-1].set_linewidth(1.5)
		# 	if alg.CanEstimateUserPreference:
		# 		axa[2].plot(time, ThetaDiffList[alg_name], label = alg_name + '_Theta')
		# 	if alg.CanEstimateW:
		# 		axa[2].plot(time, WDiffList[alg_name], label = alg_name + '_W')

		# axa[2].legend(loc='upper right',prop={'size':6})
		# axa[2].set_xlabel("Iteration")
		# axa[2].set_ylabel("Parameter estimation error")
		# axa[2].set_yscale('log')

		plt.savefig('SimulationResults/Regret' + str(timeRun_Save )+'.png')
		plt.show()

#############################################################################
if __name__ == '__main__':
	iterations =10000
	NoiseScale = .01
	matrixNoise = 0.001

	alpha  = 0.2
	lambda_ = 0.1   # Initialize A
	epsilon = 0 # initialize W
	eta_ = 0.1

	dimension = 25
	n_articles = 100
	ArticleGroups = 2

	n_users =20
	UserGroups = 5

	poolSize = 10
	batchSize = 10

	# Parameters for GOBLin
	G_alpha = alpha
	G_lambda_ = lambda_
	Gepsilon = 1
	# Epsilon_greedy parameter
	sparseLevel=0
 
	eGreedy = 0.3
	CLUB_alpha_2 = 2.0

	#####
	algName = "ALL"
	n_users = n_users
	sparseLevel = int(sparseLevel)
	NoiseScale = float(NoiseScale)
	matrixNoise = float(matrixNoise)
	RankoneInverse =True



	
	userFilename = os.path.join(sim_files_folder, "users_"+str(n_users)+"+dim-"+str(dimension)+ "Ugroups" + str(UserGroups)+".json")
	

	UM = UserManager(dimension, n_users, UserGroups = UserGroups, thetaFunc=featureUniform, argv={'l2_limit':1})
	users = UM.simulateThetafromUsers()
	UM.saveUsers(users, userFilename, force = True)
	# users = UM.loadUsers(userFilename)

	articlesFilename = os.path.join(sim_files_folder, "articles_"+str(n_articles)+"+dim"+str(dimension) + "Agroups" + str(ArticleGroups)+".json")

	AM = ArticleManager(dimension, n_articles=n_articles, ArticleGroups = ArticleGroups,
			FeatureFunc=featureUniform,  argv={'l2_limit':1})
	articles = AM.simulateArticlePool()
	AM.saveArticles(articles, articlesFilename, force=True)
	# articles = AM.loadArticles(articlesFilename)

	simExperiment = simulateOnlineData(dimension  = dimension,
						iterations = iterations,
						articles=articles,
						users = users,		
						noise = lambda : np.random.normal(scale = NoiseScale),
						matrixNoise = lambda : np.random.normal(scale = matrixNoise),
						batchSize = batchSize,
						type_ = "UniformTheta", 
						signature = AM.signature,
						poolArticleSize = poolSize, noiseLevel = NoiseScale, matrixNoiseLevel= matrixNoise, epsilon = epsilon, Gepsilon =Gepsilon, sparseLevel= sparseLevel)
	# print "Starting for ", simExperiment.simulation_signature
	algorithms = {}
	
	if algName == 'LinUCB':
		algorithms['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users, RankoneInverse = RankoneInverse)
	
	if algName =='CLUB':
		algorithms['CLUB'] = CLUBAlgorithm(dimension =dimension, alpha = alpha, lambda_ = lambda_, n = n_users, alpha_2 = CLUB_alpha_2, cluster_init = 'Erdos-Renyi')	

	if algName =='ALL':

		algorithms['LinUCB'] = LinUCBAlgorithm(dimension = dimension, alpha = alpha, lambda_ = lambda_, n = n_users, RankoneInverse = RankoneInverse)

		algorithms['CLUB'] = CLUBAlgorithm(dimension =dimension,alpha = alpha, lambda_ = lambda_, n = n_users, alpha_2 = CLUB_alpha_2, cluster_init = "Erdos-Renyi")

		# algorithms['SCLUB'] = SCLUBAlgorithm(dimension =dimension,alpha = alpha, lambda_ = lambda_, n = n_users, alpha_2 = CLUB_alpha_2, cluster_init = "complete")	

	
		
	simExperiment.runAlgorithms(algorithms)



	

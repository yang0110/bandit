import numpy as np 
from util_functions import featureUniform, gaussianFeature, fileOverWriteWarning
import json
from random import choice, randint

class User():
	def __init__(self, id, theta = None, CoTheta = None):
		self.id = id
		self.theta = theta# user parameters
		self.CoTheta = CoTheta
		self.uncertainty = 0.0
	def updateUncertainty(uncertainty):
		self.uncertainty = uncertainty


class UserManager():
	def __init__(self, dimension, userNum,  UserGroups, thetaFunc, argv = None):
		self.dimension = dimension# length of item vectors equals the lenth of user parameters
		self.thetaFunc = thetaFunc# the random distribution of user parameters at initial setting
		self.userNum = userNum# the number of users
		self.UserGroups = UserGroups# the user clusters number 
		self.argv = argv# the input from keyboard 
		self.signature = "A-"+"+PA"+"+TF-"+self.thetaFunc.__name__

	def saveUsers(self, users, filename, force = False):
		fileOverWriteWarning(filename, force=True)
		with open(filename, 'w') as f:
			for i in range(len(users)):
				# print users[i].theta
				f.write(json.dumps((users[i].id, users[i].theta.tolist())) + '\n')
				
	def loadUsers(self, filename):
		users = []
		with open(filename, 'r') as f:
			for line in f:
				id, theta = json.loads(line)
				users.append(User(id, np.array(theta)))
		return users

	def generateMasks(self):
		mask = {}
		for i in range(self.UserGroups):
			mask[i] = np.random.randint(2, size = self.dimension)
		return mask
		# the mask for each user clusters

	def simulateThetafromUsers(self):
		usersids = {}
		users = []
		mask = self.generateMasks()
		if self.UserGroups >1:
			for i in range(self.UserGroups):
				usersids[i] = range(self.userNum*i/self.UserGroups, (self.userNum*(i+1))/self.UserGroups)

				for key in usersids[i]:
					thetaVector = np.multiply(self.thetaFunc(self.dimension, argv = self.argv), mask[i])
					# l2_norm = np.linalg.norm(thetaVector, ord =2)# return the vector norm 
					# users.append(User(key, thetaVector/l2_norm))
					users.append(User(key, thetaVector))
		else:
			for i in range(self.userNum):
				thetaVector = self.thetaFunc(self.dimension, argv = self.argv)
				# theta_l2_norm = np.linalg.norm(thetaVector, ord =2)
				# users.append(User(i, thetaVector/theta_l2_norm))
				users.append(User(i, thetaVector))

		return users


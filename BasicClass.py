import pickle
import copy

class BasicClass:

	def save(self,filename):
		with open(filename,'wb') as f:
			pickle.dump(self,f)
		print('saved the class to: ' + filename)

	def load(self,filename):
		# test if class data file exist
		with open(filename,'rb') as f:
			temp = pickle.load(f)
		self.__dict__.update(temp.__dict__)

	def copy(self):
		return copy.deepcopy(self)


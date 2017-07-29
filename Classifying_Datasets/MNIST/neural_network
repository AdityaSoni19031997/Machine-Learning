'''
Using ANN Classifying Handwritten Digits
My first attempt to build a neural network....for evaluating the famous MNIST Digit Classification"
-Aditya Soni

'''
#Import Statements

import numpy as np  # for fast calculations
import matplotlib.pyplot as plt # for plotiing
import scipy.special # for sigmoid function
from sklearn.metrics import confusion_matrix

k = list()
k_ =list()
class NeuralNetworks:

	# initialising nn
	def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate ):
		self.inodes = inputnodes
		self.hnodes = hiddennodes
		self.onodes = outputnodes
		self.lr = learningrate
		#weights
		self.wih = np.random.normal(0.0 , pow(self.hnodes , -0.5),(self.hnodes, self.inodes))
		self.who = np.random.normal(0.0 , pow(self.onodes , -0.5),(self.onodes , self.hnodes))
		self.activation_function = lambda x: scipy.special.expit(x)
		pass


#train the ANN
#the subtle part....
# it is quite similar to the query function
	def train(self, input_list, target_list):
		
		#converting input to 2d array
		inputs = np.array(input_list , ndmin = 2).T 
		targets = np.array(target_list , ndmin =2).T

		#calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih , inputs)

		#calculating o/p from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		#calculating signals into final layer
		final_inputs = np.dot(self.who , hidden_outputs)

		# calculating final o/s value
		final_outputs = self.activation_function(final_inputs)

		#error is target - actual value
		output_errors = targets - final_outputs

		#applying backpropagation logic now (state of art of ANN in ML)
		# hidden layer error is the output_errors, split by weights, recombined at hidden nodes
		hidden_errors = np.dot(self.who.T, output_errors)

		#updating the weights for the link between hidden and output layers
		# the formula we apply is eta*y(1-y)*o/p
		self.who += self.lr*np.dot((output_errors * final_outputs * (1 - final_outputs)), np.transpose(hidden_outputs))
		self.wih += self.lr*np.dot((hidden_errors * hidden_outputs *(1 - hidden_outputs)), np.transpose(inputs))

		pass


	def query(self, input_list):
		
		#converting input to 2d array
		inputs = np.array(input_list , ndmin = 2).T 

		#calculate signals into hidden layer
		hidden_inputs = np.dot(self.wih , inputs)
		
		#calculating o/p from hidden layer
		hidden_outputs = self.activation_function(hidden_inputs)

		#calculating signals into final layer
		final_inputs = np.dot(self.who , hidden_outputs)

		# calculating final o/s value
		final_outputs = self.activation_function(final_inputs)

		return final_outputs

input_nodes = 784 		#28*28
hidden_nodes = 300
output_nodes = 10
learning_rate = 0.15

#creating an instance of the class.....
n = NeuralNetworks(input_nodes , hidden_nodes , output_nodes , learning_rate)

#loading the dataset.......
train_data_f = open("C:\Python27\mnist\mnist_train.csv" , 'r')
train_data_all = train_data_f.readlines()
train_data_f.close()

for rec in train_data_all:
	all_val = rec.split(',')
	inputs = (np.asfarray(all_val[1:])/255.0*.99) + 0.01
	targets = np.zeros(output_nodes) + 0.01
	targets[int(all_val[0])] = 0.99
	n.train(inputs , targets)


test_data_f = open("C:\Python27\mnist\mnist_test.csv" , 'r')
test_data_all = test_data_f.readlines()
test_data_f.close()



for rec in test_data_all:
	all_val = rec.split(',')
	p = (n.query((np.asfarray(all_val[1:])/255*.99)+0.01))
	#print max(list(p)) , list(p).index(max(list(p)))
	k.append(list(p).index(max(list(p))))
	k_.append(int(all_val[0]))
print confusion_matrix(k_ , k)
print np.trace(np.asarray(confusion_matrix(k_ , k)))/10000.0




#1st test
all_values = test_data_all[0].split(',')
print(all_values[0])
img_array = np.asfarray(all_values[1:]).reshape(28,28)
plt.imshow(img_array , cmap='Greys',interpolation="None")
plt.show()
print n.query((np.asfarray(all_values[1:])/255*.99)+0.01)

#2nd test
all_values = test_data_all[99].split(',')
print(all_values[0])
img_array = np.asfarray(all_values[1:]).reshape(28,28)
plt.imshow(img_array , cmap='Greys',interpolation="None")
plt.show()
print n.query((np.asfarray(all_values[1:])/255*.99)+0.01)

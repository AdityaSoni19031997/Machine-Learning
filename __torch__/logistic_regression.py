import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
pts = 50

x_vals = np.random.rand(50)
x_train = np.asarray(x_vals,dtype=np.float32).reshape(-1,1)
m = 1
alpha = np.random.rand(1)
beta = np.random.rand(1)
y_correct = np.asarray([2*i+m for i in x_vals], dtype=np.float32).reshape(-1,1)

# be sure with the dtypes as torch is highly sensitive to it
# adding anotheer dimension to our variables with np.reshape(-1,1) i.e matrix like structure


'''
Always we need to do the following things-:
1. Create a class (nearly always in Torch)
2. Write your forward
3. Decide the epochs, i/p, o/p, loss etc..LinearRegressionModel
Its not always so easy....
'''

class LogisticRegressionModel(nn.Module):

	def __init__(self, input_dim, output_dim):

		super(LogisticRegressionModel, self).__init__()
		self.linear = nn.Linear(input_dim, output_dim)
		criterion = nn.CrossEntropyLoss()
		learning_rate = 0.001
		optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

	def forward(self, x):

		out = self.linear(x)
		return out

class LinearRegressionModel(nn.Module):

	def __init__(self, input_dim, output_dim):

		super(LinearRegressionModel, self).__init__() # its just like c++ super in inheritance bringing imp things to out class
		self.linear = nn.Linear(input_dim, output_dim) # nn.linear is defined in nn.Module

	def forward(self, x):# here its simple as we want our model to predict the output as what it thinks is correct

		out = self.linear(x)
		return out

input_dim = 1
output_dim = 1

model = LinearRegressionModel(input_dim,output_dim)# create our model just as we do in Scikit-Learn / C / C++//
model_1 = LogisticRegressionModel(input_dim,output_dim)

criterion = nn.MSELoss()# Mean Squared Loss
l_rate = 0.01
optimiser = torch.optim.SGD(model.parameters(), lr = l_rate) #Stochastic Gradient Descent

epochs = 1000

for epoch in range(epochs):

	epoch +=1
	inputs = Variable(torch.from_numpy(x_train))
	labels = Variable(torch.from_numpy(y_correct))

	#clear grads
	optimiser.zero_grad()
	#forward to get predicted values
	outputs = model.forward(inputs)
	loss = criterion(outputs, labels)
	loss.backward()# back props
	optimiser.step()# update the parameters
	print('epoch {}, loss {}'.format(epoch,loss.data[0]))


predicted = model.forward(Variable(torch.from_numpy(x_train))).data.numpy()

plt.plot(x_train, y_correct, 'go', label = 'from data', alpha = .5)
plt.plot(x_train, predicted, label = 'prediction', alpha = 0.5)
plt.legend()
plt.show()
print(model.state_dict())
''' this will show 
OrderedDict([('linear.weight', 
OrderedDict([('linear.weight', 
 1.9940
[torch.FloatTensor of size 1x1]
), ('linear.bias', 
 1.0029
[torch.FloatTensor of size 1]
)])
'''

import numpy as np
import os


class Activation(object):

	def __init__(self):
		self.state = None

	def __call__(self, x):
		return self.forward(x)

	def forward(self, x):
		raise NotImplemented

	def derivative(self):
		raise NotImplemented

class Identity(Activation):

	def __init__(self):
		super(Identity, self).__init__()

	def forward(self, x):
		self.state = x
		return x

	def derivative(self):
		return 1.0

class Sigmoid(Activation):

	def __init__(self):
		super(Sigmoid, self).__init__()

	def forward(self, x):

		self.inp = x
		self.state = 1/(1 + np.exp(-1*x))
		return 1/(1 + np.exp(-1*x))

	def derivative(self):

		# Maybe something we need later in here...
		self.prev = self.state
		#update the state value now as we have cached the previous value. (will be useful for computing grads later)
		self.state = self.state*(1 - self.state)
		return self.state

class Tanh(Activation):

	"""
	Tanh non-linearity
	"""

	# This one's all you!

	def __init__(self):
		super(Tanh, self).__init__()

	def forward(self, x):

		self.input = x
		self.state = np.tanh(x)
		return self.state

	def derivative(self):

		self.prev = self.state
		self.state = 1 - np.power(self.state,2)
		return self.state

class ReLU(Activation):

	"""
	ReLU non-linearity
	"""

	def __init__(self):
		super(ReLU, self).__init__()

	def forward(self, x):

		self.storage = x
		self.state = x * (x > 0)
		return self.state
		
	def derivative(self):
		self.prev = self.state
		self.state = 1.0 * (self.prev > 0)
		return self.state

# Ok now things get decidedly more interesting. The following Criterion class
# will be used again as the basis for a number of loss functions (which are in the
# form of classes so that they can be exchanged easily (it's how PyTorch and other
# ML libraries do it))


class Criterion(object):

	"""
	Interface for loss functions.
	"""

	# Nothing needs done to this class, it's used by the following Criterion classes

	def __init__(self):
		self.logits = None
		self.labels = None
		self.loss = None

	def __call__(self, x, y):
		return self.forward(x, y)

	def forward(self, x, y):
		raise NotImplemented

	def derivative(self):
		raise NotImplemented


class SoftmaxCrossEntropy(Criterion):

	def __init__(self):
		super(SoftmaxCrossEntropy, self).__init__()

		self.sm = None

	def forward(self, x, y):

		self.logits = x
		self.labels = y
		temp = np.sum(np.exp(x), axis = 1).T
		self.sm = (np.exp(x).T/(temp)).T - y
		return -1 * np.sum(y * np.log(np.exp(x).T/(temp)).T, axis = 1)

	def derivative(self):

		return self.sm

class BatchNorm(object):

	def __init__(self, fan_in, alpha=0.9):

		# You shouldn't need to edit anything in init

		self.alpha = alpha
		self.eps = 1e-8
		self.x = None
		self.norm = None
		self.out = None

		# The following attributes will be tested
		self.var = np.ones((1, fan_in))
		self.mean = np.zeros((1, fan_in))

		self.gamma = np.ones((1, fan_in))
		self.dgamma = np.zeros((1, fan_in))

		self.beta = np.zeros((1, fan_in))
		self.dbeta = np.zeros((1, fan_in))

		# inference parameters
		self.running_mean = np.zeros((1, fan_in))
		self.running_var = np.ones((1, fan_in))

	def __call__(self, x, eval=False):
		return self.forward(x, eval)

	def forward(self, x, eval=False):

		# if eval:
		#    # ???

		self.x = x

		# self.mean = # ???
		# self.var = # ???
		# self.norm = # ???
		# self.out = # ???

		# update running batch statistics
		# self.running_mean = # ???
		# self.running_var = # ???

		# ...

		raise NotImplemented

	def backward(self, delta):

		raise NotImplemented


# These are both easy one-liners, don't over-think them
def random_normal_weight_init(d0, d1):
	return np.random.randn(d0,d1)


def zeros_bias_init(d):
	return np.zeros(d)


class MLP(object):

	"""
	A simple multilayer perceptron
	"""

	def __init__(self, input_size, output_size, hiddens, activations, weight_init_fn, bias_init_fn, criterion, lr, momentum=0.0, num_bn_layers=0):

		# Don't change this -->
		self.train_mode = True
		self.num_bn_layers = num_bn_layers
		self.bn = num_bn_layers > 0
		self.nlayers = len(hiddens) + 1
		self.input_size = input_size
		self.output_size = output_size
		self.activations = activations
		self.criterion = criterion
		self.lr = lr
		self.momentum = momentum
		# <---------------------

		# Don't change the name of the following class attributes,
		# the autograder will check against these attributes. But you will need to change
		# the values in order to initialize them correctly
		self.W = None
		self.dW = None
		self.b = None
		self.db = None
		# HINT: self.foo = [ bar(???) for ?? in ? ]

		# if batch norm, add batch norm parameters
		if self.bn:
			self.bn_layers = None

		# Feel free to add any other attributes useful to your implementation (input, output, ...)

	def forward(self, x):
		raise NotImplemented

	def zero_grads(self):
		raise NotImplemented

	def step(self):
		raise NotImplemented

	def backward(self, labels):
		raise NotImplemented

	def __call__(self, x):
		return self.forward(x)

	def train(self):
		self.train_mode = True

	def eval(self):
		self.train_mode = False


def get_training_stats(mlp, dset, nepochs, batch_size):

	train, val, test = dset
	trainx, trainy = train
	valx, valy = val
	testx, testy = test

	idxs = np.arange(len(trainx))

	training_losses = []
	training_errors = []
	validation_losses = []
	validation_errors = []

	# Setup ...

	for e in range(nepochs):

		# Per epoch setup ...

		for b in range(0, len(trainx), batch_size):

			pass  # Remove this line when you start implementing this
			# Train ...

		for b in range(0, len(valx), batch_size):

			pass  # Remove this line when you start implementing this
			# Val ...

		# Accumulate data...

	# Cleanup ...

	for b in range(0, len(testx), batch_size):

		pass  # Remove this line when you start implementing this
		# Test ...

	# Return results ...

	# return (training_losses, training_errors, validation_losses, validation_errors)

	raise NotImplemented

--[[

Torch Implementation of 'A Thorough Examination of the CNN/Daily Mail Reading Comprehension Task'

]]--

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'rnn'
require 'optim'
require 'xlua'
require 'sys'
require 'lmdb'

cmd = torch.CmdLine()
cmd:option('-db_name', 'mydb', [[lmdb database name containing the tensors]])
cmd:option('-seed', 123, [[seed for the random generator]])
cmd:option('-dim', 100, [[dimensionality of word embeddings]])
cmd:option('-hid_size', 128, [[GRU's hidden layer size]])
cmd:option('-num_epochs', 30, [[number of full passes through the training data]])
cmd:option('-lr', 0.1, [[sgd mini-batch size]])
cmd:option('-grad_clip', 10, [[clip gradients at this value]])
cmd:option('-dropout', 0.2, [[dropout for regularization, used before the prediction layer. 0 = no dropout]])
params = cmd:parse(arg)

torch.manualSeed(params.seed)
params.optim_state = { learningRate = params.lr }

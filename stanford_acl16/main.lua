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
tds = require('tds')
paths.dofile('model.lua')
local utils = require 'utils'

cmd = torch.CmdLine()
cmd:option('-input', 'dataset.t7', [[data tensors input file name]])
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

-- load the dataset
print('loading the dataset...')
params.dataset = torch.load(params.input)
params.bsize = (#params.dataset.train_batches[1][1])[1]

-- initiate the model & criterion
print('initializing the model...')
params.model = nn.MaskZero(get_model(params), 1):cuda()
params.criterion = nn.MaskZeroCriterion(nn.CrossEntropyCriterion(), 1):cuda()
params.question_word_lookup = utils.get_layer(params.model, 'question_word_lookup')
params.passage_word_lookup = utils.get_layer(params.model, 'passage_word_lookup')
params.question_encoder = utils.get_layer(params.model, 'question_encoder')
params.passage_encoder = utils.get_layer(params.model, 'passage_encoder')
params.question_encoder:forget() --  bring back all states to the start of the sequence buffers
params.passage_encoder:forget() --  same reason as above :)
params.passage_word_lookup:share(params.question_word_lookup, 'weight', 'bias', 'gradWeight', 'gradBias') -- share their parameters
params.pp, params.gp = params.model:getParameters() -- flatten all the parameters into one fat tensor

-- train the model
print('training...')
local optim_states, best_acc, best_acc_model, best_acc_epoch, train_start = {}, -1, -1, -1, sys.clock()
for epoch = 1, params.num_epochs do
  local epoch_start, epoch_loss, epoch_iterations, indices = sys.clock(), 0, 0, torch.randperm(#params.dataset.train_batches)
  params.model:training()
  xlua.progress(1, #params.dataset.train_batches)
  for batch = 1, #params.dataset.train_batches do
    local cur_batch_record = params.dataset.train_batches[indices[batch]]
    if (#cur_batch_record[1])[1] == params.bsize then 
      -- while defining the model, we assume the batch size is constant. (must solve this later)
      local feval = function(x)
        params.gp:zero()
        local out = params.model:forward({cur_batch_record[2], cur_batch_record[1], cur_batch_record[3]})
        local example_batch_loss = params.criterion:forward(out, cur_batch_record[4])
        epoch_loss = epoch_loss + example_batch_loss * (#cur_batch_record[4])[1]
        epoch_iterations = epoch_iterations + (#cur_batch_record[4])[1]
        local rep_grads = params.criterion:backward(out, cur_batch_record[4])
        params.model:backward({cur_batch_record[2], cur_batch_record[1], cur_batch_record[3]}, rep_grads)
        params.gp:clamp(-params.grad_clip, params.grad_clip)
        return example_batch_loss, params.gp
      end
      optim.sgd(feval, params.pp, params.optim_state, optim_states)
      -- if you don't call the following 2 lines, the previous state will be used to activate the next
      params.question_encoder:forget()
      params.passage_encoder:forget()
    end
    xlua.progress(batch, #params.dataset.train_batches)
  end
  xlua.progress(#params.dataset.train_batches, #params.dataset.train_batches)
  -- update the best performing model so far
  local cur_acc = utils.compute_accuracy(params.model, params.dataset.valid_batches)
  if best_acc < cur_acc then
    best_acc = cur_acc
    best_acc_model = params.model:clone()
    best_acc_epoch = epoch
  end
  print(string.format('epoch (%d/%d) loss = %.2f; best dev. acc = %.2f (%d); time = %.2f mins;', 
                      epoch, params.num_epochs, (epoch_loss / epoch_iterations), best_acc, 
                      best_acc_epoch, ((sys.clock() - epoch_start)/60)))
end

-- evaluate the best performing model
local final_acc = utils.compute_accuracy(best_acc_model, params.dataset.test_batches)
print(string.format('final accuracy = %.2f (%d) time = %.2f mins;', 
                    final_acc, best_acc_epoch, ((sys.clock() - train_start)/60)))

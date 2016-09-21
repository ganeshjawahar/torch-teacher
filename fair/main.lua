--[[

Torch Implementation of the paper 'The Goldilocks Principle: Reading Children's Books with Explicit Memory Representations'

Precisely we attempt to code for the model 'MemNNs (Window Memory + Self-Supervision)' mentioned in the paper

]]--

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'xlua'
require 'sys'
tds = require('tds')
paths.dofile('model.lua')

cmd = torch.CmdLine()
cmd:option('-input', 'dataset.t7', [[data tensors input file name]])
cmd:option('-seed', 123, [[seed for the random generator]])
cmd:option('-p', 300, [[dimensionality of word embeddings]])
cmd:option('-num_epochs', 5, [[number of full passes through the training data]])
cmd:option('-lr', 0.01, [[sgd learning rate]])
cmd:option('-window_compo', 'sum', [[how to compose the window rep. from the word vectors? 
                               sum or concatenation?]])
params = cmd:parse(arg)

torch.manualSeed(params.seed)

-- load the dataset
print('loading the dataset...')
params.dataset = torch.load(params.input)
params.b = (#(params.dataset.train_tensors[1][1])[1])[1]

-- initiate the model & criterion
print('initializing the model...')
local record = params.dataset.train_tensors[1]
params.model, params.criterion = get_model(params)

-- train & evaluate the model
print('training...')
local best_dev_acc, best_test_acc, best_acc_epoch, train_start = -1, -1, -1, sys.clock()
function compute_accuracy(model, data)
  model:evaluate()
  local correct, total, softmax, center = 0, 0, nn.SoftMax():cuda(), math.ceil(params.b / 2)
  for i = 1, #data do
    local record = data[i]
    local pred = model:forward({record[1], record[2]})
    local soft_pred = softmax:forward(pred:t())
    local _, max_id = soft_pred:max(2)
    if record[1][max_id[1][1]][center] == record[3] then
      correct = correct + 1
    end
    total = total + 1
  end
  return correct/total
end
for epoch = 1, params.num_epochs do
  local epoch_start, epoch_loss, epoch_iterations, indices = sys.clock(), 0, 0, torch.randperm(#params.dataset.train_tensors)
  params.model:training()
  xlua.progress(1, #params.dataset.train_tensors)
  for rec_id = 1, #params.dataset.train_tensors do
    local record = params.dataset.train_tensors[indices[rec_id]]
    local out = params.model:forward({record[1], record[2]})
    local _, m_bar = out:max(1)
    local m_o1 = nil
    if #record[4] == 1 then
      m_o1 = record[4][1]
    else
      local max_id = -1
      for mem_i = 1, #record[4] do
        if max_id == -1 or out[record[4][mem_i]][1] > out[record[4][max_id]][1] then
          max_id = mem_i
        end
      end
      m_o1 = record[4][max_id]
    end
    if m_o1 ~= m_bar[1] then
      -- update the model
      local example_loss = params.criterion:forward(out, m_o1)
      epoch_loss = epoch_loss + example_loss
      epoch_iterations = epoch_iterations + 1
      local mem_grads = params.criterion:backward(out, m_o1)
      params.model:zeroGradParameters()
      params.model:backward({record[1], record[2]}, mem_grads)
      params.model:updateParameters(params.lr)
    end
    if rec_id % 5 == 0 then xlua.progress(rec_id, #params.dataset.train_tensors) end    
  end
  xlua.progress(#params.dataset.train_tensors, #params.dataset.train_tensors)
  -- update the best performing model so far
  local cur_dev_acc = compute_accuracy(params.model, params.dataset.val_tensors)
  if best_dev_acc < cur_dev_acc then
    best_dev_acc = cur_dev_acc
    best_acc_epoch = epoch
    best_test_acc = compute_accuracy(params.model, params.dataset.test_tensors)
  end
  print(string.format('epoch (%d/%d) loss = %.2f; best dev. acc = %.2f; best test. acc = %.2f (%d); update ratio = %.2f; time = %.2f mins;', 
                      epoch, params.num_epochs, (epoch_loss / epoch_iterations), best_dev_acc, best_test_acc,
                      best_acc_epoch, (epoch_iterations / #params.dataset.train_tensors), ((sys.clock() - epoch_start)/60)))
end
print(string.format('final accuracy = %.2f (%d); time = %.2f mins;', 
                    best_test_acc, best_acc_epoch, ((sys.clock() - train_start)/60)))

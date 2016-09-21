--[[

Torch Implementation of the paper 'Text Understanding with the Attention Sum Reader Network'

Precisely we attempt to code for the model 'AS Reader (Single model)' mentioned in the paper

]]--

require 'torch'
require 'cutorch'
require 'nn'
require 'cunn'
require 'optim'
require 'xlua'
require 'sys'
require 'lfs'
tds = require('tds')
paths.dofile('model.lua')
local utils = require 'utils'

cmd = torch.CmdLine()
cmd:option('-input', 'dataset.t7', [[data tensors input file name]])
cmd:option('-seed', 123, [[seed for the random generator]])
cmd:option('-dim', 384, [[dimensionality of word embeddings]])
cmd:option('-hid_size', 384, [[GRU's hidden layer size]])
cmd:option('-num_epochs', 2, [[number of full passes through the training data]])
cmd:option('-lr', 0.001, [[adam learning rate]])
cmd:option('-bsize', 32, [[adam mini-batch size]])
cmd:option('-grad_clip', 10, [[clip gradients at this value]])
params = cmd:parse(arg)

torch.manualSeed(params.seed)
params.optim_state = { learningRate = params.lr }

-- load the dataset
print('loading the dataset...')
params.dataset = torch.load(params.input)

-- initiate the model & criterion
print('initializing the model...')
params.model = get_model(params):cuda()
params.criterion = nn.MaskZeroCriterion(nn.ClassNLLCriterion(), 1):cuda()
params.pp, params.gp = params.model:getParameters() -- flatten all the parameters into one fat tensor
params.pp:uniform(-0.1, 0.1) -- initialize the parameters from uniform distribution

-- train & evaluate the model
print('training...')
local optim_states, best_dev_acc, best_test_acc, best_acc_epoch, train_start = {}, -1, -1, -1, sys.clock()
local question_master_tensor, question_rev_master_tensor, passage_master_tensor, passage_rev_master_tensor = 
      torch.CudaTensor(params.bsize, params.dataset.max_question_len), torch.CudaTensor(params.bsize, params.dataset.max_question_len),
      torch.CudaTensor(params.bsize, params.dataset.max_passage_len), torch.CudaTensor(params.bsize, params.dataset.max_passage_len)
local sum_master_tensor, new_grads_tensor, label_master_tensor = torch.CudaTensor(params.bsize, params.dataset.max_uniq_words_per_passage),
                       torch.CudaTensor(params.bsize, params.dataset.max_passage_len), torch.CudaTensor(params.bsize)
for epoch = 1, params.num_epochs do
  local epoch_start, epoch_loss, indices, num_batches, batch_id = sys.clock(), 0, 
                    torch.randperm(#params.dataset.train_tensors[1]), math.ceil(#params.dataset.train_tensors[1] / params.bsize), 0
  params.model:training()
  xlua.progress(1, num_batches)
  for i = 1, #params.dataset.train_tensors[1], params.bsize do
    local batch_size = math.min(i + params.bsize - 1, #params.dataset.train_tensors[1]) - i + 1
    local feval = function(x)
      params.gp:zero()
      for j = 1, batch_size do
        local index = indices[i + j - 1]
        local meta_p_local_word_list, meta_local_vocab_size, meta_local_answer_id = unpack(params.dataset.train_tensors[1][index])
        local data_passage_tensor, data_passage_rev_tensor, data_question_tensor, data_question_rev_tensor = params.dataset.train_tensors[2][index],
                          params.dataset.train_tensors[3][index], params.dataset.train_tensors[4][index], params.dataset.train_tensors[5][index]
        question_master_tensor[batch_size]:copy(data_question_tensor:cuda())
        question_rev_master_tensor[batch_size]:copy(data_question_rev_tensor:cuda())
        passage_master_tensor[batch_size]:copy(data_passage_tensor:cuda())
        passage_master_tensor[batch_size]:copy(data_passage_rev_tensor:cuda())
        label_master_tensor[j]= meta_local_answer_id
      end
      local pred = params.model:forward({{passage_master_tensor[{ {1, batch_size}, {} }], passage_rev_master_tensor[{ {1, batch_size}, {} }]},
                                      {question_master_tensor[{ {1, batch_size}, {} }], question_rev_master_tensor[{ {1, batch_size}, {} }]}})
      -- compute the total probability per unique word in the passage
      sum_master_tensor:fill(0)
      for j = 1, batch_size do
        local local_word_list = params.dataset.train_tensors[1][indices[i + j - 1]][1]
        for k = 1, #local_word_list do
          local l = (params.dataset.max_passage_len - #local_word_list + k)
          sum_master_tensor[j][local_word_list[k]] = sum_master_tensor[j][local_word_list[k]] + pred[j][l]
        end
      end
      local example_loss = params.criterion:forward(sum_master_tensor[{ {1, batch_size}, {} }], label_master_tensor[{ {batch_size} }])
      epoch_loss = epoch_loss + example_loss
      local rep_grads = params.criterion:backward(sum_master_tensor[{ {1, batch_size}, {} }], label_master_tensor[{ {batch_size} }])
      new_grads_tensor:fill(0)
      for j = 1, batch_size do
        local local_word_list = params.dataset.train_tensors[1][indices[i + j - 1]][1]
        for k = 1, #local_word_list do
          local l = (params.dataset.max_passage_len - #local_word_list + k)
          new_grads_tensor[j][l] = rep_grads[j][local_word_list[k]]
        end
      end
      params.gp:clamp(-params.grad_clip, params.grad_clip)
      return example_loss, params.gp
    end
    optim.adam(feval, params.pp, params.optim_state, optim_states)
    -- if you don't call the following 2 lines, the previous state will be used to activate the next
    params.q_gru_layer:forget()
    params.p_gru_layer:forget()
    batch_id = batch_id + 1
    xlua.progress(batch_id, num_batches)
  end
  xlua.progress(num_batches, num_batches)
  -- update the best performing model so far
  local cur_dev_acc = utils.compute_accuracy(params.model, params.dataset.val_tensors, params)
  if best_dev_acc < cur_dev_acc then
    best_dev_acc = cur_dev_acc
    best_acc_epoch = epoch
    best_test_acc = utils.compute_accuracy(params.model, params.dataset.test_tensors, params)
  end
  print(string.format('epoch (%d/%d) loss = %.2f; best dev. acc = %.2f; best test. acc = %.2f (%d); time = %.2f mins;', 
                      epoch, params.num_epochs, (epoch_loss / #params.dataset.train_tensors[1]), best_dev_acc, best_test_acc,
                      best_acc_epoch, ((sys.clock() - epoch_start)/60)))
end
print(string.format('final accuracy = %.2f (%d); time = %.2f mins;', 
                    best_test_acc, best_acc_epoch, ((sys.clock() - train_start)/60)))

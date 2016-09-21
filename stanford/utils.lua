--[[

Utilities used by main.lua

]]--

local utils={}

-- Function to get any layer from nnGraph module
function utils.get_layer(model, name)
  for _, node in ipairs(model.modules[1].forwardnodes) do
    if node.data.annotations.name == name then
      return node.data.module
    end
  end
  return nil
end

-- Function to compute accuracy of the model
function utils.compute_accuracy(model, data, bsize)
  model:evaluate()
  local total, correct, softmax_model = 0, 0, nn.SoftMax():cuda()
  for i = 1, #data do
    local cur_batch_record = data[i]
    --if (#cur_batch[1])[1] == params.bsize then 
      local out = params.model:forward({cur_batch_record[3], {{cur_batch_record[1], cur_batch_record[2]}, cur_batch_record[1]}})
      local soft_out = softmax_model:forward(out)
      _, max_ids = soft_out:max(2)
      for j = 1, (#cur_batch_record[4])[1] do
        if max_ids[j] == cur_batch_record[4][j] then correct = correct + 1 end
        total = total + 1
      end
    --end
  end
  return correct / total
end

-- Function to initalize word weights
function utils.init_word_weights(params, lookup, file)
  print('initializing word lookup with the pre-trained embeddings...')
  local start_time = sys.clock()
  local ic = 0
  local begin_offset = 1 --[[ since rnn uses lookuptablemaskzero table, 
                       the first index in weight matrix corresponds to the zero padding ]]--
  for line in io.lines(file) do
    local content = stringx.split(line)
    local word = content[1]
    if params.dataset.word2index[word] ~= nil then
      local tensor = torch.Tensor(#content - 1)
      for i = 2, #content do
        tensor[i - 1] = tonumber(content[i])
      end
      lookup.weight[begin_offset + params.dataset.word2index[word]] = tensor
      ic = ic + 1
    end
  end
  print(string.format("%d out of %d words initialized. Done in %.2f minutes.", 
                      ic, #params.dataset.index2word, (sys.clock() - start_time)/60))
end

return utils

--[[

Utilities used by main.lua

]]--

local utils={}

-- Function to get any layer from nnGraph module
function utils.get_layer(model, name)
  for _, node in ipairs(model.forwardnodes) do
    if node.data.annotations.name == name then
      return node.data.module
    end
  end
  return nil
end

-- Function to compute accuracy of the model
function utils.compute_accuracy(model, data, params)
  model:evaluate()
  local correct = 0
  local question_master_tensor, question_rev_master_tensor, passage_master_tensor, passage_rev_master_tensor = 
      torch.CudaTensor(params.bsize, params.dataset.max_question_len), torch.CudaTensor(params.bsize, params.dataset.max_question_len),
      torch.CudaTensor(params.bsize, params.dataset.max_passage_len), torch.CudaTensor(params.bsize, params.dataset.max_passage_len)
  local sum_master_tensor, label_master_tensor = torch.CudaTensor(params.bsize, params.dataset.max_uniq_words_per_passage), torch.CudaTensor(params.bsize)
  for i = 1, #data, params.bsize do
    local batch_size = math.min(i + params.bsize - 1, #data) - i + 1
    for j = 1, batch_size do
      local index = i + j - 1
      local meta_p_local_word_list, meta_local_vocab_size, meta_local_answer_id = unpack(params.dataset.train_tensors[1][index])
      local data_passage_tensor, data_passage_rev_tensor, data_question_tensor, data_question_rev_tensor = params.dataset.train_tensors[2][index],
                          params.dataset.train_tensors[3][index], params.dataset.train_tensors[4][index], params.dataset.train_tensors[5][index]
      question_master_tensor[batch_size]:copy(data_question_tensor:cuda())
      question_rev_master_tensor[batch_size]:copy(data_question_rev_tensor:cuda())
      passage_master_tensor[batch_size]:copy(data_passage_tensor:cuda())
      passage_master_tensor[batch_size]:copy(data_passage_rev_tensor:cuda())
      label_master_tensor[j] = meta_local_answer_id
    end
    local pred = params.model:forward({{passage_master_tensor[{ {1, batch_size}, {} }], passage_rev_master_tensor[{ {1, batch_size}, {} }]},
                                      {question_master_tensor[{ {1, batch_size}, {} }], question_rev_master_tensor[{ {1, batch_size}, {} }]}})
    sum_master_tensor:fill(0)
    for j = 1, batch_size do
      local local_word_list = params.dataset.train_tensors[1][i + j - 1][1]
      for k = 1, #local_word_list do
        local l = (params.dataset.max_passage_len - #local_word_list + k)
        sum_master_tensor[j][local_word_list[k]] = sum_master_tensor[j][local_word_list[k]] + pred[j][l]
      end
    end
    _, ids = sum_master_tensor[{ {1, batch_size}, {} }]:max(2)
    for j = 1, batch_size do
      if ids[j][1] == label_master_tensor[j] then
        correct = correct + 1
      end
    end
  end
  return correct / #data
end

return utils
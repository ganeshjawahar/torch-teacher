--[[

Model implementation

]]--

function get_model(params)
  -- get the representations for supporting memories
  local support_dict_master = nn.ParallelTable() -- one dictionary per window position
  for i = 1, params.b do 
    support_dict_master:add(nn.LookupTable(#params.dataset.index2word, params.p)) 
  end
  local support_mem_model = nn.Sequential():add(nn.Identity()):add(nn.SplitTable(1))
                                        :add(support_dict_master):add(nn.JoinTable(1, 1))

  -- get the representation for query
  local query_dict_master = nn.ParallelTable()
  for i = 1, params.b do
    query_dict_master:add(nn.LookupTable(#params.dataset.index2word, params.p)) 
  end
  local query_mem_model = nn.Sequential():add(nn.Identity()):add(nn.SplitTable(1))
                                        :add(query_dict_master):add(nn.JoinTable(1, 1))

  -- build the final scoring model
  local model = nn.Sequential()
  model:add(nn.ParallelTable())
  model.modules[1]:add(support_mem_model)
  model.modules[1]:add(query_mem_model)
  model:add(nn.MM(false, true))

  -- ship it to gpu
  model = model:cuda()

  -- IMPORTANT! do weight sharing after model is in cuda
  for i = 1, params.b do
    query_dict_master.modules[i]:share(support_dict_master.modules[i], 'weight', 'bias', 'gradWeight', 'gradBias')
  end

  return model, nn.CrossEntropyCriterion():cuda()
end
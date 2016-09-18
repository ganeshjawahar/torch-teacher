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
function utils.compute_accuracy(model, data)
  model:evaluate()
  local total, correct = 0, 0
  for i = 1, #data do
    local cur_batch = data[i]
  	local out = model:forward({cur_batch[2], cur_batch[1], cur_batch[3]})
  	local soft_out = nn.SoftMax():forward(out)
  	_, max_ids = soft_out:max(2)
  	for j = 1, (#cur_batch[4])[1] do
  	  if max_ids[j] == cur_batch[4][j] then correct = correct + 1 end
  	  total = total + 1
  	end
  end
  return correct / total
end

return utils

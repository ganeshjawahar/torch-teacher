--[[

Pre-processing scripts which 
  (i) creates the data tensors (to be used by main.lua)
  (ii) saves them to t7 file

]]--

require 'torch'
require 'cutorch'
require 'io'
require 'xlua'
require 'lfs'
require 'pl.stringx'
require 'pl.file'
tds = require('tds')

-- Command line arguments
cmd = torch.CmdLine()
cmd:option('-data', '../cbt/', [[path to the data folder containing train, val. & test records]])
cmd:option('-word_type', 'NE', [[class type of the prediction word. (NE - Named Entity,
                                 CN - Common Noun, V - Verb, P - Preposition]])
cmd:option('-b', 5, [[size of the window memory]])
cmd:option('-out', 'dataset.t7', [[data tensors output file name]])
params = cmd:parse(arg)
params.unk = '<unk>'

-- Load all the data to memory
print('loading ' .. params.word_type .. ' data...')
params.train_lines = stringx.splitlines(file.read(params.data .. 'cbtest_' .. params.word_type .. '_train.txt')) 
params.val_lines = stringx.splitlines(file.read(params.data .. 'cbtest_' .. params.word_type .. '_valid_2000ex.txt')) 
params.test_lines = stringx.splitlines(file.read(params.data .. 'cbtest_' .. params.word_type .. '_test_2500ex.txt'))
params.train_size, params.val_size, params.test_size = #params.train_lines/22, #params.val_lines/22, #params.test_lines/22
params.num_pads = math.floor((params.b - 1) / 2)
print('found (' .. params.train_size .. '/' .. params.val_size .. '/' .. params.test_size ..') records')

-- Build the vocabulary
print('building vocab...')
local start = sys.clock()
params.index2word, params.word2index = tds.hash(), tds.hash()
-- add zero padding before and after a context sentence and question
for i = 1, params.num_pads do
  params.index2word[#params.index2word + 1] = '<begin-' .. i ..'>'
  params.word2index['<begin-' .. i ..'>'] = #params.index2word
  params.index2word[#params.index2word + 1] = '<end-' .. i ..'>'
  params.word2index['<end-' .. i ..'>'] = #params.index2word
end
params.index2word[#params.index2word + 1] = params.unk
params.word2index[params.unk] = #params.index2word
function create_vocab(lines)
  for i = 1, #lines/22 do
    local start = 1 + 22 * (i - 1)
    for j = 0, 19 do
      local words = stringx.split(lines[start + j])
      for k = 2, #words do
        if params.word2index[words[k]] == nil then
          params.index2word[#params.index2word + 1] = words[k]
          params.word2index[words[k]] = #params.index2word
        end
      end
    end
    local words = stringx.split(lines[start + 20])
    for k = 2, (#words - 2) do
      words[k] = string.lower(words[k])
      if params.word2index[words[k]] == nil then
        params.index2word[#params.index2word + 1] = words[k]
        params.word2index[words[k]] = #params.index2word
      end
    end
  end
end
create_vocab(params.train_lines)
create_vocab(params.val_lines)
create_vocab(params.test_lines)
print(string.format('vocab built in %.2f mins; # unique words = %d;', ((sys.clock() - start) / 60), #params.index2word))

-- Generate the data tensors
function gen_tensors(lines, label)
  print('generating tensors for ' .. label .. '...')
  function get_context_windows(lines, start, cands, answer)
    local windows, rel_mem_ids = {}, {}
    function tokenize(sentence, last_offset)
      local words = stringx.split(sentence)
      local last_index = #words
      if last_offset ~= nil then last_index = #words - last_offset end
      local tokens = {}
      for i = 1, params.num_pads do
        table.insert(tokens, '<begin-' .. i ..'>')
      end
      for i = 2, last_index do
        table.insert(tokens, words[i])
      end
      for i = 1, params.num_pads do
        table.insert(tokens, '<end-' .. i ..'>')
      end
      return tokens
    end
    for i = 1, #cands do
      for j = 0, 19 do
        local words = tokenize(lines[start + j])
        for k = params.num_pads + 1, #words - params.num_pads do
          if words[k] == cands[i] then
            local tensor, m = torch.CudaTensor(params.b), 0
            for l = k - params.num_pads, k + params.num_pads do
              m = m + 1
              tensor[m] = params.word2index[words[l]]
            end
            table.insert(windows, tensor)
            if cands[i] == answer then
              -- store the memories relevant to the answer
              table.insert(rel_mem_ids, #windows)
            end
          end
        end
      end  
    end
    assert(#windows ~= 0)
    assert(#rel_mem_ids ~= 0)
    local memory_tensor = torch.CudaTensor(#windows, params.b)
    for i = 1, #windows do
      memory_tensor[i] = windows[i]
    end
    return memory_tensor, rel_mem_ids
  end
  function get_query_memory(line)
    local words = tokenize(line, 2)
    for i = 1, #words do
      if words[i] == 'XXXXX' then
        local query_tensor, j = torch.CudaTensor(1, params.b), 0         
        for k = i - params.num_pads, i + params.num_pads do
          j = j + 1
          words[k] = string.lower(words[k])	
          assert(params.word2index[words[k]] ~= nil)
          query_tensor[1][j] = params.word2index[words[k]]
        end
        return query_tensor
      end
    end
    error('cannot find the placeholder token.')
  end
  local dataset = {}
  xlua.progress(1, #lines/22)
  for i = 1, #lines/22 do
    local start = 1 + 22 * (i - 1)
    local last_line = stringx.split(lines[start + 20])
    local answer, answer_cand = last_line[#last_line - 1], stringx.split(last_line[#last_line], '|')
    local supporting_memories, rel_mem_ids = get_context_windows(lines, start, answer_cand, answer)                                                
    local query_memory = get_query_memory(lines[start + 20])
    local answer_id = params.word2index[answer]
    assert(answer_id ~= nil)
    table.insert(dataset, {supporting_memories, query_memory, answer_id, rel_mem_ids})
    if i % 5 == 0 then xlua.progress(i, #lines/22) end
  end
  xlua.progress(#lines/22, #lines/22)
  return dataset
end
params.train_tensors = gen_tensors(params.train_lines, 'train')
params.val_tensors = gen_tensors(params.val_lines, 'valid')
params.test_tensors = gen_tensors(params.test_lines, 'test')

-- Save the data tensors
print('saving all the tensors...')
local save_point = {
  train_tensors = params.train_tensors,
  val_tensors = params.val_tensors,
  test_tensors = params.test_tensors,
  index2word = params.index2word,
  word2index = params.word2index
}
torch.save(params.out, save_point)

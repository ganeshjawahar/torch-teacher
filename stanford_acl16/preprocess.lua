--[[

Pre-processing scripts which 
  (i) creates the data tensors (to be used by main.lua)
  (ii) saves them to lmdb

]]--

require 'torch'
require 'cutorch'
require 'io'
require 'xlua'
require 'lfs'
require 'pl.stringx'
tds = require('tds')

-- Command line arguments
cmd = torch.CmdLine()
cmd:option('-data', '../cnn_questions/', [[path to the data folder containing train, val. & test records]])
cmd:option('-out', 'dataset.t7', [[data tensors output file name]])
cmd:option('-batch_size', 32, [[sgd mini-batch size]])
cmd:option('-vocab_size', 50000, [[size of the word vocabulary. this is constructed by taking the top <int> most frequent words.
								   rest are replaced with <unk> tokens.']])
params = cmd:parse(arg)

params.train_folder = params.data .. 'training/'
params.val_folder = params.data .. 'validation/'
params.test_folder = params.data .. 'test/'
params.unk = '<unk>'

-- Build the vocabulary
print('building vocab...')
local start = sys.clock()
params.vocab, params.global_e2i, params.global_i2e, params.unique_words, params.train_size = tds.hash(), tds.hash(), tds.hash(), 0, 0
for file in lfs.dir(params.train_folder) do
  if #file > 2 then -- pass through '.' & '..'
  	local fptr = io.open(params.train_folder .. file, 'r')
	  local url = fptr:read()
	  fptr:read() -- pass through empty line
	  local passage = fptr:read()
	  fptr:read() -- pass through empty line
	  local local_e2i, local_i2e = {}, {}
	  function add_to_vocab(tokens, update_vocab)
      for i = 1, #tokens do
        local token = tokens[i]
        if stringx.startswith(token, '@') == true then
      	  if local_e2i[token] == nil then
      	    local_i2e[#local_i2e + 1] = token
            local_e2i[token] = #local_i2e
      	  end
      	  token = '@entity' .. local_e2i[token]
      	  if params.global_e2i[token] == nil then
      	  	params.global_i2e[#params.global_i2e + 1] = token
            params.global_e2i[token] = #params.global_i2e
      	  end
        end
        if update_vocab == true then
    	    if params.vocab[tokens[i]] == nil then 
    	      params.vocab[tokens[i]] = 0
    	      params.unique_words = params.unique_words + 1
    	    end
    	    params.vocab[tokens[i]] = params.vocab[tokens[i]] + 1
        end
      end    
    end    
	  add_to_vocab(stringx.split(passage), true) -- add tokens in passage to vocab
	  local question = fptr:read() -- add tokens in question to vocab
    add_to_vocab(stringx.split(question), true)
	  fptr:read()
	  add_to_vocab({answer}, false) -- add answer token to vocab
	  fptr:read()
	  while true do
		  local line = fptr:read()
		  if line == nil then break end
		  local cand_entity = stringx.split(line, ':')[1]
		  add_to_vocab({cand_entity}, false) -- add candidate answer token to vocab
	  end
	  io.close(fptr)
	  params.train_size = params.train_size + 1
  end
end
print(string.format('vocab built in %.2f mins', ((sys.clock() - start) / 60)))
print('#unique candidate answers = '..#params.global_i2e .. ' (from ' .. params.train_size ..' questions)')
print('#unique unigrams before pruning = '..params.unique_words)
function extract_top_k_words()
  print('extracting top ' .. params.vocab_size .. ' words...')
  if params.unique_words < params.vocab_size then 
  	print('error: specified vocabulary size cannot be greater than # unique words in the dataset.') 
  	os.exit(0) 
  end
  local words, word_freq_tensor, i = {}, torch.Tensor(params.unique_words), 0
  for word, count in pairs(params.vocab) do
    table.insert(words, word)
    i = i + 1
    word_freq_tensor[i] = count
  end
  local _, idx = torch.sort(word_freq_tensor, true) -- sort the words by decreasing order of frequency
  local new_vocab = tds.hash()
  for i = 1, params.vocab_size do  	
  	new_vocab[words[idx[i]]] = params.vocab[words[idx[i]]]
  end
  return new_vocab
end
params.vocab = extract_top_k_words()
params.index2word = tds.hash()
params.word2index = tds.hash()
for word, count in pairs(params.vocab) do
  params.index2word[#params.index2word + 1] = word
  params.word2index[word] = #params.index2word
end
params.index2word[#params.index2word + 1] = params.unk
params.word2index[params.unk] = #params.index2word
params.global_i2e[#params.global_i2e + 1] = params.unk
params.global_e2i[params.unk] = #params.global_i2e

-- Generate and store the data tensors
function gen_n_save(folder, label)
  print('processing tensors for ' .. label .. '...')
  start = sys.clock()
  -- get the data id -> passage length mapping
  print('generating data id -> passage mapping for ' .. label .. '...')
  local d2plen, records = {}, {}
  for file in lfs.dir(folder) do
    if #file > 2 then
      local fptr = io.open(folder .. file, 'r')
	    local url = fptr:read()
	    fptr:read()
	    local local_e2i, local_i2e = {}, {}
	    function get_entity_text(token)
        if local_e2i[token] == nil then
          local_i2e[#local_i2e + 1] = token
          local_e2i[token] = #local_i2e
        end
        token = '@entity' .. local_e2i[token]
  	    return token
	    end
	    function get_tensor(unigrams)
	  	  local tensor = torch.CudaTensor(#unigrams)
	  	  for i = 1, #unigrams do
          local token = unigrams[i]
          if stringx.startswith(token, '@') == true then
            token = get_entity_text(token)  
          end
      	  if params.word2index[token] == nil then
      	  	token = params.unk      	  	
          end
          tensor[i] = params.word2index[token]
	  	end
	  	return tensor
	  end
	  local passage_text = fptr:read()
	  local passage_tensor = get_tensor(stringx.split(passage_text))
	  table.insert(d2plen, (#passage_tensor)[1])
	  fptr:read()
	  local question_text = fptr:read()
	  local question_tensor = get_tensor(stringx.split(question_text))
	  fptr:read()
    function get_answer_id(token)
      token = get_entity_text(token)
      if params.global_e2i[token] == nil then token = params.unk end
      return params.global_e2i[token]
    end
	  local answer_text = fptr:read()
	  local answer_id = get_answer_id(answer_text)
	  assert(answer_id ~= nil)
	  fptr:read()
	  local cand_ids = {}
	  while true do
	  	local line = fptr:read()
	  	if line == nil then break end
	  	local candidate_text = stringx.split(line, ':')[1]
	    local cand_id = get_answer_id(candidate_text)
	    assert(cand_id ~= nil)
	    table.insert(cand_ids, cand_id)
	  end
	  local cand_tensor = torch.CudaTensor(cand_ids)
	  table.insert(records, {passage_tensor, question_tensor, answer_id, cand_tensor})
	  io.close(fptr)
    end
  end
  
  local p_sizes, idx = torch.sort(torch.Tensor(d2plen), true) -- Sort the passage ids by decreasing order of length

  function get_cur_batch_stat(start, last)
    local max_passage, max_ans_cand, max_question = -1000, -1000, -1000
    for i = start, last do
      local record = records[idx[i]]
      if (#record[1])[1] > max_passage then max_passage = (#record[1])[1] end
      if (#record[2])[1] > max_question then max_question = (#record[2])[1] end
      if (#record[4])[1] > max_ans_cand then max_ans_cand = (#record[4])[1] end
    end
    return max_passage, max_ans_cand, max_question
  end

  -- create the batches
  print('creating the final batches for ' .. label .. '...')
  local batches, cur_batch, num_batches = {}, 0, math.ceil(#d2plen / params.batch_size)
  xlua.progress(1, num_batches)
  for i = 1, #d2plen, params.batch_size do
  	cur_batch = cur_batch + 1
  	local cur_batch_size = math.min(#d2plen, i + params.batch_size - 1) - i + 1
    local max_passage, max_ans_cand, max_question = get_cur_batch_stat(i, i + cur_batch_size - 1)
  	local passage_batch_tensor, question_batch_tensor, answer_cand_batch_tensor = 
              torch.CudaTensor(cur_batch_size, max_passage):fill(0),
              torch.CudaTensor(cur_batch_size, max_question):fill(0),
              torch.CudaTensor(cur_batch_size, max_ans_cand):fill(0)
  	local answer_batch_tensor = torch.CudaTensor(cur_batch_size, 1)
  	for j = 1, cur_batch_size do
  	  local record = records[idx[i + j - 1]]
      passage_batch_tensor[{ j, { 1, (#record[1])[1] }}] = record[1]
      question_batch_tensor[{ j, { 1, (#record[2])[1] }}] = record[2]
      answer_batch_tensor[j][1] = record[3]
      answer_cand_batch_tensor[{ j, { 1, (#record[4])[1] }}] = record[4]
	  end	
  	if cur_batch % 5 == 0 then xlua.progress(i, num_batches) end
  	table.insert(batches, {passage_batch_tensor, question_batch_tensor, answer_cand_batch_tensor, answer_batch_tensor})
  end
  xlua.progress(num_batches, num_batches)
  print(string.format('batches for %s processed in %.2f mins', label, ((sys.clock() - start) / 60)))
  return batches
end

params.train_batches = gen_n_save(params.train_folder, 'train')
params.valid_batches = gen_n_save(params.val_folder, 'valid')
params.test_batches = gen_n_save(params.test_folder, 'test')

-- Save the batches
print('saving all the tensors...')
local save_point = {
  train_batches = params.train_batches,
  valid_batches = params.valid_batches,
  test_batches = params.test_batches,
  index2word = params.index2word,
  word2index = params.word2index,
  global_i2e = params.global_i2e,
  global_e2i = params.global_e2i
}
torch.save(params.out, save_point)

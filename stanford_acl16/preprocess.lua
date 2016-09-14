--[[

Pre-processing scripts which 
  (i) creates the data tensors (to be used by main.lua)
  (ii) saves them to lmdb

]]--

require 'torch'
require 'cutorch'
require 'lmdb'
require 'io'
require 'xlua'
require 'lfs'
require 'os'
require 'pl.stringx'
require 'pl.file'
tds = require('tds')

-- Command line arguments
cmd = torch.CmdLine()
cmd:option('-data', '../cnn_questions/', [[path to the data folder containing train, val. & test records]])
cmd:option('-relabeling', 1, [[should we relabel the entity markers based on their first occurrence in the passage. 
							   this trick improves the convergence time of the training algorithm.]])
cmd:option('-db_name', 'mydb', [[lmdb database name]])
cmd:option('-batch_size', 32, [[sgd mini-batch size]])
cmd:option('-vocab_size', 50000, [[size of the word vocabulary. this is constructed by taking the top <int> most frequent words.
								   rest are replaced with <unk> tokens.']])
params = cmd:parse(arg)

params.train_folder = params.data .. 'test/'
params.val_folder = params.data .. 'validation/'
params.test_folder = params.data .. 'test/'
params.unk = '<unk>'

-- Build the vocabulary
print('building vocab...')
params.vocab, params.global_e2i, params.global_i2e, params.unique_words, params.train_size = tds.hash(), tds.hash(), tds.hash(), 0, 0
for file in lfs.dir(params.train_folder) do
  if #file > 2 then -- pass through '.' & '..'
  	local fptr = io.open(params.train_folder .. file, 'r')
	local url = fptr:read()
	fptr:read() -- pass through empty line
	local passage = fptr:read()
	fptr:read() -- pass through empty line
	local local_e2i, local_i2e = {}, {}
	function add_to_vocab(tokens)
      for i = 1, #tokens do
        local token = tokens[i]
        if stringx.startswith(token, '@') == true then
          if params.relabeling == 1 then
      	    if local_e2i[token] == nil then
      	      local_i2e[#local_i2e + 1] = token
              local_e2i[token] = #local_i2e
      	    end
      	    token = '@entity' .. local_e2i[token]
      	  end
      	  if params.global_e2i[token] == nil then
      	  	params.global_i2e[#params.global_i2e + 1] = token
            params.global_e2i[token] = #params.global_i2e
      	  end
        end
  	    if params.vocab[tokens[i]] == nil then 
  	      params.vocab[tokens[i]] = 0
  	      params.unique_words = params.unique_words + 1
  	    end
  	    params.vocab[tokens[i]] = params.vocab[tokens[i]] + 1
      end    
    end    
	add_to_vocab(stringx.split(passage)) -- add tokens in passage to vocab
	local question = fptr:read() -- add tokens in question to vocab
	add_to_vocab(stringx.split(question))
	fptr:read()
	io.close(fptr)
	params.train_size = params.train_size + 1
  end
end
print('#unique candidate answers = '..#params.global_i2e .. ' (from ' .. params.train_size ..' questions)')
print('#unique unigrams before pruning = '..params.unique_words)
function extract_top_k_words()
  print('extracting top ' .. params.vocab_size .. ' words...')
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

-- Generate and store the data tensors
params.db_ptr = lmdb.env{Path = './' .. params.db_name .. '_DB', Name = params.db_name .. '_DB' }
params.db_ptr:open()
params.db_reader = params.db_ptr:txn(true)
function gen_n_save(file, label)
  print('processing tensors for ' .. label .. '...')
  -- get the data id -> passage length mapping
  print('generating data id -> passage mapping for ' .. label .. '...')
  local commit_counter, d2plen = 0, {}
  params.db_writer = params.db_ptr:txn()
  for file in lfs.dir(params.train_folder) do
    if #file > 2 then
      local fptr = io.open(params.train_folder .. file, 'r')
	  local url = fptr:read()
	  fptr:read()
	  local local_e2i, local_i2e = {}, {}
	  function get_tensor(unigrams)
	  	local tensor = torch.CudaTensor(#unigrams)
	  	for i = 1, #unigrams do
          local token = unigrams[i]
          if stringx.startswith(token, '@') == true then
            if params.relabeling == 1 then
      	      if local_e2i[token] == nil then
      	        local_i2e[#local_i2e + 1] = token
                local_e2i[token] = #local_i2e
      	      end
      	      token = '@entity' .. local_e2i[token]
      	    end
      	  elseif params.vocab[token] == nil then
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
	  local answer_text = fptr:read()
	  function get_candidate_id(text)
	    if params.relabeling == 1 and stringx.startswith(text, '@') == true then 
	  	  assert(local_e2i[text] ~= nil) 
	  	  text = '@entity' .. local_e2i[text]
	    end
	    return params.global_e2i[text]
	  end
	  local answer_id = get_candidate_id(answer_text)	  
	  assert(answer_id ~= nil)
	  fptr:read()
	  local cand_ids = {}
	  while true do
	  	local line = fptr:read()
	  	if line == nil then break end
	  	local candidate_text = stringx.split(line, ':')[1]
	    local cand_id = get_candidate_id(candidate_text)	  
	    -- assert(cand_id ~= nil) MUST CHECK
	    table.insert(cand_ids, cand_id)
	  end
	  local cand_tensor = torch.CudaTensor(cand_ids)
	  params.db_writer:put('record_' .. label .. '_' .. #d2plen, {passage_tensor, question_tensor, answer_id, cand_tensor})
	  commit_counter = commit_counter + 1
	  if commit_counter % 50000 then
	  	params.db_writer:commit()
	  	params.db_writer = params.db_ptr:txn()
	  end
	  io.close(fptr)
    end
  end
  params.db_writer:put('record_' .. label .. '_size', #d2plen)
  params.db_writer:commit()
  
  local p_sizes, idx = torch.sort(torch.Tensor(d2plen), true) -- Sort the passage ids by decreasing order of length
  local num_batches = math.ceil(#d2plen / params.batch_size)
  local max_batch_len = {}
  for i = 1, #d2plen, num_batches do
  	local cur_batch_size = math.min(#d2plen, i + params.batch_size - 1) - i + 1
  	local cur_max = -1000
  	for j = 1, cur_batch_size do
  	  if p_sizes[i + j - 1] > cur_max then
  	  	cur_max = p_sizes[i + j - 1]
  	  end
  	end
  	table.insert(max_batch_len, cur_max)
  end

  -- create the batches
  print('creating batches for ' .. label .. '...')
  params.db_writer = params.db_ptr:txn()
  params.db_reader = params.db_ptr:txn(true)
  xlua.progress(1, num_batches)
  commit_counter = 0
  for i = 1, #d2plen, num_batches do
  	local cur_batch_size = math.min(#d2plen, i + params.batch_size - 1) - i + 1
  	local passage_batch_tensor = torch.CudaTensor(cur_batch_size, max_batch_len[i]):fill(0)
  	local question_batch, answer_batch, cand_batch = {}, {}, {}
  	for j = 1, cur_batch_size do
  	  print('reading ----'..j)
  	  local record = params.db_reader:get('record_' .. label .. '_' .. idx[i + j -1])
  	  passage_batch_tensor[j] = record[1]
  	  table.insert(question_batch, record[2])
  	  table.insert(answer_batch, record[3])
  	  table.insert(cand_batch, record[4])
	end
	params.db_writer:put('batch_' .. label .. '_' .. i, {passage_batch_tensor, question_batch, answer_batch, cand_batch})
  	if i % 5 == 0 then xlua.progress(i, num_batches) end
  	if commit_counter % 50000 then
  	  params.db_writer:commit()
  	  params.db_writer = params.db_ptr:txn()
    end
  end
  xlua.progress(num_batches, num_batches)
  params.db_writer:put('batches_' .. label .. '_size', #d2plen)
  params.db_writer:commit()
end

-- gen_n_save(params.train_folder, 'train')
-- gen_n_save(params.val_folder, 'valid')
gen_n_save(params.test_folder, 'test')

params.db_ptr:close()
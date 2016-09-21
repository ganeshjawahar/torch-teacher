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
cmd:option('-out', 'dataset.t7', [[data tensors output file name]])
cmd:option('-question_pad', 'left', [[which side to pad the question to make sequences in a batch to be
                                  of same size? left or right? ]])
cmd:option('-passage_pad', 'right', [[which side to pad the passage to make sequences in a batch to be 
                                  of same size? left or right? ]])
params = cmd:parse(arg)

-- Load all the data to memory
print('loading ' .. params.word_type .. ' data...')
params.train_lines = stringx.splitlines(file.read(params.data .. 'cbtest_' .. params.word_type .. '_train.txt')) 
params.val_lines = stringx.splitlines(file.read(params.data .. 'cbtest_' .. params.word_type .. '_valid_2000ex.txt')) 
params.test_lines = stringx.splitlines(file.read(params.data .. 'cbtest_' .. params.word_type .. '_test_2500ex.txt'))
params.train_size, params.val_size, params.test_size = #params.train_lines/22, #params.val_lines/22, #params.test_lines/22
print('found (' .. params.train_size .. '/' .. params.val_size .. '/' .. params.test_size ..') records')

-- Build the vocabulary
print('building vocab...')
local start = sys.clock()
params.index2word, params.word2index = tds.hash(), tds.hash()
params.max_passage_len, params.max_question_len = -1, -1
function create_vocab(lines)
  for i = 1, #lines/22 do
    local start, passage_len = (1 + 22 * (i - 1)), 0
    for j = 0, 19 do
      local words = stringx.split(lines[start + j])
      for k = 2, #words do
        if params.word2index[words[k]] == nil then
          params.index2word[#params.index2word + 1] = words[k]
          params.word2index[words[k]] = #params.index2word
        end
      end
      passage_len = passage_len + #words - 1
    end
    if params.max_passage_len < passage_len then params.max_passage_len = passage_len end
    local words = stringx.split(lines[start + 20])
    for k = 2, (#words - 2) do
      words[k] = string.lower(words[k])
      if params.word2index[words[k]] == nil then
        params.index2word[#params.index2word + 1] = words[k]
        params.word2index[words[k]] = #params.index2word
      end
    end
    if params.max_question_len < (#words - 3) then params.max_question_len = #words - 3 end
  end
end
create_vocab(params.train_lines)
create_vocab(params.val_lines)
create_vocab(params.test_lines)
print(string.format('vocab built in %.2f mins; # unique words = %d; max. passage len = %d; max. question len = %d;',
                  ((sys.clock() - start) / 60), #params.index2word, params.max_passage_len, params.max_question_len))

-- Generate the data tensors
params.max_uniq_words_per_passage = -1
function gen_tensors(lines, label)
  print('generating tensors for ' .. label .. '...')
  function tokenize(sentence, last_offset)
    local words = stringx.split(sentence)
    local last_index = #words
    if last_offset ~= nil then last_index = #words - last_offset end
    local tokens = {}
    for i = 2, last_index do
      table.insert(tokens, words[i])
    end
    return tokens
  end
  function get_question(last_line)
  	local words = tokenize(last_line, 2)
  	local question_tensor, question_rev_tensor = torch.Tensor(params.max_question_len):fill(0), 
                                                    torch.Tensor(params.max_question_len):fill(0)
  	for i = 1, #words do
  	  words[i] = string.lower(words[i])
  	  assert(params.word2index[words[i]] ~= nil)
      if params.question_pad == 'right' then
  	    question_tensor[params.max_question_len - i + 1] = params.word2index[words[i]]
        question_rev_tensor[params.max_question_len - #words + i] = params.word2index[words[i]]
      else
        question_tensor[i] = params.word2index[words[i]]
        question_rev_tensor[#words - i + 1] = params.word2index[words[i]]
      end
  	end
  	return {question_tensor, question_rev_tensor}
  end
  function get_passage_tensor_n_answer(lines, start, answer, answer_cand)
    local p_global_word_list, p_local_word_list, local_index2word, local_word2index, local_answer_id = tds.hash(), tds.hash(), 
                                                                            tds.hash(), tds.hash(), -1
    for i = 0, 19 do
      local words = tokenize(lines[start + i])
      for j = 1, #words do
      	assert(params.word2index[words[j]] ~= nil)
        p_global_word_list[#p_global_word_list + 1] = params.word2index[words[j]]
        if local_word2index[words[j]] == nil then
          local_index2word[#local_index2word + 1] = words[j]
          local_word2index[words[j]] = #local_index2word
          if words[j] == answer then
            local_answer_id = local_word2index[answer]
          end
        end
        assert(local_word2index[words[j]] ~= nil)
        p_local_word_list[#p_local_word_list + 1] = local_word2index[words[j]]
      end
    end
    assert(local_answer_id ~= -1) 
    assert(#p_global_word_list == #p_local_word_list)
    local passage_tensor, passage_rev_tensor = torch.Tensor(params.max_passage_len):fill(0), 
                                                        torch.Tensor(params.max_passage_len):fill(0)
    for i = 1, #p_global_word_list do
      if params.passage_pad == 'right' then
        passage_tensor[params.max_passage_len - i + 1] = p_global_word_list[i]
        passage_rev_tensor[params.max_passage_len - #p_global_word_list + i] = p_global_word_list[i] 
      else
        passage_tensor[i] = p_global_word_list[i]
        passage_rev_tensor[#p_global_word_list - i + 1] = p_global_word_list[i]
      end
    end

    if #local_index2word > params.max_uniq_words_per_passage then
      params.max_uniq_words_per_passage = #local_index2word
    end

    return {passage_tensor, passage_rev_tensor}, p_local_word_list, #local_index2word, local_answer_id
  end
  local meta_info, passage_tensor_master, passage_rev_tensor_master, question_tensor_master, question_rev_tensor_master = {},
                                              tds.hash(), tds.hash(), tds.hash(), tds.hash()
  xlua.progress(1, #lines/22)
  for i = 1, #lines/22 do
    local start = 1 + 22 * (i - 1)
    local last_line = stringx.split(lines[start + 20])
    local answer, answer_cand = last_line[#last_line - 1], stringx.split(last_line[#last_line], '|')
    local question_tensor = get_question(lines[start + 20])
    local passage_tensor, p_local_word_list, local_vocab_size, local_answer_id = get_passage_tensor_n_answer(
                                                                                lines, start, answer, answer_cand)
    passage_tensor_master[#passage_tensor_master + 1] = passage_tensor[1]
    passage_rev_tensor_master[#passage_rev_tensor_master + 1] = passage_tensor[2]
    question_tensor_master[#question_tensor_master + 1] = question_tensor[1]
    question_rev_tensor_master[#question_rev_tensor_master + 1] = question_tensor[2]
    table.insert(meta_info, {p_local_word_list, local_vocab_size, local_answer_id})
    if i % 5 == 0 then xlua.progress(i, #lines/22) end
  end
  xlua.progress(#lines/22, #lines/22)
  return {meta_info, passage_tensor_master, passage_rev_tensor_master, question_tensor_master, question_rev_tensor_master}
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
  word2index = params.word2index,
  max_passage_len = params.max_passage_len,
  max_question_len = params.max_question_len,
  max_uniq_words_per_passage = params.max_uniq_words_per_passage
}
torch.save(params.out, save_point)

--[[

Nngraph based implementation of the NN

]]--

require 'nngraph'
require 'rnn'

function get_model(params)
  local inputs = { nn.Identity()(), nn.Identity()()}
  local question = inputs[1]
  local passage = inputs[2]

  -- encode the question
  local question_word_vectors = nn.LookupTable(#params.dataset.index2word, params.dim)(question):annotate{name = 'question_word_lookup'}
  local question_encoder = nn.BiSequencer(nn.GRU(params.dim, params.hid_size, nil, 0), 
                   nn.GRU(params.dim, params.hid_size, nil, 0):sharedClone(), nn.JoinTable(1, 1))
  								(nn.SplitTable(1, 2)(question_word_vectors)):annotate{name = 'question_encoder'}
  local final_q_out = nn.SelectTable(-1)(question_encoder)   -- get the last step output

  -- encode the passage
  local passage_word_vectors = nn.LookupTable(#params.dataset.index2word, params.dim)(passage):annotate{name = 'passage_word_lookup'}
  local passage_encoder = nn.BiSequencer(nn.GRU(params.dim, params.hid_size, nil, 0), 
                   nn.GRU(params.dim, params.hid_size, nil, 0):sharedClone(), nn.JoinTable(1, 1))
  								(nn.SplitTable(1, 2)(passage_word_vectors)):annotate{name = 'passage_encoder'}
  local final_p_out = nn.View(-1, 2 * params.hid_size)(nn.JoinTable(2)(passage_encoder)) -- combine the forward and backward rnns' output

  -- calculate the final prob
  local soft_out = nn.SoftMax()(nn.Squeeze()(nn.MM(false, true){final_p_out, final_q_out}))

  return nn.gModule(inputs, {soft_out})
end

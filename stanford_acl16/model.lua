--[[

Nngraph based implementation of the NN

]]--

require('nngraph')

function get_model(params)
  local inputs = { nn.Identity()(), nn.Identity()(), nn.Identity()()}
  local question = inputs[1]
  local passage = inputs[2]
  local answer_candidates = inputs[3]

  -- encode the question
  local question_word_vectors = nn.LookupTableMaskZero(#params.dataset.index2word, params.dim)(question):annotate{name = 'question_word_lookup'}
  local question_encoder = nn.BiSequencer(nn.GRU(params.dim, params.hid_size, nil, 0), nn.GRU(params.dim, params.hid_size, nil, 0):sharedClone(), nn.JoinTable(1, 1))
  								(nn.SplitTable(1, 2)(question_word_vectors)):annotate{name = 'question_encoder'}
  local final_q_out = nn.Dropout(params.dropout)(nn.Unsqueeze(3)(nn.SelectTable(-1)(question_encoder)))   -- get the last step output

  -- encode the passage
  local passage_word_vectors = nn.LookupTableMaskZero(#params.dataset.index2word, params.dim)(passage):annotate{name = 'passage_word_lookup'}
  local passage_encoder = nn.BiSequencer(nn.GRU(params.dim, params.hid_size, nil, 0), nn.GRU(params.dim, params.hid_size, nil, 0):sharedClone(), nn.JoinTable(1, 1))
  								(nn.SplitTable(1, 2)(passage_word_vectors)):annotate{name = 'passage_encoder'}
  local final_p_out = nn.Dropout(params.dropout)(nn.View(params.bsize, -1, 2 * params.hid_size)(nn.JoinTable(2)(passage_encoder))) -- combine the forward and backward rnns' output

  -- calculate the attention
  local attention_probs = nn.SoftMax()(nn.MM(false, false)({final_p_out, 
  									nn.Unsqueeze(3)(nn.Linear(2 * params.hid_size, 2 * params.hid_size)(nn.View(2 * params.hid_size)(final_q_out)))}))
  local weighted_out = nn.MM(true, false){final_p_out, attention_probs}

  -- do prediction
  local answer_output_lookup = nn.LookupTableMaskZero(#params.dataset.global_i2e, 2 * params.hid_size)
  local cand_answer_vectors = answer_output_lookup(answer_candidates):annotate{name = 'cand_lookup'}
  local prediction_out = nn.Squeeze()(nn.MM(false, false){cand_answer_vectors, weighted_out})

  return nn.gModule(inputs, {prediction_out})
end

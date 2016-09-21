require 'rnn'

function get_model(params)
  -- question encoder
  local word_lookup = nn.LookupTableMaskZero(#params.dataset.index2word, params.dim)
  local q_gru_layer = nn.GRU(params.dim, params.hid_size, params.max_question_len, 0):maskZero(1)
  local q_fwd_gru = nn.Sequential():add(word_lookup)
                                   :add(q_gru_layer)
  local q_fwd_seq = nn.Sequential():add(nn.SplitTable(1, 2))
                                   :add(nn.Sequencer(q_fwd_gru))
  local q_bwd_gru = nn.Sequential():add(word_lookup:sharedClone())
                                   :add(q_gru_layer:sharedClone())
  local q_bwd_seq = nn.Sequential():add(nn.SplitTable(1, 2))
                                   :add(nn.Sequencer(q_bwd_gru))
  local q_parallel = nn.ParallelTable():add(q_fwd_seq)
                                       :add(q_bwd_seq)
  local q_encoder = nn.Sequential():add(q_parallel)
                                   :add(nn.MaskZero(nn.ZipTable(), 1))
                                   :add(nn.Sequencer(nn.MaskZero(nn.JoinTable(1, 1), 1))) -- merges the fwd, seq out at each time step
                                   :add(nn.Sequencer(nn.MaskZero(nn.Select(1, -1), 2))) -- get the last step output
                                   :add(nn.MaskZero(nn.JoinTable(1), 1))
                                   :add(nn.MaskZero(nn.View(-1, 2 * params.hid_size), 1))
                                   :add(nn.MaskZero(nn.Unsqueeze(3), 2))

  -- passage encoder
  local p_gru_layer = nn.GRU(params.dim, params.hid_size, params.max_passage_len, 0):maskZero(1)
  local p_fwd_gru = nn.Sequential():add(word_lookup:sharedClone())
                                   :add(p_gru_layer)
  local p_fwd_seq = nn.Sequential():add(nn.SplitTable(1, 2))
                                   :add(nn.Sequencer(p_fwd_gru))
  local p_bwd_gru = nn.Sequential():add(word_lookup:sharedClone())
                                   :add(p_gru_layer:sharedClone())
  local p_bwd_seq = nn.Sequential():add(nn.SplitTable(1, 2))
                                   :add(nn.Sequencer(p_bwd_gru))
  local p_parallel = nn.ParallelTable():add(p_fwd_seq)
                                       :add(p_bwd_seq)
  local p_encoder = nn.Sequential():add(p_parallel)
                                   :add(nn.MaskZero(nn.ZipTable(), 1))
                                   :add(nn.Sequencer(nn.MaskZero(nn.JoinTable(1, 1), 1))) -- merges the fwd, seq out at each time step
                                   :add(nn.Sequencer(nn.MaskZero(nn.View(1, -1, 2 * params.hid_size), 1)))
                                   :add(nn.MaskZero(nn.JoinTable(1), 3))
  
  -- build the attention model
  local combiner = nn.ParallelTable():add(p_encoder)
                                     :add(q_encoder)
  local model = nn.Sequential():add(combiner):add(nn.MaskZero(nn.MM(false, false), 2))
                                             :add(nn.MaskZero(nn.Squeeze(), 3))
                                             :add(nn.MaskZero(nn.SoftMax(), 1))
  
  --  bring back all states to the start of the sequence buffers
  params.q_gru_layer = q_gru_layer
  params.q_gru_layer:forget()
  params.p_gru_layer = p_gru_layer
  params.p_gru_layer:forget()

  return model  
end
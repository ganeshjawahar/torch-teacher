require 'rnn'

function get_model(params)
  -- question encoder
  local word_lookup = nn.LookupTableMaskZero(#params.dataset.index2word, params.dim)
  local q_gru_layer = nn.GRU(params.dim, params.hid_size, nil, 0):maskZero(1)
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

  -- passage encoder
  local p_gru_layer = nn.GRU(params.dim, params.hid_size, nil, 0):maskZero(1)
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
  local bilinear_layer = nn.Sequential():add(q_encoder)
                                        :add(nn.MaskZero(nn.Linear(2 * params.hid_size, 2 * params.hid_size), 2))
                                        :add(nn.MaskZero(nn.Unsqueeze(3), 2))
  local alpha_layer_0 = nn.ParallelTable():add(p_encoder)
                                        :add(bilinear_layer)
  local alpha_layer = nn.Sequential():add(alpha_layer_0)
                                     :add(nn.MaskZero(nn.MM(false, false), 2))
                                     :add(nn.MaskZero(nn.Squeeze(), 3))
                                     :add(nn.MaskZero(nn.SoftMax(), 1))
                                     :add(nn.MaskZero(nn.Unsqueeze(3), 2))
  local output_layer_0 = nn.ParallelTable():add(alpha_layer)
                                     :add(p_encoder:sharedClone())
  local output_layer = nn.Sequential():add(output_layer_0)
                                      :add(nn.MaskZero(nn.MM(true, false), 3))
  local output_lookup = nn.LookupTableMaskZero(#params.dataset.global_i2e, 2 * params.hid_size)
  local combiner = nn.ParallelTable():add(output_lookup)
                                     :add(output_layer)
  local model = nn.Sequential():add(combiner)
                               :add(nn.MaskZero(nn.MM(false, true), 3))
                               :add(nn.MaskZero(nn.Squeeze(), 3))
  model = model:cuda()

  --  bring back all states to the start of the sequence buffers
  params.question_encoder = q_gru_layer
  params.question_encoder:forget()
  params.passage_encoder = p_gru_layer
  params.passage_encoder:forget()

  return model  
end
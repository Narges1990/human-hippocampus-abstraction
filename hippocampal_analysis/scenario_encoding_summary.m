
function enc = scenario_encoding_summary(hpc_encoding, neuron_ids)

    idx = ismember(hpc_encoding.neuron, neuron_ids);

    enc.stimulus = mean(hpc_encoding.deltaR2_stim(idx), 'omitnan');
    enc.context  = mean(hpc_encoding.deltaR2_context(idx), 'omitnan');
    enc.response = mean(hpc_encoding.deltaR2_response(idx), 'omitnan');
    enc.reward   = mean(hpc_encoding.deltaR2_reward(idx), 'omitnan');

end

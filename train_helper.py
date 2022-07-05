import itertools
import torch
import torch.autograd as autograd
from torch.autograd import Variable

def do_supervised_minibatch(model,
                            # Source
                            inp_grids, out_grids,
                            # Target
                            in_tgt_seq, in_tgt_seq_list, out_tgt_seq,
                            # Criterion
                            criterion):

    # Get the log probability of each token in the ground truth sequence of tokens.
    #decoder_logit, _ = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)
    io_embedding = model(inp_grids, out_grids, in_tgt_seq, in_tgt_seq_list)
    
    nb_predictions = torch.numel(out_tgt_seq.data)
    # criterion is a weighted CrossEntropyLoss. The weights are used to not penalize
    # the padding prediction used to make the batch of the appropriate size.
    loss = criterion(
        decoder_logit.contiguous().view(nb_predictions, decoder_logit.size(2)),
        out_tgt_seq.view(nb_predictions)
    )

    # Do the backward pass over the loss
    loss.backward()

    # Return the value of the loss over the minibatch for monitoring
    return loss.data[0]
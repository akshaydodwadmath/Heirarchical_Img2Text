
# External imports
import copy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from dataloader import IMG_SIZE

### A module to ensure convolution happens in the correct manner and the required dimensions are obtained 
### after convolution. Example:
# inp_grids torch.Size([16, 5, 16, 18, 18])
# x_feat_shape torch.Size([16, 18, 18])
# flat_x_shape (-1, 16, 18, 18)
# flat_x torch.Size([80, 16, 18, 18]) : x is input
# flat_y torch.Size([80, 32, 18, 18]) : y is output
# y_feat_shape torch.Size([32, 18, 18])
# y_shape torch.Size([16, 5, 32, 18, 18])
class MapModule(nn.Module):
    '''
    Takes as argument a module `elt_module` that as a signature:
    B1 x I1 x I2 x I3 x ... -> B x O1 x O2 x O3 x ...
    This becomes a module with signature:
    B1 x B2 x B3 ... X I1 x I2 x I3 -> B1 x B2 x B3 x ... X O1 x O2 x O3
    '''
    def __init__(self, elt_module, nb_mod_dim):
        super(MapModule, self).__init__()
        self.elt_module = elt_module
        self.nb_mod_dim = nb_mod_dim

    def forward(self, x):
        x_batch_shape = x.size()[:-self.nb_mod_dim]
        x_feat_shape = x.size()[-self.nb_mod_dim:]

        flat_x_shape = (-1, ) + x_feat_shape
        flat_x = x.contiguous().view(flat_x_shape)
        flat_y = self.elt_module(flat_x)

        y_feat_shape = flat_y.size()[1:]
        y_shape = x_batch_shape + y_feat_shape
        y = flat_y.view(y_shape)

        return y

class GridEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack):
        '''
        kernel_size: width of the kernels
        conv_stack: Number of channels at each point of the convolutional part of
                    the network (includes the input)
        fc_stack: number of channels in the fully connected part of the network
        '''
        super(GridEncoder, self).__init__()
        self.conv_layers = []
        for i in range(1, len(conv_stack)):
            if conv_stack[i-1] != conv_stack[i]:
                block = nn.Sequential(
                    ResBlock(kernel_size, conv_stack[i-1]),
                    nn.Conv2d(conv_stack[i-1], conv_stack[i],
                              kernel_size=kernel_size, padding=(kernel_size-1)/2 ),
                    nn.ReLU(inplace=True)
                )
            else:
                block = ResBlock(kernel_size, conv_stack[i-1])
            self.conv_layers.append(block)
            self.add_module("ConvBlock-" + str(i-1), self.conv_layers[-1])

        # We have operated so far to preserve all of the spatial dimensions so
        # we can estimate the flattened dimension.
        first_fc_dim = conv_stack[-1] * IMG_SIZE[-1]* IMG_SIZE[-2]
        adjusted_fc_stack = [first_fc_dim] + fc_stack
        self.fc_layers = []
       
        for i in range(1, len(adjusted_fc_stack)):
            self.fc_layers.append(nn.Linear(adjusted_fc_stack[i-1],
                                            adjusted_fc_stack[i]))
            self.add_module("FC-" + str(i-1), self.fc_layers[-1])
        # TODO: init?

    def forward(self, x):
        '''
        x: batch_size x channels x Height x Width
        '''
        batch_size = x.size(0)

        # Convolutional part
        for conv_layer in self.conv_layers:
            x = conv_layer(x)

        # Flatten for the fully connected part
        x = x.view(batch_size, -1)
        # Fully connected part
        for i in range(len(self.fc_layers) - 1):
            x = F.relu(self.fc_layers[i](x))
        x = self.fc_layers[-1](x)

        return x

class ResBlock(nn.Module):
    def __init__(self, kernel_size, in_feats):
        '''
        kernel_size: width of the kernels
        in_feats: number of channels in inputs
        '''
        super(ResBlock, self).__init__()
        self.feat_size = in_feats
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) / 2

        self.conv1 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv2 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.conv3 = nn.Conv2d(self.feat_size, self.feat_size,
                               kernel_size=self.kernel_size,
                               padding=self.padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.relu(out)

        out = self.conv3(out)

        out += residual
        out = self.relu(out)

        return out
        
class IOsEncoder(nn.Module):
    def __init__(self, kernel_size, conv_stack, fc_stack):
        super(IOsEncoder, self).__init__()

        ## Do one layer of convolution before stacking

        # Deduce the size of the embedding for each grid
        initial_dim = conv_stack[0] // 2  # Because we are going to get dim from I and dim from O

        # TODO: we know that our grids are mostly sparse, and only positive.
        # That means that a different initialisation might be more appropriate.
        self.in_grid_enc = MapModule(nn.Sequential(
            nn.Conv2d(IMG_SIZE[0], initial_dim, kernel_size=kernel_size, padding=(kernel_size -1)//2),
            nn.ReLU(inplace=True)
        ), 3)
        self.out_grid_enc = MapModule(nn.Sequential(
            nn.Conv2d(IMG_SIZE[0], initial_dim,kernel_size=kernel_size, padding=(kernel_size -1)//2),
            nn.ReLU(inplace=True)
        ), 3)

        # Define the model that works on the stacking
        self.joint_enc = MapModule(nn.Sequential(
            GridEncoder(kernel_size, conv_stack, fc_stack)
        ), 3)

    def forward(self, input_grids, output_grids):
        '''
        {input, output}_grids: batch_size x nb_ios x channels x height x width
        '''
        inp_emb = self.in_grid_enc(input_grids)
        out_emb = self.out_grid_enc(output_grids)
        # {inp, out}_emb: batch_size x nb_ios x feats x height x width

        io_emb = torch.cat([inp_emb, out_emb], 2)
        # io_emb: batch_size x nb_ios x 2 * feats x height x width
        joint_emb = self.joint_enc(io_emb)
        # return joint_emb
        return joint_emb
        
class IOs2Seq(nn.Module):

    def __init__(self,
                 # IO encoder
                 kernel_size, conv_stack, fc_stack,
                 # Program Decoder
                 tgt_vocabulary_size,
                 tgt_embedding_dim,
                 decoder_lstm_hidden_size,
                 decoder_nb_lstm_layers,
                 learn_syntax):
        super(IOs2Seq, self).__init__()
        self.encoder = IOsEncoder(kernel_size, conv_stack, fc_stack)
        io_emb_size = fc_stack[-1]
        # self.decoder = MultiIOProgramDecoder(tgt_vocabulary_size,
                                             # tgt_embedding_dim,
                                             # io_emb_size,
                                             # decoder_lstm_hidden_size,
                                             # decoder_nb_lstm_layers,
                                             # learn_syntax)
                                             
                                             
    def forward(self, input_grids, output_grids, tgt_inp_sequences, list_inp_sequences):

        io_embedding = self.encoder(input_grids, output_grids)
        # dec_outs, _, _, syntax_mask = self.decoder(tgt_inp_sequences,
                                                   # io_embedding,
        return  io_embedding                                          # list_inp_sequences)
       # return dec_outs, syntax_mask
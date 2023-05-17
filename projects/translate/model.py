import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import random
import pickle
import os
import re

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## model architectures

### Encoder

class Encoder(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim,
                 dropout, 
                 norm_first,
                 device,
                 max_length = 100):
        super().__init__()

        self.device = device
        
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim,
                                                  dropout,
                                                  norm_first,
                                                  device) 
                                     for _ in range(n_layers)])
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len]
        #src_mask = [batch size, 1, 1, src len]
        
        batch_size = src.shape[0]
        src_len = src.shape[1]
        
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        
        #pos = [batch size, src len]
        
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        
        #src = [batch size, src len, hid dim]
        
        for layer in self.layers:
            src = layer(src, src_mask)
            
        #src = [batch size, src len, hid dim]
            
        return src

### Encoder Layer


class EncoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim,  
                 dropout,                               
                 norm_first,
                 device):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        
    def forward(self, src, src_mask):
        
        #src = [batch size, src len, hid dim]
        #src_mask = [batch size, 1, 1, src len] 
        
        if not self.norm_first:
            #self attention
            _src, _ = self.self_attention(src, src, src, src_mask)
            
            #dropout, residual connection and layer norm
            src = self.norm1(src + self.dropout(_src))
            
            #src = [batch size, src len, hid dim]
            
            #positionwise feedforward
            _src = self.positionwise_feedforward(src)
            
            #dropout, residual and layer norm
            src = self.norm2(src + self.dropout(_src))
            
            #src = [batch size, src len, hid dim]

        else:
            _src = self.norm1(src)
            _src, _ = self.self_attention(_src, _src, _src, src_mask)
            src = src + self.dropout(_src)

            _src = self.norm2(src) 
            _src = self.positionwise_feedforward(_src)   
            src = src + self.dropout(_src)
        
        return src

"""### Mutli Head Attention Layer"""

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        
        assert hid_dim % n_heads == 0
        
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        
    def forward(self, query, key, value, mask = None):
        
        batch_size = query.shape[0]
        
        #query = [batch size, query len, hid dim]
        #key = [batch size, key len, hid dim]
        #value = [batch size, value len, hid dim]
                
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        
        #Q = [batch size, query len, hid dim]
        #K = [batch size, key len, hid dim]
        #V = [batch size, value len, hid dim]
                
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        #Q = [batch size, n heads, query len, head dim]
        #K = [batch size, n heads, key len, head dim]
        #V = [batch size, n heads, value len, head dim]
                
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        #energy = [batch size, n heads, query len, key len]
        
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        
        attention = torch.softmax(energy, dim = -1)
                
        #attention = [batch size, n heads, query len, key len]
                
        x = torch.matmul(self.dropout(attention), V)
        
        #x = [batch size, n heads, query len, head dim]
        
        x = x.permute(0, 2, 1, 3).contiguous()
        
        #x = [batch size, query len, n heads, head dim]
        
        x = x.view(batch_size, -1, self.hid_dim)
        
        #x = [batch size, query len, hid dim]
        
        x = self.fc_o(x)
        
        #x = [batch size, query len, hid dim]
        
        return x, attention

### Position-wise Feedforward Layer


class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        
        self.fc_1 = nn.Linear(hid_dim, pf_dim)
        self.fc_2 = nn.Linear(pf_dim, hid_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        
        #x = [batch size, seq len, hid dim]
        
        x = self.dropout(torch.relu(self.fc_1(x)))
        
        #x = [batch size, seq len, pf dim]
        
        x = self.fc_2(x)
        
        #x = [batch size, seq len, hid dim]
        
        return x

### Decoder

class Decoder(nn.Module):
    def __init__(self, 
                 output_dim, 
                 hid_dim, 
                 n_layers, 
                 n_heads, 
                 pf_dim, 
                 dropout, 
                 norm_first,
                 device,                
                 max_length = 100):
        super().__init__()
        
        self.device = device
        
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, 
                                                  n_heads, 
                                                  pf_dim, 
                                                  dropout,                                                   
                                                  norm_first,
                                                  device)
                                     for _ in range(n_layers)])
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        #trg = [batch size, trg len]
        #enc_src = [batch size, src len, hid dim]
        #trg_mask = [batch size, 1, trg len, trg len]
        #src_mask = [batch size, 1, 1, src len]
                
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
                            
        #pos = [batch size, trg len]
            
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
                
        #trg = [batch size, trg len, hid dim]
        
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        
        #trg = [batch size, trg len, hid dim]
        #attention = [batch size, n heads, trg len, src len]
        
        output = self.fc_out(trg)
        
        #output = [batch size, trg len, output dim]
            
        return output, attention

### Decoder Layer

class DecoderLayer(nn.Module):
    def __init__(self, 
                 hid_dim, 
                 n_heads, 
                 pf_dim, 
                 dropout,
                 norm_first,
                 device):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hid_dim)
        self.norm2 = nn.LayerNorm(hid_dim)
        self.norm3 = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforwardLayer(hid_dim, 
                                                                     pf_dim, 
                                                                     dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_first = norm_first
        
    def forward(self, trg, enc_src, trg_mask, src_mask):
        
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, 1, trg len, trg len]
        # src_mask = [batch size, 1, 1, src len]
        
        if not self.norm_first:
            #self attention
            _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
            
            #dropout, residual connection and layer norm
            trg = self.norm1(trg + self.dropout(_trg))
                
            #trg = [batch size, trg len, hid dim]
                
            #encoder attention
            _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
            
            #dropout, residual connection and layer norm
            trg = self.norm2(trg + self.dropout(_trg))
                        
            #trg = [batch size, trg len, hid dim]
            
            #positionwise feedforward
            _trg = self.positionwise_feedforward(trg)
            
            #dropout, residual and layer norm
            trg = self.norm3(trg + self.dropout(_trg))
            
            # trg = [batch size, trg len, hid dim]
            # attention = [batch size, n heads, trg len, src len]
        else:
            _trg = self.norm1(trg)
            _trg, _ = self.self_attention(_trg, _trg, _trg, trg_mask)
            trg = trg + self.dropout(_trg)

            _trg = self.norm2(trg)
            _trg, attention = self.encoder_attention(_trg, enc_src, enc_src, src_mask)
            trg = trg + self.dropout(_trg)

            _trg = self.norm3(trg)
            _trg = self.positionwise_feedforward(_trg)
            trg = trg + self.dropout(_trg)
        
        return trg, attention

### Seq2Seq

class Seq2Seq(nn.Module):
    def __init__(self, 
                 encoder, 
                 decoder, 
                 src_pad_idx, 
                 trg_pad_idx, 
                 device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device
        
    def make_src_mask(self, src):
        
        #src = [batch size, src len]
        
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)

        #src_mask = [batch size, 1, 1, src len]

        return src_mask
    
    def make_trg_mask(self, trg):
        
        #trg = [batch size, trg len]
        
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        
        #trg_pad_mask = [batch size, 1, 1, trg len]
        
        trg_len = trg.shape[1]
        
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device = self.device)).bool()
        
        #trg_sub_mask = [trg len, trg len]
            
        trg_mask = trg_pad_mask & trg_sub_mask
        
        #trg_mask = [batch size, 1, trg len, trg len]
        
        return trg_mask

    def forward(self, src, trg):
        
        #src = [batch size, src len]
        #trg = [batch size, trg len]
                
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        
        #src_mask = [batch size, 1, 1, src len]
        #trg_mask = [batch size, 1, trg len, trg len]
        
        enc_src = self.encoder(src, src_mask)
        
        #enc_src = [batch size, src len, hid dim]
                
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        
        #output = [batch size, trg len, output dim]
        #attention = [batch size, n heads, trg len, src len]
        
        return output, attention



def read_vocab(path):
    #read vocabulary pkl 
    pkl_file = open(path, 'rb')
    vocab = pickle.load(pkl_file)
    pkl_file.close()
    return vocab

def load_model_and_vocab():

    dir_path = os.path.abspath(os.path.dirname(__file__))

    #Load Vocab
    SRC_vocab = read_vocab(dir_path + '/src_vocab.pkl')
    TRG_vocab = read_vocab(dir_path + '/trg_vocab.pkl')

    #Load Model
    INPUT_DIM = len(SRC_vocab)
    OUTPUT_DIM = len(TRG_vocab)
    HID_DIM = 256
    ENC_LAYERS = 3
    DEC_LAYERS = 3
    ENC_HEADS = 8
    DEC_HEADS = 8
    ENC_PF_DIM = 512
    DEC_PF_DIM = 512
    ENC_DROPOUT = 0.1
    DEC_DROPOUT = 0.1
    ENC_NORM_FIRST = True 
    DEC_NORM_FIRST = True

    enc = Encoder(INPUT_DIM, 
              HID_DIM, 
              ENC_LAYERS, 
              ENC_HEADS, 
              ENC_PF_DIM, 
              ENC_DROPOUT,
              ENC_NORM_FIRST,
              device)

    dec = Decoder(OUTPUT_DIM, 
                HID_DIM, 
                DEC_LAYERS, 
                DEC_HEADS, 
                DEC_PF_DIM, 
                DEC_DROPOUT,
                DEC_NORM_FIRST,
                device)

    """Then, use them to define our whole sequence-to-sequence encapsulating model."""

    SRC_PAD_IDX = SRC_vocab.stoi['<pad>']
    TRG_PAD_IDX = TRG_vocab.stoi['<pad>']

    model = Seq2Seq(enc, dec, SRC_PAD_IDX, TRG_PAD_IDX, device).to(device)
    model.load_state_dict(torch.load(dir_path + '/min-zh.pt', map_location=torch.device('cpu')))

    return SRC_vocab, TRG_vocab, model


def translate_sentence_beam_search(sentence, src_field, trg_field, model, beam_width, alpha, return_sent=3):
    
    model.eval()
    m = nn.Softmax(dim=-1)
        
    if isinstance(sentence, str):       
        tokens = re.split("-| ", sentence) #tokenize_min(sentence)
    else: 
        tokens = [token for token in sentence]

    max_len = len(tokens) 
    
    tokens = ['<sos>'] + tokens + ['<eos>']
        
    src_indexes = [src_field.stoi[token] for token in tokens]
  
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    B = beam_width   

    score_arr = torch.zeros((B)).to(device)
    token_arr = torch.full((B, 1), trg_field.stoi['<sos>']).to(device)
    complete_seqs = []
    complete_seqs_scores = []
    
    for i in range(max_len + 3):
        if i == 0:
            trg_tensor = token_arr[0].clone().unsqueeze(0)
            trg_mask = model.make_trg_mask(trg_tensor)
            with torch.no_grad():
                output, _ = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
            preds = m(output).topk(B)
        
            score_arr = preds.values.log2().view(-1)
            token_arr = torch.cat((token_arr, preds.indices.view(-1,1)), dim=1) # 每句長度為 2

        else:
            trg_mask = model.make_trg_mask(token_arr)

            with torch.no_grad():
                output, _ = model.decoder(token_arr, enc_src.repeat(B, 1, 1), trg_mask, src_mask.repeat(B, 1, 1, 1))

            preds =  m(output).topk(B)
            next_word_indexes =  preds.indices[:,-1]# Get B * B candidates
            next_word_scores = preds.values[:,-1].log2()

            new_seqs = torch.cat((torch.repeat_interleave(token_arr, B, dim=0), next_word_indexes.reshape(-1,1)), dim=-1)
            new_scores = torch.add(torch.repeat_interleave(score_arr, B, dim=0), next_word_scores.flatten())
            
            topB = new_scores.topk(B)               
            score_arr = topB.values
        
            token_arr = torch.index_select(new_seqs, 0, topB.indices)
            
            incomplete_idxs = []
            for b in range(B):
                if token_arr[b][-1] == trg_field.stoi['<eos>']:
                    complete_seqs.append(token_arr[b].tolist())
                    complete_seqs_scores.append(float(score_arr[b] / math.pow(len(token_arr[b]), alpha)))
                    B -= 1
                else:
                    incomplete_idxs.append(b)

            B -= (B - len(incomplete_idxs))
            token_arr = token_arr[incomplete_idxs]
            score_arr = score_arr[incomplete_idxs]

            if B == 0:
                break
    
    topk_seqs = []
    topk_scores = list(sorted(complete_seqs_scores, reverse=True))[:return_sent]

    for score in topk_scores:
        seq_idx = complete_seqs_scores.index(score)
        topk_seqs.append([trg_field.itos[i] for i in complete_seqs[seq_idx]][1:-1])
    
    if len(topk_seqs) != return_sent:
        return_sent -= len(topk_seqs)
        topB = score_arr.topk(return_sent)        
        incomplete_seqs = torch.index_select(token_arr, 0, topB.indices)
        
        for s in incomplete_seqs:
            topk_seqs.append([trg_field.itos[i] for i in s[1:]])
    
    # base = torch.full((return_sent,), 2).to(device)
    # topk_scores = torch.pow(base, torch.tensor(topk_scores).to(device)).tolist()

    return topk_seqs


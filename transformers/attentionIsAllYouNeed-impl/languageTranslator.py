import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x):
        return self.embedding * math.sqrt(self.d_model)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model :int, seq_len, dropout: float) -> None: 
        ## seq_len -- max length of sentence to create that many vectors, dropout -- to reduce overfitting 
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = dropout
        
        ## Create matrix of shape (seq_len, d_model) -- positional embeddig for each word in sentence
        pe = torch.zeros(seq_len, d_model)
        ## Create matrix of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) ## Numerator
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) ## Denominator
        # Apply sin to even position and cos to odd position
        pe[:, 0::2] = torch.sin(position * div_term) ## [:,0::2] apply for all elements starting from 2, skipping 2
        pe[:, 1::2] = torch.cos(position * div_term)
        
        ## Add batch dimension to this pe so that it can be trained in batches
        pe = pe.unsqueeze(0) ## Shape changes to (1, seq_len, d_model)
        
        self.register_buffer('pe',pe) ## Tensor will be saved with the file along with model
        
    def forward(self, x):
        ## Add positional encoding to every word in the sentenc
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) ## Requires grad is used to tell that this should not be learnt and should be fixed
        return self.dropout(x)
    
class LayerNormilization(nn.Module):
    def __init__(self, eps :float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) ## nn.Parameter makes it learnable ## Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) ## added
        
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
        
class FeedForwardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(d_model, d_ff) ## W1 and B1
        self.linear_2 = nn.Linear(d_ff, d_model) ## W2 and B2
        
    def forward(self, x):
        ## (Batch, seq_len, d_model) --> linear1 to get (Batch, seq_len, d_ff) --> linear2 to get (Batch, seq_len, d_model)
        return self.linear_2(
            self.dropout(
                torch.relu(
                    self.linear_1(x)
                    )))
        
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by 0"
        
        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) ## Wq
        self.w_k = nn.Linear(d_model, d_model) ## Wk
        self.w_v = nn.Linear(d_model, d_model) ## Wv
        
        self.w_o = nn.Linear(d_model, d_model) ## Wo
    
    @staticmethod    
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]
        
        ## (Batch, h, seq_len, d_k) -->  (Batch, h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) ## query @ does matrix multiplication
        if mask is not None: 
            attention_score.masked_fill_(mask == 0 , -1e9)
            
        attention_score = attention_score.softmax(dim=-1) ## (Batch, h, seq_len, seq_len)
        
        if dropout is not None:
            attention_score = dropout(attention_score)
            
        return (attention_score @ value), attention_score
        
    def forward(self, q, k, v, mask):
        query = self.w_q(q) ## (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 
        key = self.w_k(k) ## (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 
        value = self.w_v(v) ## (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) 
        
        ## (Batch, seq_len, d_model) --> (Batch, seq_len, h, d_k) --transpose--> (Batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        ## (Batch, h, seq_len, d_k) --> ## (Batch, seq_len, h, d_k) --> ## (Batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        
         ## (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
        return self.w_o(x)
    
class ResidualConnection(nn.Module): ## For skip connection from MultiHeadAttentionBlock & FeedforwardBlock to LayerNormalizationBlock
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormilization()
    
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    
class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])
        
    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x
    
class Encoder(nn.Module):
    def __init__(self,  layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormilization()
        
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
            
        return self.norm(x)
    
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, 
                 cross_attention_block: MultiHeadAttentionBlock, 
                 feed_forward_block: FeedForwardBlock,
                 dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    
class Decoder(nn.Module): 
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers =  layers
        self.norm = LayerNormilization()
        
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers: 
            x = layer(x, encoder_output, src_mask, tgt_mask)
        
        return self.norm(x)
    
## Project vocabulary on embeddings produced. converts back to language
class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__( )
        self.proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, vocab_size)
        return torch.log_softmax(self.proj(x), dim = -1)
    
class Transformer(nn.Module):
    
    def __init__(self, encoder: Encoder, 
                 decoder: Decoder, 
                 src_embedding: InputEmbeddings, 
                 tgt_embedding: InputEmbeddings,
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer
                 ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embedding = src_embedding
        self.tgt_embedding = tgt_embedding
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
        
    def encode(self, src, src_mask):
        src = self.src_embedding(src)
        src = self.src_pos(src)
        src = self.encoder(src, src_mask)
        return src
    
    def decode(self, encoder_output, src_mask, tgt_mask):
        tgt = self.tgt_embedding(tgt)
        tgt = self.tgt_pos(tgt)
        tgt = self.decoder(tgt, encoder_output, src_mask, tgt_mask)
        return tgt
    
    def project(self, x):
        return self.projection_layer(x)
    
def build_transformer(src_vocab_size: int, 
                      tgt_vocab_size: int, 
                      src_seq_len: int, 
                      tgt_seq_len: int, 
                      d_model: int = 512, 
                      N: int = 6,
                      h: int = 8,
                      dropout: float = 0.1, 
                      d_ff = 2048):
    ## Create embedding layers
    src_embed = InputEmbeddings(d_model, src_vocab_size)
    tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)
    
    ##Create positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)
    
    ## Create encoder block
    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)
        
    ##Create Decoder block
    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks = decoder_blocks.append(decoder_block)
        
    ## Create encoder and decoder using arrays
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))
    
    #Create project layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)
    
    #Create transformer
    transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)
    
    ## Initialize parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
            
    return transformer


    

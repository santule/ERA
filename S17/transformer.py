import torch
from torch import nn
from torch.nn import functional as F

# PATCH EMBEDDING
class PatchEmbedding(nn.Module):

  def __init__(self,in_channels:int=3, patch_size:int=16,embedding_dim:int=768):
    super().__init__()

    # ( (224 - 16) / 16)  + 1
    self.patcher = nn.Conv2d(in_channels=in_channels,out_channels=embedding_dim,kernel_size=patch_size,stride=patch_size,padding=0)
    self.flatten = nn.Flatten(start_dim=2,end_dim=3)
    self.patch_size = patch_size

  def forward(self,x):
    image_resolution = x.shape[-1]
    assert image_resolution % self.patch_size ==0,f"Input image must be divisible by patch size"

    x_patched = self.patcher(x)
    #print(f"x_patched shape {x_patched.shape}")
    x_flattened = self.flatten(x_patched)
    #print(f"x_flattened shape {x_flattened.shape}")

    return x_flattened.permute(0, 2, 1)

# SELF ATTENTION - DECODER
class AttentionHead(nn.Module):
    """
    One head of the self-attention layer
    """

    def __init__(self, head_size, num_embed, seq_len, dropout):
        super().__init__()
        self.key = nn.Linear(num_embed, head_size, bias=False)
        self.query = nn.Linear(num_embed, head_size, bias=False)
        self.value = nn.Linear(num_embed, head_size, bias=False)
        # tril is a lower triangular matrix. it is not a parameter
        # of the model, so we assign it to the module using register_buffer
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

        # let's also add dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        # compute attention scores
        # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        # Tril matrix (lower triagular matrix) is used to mask 
        # future positions (setting them to -inf) so that the
        # decoder "learns" to predict next words
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B,T,T)
        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)
        # weighted aggregation of the values
        v = self.value(x)
        out = wei @ v  # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out


# MULTI HEAD ATTENTION - DECODER
class MultiHeadAttentionBlock_Decoder(nn.Module):
    """
    Multiple Heads of self-attention in parallel
    """
    def __init__(self, embedding_dim, num_heads, attn_dropout, seq_len):
        super().__init__()
        self.head_size = embedding_dim // num_heads

        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    head_size= self.head_size,
                    num_embed=embedding_dim,
                    seq_len=seq_len,
                    dropout=attn_dropout,
                )
                for _ in range(num_heads)
            ]
        )
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        # output of the self-attention
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # apply the linear projection layer
        out = self.dropout(self.proj(out))
        return out


# MULTIHEAD ATTENTION BLOCK
class MultiHeadAttentionBlock(nn.Module):
  def __init__(self, embedding_dim:int=768, num_heads:int=12,attn_dropout:float=0,mask=None):
    super().__init__()
    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
    self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim, num_heads=num_heads,dropout=attn_dropout,batch_first=True)

  def forward(self,x):
    x = self.layer_norm(x)
    attn_output,_ = self.multihead_attn(query=x,key=x,value=x,need_weights=False)

    return attn_output
  
# MLP BLOCK
class MLPBlock(nn.Module):
  def __init__(self, embedding_dim:int=768, mlp_size:int=3072,dropout:float=0.1):
    super().__init__()

    self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
    self.mlp = nn.Sequential(
        nn.Linear(in_features=embedding_dim,
                  out_features=mlp_size),
        nn.GELU(),
        nn.Dropout(p=dropout),
        nn.Linear(in_features = mlp_size, out_features=embedding_dim),
        nn.Dropout(p=dropout)
    )

  def forward(self,x):
    x = self.layer_norm(x)
    x = self.mlp(x)

    return x
  
# TRANSFORMER ENCODER BLOCK
class TransformerEncoderBlock(nn.Module):
  def __init__(self,embedding_dim:int=768, num_heads:int=12,mlp_size:int=3072,mlp_dropout:float=0.1,attn_dropout:float=0):

    super().__init__()
    self.msa_block = MultiHeadAttentionBlock(embedding_dim=embedding_dim, num_heads=num_heads,attn_dropout=attn_dropout)
    self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size,dropout=mlp_dropout)

  def forward(self,x):
    x = self.msa_block(x) + x
    x = self.mlp_block(x) + x

    return x

# TRANSFORMER DECODER BLOCK
class TransformerDecoderBlock(nn.Module):
  def __init__(self,embedding_dim:int=768, num_heads:int=12,mlp_size:int=3072,mlp_dropout:float=0.1,attn_dropout:float=0,seq_len:int=64):

    super().__init__()
    self.msa_block = MultiHeadAttentionBlock_Decoder(embedding_dim=embedding_dim, num_heads=num_heads,attn_dropout=attn_dropout,seq_len=seq_len)
    self.mlp_block = MLPBlock(embedding_dim=embedding_dim, mlp_size=mlp_size,dropout=mlp_dropout)

  def forward(self,x):
    x = self.msa_block(x) + x
    x = self.mlp_block(x) + x

    return x

# MODEL 1 - VIT
class ViT(nn.Module):
  def __init__(self,
               img_size:int=224,
               in_channels:int=3,
               patch_size:int=16,
               num_transformer_layers:int=12,
               embedding_dim:int=768,
               mlp_size:int=3072,
               num_heads:int=12,
               attn_dropout:float=0,
               mlp_dropout:float=0.1,
               embedding_dropout:float=0.1,
               num_classes:int=1000):
    
      super().__init__()

      assert img_size % patch_size == 0,f"Input image must be divisible by patch size"

      self.num_patches = (img_size *img_size) // patch_size**2 # 224*224 / 16 * 16
      self.patch_embedding = PatchEmbedding(in_channels = in_channels,patch_size=patch_size,embedding_dim= embedding_dim)
      self.positional_embedding = nn.Parameter(data = torch.rand(1,self.num_patches + 1, embedding_dim), requires_grad=True)
      self.class_embedding = nn.Parameter(data = torch.rand(1,1,embedding_dim), requires_grad = True)
      self.embedding_dropout = nn.Dropout(p= embedding_dropout)

      self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim= embedding_dim, num_heads=num_heads,mlp_size=mlp_size,mlp_dropout=mlp_dropout,attn_dropout=attn_dropout)
                                                for _ in range(num_transformer_layers)])

      self.classifier = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim), nn.Linear(in_features = embedding_dim,out_features = num_classes))

  def forward(self,x):

    batch_size = x.shape[0]
    class_token = self.class_embedding.expand(batch_size,-1,-1)

    # embedding
    x = self.patch_embedding(x)
    x = torch.cat((class_token,x),dim=1)
    x = self.positional_embedding + x
    x = self.embedding_dropout(x)

    # transformer encoder
    x = self.transformer_encoder(x)
    x = self.classifier(x[:,0])

    return x

# MODEL 2 - BERT
class Bert(nn.Module):
  def __init__(self,
               n_embeddings:int=40000,
               seq_len:int=20,
               num_transformer_layers:int=8,
               embedding_dim:int=128,
               mlp_size:int=128 * 4,
               num_heads:int=8,
               attn_dropout:float=0.1,
               mlp_dropout:float=0.1,
               embedding_dropout:float=0):
    
      super().__init__()

      self.embeddings = nn.Embedding(n_embeddings,embedding_dim)
      self.positional_embedding = nn.Parameter(data = torch.rand(1,seq_len,embedding_dim), requires_grad=True)
      self.embedding_dropout = nn.Dropout(p= embedding_dropout)

      self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim= embedding_dim, num_heads=num_heads,mlp_size=mlp_size,mlp_dropout=mlp_dropout,attn_dropout=attn_dropout)
                                                for _ in range(num_transformer_layers)])
      self.output_embedding = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim), nn.Linear(in_features = embedding_dim,out_features = n_embeddings,bias=False))


  def forward(self,x):

    # embedding
    x = self.embeddings(x)
    x = self.positional_embedding + x
    x = self.embedding_dropout(x)

    # transformer encoder
    x = self.transformer_encoder(x)
    x = self.output_embedding(x)
    return x
  

# MODEL 3 -GPT
class Gpt(nn.Module):
  def __init__(self,
               n_embeddings:int=40000,
               seq_len:int=64,
               num_transformer_layers:int=6,
               embedding_dim:int=128 * 6,
               mlp_size:int=128 * 4,
               num_heads:int=6,
               attn_dropout:float=0.1,
               mlp_dropout:float=0.1,
               embedding_dropout:float=0):

      super().__init__()

      self.embeddings = nn.Embedding(n_embeddings,embedding_dim)
      self.positional_embedding = nn.Parameter(data = torch.rand(1,seq_len,embedding_dim), requires_grad=True) #nn.Embedding(seq_len,embedding_dim)
      self.embedding_dropout = nn.Dropout(p= embedding_dropout)

      self.transformer_decoder = nn.Sequential(*[TransformerDecoderBlock(embedding_dim= embedding_dim, num_heads=num_heads,mlp_size=mlp_size,mlp_dropout=mlp_dropout,attn_dropout=attn_dropout,seq_len=seq_len)
                                                for _ in range(num_transformer_layers)])
      self.output_logits = nn.Sequential(nn.LayerNorm(normalized_shape=embedding_dim), nn.Linear(in_features = embedding_dim,out_features = n_embeddings,bias=False))


  def forward(self,x,targets=None):

    B,T = x.shape # batch,block or seq length

    # embedding
    x = self.embeddings(x)
    #pos_embed = self.positional_embedding(torch.arange(T))
    #x = token_embed + pos_embed
    x = self.positional_embedding + x
    x = self.embedding_dropout(x)

    # transformer encoder
    x = self.transformer_decoder(x)
    logits = self.output_logits(x)

    if targets != None:
        B, T, C = logits.shape
        logits = torch.reshape(logits, (B * T, C))
        targets = torch.reshape(targets, (B * T,))
        loss = F.cross_entropy(logits, targets)
    else:
        loss = None
    return logits, loss


  def generate(self,idx: torch.Tensor, max_new_tokens: int,block_size: int):

    for _ in range(max_new_tokens):
      idx_crop = idx[:, -block_size:]
      logits, loss = self.forward(idx_crop)

      logits = logits[:,-1,:]
      probs = F.softmax(logits,dim=-1)
      idx_next = torch.multinomial(probs,num_samples=1)
      idx = torch.cat((idx, idx_next), dim=1)

    return idx
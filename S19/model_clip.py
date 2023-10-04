from torch import nn
from PIL import Image
from tqdm.auto import tqdm
import albumentations as A
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import torch.optim as optim
import timm
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ImageEncoder(nn.Module):
  def __init__(
      self
  ):
    super().__init__()
    self.model = timm.create_model('resnet50', pretrained = True,
                                   num_classes = 0, global_pool = 'avg')

    for p in self.model.parameters():
      p.requires_grad = True

  def forward(self,x):
    return self.model(x)

class TextEncoder(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    for p in self.model.parameters():
      p.requires_grad = True

  def forward(self,input_ids, attention_mask):
    output = self.model(input_ids=input_ids,attention_mask = attention_mask)
    last_hidden_state = output.last_hidden_state
    return last_hidden_state[:,0,:] # CLS TOKEN

class ProjectionHead(nn.Module):
    
  def __init__(
      self,
      embedding_dim,
      projection_dim = 256,
      dropout = 0.1
  ):
    super().__init__()
    self.projection = nn.Linear(embedding_dim,projection_dim)
    self.gelu       = nn.GELU()
    self.fc         = nn.Linear(projection_dim,projection_dim)
    self.dropout    = nn.Dropout(dropout)
    self.layer_norm = nn.LayerNorm(projection_dim)

  def forward(self,x):
    projected = self.projection(x)
    x = self.gelu(projected)
    x = self.fc(x)
    x = self.dropout(x)
    x = x + projected
    x = self.layer_norm(x)

    return x
  
class CLIPModel(nn.Module):
  def __init__(
      self,
      image_embedding = 2048,
      text_embedding  = 768
  ):
    super().__init__()
    self.image_encoder = ImageEncoder()
    self.text_encoder = TextEncoder()

    self.image_projection = ProjectionHead(embedding_dim=image_embedding)
    self.text_projection  = ProjectionHead(embedding_dim=text_embedding)


  def forward(self,batch):
    image_features = self.image_encoder(batch['image'].to(device))
    text_features  = self.text_encoder(
        input_ids  = batch['input_ids'].to(device),attention_mask = batch["attention_mask"].to(device)
    )
    image_embeddings = self.image_projection(image_features)
    text_embeddings = self.text_projection(text_features)

    return text_embeddings, image_embeddings

def loss_fn(text_embeddings,image_embeddings,temperature):
    logits = (text_embeddings @ image_embeddings.T) / temperature
    image_similarity = image_embeddings @ image_embeddings.T
    text_similarity  = text_embeddings @ text_embeddings.T

    targets = F.softmax(
        (image_similarity + text_similarity)/(2 * temperature), dim=-1
    )
    texts_loss  = cross_entropy(logits, targets)
    images_loss = cross_entropy(logits.T, targets.T)
    loss = (images_loss + texts_loss) / 2.0
    return loss.mean()

def cross_entropy(preds, targets):
    log_softmax = nn.LogSoftmax(dim=-1)
    loss = (- targets * log_softmax(preds)).sum(1)
    return loss
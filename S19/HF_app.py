import gradio as gr
import torch
import numpy as np
import os
import csv
import timm
import torch
import requests
import numpy as np
from torch import nn
from PIL import Image
import torch.nn.functional as F
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import pickle
import model_clip

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# load validation data (400)
file = open("val_genre.pkl",'rb')
genre = pickle.load(file)
file.close()
file = open("val_images_url.pkl",'rb')
poster_url = pickle.load(file)
file.close()
movie_poster_embeddings = torch.load('val_image_embeddings.pt',map_location=torch.device(device))
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

my_clip = model_clip.CLIPModel().to(device)
my_clip.load_state_dict(torch.load('my_clip',map_location=torch.device(device)))
my_clip.eval()


# gradio app
with gr.Blocks() as demo:

    def movie_recommend(query):
      movie_list = []

      encoded_query = tokenizer([query])
      batch = {
        key: torch.tensor(values).to(device)
        for key, values in encoded_query.items()
      }
      text_features   = my_clip.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
      text_embeddings = my_clip.text_projection(text_features)

      image_embeddings_n = F.normalize(movie_poster_embeddings, p=2,  dim=-1)
      text_embeddings_n  = F.normalize(text_embeddings,  p=2,  dim=-1)
      dot_similarity = text_embeddings_n @ image_embeddings_n.T
      values, indices = torch.topk(dot_similarity.squeeze(0), 10)

      indices = np.random.choice(indices.to('cpu'),4,replace=False)
      matches = [poster_url[idx] for idx in indices]
      matches_genre = [genre[idx] for idx in indices]

      for m in matches:
        img_https_link = 'https://images-na.ssl-images-amazon.com/images/M/' + m.split('/')[-1]
        movie_list.append(np.array(Image.open(requests.get(img_https_link, stream=True).raw).convert('RGB')))

      return movie_list #x[::-1]


    gr.Markdown(
        """
    # Discover your next Movie !!!
    Discover your next movie by providing prompt based on genre or combination of genres.
    """
    )
    with gr.Column(variant="panel"):
        with gr.Row():
            text = gr.Textbox(
                label="Enter your prompt",
                max_lines=1,
                placeholder="Show me movies of genres ....",
                container=False,
            )
            btn = gr.Button("Show Movies", scale=0)

        gallery = gr.Gallery(
            label="Movies", show_label=False, elem_id="gallery"
        , columns=[4], rows=[1], object_fit="contain", height="auto")

    btn.click(movie_recommend, text, gallery)

if __name__ == "__main__":
    demo.launch()


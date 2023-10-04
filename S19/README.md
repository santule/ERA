# Building and Training CLIP model on movie dataset

## Motivation:
Suggest movies based on user prompts. Currently it supports prompts based on genre.

![Screenshot 2023-10-04 at 11 52 13 pm](https://github.com/santule/ERA/assets/20509836/af5e40da-f479-42d7-9480-3eabd15dd938)


## Model details:
In this repo, we build CLIP model using Distilbert text encoder and RESNET50 image decoder on Movie Poster dataset. 
The projection head with 256 dimension on the top of the encoders is trained for 20 epochs with the CLIP contrastive loss function.


## Dataset:
Movie Poster dataset contains approx. 40k movies posters and its genre. It is available https://www.kaggle.com/datasets/neha1703/movie-genre-from-its-poster
The genre text was modified with different kinds of user prompts while training the model.

## App:
The gradio app is available at https://huggingface.co/spaces/sanjanatule/clipmovie

## Future work:
The movie plot, actors and producers/directors can be included as a part of the training the model.






## Multimodal GPT training

### STEP 1 - PRETRAIN FOR IMAGE - LANGUAGE CAPTIONS

1. Dataset : Coco 2017 train dataset (URL:: http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
2. CLIP Model: clip-vit-base-patch32 (URL:: https://huggingface.co/openai/clip-vit-base-patch32)
3. LLM Model: Phi2 model ((URL:: https://huggingface.co/microsoft/phi-2)
4. 48 GB RAM with batch size 2


### TRAIN LOSS
![step1_pretraining_train_loss](https://github.com/santule/ERA/assets/20509836/ae8feb28-6ae8-44ba-9aa7-4732be97b3bf)

### EXAMPLE PREDICTIONS FROM THE MODEL
```
STEP 4000 (BATCH SIZE 2)
0 - Target captions:
 a woman sitting at a table with some scrapbook items <|endoftext|>  
0 - predicted_captions:
 A man is sitting at a desk with a a a a a a a a a a a<|endoftext|> 
1 - Target captions:
 A desk with two laptops at a computer monitor.<|endoftext|><|endoftext|><|endoftext|>  
1 - predicted_captions:
 A desk with a computer and a a a a a a a a a a a a a<|endoftext|> 
2 - Target captions:
 A woman hitting a tennis ball with a tennis racket.<|endoftext|><|endoftext|>  
2 - predicted_captions:
 A baseball game is being played on a field..........<|endoftext|> 
3 - Target captions:
 Refrigerator door open with a lot of food inside.   
3 - predicted_captions:
 A refrigerator with a variety of food inside of it.........<|endoftext|>
```

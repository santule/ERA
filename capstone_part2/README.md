## Multimodal GPT training


<img width="1024" alt="Screenshot 2024-01-27 at 10 17 49 pm" src="https://github.com/santule/ERA/assets/20509836/a32351e8-5c76-4c78-b607-65e333424653">


#### STEP 1 - PRETRAIN FOR IMAGE - LANGUAGE CAPTIONS

1. Dataset : Coco 2017 train dataset (URL:: http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
2. CLIP Model: clip-vit-base-patch32 (URL:: https://huggingface.co/openai/clip-vit-base-patch32)
3. LLM Model: Phi2 model ((URL:: https://huggingface.co/microsoft/phi-2)
4. 48 GB RAM with batch size 2


#### Train loss
![step1_pretraining_train_loss](https://github.com/santule/ERA/assets/20509836/ae8feb28-6ae8-44ba-9aa7-4732be97b3bf)

#### Example predictions 
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
#### STEP 2 - FINETUNING FOR IMAGE - LANGUAGE QUESTION N ANSWERS

1. Dataset : Instruct150k Dataset (URL:: https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K)
2. CLIP Model: clip-vit-base-patch32 (URL:: https://huggingface.co/openai/clip-vit-base-patch32)
3. LLM Model: Phi2 model ((URL:: https://huggingface.co/microsoft/phi-2)
4. 48 GB RAM with batch size 32


#### Train loss
![W B Chart 27_01_2024, 23_05_51](https://github.com/santule/ERA/assets/20509836/813853d6-5424-4d5f-8f7f-c25a7c496f7b)

#### Example predictions
```
Image: http://images.cocodataset.org/train2017/000000410743.jpg
Question: Is there any specific direction indicated on one of the street signs?
Answer:   Yes, one of the street signs presents an arrow pointing to the left.
Model Predicted Ans: ['The street sign reads "No Left Turn" in English.\n\nThe street sign reads "No Left Turn" in English.\n\nThe street sign reads "No Left Turn" in English.\n\nThe street sign reads "No Left Turn" in English.\n\nThe street sign reads "No Left Turn" in English.\n\nThe street sign reads "No Left Turn" in English.\n\nThe street sign reads "No Left Turn" in English.\n\nThe street']
```


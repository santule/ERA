## Multimodal GPT training

Mulitmodal GPT training using CLIP and Phi2 models. Phi2 model finetuned using QLORA.
![Screenshot 2024-01-30 at 6 59 10 pm](https://github.com/santule/ERA/assets/20509836/5e9c42be-77d0-4439-8e6d-c92023f9787b)

Hugging Face Space - https://huggingface.co/spaces/sanjanatule/mmgpt

Input:
Text / Audio / Image

#### STEP 1 - PRETRAIN FOR IMAGE - LANGUAGE CAPTIONS

1. Dataset : Coco 2017 train dataset (URL:: http://images.cocodataset.org/annotations/annotations_trainval2017.zip)
2. CLIP Model: clip-vit-base-patch32 (URL:: https://huggingface.co/openai/clip-vit-base-patch32)
3. LLM Model: Phi2 model ((URL:: https://huggingface.co/microsoft/phi-2)
4. 48 GB RAM with batch size 2


#### Train loss
Multiple loss curves as the training failed due to some error. So training was resumed from the last checkpoint.

![step1_multi_train_loss](https://github.com/santule/ERA/assets/20509836/2c4ee048-1148-43aa-8508-b01417c3539a)

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
Multiple loss curves as the training failed due to some error. So training was resumed from the last checkpoint.

![step2_multiple_train_loss](https://github.com/santule/ERA/assets/20509836/0f5c1ca8-5287-4855-baf5-b29cd06801e1)

#### Example predictions
```
Image: http://images.cocodataset.org/train2017/000000107535.jpg
Question: What is the woman doing on the street corner? [QA]
Answer:   The woman is standing near a pole on the sidewalk at the corner of a street, looking at ads posted on the pole and pushing a walk signal button on the street corner to safely cross the road.<|endoftext|>
Model Predicted Ans: a woman is looking at a street sign in the middle of the street.
The woman in the image is looking at a street sign, which suggests that she is in a street with streetlights and traffic.

Image: http://images.cocodataset.org/train2017/000000026263.jpg
Question: Is there any writing or message on the banana? If so, what does it say? [QA]
Answer:   Yes, there is a message written on the banana. It says, "Sometimes a bit green; Often slippery; But always good for you!"<|endoftext|>
Model Predicted Ans: 
The banana has a label that reads "Banana: The Fruit of the Month".
```

#### Future Improvements
1. Train longer for step 1 and step 2.
2. Better quality input data ( [example 558 ](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain)
3. One cycle policy for learning rate for step 1.

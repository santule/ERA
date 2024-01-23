

## STEP 1 - PRETRAIN FOR IMAGE - LANGUAGE CAPTIONS
1. Dataset : CC3M dataset https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K
2. CLIP Model: 
3. LLM Model: Phi2 model

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

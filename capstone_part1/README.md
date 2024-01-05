
# Training Microsoft Phi2 GPT model from scratch

## Features:
1. Phi2 Model - https://huggingface.co/microsoft/phi-2
2. Trained on 4 A100 80 GB GPU Ram.
3. Loss reduced from 11 to 4.
4. Used 100 MB zipped clean data - wikipedia, archive and book. Sample from https://github.com/togethercomputer/RedPajama-Data
   


## Training logs:
```
iter 0 step 1: loss 11.1288, LR: 0.000000, iter time: 2820.98ms (optimizer.step) 

iter 10 step 11: loss 6.6115, LR: 0.000030, iter time: 2432.98ms (optimizer.step) 

iter 20 step 21: loss 4.9748, LR: 0.000060, iter time: 2433.09ms (optimizer.step) 

iter 30 step 31: loss 3.5289, LR: 0.000090, iter time: 2431.82ms (optimizer.step) 

iter 40 step 41: loss 7.5937, LR: 0.000120, iter time: 2434.67ms (optimizer.step) 

iter 50 step 51: loss 5.7037, LR: 0.000150, iter time: 2435.77ms (optimizer.step) 

iter 60 step 61: loss 5.0583, LR: 0.000180, iter time: 2481.01ms (optimizer.step) 

iter 70 step 71: loss 4.5447, LR: 0.000210, iter time: 2435.36ms (optimizer.step) 

iter 80 step 81: loss 3.1150, LR: 0.000240, iter time: 2434.76ms (optimizer.step) 

iter 90 step 91: loss 5.3811, LR: 0.000270, iter time: 2435.89ms (optimizer.step) 

iter 100 step 101: loss 4.3036, LR: 0.000300, iter time: 2437.67ms (optimizer.step) 
```


   

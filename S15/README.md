
# Speeding up transformers training using pytorch lightning.


In this repo, we speed up training transformers on the english to french translation by:

1. Parameter Sharing.
2. One Cycle Policy. (Model trained for 18 epochs only).
3. Reducing the hidden layer in feed forward network from 1024 to 128.
4. Dynamic Padding.
5. Removing english sentences with > 150 tokens and french sentences with tokens > 150 + 10.


Training logs




Sample Prediction from model

```


```

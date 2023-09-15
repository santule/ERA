# Building BERT, GPT and  ViT using modular code.


In this repo, we build all 3 models using single transformer.py file

BERT
```
my_gpt = transformer.Gpt(n_embeddings = vocab_size)
summary(model= my_gpt, input_size=(32,64), dtypes = [torch.int32],col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])

```

![Screenshot 2023-09-15 at 5 17 24 pm](https://github.com/santule/ERA/assets/20509836/8d5d5f26-af7a-4237-bc8d-f9978757900f)


GPT
```
my_gpt = transformer.Gpt(n_embeddings = vocab_size)
summary(model= my_gpt, input_size=(32,64), dtypes = [torch.int32],col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])
```

![Screenshot 2023-09-15 at 5 18 07 pm](https://github.com/santule/ERA/assets/20509836/e0a961d9-04d2-4bfd-b4e4-4c0d489cd3d9)

ViT
```
vit_model = transformer.ViT(num_classes = 3)
summary(model= vit_model, input_size=(32,3,224,224), col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])

```

![Screenshot 2023-09-15 at 5 18 47 pm](https://github.com/santule/ERA/assets/20509836/ac6ba77b-bbbc-4cf6-8255-6f875df2cf37)

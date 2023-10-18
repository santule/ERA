# Art Guidance using Stable Diffusion 

## Key Features:
  1. Application of textual inversion concepts Library
  2. Additional guidance using custom image loss


## Load the stable diffusion model:
```
pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    torch_dtype=torch.float16
).to("cuda")
```

## Load the textual inversion concepts library
```
pipe.load_textual_inversion("sd-concepts-library/dreams") 
pipe.load_textual_inversion("sd-concepts-library/midjourney-style") 
pipe.load_textual_inversion("sd-concepts-library/moebius") 
pipe.load_textual_inversion("sd-concepts-library/style-of-marc-allante") 
pipe.load_textual_inversion("sd-concepts-library/wlop-style")
```

## Generate Image before and after additional guidance using elastic distortion image loss and dreams concepts library

```
prompt = 'A beautiful sorceress in the style of <meeg>' 
```
<img width="321" alt="Screenshot 2023-10-16 at 12 29 28 am" src="https://github.com/santule/ERA/assets/20509836/53139697-d132-460e-b753-abdc9c5854ee">

## Generate Image using elastic distortion image loss using different concepts library
![Screenshot 2023-10-18 at 10 09 08 pm](https://github.com/santule/ERA/assets/20509836/292e76ce-41e6-43cf-b58b-35cb60258855)

## Generate Image after additional guidance using elastic distortion image loss
![Screenshot 2023-10-18 at 10 08 57 pm](https://github.com/santule/ERA/assets/20509836/8fb95f78-3ac6-4918-a1eb-ea924d3f3502)

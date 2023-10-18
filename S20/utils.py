import PIL
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
from diffusers import LMSDiscreteScheduler, DiffusionPipeline

# configurations
torch_device        = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
height, width       = 512, 512
guidance_scale      = 8
blue_loss_scale     = 200
num_inference_steps = 50

elastic_transformer = T.ElasticTransform(alpha=550.0, sigma=5.0)



pretrained_model_name_or_path = "segmind/tiny-sd"
pipe = DiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    low_cpu_mem_usage = True,
    torch_dtype=torch.float16
).to(torch_device)


pipe.load_textual_inversion("sd-concepts-library/dreams")
pipe.load_textual_inversion("sd-concepts-library/midjourney-style")
pipe.load_textual_inversion("sd-concepts-library/moebius")
pipe.load_textual_inversion("sd-concepts-library/style-of-marc-allante")
pipe.load_textual_inversion("sd-concepts-library/wlop-style")


concepts_mapping = {
    "Dream": '<meeg>', "Midjourney":'<midjourney-style>',
    "Marc Allante": '<Marc_Allante>', "Moebius": '<moebius>',
    "Wlop": '<wlop-style>'
}


def image_loss(images, method='elastic'):

    # elastic loss
    if method == 'elastic':
      transformed_imgs = elastic_transformer(images)
      error = torch.abs(transformed_imgs - images).mean()

    # symmetry loss - Flip the image along the width
    elif method == "symmetry":
      flipped_image = torch.flip(images, [3])
      error = F.mse_loss(images, flipped_image)

    # saturation loss
    elif method == 'saturation':
      transformed_imgs = T.functional.adjust_saturation(images,saturation_factor = 10)
      error = torch.abs(transformed_imgs - images).mean()

    # blue loss
    elif method == 'blue':
      error = torch.abs(images[:,2] - 0.9).mean() # [:,2] -> all images in batch, only the blue channel

    return error


HTML_TEMPLATE = """
<style>
    body {
        background: linear-gradient(135deg, #f5f7fa, #c3cfe2);
    }
    #app-header {
        text-align: center;
        background: rgba(255, 255, 255, 0.8); /* Semi-transparent white */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative; /* To position the artifacts */
    }
    #app-header h1 {
        color: #4CAF50;
        font-size: 2em;
        margin-bottom: 10px;
    }
    .concept {
        position: relative;
        transition: transform 0.3s;
    }
    .concept:hover {
        transform: scale(1.1);
    }
    .concept img {
        width: 100px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .concept-description {
        position: absolute;
        bottom: -30px;
        left: 50%;
        transform: translateX(-50%);
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 5px;
        opacity: 0;
        transition: opacity 0.3s;
    }
    .concept:hover .concept-description {
        opacity: 1;
    }
    /* Artifacts */
    .artifact {
        position: absolute;
        background: rgba(76, 175, 80, 0.1); /* Semi-transparent green */
        border-radius: 50%; /* Make it circular */
    }
    .artifact.large {
        width: 300px;
        height: 300px;
        top: -50px;
        left: -150px;
    }
    .artifact.medium {
        width: 200px;
        height: 200px;
        bottom: -50px;
        right: -100px;
    }
    .artifact.small {
        width: 100px;
        height: 100px;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
    }
</style>
<div id="app-header">
    <!-- Artifacts -->
    <div class="artifact large"></div>
    <div class="artifact medium"></div>
    <div class="artifact small"></div>
    <!-- Content -->
    <h1>Art Generator</h1>
    <p>Generate new art in five different styles by providing a prompt.</p>
    <div style="display: flex; justify-content: center; gap: 20px; margin-top: 20px;">
        <div class="concept">
            <img src="https://github.com/Delve-ERAV1/S20/assets/11761529/30ac92f8-fc62-4aab-9221-043865c6fe7c" alt="Midjourney">
            <div class="concept-description">Midjourney Style</div>
        </div>
        <div class="concept">
            <img src="https://github.com/Delve-ERAV1/S20/assets/11761529/54c9a61e-df9f-4054-835b-ec2c6ba5916c" alt="Dreams">
            <div class="concept-description">Dreams Style</div>
        </div>
        <div class="concept">
            <img src="https://github.com/Delve-ERAV1/S20/assets/11761529/2f37e402-15d1-4a74-ba85-bb1566da930e" alt="Moebius">
            <div class="concept-description">Moebius Style</div>
        </div>
        <div class="concept">
            <img src="https://github.com/Delve-ERAV1/S20/assets/11761529/f838e767-ac20-4996-b5be-65c61b365ce0" alt="Allante">
            <div class="concept-description">Hong Kong born artist inspired by western and eastern influences</div>
        </div>
        <div class="concept">
            <img src="https://github.com/Delve-ERAV1/S20/assets/11761529/9958140a-1b62-4972-83ca-85b023e3863f" alt="Wlop">
            <div class="concept-description">WLOP (Born 1987) is known for Digital Art (NFTs)</div>
        </div>
    </div>
</div>
"""


def get_examples():
   examples = [
      ['A powerful man in dreadlocks', 'Dream', 'Symmetry', 45],
      ['World Peace', 'Marc Allante', 'Saturation', 147],
      ['Storm trooper in the desert, dramatic lighting, high-detail', 'Moebius', 'Elastic', 28],
      ['Delicious Italian pizza on a table, a window in the background overlooking a city skyline', 'Wlop', 'Blue', 50],
   ]
   return(examples)


def latents_to_pil(latents):
    # bath of latents -> list of images
    latents = (1 / 0.18215) * latents
    with torch.no_grad():
        image = pipe.vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1) # 0 to 1
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).round().astype("uint8")
    return Image.fromarray(image[0])


def generate_art(prompt, concept, method, seed):

  prompt = f"{prompt} in the style of {concepts_mapping[concept]}"
  img_no_loss = latents_to_pil(generate_image(prompt, method, seed))
  img_loss = latents_to_pil(generate_image(prompt, method, seed, loss_apply=True))
  return([img_no_loss, img_loss])


def generate_image(prompt, method, seed, loss_apply=False):

    generator           = torch.manual_seed(seed)
    batch_size          = 1
    method              = method.lower()

    # scheduler
    scheduler    = LMSDiscreteScheduler(beta_start = 0.00085, beta_end = 0.012, beta_schedule = "scaled_linear", num_train_timesteps = 1000)
    scheduler.set_timesteps(50)
    scheduler.timesteps = scheduler.timesteps.to(torch.float32)

    # text embeddings of the prompt
    text_input = pipe.tokenizer([prompt], padding='max_length', max_length = pipe.tokenizer.model_max_length, truncation= True, return_tensors="pt")
    input_ids = text_input.input_ids.to(torch_device)

    with torch.no_grad():
        text_embeddings = pipe.text_encoder(text_input.input_ids.to(torch_device))[0]

    max_length = text_input.input_ids.shape[-1]
    uncond_input = pipe.tokenizer(
          [""] * 1, padding="max_length", max_length= max_length, return_tensors="pt"
    )

    with torch.no_grad():
        uncond_embeddings = pipe.text_encoder(uncond_input.input_ids.to(torch_device))[0]

    text_embeddings = torch.cat([uncond_embeddings,text_embeddings]) 

    # random latent
    latents = torch.randn(
        (batch_size, pipe.unet.config.in_channels, height// 8, width //8),
        generator = generator,
    ).to(torch.float16)


    latents = latents.to(torch_device)
    latents = latents * scheduler.init_noise_sigma

    for i, t in tqdm(enumerate(scheduler.timesteps), total = len(scheduler.timesteps)):

        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)

        with torch.no_grad():
            noise_pred = pipe.unet(latent_model_input.to(torch.float16), t, encoder_hidden_states=text_embeddings)["sample"]

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        if loss_apply and i%5 == 0:

            latents = latents.detach().requires_grad_()
            latents_x0 = latents - sigma * noise_pred

            # use vae to decode the image
            denoised_images = pipe.vae.decode((1/ 0.18215) * latents_x0).sample / 2 + 0.5 # range(0,1)

            loss = image_loss(denoised_images, method) * blue_loss_scale

            cond_grad = torch.autograd.grad(loss, latents)[0]
            latents = latents.detach() - cond_grad * sigma**2

        latents = scheduler.step(noise_pred,t, latents).prev_sample

    return latents
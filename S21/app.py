import gradio as gr
import torch
from torch import nn
import lightning.pytorch as pl
from torch.nn import functional as F
from utils import GPTLM,encode,decode

newmodel = GPTLM.load_from_checkpoint('shakespeare_gpt.pth')

def generate_dialogue(character_dropdown):
  if character_dropdown == "NONE":
    context = torch.zeros((1, 1), dtype=torch.long)
    return decode(newmodel.model.generate(context, max_new_tokens=100)[0].tolist())
  else:
    context = torch.tensor([encode(character_dropdown)], dtype=torch.long)
    output_dialogue = decode(newmodel.model.generate(context, max_new_tokens=100)[0].tolist())
    # remove extra dialogue returned
    output_dialogue = str(output_dialogue.split("\n\n")[0])
    return output_dialogue

      

HTML_TEMPLATE = """    
<style>
    
    #app-header {
        text-align: center;
        background: rgba(255, 255, 255, 0.3); /* Semi-transparent white */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        position: relative; /* To position the artifacts */
    }
    #app-header h1 {
        color: #FF0000;
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
    
</style>
<div id="app-header">
    <!-- Artifacts -->
    <div class="artifact large"></div>
    <div class="artifact large"></div>
    <div class="artifact large"></div>
    <div class="artifact large"></div>
    <!-- Content -->
    <h1>SHAKESPEARE  DIALOGUE  GENERATOR</h1>
    <p>Generate dialogue for Shakespearean character by selecting character from dropdown.</p>
    <p>Model: GPT, Dataset: Tiny Shakespeare, Token limit: 100.</p>
"""

with gr.Blocks(theme=gr.themes.Glass(),css=".gradio-container {background: url('file=https://github.com/Delve-ERAV1/S20/assets/11761529/c0ff84a4-dde6-473e-a820-d3797040eb9d')}") as interface:
    gr.HTML(value=HTML_TEMPLATE, show_label=False)

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")
    gr.Markdown("")

    
    with gr.Row(scale=1):
        character_dropdown = gr.Dropdown(
            label="Select a Character",
            choices=["NONE","ROMEO","JULIET","MENENIUS","ANTONIO"],
            value='Dream'
        )
        outputs = gr.Textbox(
            label="Generated Dialogue"
        )
        inputs = [character_dropdown]
   
    with gr.Column(scale=1):
        button = gr.Button("Generate")
        button.click(generate_dialogue, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    interface.launch(enable_queue=True)
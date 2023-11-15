import torch
import gradio as gr
from utils import *
from torch import nn
import lightning.pytorch as pl
from torch.nn import functional as F

device     = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    <h1>GPT NEXT WORD GENERATOR</h1>
    <p>Generate dialogue for given some initial prompt for context.</p>
    <p>Model: GPT, Dataset: arxiv + book + cc, Parameter Count: 160M</p>
"""

with gr.Blocks(theme=gr.themes.Glass(),css=".gradio-container {background: url('file=https://github.com/santule/ERA/assets/20509836/e78f2bb3-ddd8-4ce9-a941-3d3d7ef7a272')}") as interface:
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
    
    with gr.Row():

        input_text = gr.Textbox(
            label="Input Text", 
            value="Enter your prompt here: This text will set the context for the AI's response."
        )

        temperature_dropdown = gr.Slider(0, 1, value=0.8, label="Temperature", info="Set the creativity level: Higher values produce more varied results, lower values generate more predictable text.")
        top_k_dropdown = gr.Slider(200, 300, value=200, label="Top K", info="Control the randomness: Limits the AI to consider only the top K most likely next words.")
        max_new_tokens = gr.Slider(10, 100, value=50, label="Max Tokens", info="Choose the length: This determines the maximum number of words the AI will generate.")


        outputs = gr.Textbox(
            label="Generated Dialogue"
        )
        inputs = [input_text, temperature_dropdown, top_k_dropdown, max_new_tokens]
   
    with gr.Column():
        button = gr.Button("Generate")
        button.click(generate_dialogue, inputs=inputs, outputs=outputs)

    with gr.Row():
         gr.Examples(examples=examples, inputs=inputs, outputs=outputs, fn=generate_dialogue, cache_examples=True,)


interface.launch()
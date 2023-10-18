import gradio as gr
from utils import *


with gr.Blocks() as interface:
    gr.HTML(value=HTML_TEMPLATE, show_label=False)
    with gr.Row():
        text_input = gr.Textbox(
            label="Enter your prompt",
            placeholder="A powerful mysterious sorceress..........",
        )
        concept_dropdown = gr.Dropdown(
            label="Select a Concept",
            choices=["Midjourney", "Dream", "Moebius", "Marc Allante", "Wlop"],
            value='Dream'
        )

        method_dropdown = gr.Dropdown(
            label="Select Guidance Method",
            choices=["Elastic", "Symmetry", "Saturation", "Blue"],
            value='Elastic'
        )

        seed_slider = gr.Slider(
            label="Random Seed",
            minimum=0,
            maximum=1000,
            step=1,
            value=42
        )
        inputs = [text_input, concept_dropdown, method_dropdown, seed_slider]

    with gr.Row():
        outputs = gr.Gallery(
            label="Generated Art", show_label=True,
            columns=[2], rows=[1], object_fit="contain"
        )

    with gr.Row():
        button = gr.Button("Generate Art")
        button.click(generate_art, inputs=inputs, outputs=outputs)

    with gr.Row():
        gr.Examples(examples=get_examples(), inputs=inputs, outputs=outputs, fn=generate_art, cache_examples=True)


if __name__ == "__main__":
    interface.launch(enable_queue=True)
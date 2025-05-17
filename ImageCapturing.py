#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install gradio transformers torch')


# In[ ]:


import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):
    inputs = processor(images=image, return_tensors="pt")
    out = model.generate(**inputs)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

iface = gr.Interface(
    fn=generate_caption, 
    inputs=gr.Image(type="pil"), 
    outputs=gr.Textbox(), 
    live=True,
    title="BLIP Image Captioning",
    description="Upload an image and get a generated caption."
)

iface.launch()


# In[ ]:





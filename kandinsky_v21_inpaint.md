# Kandinsky 2.1 Inpainting 

The Kandinsky models are a series of multilingual text-to-image generation models. The Kandinsky 2.0 model uses two multilingual text encoders and concatenates those results for the UNet.

[Kandinsky 2.1](https://github.com/ai-forever/Kandinsky-2) changes the architecture to include an image prior model ([`CLIP`](https://huggingface.co/docs/transformers/model_doc/clip)) to generate a mapping between text and image embeddings. The mapping provides better text-image alignment and it is used with the text embeddings during training, leading to higher quality results. Finally, Kandinsky 2.1 uses a [Modulating Quantized Vectors (MoVQ)](https://huggingface.co/papers/2209.09002) decoder - which adds a spatial conditional normalization layer to increase photorealism - to decode latents into images.

To use the Kandinsky models for any task, you always start by setting up the prior pipeline to encode the prompt and generate the image embeddings. The prior pipeline also generates `negative_image_embeds` that correspond to the negative prompt `""`. For better results, you can pass an actual `negative_prompt` to the prior pipeline, but that'll increase the effective batch size of the prior pipeline by 2x.

## 🧨 mindone.diffusers 

For Kandinsky 2.1 inpainting in mindone.diffusers, you’ll need the original image, a mask of the area to replace the original image, and a text prompt of what to inpaint. Load the prior pipeline,

```python
from mindone.diffusers import KandinskyInpaintPipeline, KandinskyPriorPipeline
from mindone.diffusers.utils import load_image, make_image_grid
import mindspore as ms
import numpy as np
from PIL import Image

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms.float16, use_safetensors=True)
pipeline = KandinskyInpaintPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-inpaint", mindspore_dtype=ms.float16, use_safetensors=True)
```

Load an initial image and create a mask,

```python
init_image = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
mask = np.zeros((768, 768), dtype=np.float32)
# mask area above cat's head
mask[:250, 250:-250] = 1
```

Generate the embeddings with the prior pipeline:

```python
prompt = "a hat"
image_emb, zero_image_emb = prior_pipeline(prompt)
```

Now pass the initial image, mask, and prompt and embeddings to the pipeline to generate an image,

```python
output_image = pipeline(
prompt,
image=init_image,
mask_image=mask,
image_embeds=image_emb,
negative_image_embeds=zero_image_emb,
height=768,
width=768,
num_inference_steps=150
)[0][0]
mask = Image.fromarray((mask*255).astype('uint8'), 'L')
make_image_grid([init_image, mask, output_image], rows=1, cols=3)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
<img class="rounded-xl" src="https://github.com/user-attachments/assets/af2022b7-d602-4352-9406-e65f2aab80ff"/>
</div>


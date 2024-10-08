# Kandinsky 2.1 Text-to-image

The Kandinsky models are a series of multilingual text-to-image generation models. The Kandinsky 2.0 model uses two multilingual text encoders and concatenates those results for the UNet.

[Kandinsky 2.1](https://github.com/ai-forever/Kandinsky-2) changes the architecture to include an image prior model ([`CLIP`](https://huggingface.co/docs/transformers/model_doc/clip)) to generate a mapping between text and image embeddings. The mapping provides better text-image alignment and it is used with the text embeddings during training, leading to higher quality results. Finally, Kandinsky 2.1 uses a [Modulating Quantized Vectors (MoVQ)](https://huggingface.co/papers/2209.09002) decoder - which adds a spatial conditional normalization layer to increase photorealism - to decode latents into images.

To use the Kandinsky models for any task, you always start by setting up the prior pipeline to encode the prompt and generate the image embeddings. The prior pipeline also generates `negative_image_embeds` that correspond to the negative prompt `""`. For better results, you can pass an actual `negative_prompt` to the prior pipeline, but that'll increase the effective batch size of the prior pipeline by 2x.

## 🧨 mindone.diffusers 

Use Kandinsky 2.1 in mindone.diffusers for text-to-image generation tasks,

```py
from mindone.diffusers import KandinskyPriorPipeline, KandinskyPipeline
import mindspore as ms

prior_pipeline = KandinskyPriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", mindspore_dtype=ms.float16)
pipeline = KandinskyPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", mindspore_dtype=ms.float16)

prompt = "A alien cheeseburger creature eating itself, claymation, cinematic, moody lighting"
negative_prompt = "low quality, bad quality" # optional to include a negative prompt, but results are usually better
image_embeds, negative_image_embeds = prior_pipeline(prompt, negative_prompt, guidance_scale=1.0)
```

Now pass all the prompts and embeddings to the [`KandinskyPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky/#mindone.diffusers.KandinskyPipeline) to generate an image:

```py
image = pipeline(prompt, image_embeds=image_embeds, negative_prompt=negative_prompt, negative_image_embeds=negative_image_embeds, height=768, width=768)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/ec761cf3-8781-4ae7-8b06-c6940856f45f"/>
</div>






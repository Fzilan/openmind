# Kandinsky 2.2 Interpolation 

The Kandinsky models are a series of multilingual text-to-image generation models. The Kandinsky 2.0 model uses two multilingual text encoders and concatenates those results for the UNet.

[Kandinsky 2.2](https://github.com/ai-forever/Kandinsky-2) improves on the previous model by replacing the image encoder of the image prior model with a larger CLIP-ViT-G model to improve quality. The image prior model was also retrained on images with different resolutions and aspect ratios to generate higher-resolution images and different image sizes.

To use the Kandinsky models for any task, you always start by setting up the prior pipeline to encode the prompt and generate the image embeddings. The prior pipeline also generates `negative_image_embeds` that correspond to the negative prompt `""`. For better results, you can pass an actual `negative_prompt` to the prior pipeline, but that'll increase the effective batch size of the prior pipeline by 2x.

## 🧨 mindone.diffusers 

Kandinsky 2.1 and 2.2 usage is very similar! The only difference is Kandinsky 2.2 doesn't accept `prompt` as an input when decoding the latents. Instead, Kandinsky 2.2 only accepts `image_embeds` during decoding.

To use mindone.diffusers for Kandinsky 2.2 interpolation tasks, load the prior pipeline and two images you’d like to interpolate: 

```python
from mindone.diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
from mindone.diffusers.utils import load_image, make_image_grid
import mindspore as ms

prior_pipeline = KandinskyV22PriorPipeline.from_pretrained("kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16, use_safetensors=True)
img_1 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png")
img_2 = load_image("https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg")
make_image_grid([img_1.resize((512,512)), img_2.resize((512,512))], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/cat.png"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">a cat</figcaption>
  </div>
  <div>
    <img class="rounded-xl" src="https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinsky/starry_night.jpeg"/>
    <figcaption class="mt-2 text-center text-sm text-gray-500">Van Gogh's Starry Night painting</figcaption>
  </div>
</div>



Specify the text or images to interpolate, and set the weights for each text or image. Experiment with the weights to see how they affect the interpolation!

```py
images_texts = ["a cat", img_1, img_2]
weights = [0.3, 0.3, 0.4]
```


Call the `interpolate` function to generate the embeddings, and then pass them to the pipeline to generate the image:

```python
# prompt can be left empty
prompt = ""
image_embeds, negative_image_embeds = prior_pipeline.interpolate(images_texts, weights)

pipeline = KandinskyV22Pipeline.from_pretrained("kandinsky-community/kandinsky-2-2-decoder", mindspore_dtype=ms.float16, use_safetensors=True)

image = pipeline(image_embeds=image_embeds, negative_image_embeds=negative_image_embeds, height=768, width=768)[0][0]
image
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/4cad5e59-36f0-4b27-9bc9-6cb269074ebd"/>
</div>


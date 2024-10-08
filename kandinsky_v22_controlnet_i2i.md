# Kandinsky 2.2 ControlNet Image-to-image

The Kandinsky models are a series of multilingual text-to-image generation models. 

[Kandinsky 2.2](https://github.com/ai-forever/Kandinsky-2) improves on the previous model by replacing the image encoder of the image prior model with a larger CLIP-ViT-G model to improve quality. The image prior model was also retrained on images with different resolutions and aspect ratios to generate higher-resolution images and different image sizes.

ControlNet enables conditioning large pretrained diffusion models with additional inputs such as a depth map or edge detection. For example, you can condition Kandinsky 2.2 with a depth map so the model understands and preserves the structure of the depth image.

## 🧨 mindone.diffusers 

For image-to-image with ControlNet, you'll need to use the:

- [`KandinskyV22PriorEmb2EmbPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22PriorEmb2EmbPipeline) to generate the image embeddings from a text prompt and an image
- [`KandinskyV22ControlnetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetImg2ImgPipeline) to generate an image from the initial image and the image embeddings

⚠️ MindOne currently does not support the full process for extracting the depth map, as MindOne does not yet support depth-estimation [~transformers.Pipeline] from mindone.transformers. You need to prepare the depth map in advance to continue the process.

Here's an example of using ControlNet image-to-image task for Kandinsky 2.2 in mindone.diffusers.

Process the depth map extracted from the initial image of a cat, which you prepared in advance.

```python
import mindspore as ms
import numpy as np

from mindone.diffusers import KandinskyV22PriorEmb2EmbPipeline, KandinskyV22ControlnetImg2ImgPipeline
from mindone.diffusers.utils import load_image

img = load_image(
    "https://huggingface.co/datasets/hf-internal-testing/diffusers-images/resolve/main/kandinskyv22/cat.png"
).resize((768, 768))

def make_hint(depth_image):
    image = depth_image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    detected_map = ms.Tensor.from_numpy(image).float() / 255.0
    hint = detected_map.permute(2, 0, 1)
    return hint

hint = make_hint(depth_image).unsqueeze(0).half()
```

Load the prior pipeline and the [`KandinskyV22ControlnetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetImg2ImgPipeline):

```python
prior_pipeline = KandinskyV22PriorEmb2EmbPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-prior", mindspore_dtype=ms.float16, use_safetensors=True
)

pipeline = KandinskyV22ControlnetImg2ImgPipeline.from_pretrained(
    "kandinsky-community/kandinsky-2-2-controlnet-depth", revision="refs/pr/7", mindspore_dtype=ms.float16
)
```

Pass a text prompt and the initial image to the prior pipeline to generate the image embeddings:

```python
prompt = "A robot, 4k photo"
negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

generator = np.random.Generator(np.random.PCG64(43))

img_emb = prior_pipeline(prompt=prompt, image=img, strength=0.85, generator=generator)
negative_emb = prior_pipeline(prompt=negative_prior_prompt, image=img, strength=1, generator=generator)
```

Now you can run the [`KandinskyV22ControlnetImg2ImgPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetImg2ImgPipeline) to generate an image from the initial image and the image embeddings:

```python
prompt = "A robot, 4k photo"
negative_prior_prompt = "lowres, text, error, cropped, worst quality, low quality, jpeg artifacts, ugly, duplicate, morbid, mutilated, out of frame, extra fingers, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, dehydrated, bad anatomy, bad proportions, extra limbs, cloned face, disfigured, gross proportions, malformed limbs, missing arms, missing legs, extra arms, extra legs, fused fingers, too many fingers, long neck, username, watermark, signature"

generator = np.random.Generator(np.random.PCG64(43))

image_emb, zero_image_emb = prior_pipeline(
    prompt=prompt, negative_prompt=negative_prior_prompt, generator=generator
)
```

Finally, pass the image embeddings and the depth image to the [`KandinskyV22ControlnetPipeline`](https://mindspore-lab.github.io/mindone/latest/diffusers/api/pipelines/kandinsky_v22/#mindone.diffusers.KandinskyV22ControlnetPipeline) to generate an image:

```python
image = pipeline(image=img, strength=0.5, image_embeds=img_emb[0], negative_image_embeds=negative_emb[0], hint=hint, num_inference_steps=50, generator=generator, height=768, width=768)[0][0]
make_image_grid([img.resize((512, 512)), image.resize((512, 512))], rows=1, cols=2)
```

<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="rounded-xl" src="https://github.com/user-attachments/assets/b67abe65-5874-4795-bb74-3eff17e911ca"/>
</div>
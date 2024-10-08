# Hunyuan-DiT 

Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding from Tencent Hunyuan.

The abstract from the paper is:

*We present Hunyuan-DiT, a text-to-image diffusion transformer with fine-grained understanding of both English and Chinese. To construct Hunyuan-DiT, we carefully design the transformer structure, text encoder, and positional encoding. We also build from scratch a whole data pipeline to update and evaluate data for iterative model optimization. For fine-grained language understanding, we train a Multimodal Large Language Model to refine the captions of the images. Finally, Hunyuan-DiT can perform multi-turn multimodal dialogue with users, generating and refining images according to the context. Through our holistic human evaluation protocol with more than 50 professional human evaluators, Hunyuan-DiT sets a new state-of-the-art in Chinese-to-image generation compared with other open-source models.*

You can find the original codebase at [Tencent/HunyuanDiT](https://github.com/Tencent/HunyuanDiT) and all the available checkpoints at [Tencent-Hunyuan](https://huggingface.co/Tencent-Hunyuan/HunyuanDiT).

**Highlights**: HunyuanDiT supports Chinese/English-to-image, multi-resolution generation.

HunyuanDiT has the following components:

- It uses a diffusion transformer as the backbone
- It combines two text encoders, a bilingual CLIP, and a multilingual T5 encoder

## 🧨 mindone.diffusers 

Use Hunyuan-dit in mindone.diffusers for text-to-image generation tasks :

```bash
import mindspore as ms
from mindone.diffusers import HunyuanDiTPipeline

pipe = HunyuanDiTPipeline.from_pretrained(
    "Tencent-Hunyuan/HunyuanDiT-Diffusers", mindspore_dtype=ms.float32
)

# You may also use English prompt as HunyuanDiT supports both English and Chinese
# prompt = "An astronaut riding a horse"
prompt = "一个宇航员在骑马"
image = pipe(prompt).images[0]
image.save("hunyuandit_t2i.png")
```



<div style="display: flex; justify-content: center; align-items: flex-start; text-align: center; max-width: 98%; margin: 0 auto; gap: 1vw;">
    <img class="hunyuandit_t2i" src="https://github.com/user-attachments/assets/93552253-1030-4352-bab9-1cc74dcd4d1a"/>
</div>
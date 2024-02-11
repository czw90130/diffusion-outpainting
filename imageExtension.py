# 导入必要的库
# UNet2DConditionModel
# 这是一个条件UNet模型，用于生成或编辑图像
# 在Stable Diffusion中，UNet是负责图像生成的核心网络，可以根据给定的条件（如文本描述）生成相应的图像
# 它通过接收嵌入文本的提示和可能的其他输入（如遮罩区域的图像），来生成或编辑图像。
from diffusers import UNet2DConditionModel
# AutoencoderKL
# 这是一个变分自编码器（VAE），在Stable Diffusion中用于图像的编码和解码。
# 这个模型可以将图像编码到一个低维的隐空间表示中，并能从这个隐空间表示中重构图像。
# 在图像编辑任务中，VAE用于将输入图像编码到隐空间，进行处理后再解码回图像空间。
from diffusers import AutoencoderKL
# DDIMScheduler
# 这是一个确定性的差分图像生成（DDIM）调度器，用于控制图像生成过程中的噪声减少步骤。
# 调度器负责在图像生成过程中安排时间步骤，以便逐步从噪声图像过渡到清晰图像。
# 这种方法有助于生成高质量的图像，并可以调整以生成多样化的结果。
from diffusers import DDIMScheduler
# VaeImageProcessor
# 这是一个图像处理器，专为与VAE模型一起使用而设计。
# 它提供了一系列的预处理和后处理功能，使得图像能够被模型正确处理。
# 预处理功能包括调整图像大小、标准化等，而后处理功能则负责将模型输出转换回可视化图像。
from diffusers.image_processor import VaeImageProcessor
# CLIPTokenizer & CLIPTextModel
# 这两个来自transformers库的组件与diffusers库紧密协作，用于处理文本输入。
# CLIPTokenizer用于将文本提示分词并编码为模型可以理解的格式，而CLIPTextModel则根据这些编码生成文本的嵌入表示。
# 这些嵌入表示随后被用作生成或编辑图像时的条件输入。
from transformers import CLIPTokenizer, CLIPTextModel

import torch
from tqdm import tqdm
import cv2 as cv
import numpy as np
import time
from safetensors.torch import load_file
from typing import List

class ImageExtension():
    """
    图像扩展类，用于处理图像生成和编辑任务，特别是使用Stable Diffusion模型进行图像的生成、编辑和修复。
    """
    # 设置默认模型路径和本地缓存路径
    DEFAULT_MODEL = "stabilityai/stable-diffusion-2-inpainting"
    cache_dir = "F:/huggingface_model/"    # diffusers的本地缓存路径
    def __init__(self, 
                 sampleTimeStep:int = 30, 
                 only_local_files:bool = False):
        """
        初始化ImageExtension类的实例。

        参数:
        - sampleTimeStep: int, 定义采样过程中使用的时间步长，默认为30。
        - only_local_files: bool, 指定是否仅使用本地文件加载模型，以避免从网络下载，默认为False。
        """
        # 初始化模型参数
        self.base_model_name = self.DEFAULT_MODEL # 指定使用的模型
        self.only_local_files = only_local_files # 设置是否仅使用本地文件
        set_step(sampleTimeStep) # 设置采样时间步长

        # 加载必要的模型和组件
        self.load_model()  # 加载条件UNet模型
        self.getLatent_model()  # 加载VAE模型，用于图像的编码和解码
        self.load_text_encoder()  # 加载文本编码器，用于处理文本提示
        self.load_scheduler()  # 加载调度器，用于控制图像生成过程中的时间步
        
        self.allTimestep = self.ddim.timesteps  # 获取调度器定义的所有时间步

        self.image_processor = VaeImageProcessor()  # 初始化图像处理器，用于预处理和后处理图像


    def addNoise(self, latent:torch.Tensor, noise: torch.Tensor, timestep:torch.Tensor):
        """
        向隐空间表示添加噪声。这一操作是图像生成过程的关键步骤之一，有助于引入随机性。

        参数:
        - latent: torch.Tensor, 当前的隐空间表示，它是图像在高维空间的表示形式。
        - noise: torch.Tensor, 要添加到隐空间表示中的噪声。这个噪声通常是随机生成的，有助于在生成过程中引入变异。
        - timestep: torch.Tensor, 当前的时间步。在随时间步进行的图像生成过程中，不同的时间步可能需要添加不同程度的噪声。

        返回:
        - torch.Tensor: 添加噪声后的新的隐空间表示。
        """
        # 使用DDIM调度器的add_noise方法将噪声添加到隐空间表示中。
        # 这个方法根据当前时间步计算噪声的加权，确保噪声的添加与生成过程的阶段相匹配。
        latent = self.ddim.add_noise(latent, noise, timestep)
        # 对加入噪声后的隐空间表示进行缩放，使用init_noise_sigma进行调整。
        # init_noise_sigma是一个预先定义的缩放因子，它根据模型的配置和训练过程中使用的噪声水平来设定。
        latent = latent * self.ddim.init_noise_sigma
    
        return latent
    
    def set_step(self, sampleTimeStep:int):
        # 设置采样时间步
        self.sampleTimeStep = sampleTimeStep
        self.ddim.set_timesteps(self.sampleTimeStep, device="cuda:0")
        self.allTimestep = self.ddim.timesteps
    
        
    def sample_step(self, latent: torch.Tensor, niose: torch.Tensor, timestep: torch.Tensor):
        """
        进行单步采样，根据当前的隐空间表示、噪声和时间步更新隐空间表示。

        参数:
        - latent: torch.Tensor，当前的隐空间表示。
        - noise: torch.Tensor，当前步骤要添加的噪声。
        - timestep: torch.Tensor，当前的时间步。

        返回:
        - torch.Tensor: 更新后的隐空间表示。
        """
        # 使用DDIM调度器的step方法更新隐空间表示
        return self.ddim.step(niose, timestep, latent)['prev_sample']


    def sample_block(self, latent:torch.Tensor, masked_latent:torch.Tensor, mask: torch.Tensor, prompt_embeds:torch.Tensor, timestep: torch.Tensor, guidance_scale:int=7):
        """
        对指定区块进行采样，根据给定的文本提示通过模型生成对应的噪声，然后应用噪声生成最终的图像。

        参数:
        - latent: torch.Tensor，原始图像的隐空间表示。
        - masked_latent: torch.Tensor，被遮罩的区域的隐空间表示。
        - mask: torch.Tensor，遮罩，用于指定需要生成内容的区域。
        - prompt_embeds: torch.Tensor，文本提示的嵌入表示。
        - timestep: torch.Tensor，当前的时间步。
        - guidance_scale: int，指导比例，用于调整条件生成和无条件生成的平衡。

        返回:
        - torch.Tensor: 更新后的隐空间表示。
        """
        # 准备输入数据：复制latent、mask和masked_latent以适应UNet的输入要求
        latent_model_input = torch.cat([latent] * 2)
        mask_input = torch.cat([mask]*2)
        masked_latent_input = torch.cat([masked_latent]*2)

        # 调整输入数据的尺寸并合并，以符合模型的输入格式
        latent_model_input = self.ddim.scale_model_input(latent_model_input, timestep)
        latent_model_input = torch.cat([latent_model_input, mask_input, masked_latent_input], dim=1)     # inpaint模型拥有额外的输入信息，通道数为9
        
        # 通过UNet模型预测噪声，并根据指导比例调整
        noise_pred = self.unet(latent_model_input, timestep, encoder_hidden_states = prompt_embeds).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # 使用预测的噪声更新隐空间表示
        latent = self.sample_step(latent, noise_pred, timestep)

        return latent
    
    def sample(self, latent:torch.Tensor, masked_latent:torch.Tensor, mask: torch.Tensor, chunk_list: List[int], prompt_embeds:torch.Tensor, guidance_scale:int=7):
        """
        对整个图像进行分块采样，遍历所有时间步，对每个块进行采样，最后合并结果。

        参数:
        - latent: torch.Tensor，原始图像的隐空间表示。
        - masked_latent: torch.Tensor，被遮罩区域的隐空间表示。
        - mask: torch.Tensor，遮罩，用于指定需要生成内容的区域。
        - chunk_list: List[int]，需要采样的图像块的列表。
        - prompt_embeds: torch.Tensor，文本提示的嵌入表示。
        - guidance_scale: int，指导比例，用于调整条件生成和无条件生成的平衡。

        返回:
        - torch.Tensor: 更新后的整个图像的隐空间表示。
        """
        # print(prompt_embeds.shape)
        # print(latent.shape)
        count = torch.zeros_like(latent) # 初始化计数器，用于记录每个像素被更新的次数
        full_latent = torch.zeros_like(latent) # 初始化用于累加更新结果的张量

        # 遍历所有时间步
        for Tin in tqdm(range(0, len(self.allTimestep))):
            Ti = self.allTimestep[Tin]  # 获取当前时间步
            count.zero_()  # 重置计数器
            full_latent.zero_()  # 重置累加器
            
            # 遍历所有需要处理的图像块
            for chunk_block in chunk_list:

                # 提取当前块的隐空间表示和遮罩
                sample_latent = latent[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]]
                sample_mask = mask[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]]
                sample_masked_latent = masked_latent[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]]

                # 对当前块进行采样
                pred_noise = self.sample_block(sample_latent, sample_masked_latent, sample_mask, prompt_embeds, Ti, guidance_scale)   # 每一个时间步的采样过程
                
                # 累加更新结果，并更新计数器
                full_latent[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]] += pred_noise
                count[:, :, chunk_block[0]:chunk_block[1], chunk_block[2]:chunk_block[3]] += 1

            # 使用累加结果和计数器更新最终的隐空间表示
            latent = torch.where(count > 0, full_latent / count, full_latent)

        return latent
    
    def load_scheduler(self):
        # 加载调度器
        # 加载DDIMScheduler用于控制图像生成过程中的噪声减少步骤
        MN = self.base_model_name
        self.ddim = DDIMScheduler.from_pretrained(MN, 
                                                  subfolder="scheduler", 
                                                  local_files_only=self.only_local_files, 
                                                #   torch_dtype=torch.float16, 
                                                  use_safetensors=True, 
                                                  cache_dir = self.cache_dir)
        

        self.ddim.set_timesteps(self.sampleTimeStep, device="cuda:0")

    def load_model(self):
        # 加载UNet模型
        # 加载条件UNet模型用于图像生成
        self.unet = UNet2DConditionModel.from_pretrained(self.base_model_name, 
                                                            local_files_only = self.only_local_files, 
                                                            torch_dtype=torch.float16, 
                                                            # use_safetensors=True, 
                                                            subfolder = "unet",
                                                            cache_dir = self.cache_dir).cuda()
        
     
        self.unet.enable_xformers_memory_efficient_attention()
        
    def getLatent_model(self):
        # 加载VAE模型
        # 加载VAE模型用于编码和解码图像到隐空间
        MN = self.base_model_name
        self.vae = AutoencoderKL.from_pretrained(MN, 
                                                 local_files_only = self.only_local_files,
                                                 torch_dtype=torch.float16,
                                                #  use_safetensors=True,
                                                 subfolder = "vae",
                                                 cache_dir = self.cache_dir).cuda()
        

    def load_text_encoder(self):
        # 加载文本编码器
        # 加载CLIP的文本模型和分词器，用于将文本转换为嵌入向量
        MN = self.base_model_name
        self.text_encoder = CLIPTextModel.from_pretrained(MN, 
                                                          local_files_only = self.only_local_files,
                                                          torch_dtype=torch.float16,
                                                        #   use_safetensors=True,
                                                          subfolder = "text_encoder",
                                                          cache_dir = self.cache_dir).cuda()
    
        self.tokenizer = CLIPTokenizer.from_pretrained(MN,
                                                         local_files_only = self.only_local_files,
                                                         subfolder = "tokenizer",
                                                         cache_dir = self.cache_dir)
        
        
    @staticmethod
    def tokenize_prompt(tokenizer, prompt):
        # 分词处理文本提示
        # 使用CLIPTokenizer处理文本提示
        text_inputs = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        return text_input_ids 

    
    def encode_prompt(self, prompt:str, neg_prompt:str = None):
        # 编码文本提示
        # 将正面和负面提示编码为嵌入向量
        text_input_ids = self.tokenize_prompt(self.tokenizer, prompt)

        prompt_embeds = self.text_encoder(
            text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )
        encoder_hidden_states = prompt_embeds.hidden_states[-2]
        prompt_embeds = self.text_encoder.text_model.final_layer_norm(encoder_hidden_states)
        # prompt_embeds = prompt_embeds[0]

        if neg_prompt is None:
            neg_prompt = ""
        negative_text_input_ids = self.tokenize_prompt(self.tokenizer, neg_prompt)
        negative_prompt_embeds = self.text_encoder(
            negative_text_input_ids.to(self.text_encoder.device),
            output_hidden_states=True,
        )
        negative_prompt_embeds = negative_prompt_embeds[0]
    
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds

    def get_text_embedding(self, prompt:str, neg_prompt:str = None):
        # 获取文本嵌入向量
        # 对外提供的接口，内部调用encode_prompt方法
        return self.encode_prompt(prompt, neg_prompt)

        
    def getImgLatent(self, img:torch.Tensor):
        # 将图像编码到隐空间
        # 使用VAE模型编码图像
        # img = self.image_processor.preprocess(img)
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
    
    def getImg(self, latent:torch.Tensor):
        # 从隐空间解码图像
        # 使用VAE模型解码隐向量到图像
        image = self.vae.decode(latent / self.vae.config.scaling_factor)[0]
        image = image.detach()
        image = self.image_processor.postprocess(image, output_type="np", do_denormalize=[True])
        return image

    
def main():
    pass

if __name__ == "__main__":
    main()


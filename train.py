import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import PIL.Image
import numpy as np
import os
from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTokenizer, CLIPTextModel

from diffusers import PNDMScheduler, StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPFeatureExtractor

import random

# 文本描述，包含新增的词汇
texts = [
    'a photo of a <cat-toy>',
    'a rendering of a <cat-toy>',
    'a cropped photo of the <cat-toy>',
    'the photo of a <cat-toy>',
]

# 图片文件路径
img_files = [
    'datas/cat_toy/0.jpeg',
    'datas/cat_toy/1.jpeg',
    'datas/cat_toy/2.jpeg',
    'datas/cat_toy/3.jpeg',
]

# 对应的遮罩文件路径，这需要你根据实际情况进行设置
mask_files = [
    'datas/cat_toy_masks/0_mask.jpeg',
    'datas/cat_toy_masks/1_mask.jpeg',
    'datas/cat_toy_masks/2_mask.jpeg',
    'datas/cat_toy_masks/3_mask.jpeg',
]

# 设置模型和tokenizer
DEFAULT_MODEL = "stabilityai/stable-diffusion-2-inpainting"

class TokenizerManager:
    def __init__(self, model_name, new_tokens):
        """
        初始化TokenizerManager类。

        参数:
        - model_name: 预训练模型的名称，用于加载tokenizer。
        - new_tokens: 一个列表，包含要添加到tokenizer中的新词。
        """
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, subfolder='tokenizer')
        self.new_tokens = new_tokens
        self.new_token_ids = self.add_new_tokens(new_tokens)

    def add_new_tokens(self, new_tokens):
        """
        向tokenizer中添加新词，并返回新词的token id列表。

        参数:
        - new_tokens: 一个列表，包含要添加的新词。
        
        返回:
        - 一个列表，包含新词的token id。
        """
        added_tokens = self.tokenizer.add_tokens(new_tokens)
        if added_tokens > 0:
            print(f"Added {added_tokens} new tokens.")
        return [self.tokenizer.convert_tokens_to_ids(token) for token in new_tokens]

# 定义数据集类
class InpaintingDataset(torch.utils.data.Dataset):
    def __init__(self, img_files, mask_files, texts, tokenizer):
        self.img_files = img_files
        self.mask_files = mask_files
        self.texts = texts
        self.tokenizer = tokenizer
        # 定义transformations
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),  # 调整图像大小
            transforms.ToTensor(),  # 转换为tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
        ])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 随机选择一段文本并进行tokenize处理
        text = random.choice(self.texts)
        input_ids = self.tokenizer(text, padding='max_length', truncation=True, max_length=77, return_tensors='pt')['input_ids'][0]

        # 加载图像
        img = PIL.Image.open(self.img_files[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 生成遮罩
        mask = self.generate_mask(self.img_files[idx])

        # 生成被遮罩的图像版本
        # 注意：这里假设遮罩值为1的区域是需要被遮挡的区域
        masked_img = img * (1 - mask)

        # # 需要确保mask的维度与img一致，或者进行适当的调整
        # # 如果使用单通道遮罩，则需要展开遮罩以匹配图像的通道数
        # if mask.shape[0] == 1:  # 如果遮罩是单通道的
        #     mask = mask.repeat(3, 1, 1)  # 重复遮罩以匹配图像的通道数

        return {"input_ids": input_ids, "pixel": img, "masked_pixel": masked_img, "mask": mask}
    
    def generate_mask(self, file_name):
        # 假设文件名中包含数字，根据数字的奇偶性选择遮罩生成策略
        num = int(''.join(filter(str.isdigit, file_name)))
        mask = torch.zeros((1, 512, 512))  # 创建一个与图像同尺寸的遮罩tensor

        if num % 2 == 0:  # 偶数文件名使用策略A
            mask[:, 256:, :] = 1  # 例如，将图像右半部分设为需要填充的区域
        else:  # 奇数文件名使用策略B
            mask[:, :256, :] = 1  # 将图像左半部分设为需要填充的区域

        return mask

class InpaintingTrainer:
    def __init__(self, model_name, dataset, tokenizer_manager, batch_size=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.dataset = dataset
        self.loader = DataLoader(dataset, batch_size, shuffle=True)
        self.tokenizer_manager = tokenizer_manager
        
        self.device = device
        self._init_models(model_name)
        self._init_scheduler()

    def _freeze_model_parameters(self, model, layers_to_freeze=[]):
        """
        冻结指定模型的部分参数。
        :param model: 要冻结参数的模型。
        :param layers_to_freeze: 一个字符串列表，指定要冻结的层的名称关键词。
        """
        for name, param in model.named_parameters():
            param.requires_grad = True  # 默认设置为可训练
            if any(layer_name in name for layer_name in layers_to_freeze):
                param.requires_grad = False  # 冻结指定层的参数

    def _freeze_encoder_parameters(self, freeze_all_but_embedding=False):
        """
        冻结文本编码器的参数，可选择保留嵌入层为可训练。
        :param freeze_all_but_embedding: 是否保留嵌入层为可训练状态。
        """
        for name, param in self.text_encoder.named_parameters():
            if freeze_all_but_embedding and 'embeddings' in name:
                continue  # 保留嵌入层参数为可训练状态
            param.requires_grad = False  # 冻结参数

    def getImgLatent(self, img:torch.Tensor):
        # 将图像编码到隐空间
        # 使用VAE模型编码图像
        # img = self.image_processor.preprocess(img)
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor
    
    def _init_models(self, model_name):
        # 加载并初始化模型
        self.text_encoder = CLIPTextModel.from_pretrained(model_name, subfolder='text_encoder').to(self.device)
        self.vae = AutoencoderKL.from_pretrained(model_name, subfolder='vae').to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_name, subfolder='unet').to(self.device)
        
        # 调整文本编码器以适应新词
        self.text_encoder.resize_token_embeddings(len(self.tokenizer_manager.tokenizer))
        
        # 准备模型状态
        self.text_encoder.train()
        self.vae.train()
        self.unet.train()

        # 冻结文本编码器的部分参数
        self._freeze_encoder_parameters(freeze_all_but_embedding=True)

        # 选择性冻结VAE的部分参数
        self._freeze_model_parameters(self.vae, layers_to_freeze=['encoder'])  # 示例：冻结编码器部分

        # 选择性冻结UNet的部分参数
        self._freeze_model_parameters(self.unet, layers_to_freeze=['downsampling', 'middle_conv'])  # 示例：冻结下采样和中间卷积层

        # 定义优化器
        self.optimizer = torch.optim.AdamW([
            {'params': self.text_encoder.parameters(), 'lr': 2e-5},
            {'params': self.vae.parameters(), 'lr': 1e-4},
            {'params': self.unet.parameters(), 'lr': 1e-4}
        ])

    def _init_scheduler(self):
        # 初始化噪声调度器
        self.noise_scheduler = DDPMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule='scaled_linear', num_train_timesteps=1000, tensor_format='pt')
    
    def forward(self, data):
        """
        根据给定的数据进行一次前向传播，并计算损失。
        """
        # 使用vae压缩原图像
        # [1, 3, 512, 512] -> [1, 4, 64, 64]
        latents = self.getImgLatent(data['pixel'].to(self.device))
        masked_latent = self.getImgLatent(data['masked_pixel'].to(self.device))

        # 为inpainting任务引入遮罩处理，将遮罩调整为与隐空间(latents)相同的尺寸
        mask = data['mask'].to(self.device)
        mask = torch.nn.functional.interpolate(mask.float(), size=(latents.shape[2], latent.shape[3]), mode='nearest')
        # mask = torch.nn.functional.interpolate(mask, size=(64, 64), mode='nearest')
        
        # 将遮罩应用到隐空间表示上，模拟inpainting任务中被遮挡的区域
        # 这里简单地将遮罩区域的隐空间表示置零
        masked_latents = latents * (mask_resized < 0.5)

        # 随机b张噪声图
        # [1, 4, 64, 64]
        noise = torch.randn(latents.shape).to(self.device)

        # 随机采样0-1000之间的b个数字，为每张图片随机一个步数
        # [1]
        timesteps = torch.randint(0, 1000, (1, ), device=self.device).long()

        # 把噪声添加到压缩图中，维度不变
        noisy_latents = self.noise_scheduler.add_noise(masked_latents, noise, timesteps)

        # 编码文字
        # [1, 77, 768]
        encoder_hidden_states = self.text_encoder(data['input_ids'])[0]

        # 根据文字信息，从混合图中把噪声图给抽取出来
        # [1, 4, 64, 64]
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, mask=mask).sample

        # 求mse loss即可
        # [1, 4, 64, 64]
        loss = torch.nn.functional.mse_loss(noise_pred, noise, reduction='none')
        # [1, 4, 64, 64] -> [1]
        loss = loss.mean(dim=[1, 2, 3])

        return loss

    def train(self, epochs=6000, save_interval=100):
        loss_mean = []
        for epoch in range(epochs):
            for i, data in enumerate(self.loader):
                # 确保所有数据都移动到了正确的设备
                data = {k: v.to(self.device) for k, v in data.items()}

                loss = self.forward(data)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                loss_mean.append(loss.item())

            # 每隔一定数量的epochs保存模型
            if (epoch + 1) % save_interval == 0:
                print(f"Epoch {epoch}, Average Loss: {np.mean(loss_mean)}")
                self.save_model(model_path=f'models/inpainting_model_epoch_{epoch}')
                loss_mean = []
            
            if epoch % 30 == 0: #每30个epoch的平均损失打印
                print(f"Epoch {epoch}, Average Loss: {np.mean(loss_mean)}")
                loss_mean = []

    def save_model(self, model_path='models/inpainting_model'):
        #保存模型
        pipeline = StableDiffusionPipeline(
            text_encoder=self.text_encoder,
            vae=self.vae,
            unet=self.unet,
            tokenizer=self.tokenizer,
            scheduler=PNDMScheduler(beta_start=0.00085,
                                    beta_end=0.012,
                                    beta_schedule='scaled_linear',
                                    skip_prk_steps=True),
            safety_checker=None, # inpainting任务可能不需要safety_checker
            feature_extractor=CLIPFeatureExtractor.from_pretrained(
                'openai/clip-vit-base-patch32'),
        )

        # 保存整个pipeline到指定目录，包括所有组件和配置
        pipeline.save_pretrained(model_path)
        # 保存新词的嵌入向量
        learned_embeds = {}
        for new_token_id in self.tokenizer_manager.new_token_ids:  # 遍历所有新词的ID
            embed = self.text_encoder.get_input_embeddings().weight[new_token_id].detach().cpu()
            learned_embeds[new_token_id] = embed
        
        # 如果有新词嵌入需要保存
        if learned_embeds:
            torch.save(learned_embeds, os.path.join(model_path, 'learned_embeds.bin'))

if __name__ == '__main__':
    # 初始化TokenizerManager
    model_name = DEFAULT_MODEL
    new_tokens = ["<cat-toy>", "<another-token>"]  # 示例：添加多个新词
    tokenizer_manager = TokenizerManager(model_name, new_tokens)

    # 实例化数据集
    dataset = InpaintingDataset(img_files, mask_files, texts, tokenizer_manager.tokenizer)

    trainer = InpaintingTrainer(DEFAULT_MODEL, dataset, tokenizer_manager)

    trainer.train()
    trainer.save_model()


########################################################################################

# def test(prompt):
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     #节省显存
#     device = 'cpu'

#     #加载
#     pipe = StableDiffusionPipeline.from_pretrained('models/cat_toy',
#                                                    torch_dtype=torch.float32)
#     pipe = pipe.to(device)

#     #运算
#     images = pipe([prompt] * 4, num_inference_steps=50,
#                   guidance_scale=7.5).images

#     #画图
#     def show(image, idx):
#         plt.subplot(1, 4, idx)
#         plt.imshow(image)
#         plt.axis('off')

#     plt.figure(figsize=[8, 3])
#     for i in range(len(images)):
#         show(images[i], i + 1)
#     plt.show()


# test('a grafitti in a wall with a <cat-toy> on it')
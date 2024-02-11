import gradio as gr
import numpy as np
import cv2 as cv
import torch

from imageExtension import ImageExtension

class Event():
    def __init__(self) -> None:
        self.col_move = 0
        self.row_move = 0
        self.grid_size = 128

        self.img_ext = ImageExtension(only_local_files=False)

    def set_grid_size(self, grid_size:int):
        self.grid_size = int(grid_size)

    def image_change(self, img:np.ndarray):
        selection = [([0, 0, img.shape[1], img.shape[0]], "image")]
        return img, selection
    
    def render_box(self, img:np.ndarray):
        # 获取输入图像的形状
        imgShape = img.shape

        # 初始化selection列表，用于存储原始图像区域和扩展区域的坐标及类型
        selection = [([max(-self.col_move, 0) * self.grid_size, # 计算原始图像左上角的x坐标
                       max(-self.row_move, 0) * self.grid_size, # 计算原始图像左上角的y坐标
                       max(-self.col_move, 0) * self.grid_size + imgShape[1], # 计算原始图像右下角的x坐标
                       max(-self.row_move, 0) * self.grid_size + imgShape[0]], "image")] # 计算原始图像右下角的y坐标
        
        # 如果图像需要在任一方向上进行移动，则计算扩展区域的坐标，并添加到selection中
        if self.col_move != 0 or self. row_move != 0:
            selection.append(
                     ([max(self.col_move, 0) * self.grid_size, # 计算扩展区域左上角的x坐标
                       max(self.row_move, 0) * self.grid_size, # 计算扩展区域左上角的y坐标
                       max(self.col_move, 0) * self.grid_size + imgShape[1], # 计算扩展区域右下角的x坐标
                       max(self.row_move, 0) * self.grid_size + imgShape[0]], "extension") # 计算扩展区域右下角的y坐标
            )

        # 根据移动的行列数计算新图像的尺寸
        newImgShape = [
            abs(self.row_move) * self.grid_size + imgShape[0],
            abs(self.col_move) * self.grid_size + imgShape[1]
        ]
        # 创建一个新的空白图像
        newImg = np.zeros([*newImgShape, 3], dtype=np.uint8)

        # 在新图像上复制原始图像到指定的位置
        newImg[selection[0][0][1] : selection[0][0][3], selection[0][0][0] : selection[0][0][2]] = img

        # 返回新图像和边界框信息
        return (newImg, selection)
    
    def left_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.col_move -= 1
        self.col_move = self.col_move if abs(self.col_move * self.grid_size) < imgShape[1]//2 else self.col_move + 1

        return self.render_box(img)
    
    def right_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.col_move += 1
        self.col_move = self.col_move if self.col_move * self.grid_size < imgShape[1]//2 else self.col_move - 1

        return self.render_box(img)
    
    def up_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.row_move -= 1
        self.row_move = self.row_move if abs(self.row_move * self.grid_size) < imgShape[0]//2 else self.row_move + 1

        return self.render_box(img)
    
    def down_ext(self, img:np.ndarray):
        imgShape = img.shape
        
        self.row_move += 1
        self.row_move = self.row_move if self.row_move * self.grid_size < imgShape[0]//2 else self.row_move - 1

        return self.render_box(img)
    
    def process(self, img:np.ndarray, prompt:str, neg_prompt:str, sampleStep:int, guidance_scale:int):
        """
        处理图像，根据给定的文本提示进行图像生成或编辑。

        参数:
        - img: np.ndarray, 原始图像数组。
        - prompt: str, 正面文本提示，描述希望生成或编辑到图像中的内容。
        - neg_prompt: str, 负面文本提示，描述不希望出现在图像中的内容。
        - sampleStep: int, 采样步骤的数量，影响生成过程的细节程度。
        - guidance_scale: int, 指导缩放比例，控制正面提示与负面提示的影响力度。

        返回:
        - out_img: 处理后的图像数组。
        """
        # 调用render_box函数获取预处理后的图像和边界框信息
        newimg, bbox = self.render_box(img)
        
        # 清除CUDA缓存，以避免显存溢出
        torch.cuda.empty_cache()

        # 设置采样步骤
        self.img_ext.set_step(sampleStep)
        with torch.no_grad():
            # 将输入图像从NumPy数组转换为PyTorch张量，并调整为模型所需的形状和类型
            torchImg = torch.from_numpy(newimg).permute(2, 0, 1).unsqueeze(0).half()
            # 根据边界框信息提取需要处理的图像区域，并进行归一化处理
            torchUseImg = torchImg[:, :, bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] / 127.5 - 1.0

            # 将处理后的区域放回原图像对应位置
            torchImg[:, :, bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] = torchUseImg

            # 将图像数据转移到CUDA设备上
            torchImg = torchImg.cuda().half()
            # 创建一个遮罩，初始化为全0，然后根据bbox设置需要编辑的区域为1
            mask = np.zeros([torchImg.shape[2], torchImg.shape[3]], dtype=np.uint8)
            mask[bbox[1][0][1] : bbox[1][0][3], bbox[1][0][0] : bbox[1][0][2]] = 1      # 需要扩展的地方设置为mask 1
            mask[bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] = 0       # 重叠区域再次置零

            # 使用中值滤波平滑遮罩边缘
            mask = cv.medianBlur(mask, 3)
            mask = mask[None, None, :, :]
            # 将遮罩转换为torch张量，并传输到CUDA设备
            mask = torch.from_numpy(mask).cuda().half()

            # 应用遮罩生成被遮挡的图像版本
            masked_img = torchImg * (mask < 0.5)  # 反向掩码

            # 将原图和被遮挡的图像转换为隐空间表示
            latent = self.img_ext.getImgLatent(torchImg)
            masked_latent = self.img_ext.getImgLatent(masked_img)

            # 将遮罩大小调整至隐空间的尺寸
            mask = torch.nn.functional.interpolate(
                mask, size=(mask.shape[2] // 8, mask.shape[3] // 8) # mask 转换为latent尺寸
            )

            # 为隐空间表示添加噪声
            noise = torch.randn_like(latent)
            extend_latent = self.img_ext.addNoise(latent, noise, self.img_ext.allTimestep[0])  # 加噪

            # 计算处理块的列表，这些块基于bbox和缩放，转换为latent尺寸
            chunkList = [
                [bbox[0][0][1]//8, bbox[0][0][3]//8, bbox[0][0][0]//8, bbox[0][0][2]//8], 
                [bbox[1][0][1]//8, bbox[1][0][3]//8, bbox[1][0][0]//8, bbox[1][0][2]//8]
                ]

            # 获取正面和负面文本提示的嵌入表示
            prompt_embeds = self.img_ext.get_text_embedding(prompt, neg_prompt)
        
            # 使用嵌入表示、隐空间表示和遮罩进行图像采样
            result = self.img_ext.sample(extend_latent, masked_latent, mask, chunkList, prompt_embeds, guidance_scale=guidance_scale)    
            # 将隐空间表示解码回图像
            out_img = self.img_ext.getImg(result)[0]

        # 调整输出图像大小以匹配原始图像
        out_img = cv.resize(out_img, (newimg.shape[1], newimg.shape[0]))

        # 创建输出遮罩以确定哪些部分应从原始图像中保留
        outPaintMask = np.zeros_like(out_img, dtype=np.uint8)
        outPaintMask[bbox[1][0][1] : bbox[1][0][3], bbox[1][0][0] : bbox[1][0][2]] = 1   
        outPaintMask[bbox[0][0][1] : bbox[0][0][3], bbox[0][0][0] : bbox[0][0][2]] = 1

        # 结合原图和生成的图像部分
        out_img = np.where(outPaintMask < 0.5, newimg, out_img)   

        return out_img
    
    def start_process(self, img:np.ndarray, prompt:str, neg_prompt:str, sampleStep:int, guidance_scale:int):
        return self.process(img, prompt, neg_prompt, sampleStep, guidance_scale)
    
    def apply_to(self, img):
        self.col_move = 0
        self.row_move = 0

        return img


def setupUI(event_process:Event):
    """
    使用Gradio库设置用户界面，允许用户上传图片、输入文本提示、调整参数，并通过事件处理函数处理图像。

    参数:
    - event_process: Event 类的实例，包含了处理各种UI事件的函数。

    返回:
    - UI: Gradio界面实例。
    """
    with gr.Blocks() as UI:
        with gr.Row():
            # 创建一行用于放置图像上传和编辑区域
            with gr.Column():
                # 创建一列用于放置图像上传组件
                image = gr.Image(height=300) # 图像上传组件
                img_window = gr.AnnotatedImage( # 图像编辑组件，用于显示和标记图像的不同区域
                    color_map={"image": "#a89a00", "extension": "#00aeff"},
                    height = 300
                )
                # 设置图像上传时的回调函数，上传图像后更新img_window组件
                image.upload(event_process.image_change, inputs=[image], outputs=[img_window])
                # 设置应用按钮点击时清空image组件的内容
                image.clear(event_process.apply_to, inputs=[image])
            # 预览处理后的图像
            pre_image = gr.Image(height=600, interactive=False)

        # 创建另一行用于放置文本输入框、滑块和按钮
        with gr.Row():
            # 创建一列用于放置文本输入框和滑块
            with gr.Column():
                # 正面文本提示输入框
                prompt = gr.Textbox(interactive=True, label="prompt")
                # 负面文本提示输入框
                neg_prompt = gr.Textbox(interactive=True, value="Blurred, incomplete, distorted, misplaced", label="negative prompt")
                # 指导比例滑块
                guidance_scale = gr.Slider(0, 20, value=7, label="guidance scale", step=0.5)
            # 创建另一列用于放置方向控制按钮
            with gr.Column():
                with gr.Row():
                    # 方向控制按钮
                    left = gr.Button("LEFT")
                    with gr.Column():
                        up = gr.Button("UP")
                        down = gr.Button("DOWN")
                    right = gr.Button("right")

                    # 设置方向按钮的点击事件
                    left.click(event_process.left_ext, inputs=[image], outputs=[img_window])
                    right.click(event_process.right_ext, inputs=[image], outputs=[img_window])
                    up.click(event_process.up_ext, inputs=[image], outputs=[img_window])
                    down.click(event_process.down_ext, inputs=[image], outputs=[img_window])
                # 网格大小数字输入框
                grid_size = gr.Number(128, label="grid size")
                # 设置网格大小改变时的回调函数
                grid_size.change(event_process.set_grid_size, inputs=[grid_size])
        
        # 创建一行用于放置处理和应用按钮
        with gr.Row():
            with gr.Row():
                # 处理按钮，点击时调用start_process函数处理图像
                process = gr.Button("start")
                # 应用按钮，点击时将处理后的图像应用到原图像上传组件
                apply = gr.Button("apply")
            # 采样步骤滑块
            sample_step = gr.Slider(0, 100, value=30, label="sample step", step=1)
            # 设置处理按钮的点击事件
            process.click(event_process.start_process, inputs=[image, prompt, neg_prompt, sample_step, guidance_scale], outputs=[pre_image])
            # 设置应用按钮的点击事件
            apply.click(event_process.apply_to, inputs=[pre_image], outputs=[image])


    return UI


def main():
    event_process = Event()
    UI = setupUI(event_process)
    UI.launch()

if __name__ == "__main__":
    main() 


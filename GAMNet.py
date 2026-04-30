import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from CompareNets.models.vmamba import VSSBlock
except ImportError:
    pass

class MBEUpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(MBEUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)

        self.sculpting_gate = nn.Sequential(
            nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.boundary_extractor = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        self.conv = nn.Sequential(
            nn.Conv2d(skip_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):

        x_up = self.up(x)

        gate = self.sculpting_gate(torch.cat([x_up, skip], dim=1))
        x_sculpted = x_up * gate # “虚胖”溢出到平坦森林区域的深层激活被强行置为 0！
        edge_mask = self.boundary_extractor(skip)
        skip_enhanced = skip + skip * edge_mask
        out = self.conv(torch.cat([x_sculpted, skip_enhanced], dim=1))

        return out

class MCSF(nn.Module):

    def __init__(self, channels, reduction=4):
        super(MCSF, self).__init__()


        self.mamba_channels = channels // reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channels, self.mamba_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mamba_channels)
        )

        self.mamba_scanner = VSSBlock(
            hidden_dim=self.mamba_channels,
            drop_path=0.1,
            d_state=16 # 保持轻量，SSM 的状态维度
        )
        self.mamba_norm_out = nn.LayerNorm(self.mamba_channels)

        joint_dim = channels * 2 + self.mamba_channels
        mid_channels = max(16, channels // reduction)

        self.joint_excitation = nn.Sequential(
            nn.Linear(joint_dim, mid_channels, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels * 2, bias=False),
            nn.Sigmoid() # 生成 0~1 的通道放大系数
        )

    def forward(self, rgb, dem):
        B, C, H, W = rgb.size()

        x_base = rgb + dem

        x_mamba_in = self.reduce_conv(x_base)
        x_mamba_in = x_mamba_in.permute(0, 2, 3, 1).contiguous()

        x_mamba = self.mamba_scanner(x_mamba_in)

        x_mamba = self.mamba_norm_out(x_mamba).permute(0, 3, 1, 2).contiguous()

        z_mamba = F.adaptive_avg_pool2d(x_mamba, 1).view(B, self.mamba_channels)

        z_rgb = F.adaptive_avg_pool2d(rgb, 1).view(B, C)
        z_dem = F.adaptive_avg_pool2d(dem, 1).view(B, C)


        z_joint = torch.cat([z_rgb, z_dem, z_mamba], dim=1)
        weights = self.joint_excitation(z_joint)

        weight_rgb = weights[:, :C].view(B, C, 1, 1)
        weight_dem = weights[:, C:].view(B, C, 1, 1)


        out_rgb = rgb * (1.0 + weight_rgb)
        out_dem = dem * (1.0 + weight_dem)

        # 纯粹的逐元素相加，0 空间溢出，完美保住所有边界，下限极高！
        out = out_rgb + out_dem

        return out

class BMIM(nn.Module):
    def __init__(self, channels, reduction=4):
        super(BMIM, self).__init__()

        mip = max(8, channels // reduction)

        self.rgb_weight_extractor = nn.Sequential(
            nn.Linear(channels, mip, bias=False),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Linear(mip, channels, bias=False),
            nn.Sigmoid()
        )

        self.dem_weight_extractor = nn.Sequential(
            nn.Linear(channels, mip, bias=False),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Linear(mip, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, rgb, dem):
        B, C, H, W = rgb.size()
        N = H * W

        rgb_flat = rgb.view(B, C, N)
        dem_flat = dem.view(B, C, N)

        rgb_norm = F.normalize(rgb_flat, dim=-1)
        dem_norm = F.normalize(dem_flat, dim=-1)

        cross_covariance = torch.bmm(rgb_norm, dem_norm.transpose(1, 2))

        rgb_resonance = cross_covariance.sum(dim=2)

        dem_resonance = cross_covariance.sum(dim=1)

        weight_rgb = self.rgb_weight_extractor(rgb_resonance).view(B, C, 1, 1)
        weight_dem = self.dem_weight_extractor(dem_resonance).view(B, C, 1, 1)

        out_rgb = rgb + rgb * weight_rgb
        out_dem = dem + dem * weight_dem

        return out_rgb, out_dem

class VectorSobel(nn.Module):

    def __init__(self, channels):
        super(VectorSobel, self).__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]])
        self.sobel_x = sobel_x.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.sobel_y = sobel_y.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.register_buffer('filter_x', self.sobel_x)
        self.register_buffer('filter_y', self.sobel_y)

    def forward(self, x):
        g_x = F.conv2d(x, self.filter_x, padding=1, groups=x.shape[1])
        g_y = F.conv2d(x, self.filter_y, padding=1, groups=x.shape[1])
        return g_x, g_y

class CGDFusion(nn.Module):

    def __init__(self, channels):
        super(CGDFusion, self).__init__()

        self.bottleneck_dim = min(16, channels)

        self.rgb_proj = nn.Sequential(
            nn.Conv2d(channels, self.bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.bottleneck_dim),
            nn.ReLU(inplace=True)
        )
        self.dem_proj = nn.Sequential(
            nn.Conv2d(channels, self.bottleneck_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.bottleneck_dim),
            nn.ReLU(inplace=True)
        )

        self.vector_extractor = VectorSobel(self.bottleneck_dim)


        self.alignment_gate = nn.Sequential(
            nn.Conv2d(self.bottleneck_dim * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        self.dem_injector = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )


        self.fusion_out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, dem):
        rgb_sub = self.rgb_proj(rgb)
        dem_sub = self.dem_proj(dem)

        r_gx, r_gy = self.vector_extractor(rgb_sub)
        d_gx, d_gy = self.vector_extractor(dem_sub)

        eps = 1e-6
        # 1. 计算幅值 (表示边界的强度)
        mag_rgb = torch.sqrt(r_gx**2 + r_gy**2 + eps)
        mag_dem = torch.sqrt(d_gx**2 + d_gy**2 + eps)

        dot_product = r_gx * d_gx + r_gy * d_gy
        cos_sim = torch.abs(dot_product / (mag_rgb * mag_dem + eps))

        alignment_features = torch.cat([mag_rgb, mag_dem, cos_sim], dim=1)
        align_gate = self.alignment_gate(alignment_features) #[B, Channels, H, W]


        dem_injected = self.dem_injector(dem) * align_gate
        rgb_refined = rgb + dem_injected

        out = self.fusion_out(torch.cat([rgb_refined, dem], dim=1))
        return out

class GAMnet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(GAMnet, self).__init__()

        # =================================================
        # 1. RGB 分支 (ResNet34)
        # =================================================
        rgb_backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.rgb_initial = nn.Sequential(rgb_backbone.conv1, rgb_backbone.bn1, rgb_backbone.relu)
        self.rgb_maxpool = rgb_backbone.maxpool
        self.rgb_layer1 = rgb_backbone.layer1
        self.rgb_layer2 = rgb_backbone.layer2
        self.rgb_layer3 = rgb_backbone.layer3
        self.rgb_layer4 = rgb_backbone.layer4

        # =================================================
        # 2. DEM 分支 (ResNet34)
        # =================================================
        # 同样加载预训练权重，利用 Transfer Learning
        dem_backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)

        # 🔥 修改 DEM 第一层卷积：3通道 -> 1通道
        # 策略：取原 RGB 权重的平均值，保留边缘提取能力
        original_conv = dem_backbone.conv1
        new_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            new_conv.weight.copy_(torch.mean(original_conv.weight, dim=1, keepdim=True))

        self.dem_initial = nn.Sequential(new_conv, dem_backbone.bn1, dem_backbone.relu)
        self.dem_maxpool = dem_backbone.maxpool
        self.dem_layer1 = dem_backbone.layer1
        self.dem_layer2 = dem_backbone.layer2
        self.dem_layer3 = dem_backbone.layer3
        self.dem_layer4 = dem_backbone.layer4

        self.interaction1 = BMIM(64)
        self.interaction2 = BMIM(128)
        self.interaction3 = BMIM(256)

        self.fuse0 = CGDFusion(64)
        self.fuse1 = CGDFusion(64)
        self.fuse2 = CGDFusion(128)
        self.fuse3 = MCSF(256)
        self.fuse4 = MCSF(512)

        # --- Decoder ---
        self.decode4 = MBEUpBlock(512, 256, 256)
        self.decode3 = MBEUpBlock(256, 128, 128)
        self.decode2 = MBEUpBlock(128, 64, 64)
        self.decode1 = MBEUpBlock(64, 64, 64)

        self.final_up = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x):
        rgb = x[:, 0:3]
        dem = x[:, 3:4]

        r0 = self.rgb_initial(rgb)
        d0 = self.dem_initial(dem)
        f0 = self.fuse0(r0, d0)

        r_stem = self.rgb_maxpool(r0)
        d_stem = self.dem_maxpool(d0)

        r1 = self.rgb_layer1(r_stem)
        d1 = self.dem_layer1(d_stem)
        r1, d1 = self.interaction1(r1, d1)
        f1 = self.fuse1(r1, d1)

        r2 = self.rgb_layer2(r1)
        d2 = self.dem_layer2(d1)
        r2, d2 = self.interaction2(r2, d2)
        f2 = self.fuse2(r2, d2)

        r3 = self.rgb_layer3(r2)
        d3 = self.dem_layer3(d2)
        r3, d3 = self.interaction3(r3, d3)
        f3 = self.fuse3(r3, d3)

        r4 = self.rgb_layer4(r3)
        d4 = self.dem_layer4(d3)
        f4 = self.fuse4(r4, d4)

        # Decoding
        x = self.decode4(f4, f3)
        x = self.decode3(x, f2)
        x = self.decode2(x, f1)
        x = self.decode1(x, f0)

        x = self.final_up(x)
        output = self.final_conv(x)
        heatmap = torch.sigmoid(output[:, 1:2, :, :])  # 只取landslide类别的概率
        return output,heatmap

# if __name__ == '__main__':
#     # 测试代码
#     model = DualResNet34Unet(num_classes=2,pretrained=False).cuda()
#
#     print(model)
#
#     # 模拟输入 (Batch=2, Channel=4, 256x256)
#     # 前3通道RGB，第4通道DEM
#     inputs = torch.randn(2, 4, 256, 256).cuda()
#
#     outputs = model(inputs)
#     print(f"Input: {inputs.shape}")
#     print(f"Output: {outputs.shape}")
#
#     # 计算参数量
#     total_params = sum(p.numel() for p in model.parameters())
#     print(f"Total Parameters: {total_params / 1e6:.2f} M")

import torch
import time
from thop import profile
from thop import clever_format

if __name__ == '__main__':
    # ========================================================
    # 1. 初始化模型与输入
    # ========================================================
    # 设定设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 你的模型输入通道为 3 (AFENet默认可能只接收RGB，如果是多模态请改为4)
    input_tensor = torch.rand(1, 4, 256, 256).to(device)

    model = GAMnet(num_classes=2,pretrained=False).to(device)

    # 【必须】切换到推理模式
    model.eval()

    print("================ Model Info ================")
    # 测试一次前向传播
    with torch.no_grad():
        output = model(input_tensor)
    print(f"Input shape: {input_tensor.size()}")
    #print(f"Output shape: {output.size()}")

    # ========================================================
    # 2. 测量理论复杂度 (FLOPs & Params)
    # ========================================================
    print("\n================ Complexity ================")
    # 注意：thop 最好在 CPU 上算，或者确保模型和输入在同设备
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"FLOPs:  {flops_str}")
    print(f"Params: {params_str}")

    # ========================================================
    # 3. 严谨的推理速度测试 (FPS & Inference Time)
    # ========================================================
    print("\n================ Speed Test ================")

    # a. 预热阶段 (Warm-up)
    # 唤醒 GPU 并让 cudnn 寻找最优卷积算法
    print("Warming up GPU...")
    warmup_iters = 50
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)

    # b. 正式测量阶段
    measure_iters = 200
    print(f"Measuring over {measure_iters} iterations...")

    # 【必须】在开始计时前同步 GPU，确保之前的操作全跑完
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(input_tensor)

    # 【必须】在结束计时前同步 GPU，确保所有的计算流都执行完毕
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    # c. 计算指标
    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / measure_iters) * 1000  # 单位：毫秒
    fps = measure_iters / total_time                             # 单位：帧/秒

    print(f"Avg Inference Time: {avg_inference_time_ms:.2f} ms/image")
    print(f"FPS (Batch Size 1): {fps:.2f} frames/s")
    print("============================================")

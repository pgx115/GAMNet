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
    """
    Multilevel Boundary Enhancement UpBlock (跨层边界雕刻解码模块)
    TGRS 顶级创新点 (Story):
    1. 抛弃传统 UNet 盲目的 Concat 拼接，解决深层低频语义向非滑坡区域“溢出(Bleeding)”导致的误报问题。
    2. 引入“物理边界手术刀(Physical Boundary Scalpel)”：利用浅层提取的高分辨率地形结构，对深层虚胖的语义进行精准裁剪。
    3. 边缘强化双流融合：在消除误报的同时，强制网络贴合滑坡真实的物理断裂带。
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super(MBEUpBlock, self).__init__()

        # 基础的上采样层
        self.up = nn.ConvTranspose2d(in_channels, skip_channels, kernel_size=2, stride=2)

        # =========================================================
        # 1. 误报剔除门控 (False Positive Suppression Gate)
        # 联合深层宏观语义与浅层物理结构，判断深层的激活是否为幻觉
        # =========================================================
        self.sculpting_gate = nn.Sequential(
            nn.Conv2d(skip_channels * 2, skip_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels, skip_channels, kernel_size=1, bias=False),
            nn.Sigmoid() # 输出 0~1：0代表跨层语义冲突(误报)，强行抹杀；1代表达成共识，保留。
        )

        # =========================================================
        # 2. 浅层物理边界锐化 (Physical Boundary Sharpening)
        # 从浅层特征中提取纯粹的物理边缘（高频显著性）
        # =========================================================
        self.boundary_extractor = nn.Sequential(
            nn.Conv2d(skip_channels, skip_channels // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(skip_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(skip_channels // 2, 1, kernel_size=1, bias=False),
            nn.Sigmoid() # 输出 [B, 1, H, W] 的高分辨率边缘掩膜
        )

        # 最终的平滑与特征映射
        self.conv = nn.Sequential(
            nn.Conv2d(skip_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        # x: 深层传来的特征 (低分辨率，语义强，边界糊，有误报)
        # skip: 编码器同层传来的跳跃连接特征 (高分辨率，物理边界极度清晰)

        x_up = self.up(x) #[B, skip_channels, H, W]

        # ---------------------------------------------------------
        # Step 1: 语义雕刻 (Semantic Sculpting) - 专治“满屏误报”
        # 让高分辨率的 skip 检查 x_up 的合法性。
        # ---------------------------------------------------------
        gate = self.sculpting_gate(torch.cat([x_up, skip], dim=1))
        x_sculpted = x_up * gate # “虚胖”溢出到平坦森林区域的深层激活被强行置为 0！

        # ---------------------------------------------------------
        # Step 2: 边界强化 (Boundary Enhancement) - 专治“边界丢失”
        # 提取浅层特征中的隐式地形突变带，自我增强
        # ---------------------------------------------------------
        edge_mask = self.boundary_extractor(skip)
        skip_enhanced = skip + skip * edge_mask

        # ---------------------------------------------------------
        # Step 3: 重组与平滑 (Recombination)
        # ---------------------------------------------------------
        # 拼接【被裁剪干净的深层语义】与【被强化的浅层物理结构】
        out = self.conv(torch.cat([x_sculpted, skip_enhanced], dim=1))

        return out

class MCSF(nn.Module):
    """
     Mamba-driven Cross Semantic Fusion Mamba驱动的交叉语义融合模块)

    TGRS 顶级创新点 (Story):
    1. 延续 80.1% 成功的 "Spatially-Agnostic (空间不可知)" 黄金法则：绝对不使用空间门控去乘以特征，杜绝边界模糊与细节丢失。
    2. 克服全局池化的“拓扑盲区”：引入 VSSBlock (Mamba) 进行二维序列扫描，追踪长条形滑坡的连通性。
    3. 拓扑序列描述符 (Topological Sequence Descriptor)：提取 Mamba 的全局扫描结果，生成高级拓扑向量，以指导跨模态通道共振，进一步推高精度上限。
    """
    def __init__(self, channels, reduction=4):
        super(MCSF, self).__init__()

        # =========================================================
        # 1. Mamba 拓扑扫描分支 (Mamba Topological Scanner)
        # =========================================================
        self.mamba_channels = channels // reduction
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(channels, self.mamba_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(self.mamba_channels)
        )

        # 实例化真正的 Mamba Scanner (参照你提供的用法)
        self.mamba_scanner = VSSBlock(
            hidden_dim=self.mamba_channels,
            drop_path=0.1,
            d_state=16 # 保持轻量，SSM 的状态维度
        )
        self.mamba_norm_out = nn.LayerNorm(self.mamba_channels)

        # =========================================================
        # 2. Mamba 增强的联合通道共振分支 (DCSR 核心理念)
        # =========================================================
        # 接收: RGB 基础语义(C) + DEM 基础语义(C) + Mamba 拓扑先验(C/reduction)
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

        # =========================================================
        # Step 1: Mamba 全局连通性扫描 (Topological Scanning)
        # =========================================================
        # 以最安全的 rgb + dem 作为物理底座进行扫描，让 Mamba 去寻找长距离依赖
        x_base = rgb + dem

        # 降维并准备进入 Mamba: 需要[B, H, W, C] 输入
        x_mamba_in = self.reduce_conv(x_base)
        x_mamba_in = x_mamba_in.permute(0, 2, 3, 1).contiguous()

        # Mamba 状态空间扫描
        x_mamba = self.mamba_scanner(x_mamba_in)

        # 归一化并恢复形状[B, mamba_channels, H, W]
        x_mamba = self.mamba_norm_out(x_mamba).permute(0, 3, 1, 2).contiguous()

        # =========================================================
        # Step 2: 提取 "拓扑序列描述符" 和 "基础语义描述符"
        # 【核心操作】: 严格遵循 80.1% 成功经验，用 GAP 剥离空间属性，绝对不做空间乘法！
        # =========================================================
        # z_mamba 不再是简单的均值，而是包含了 Mamba 贪吃蛇扫描后的连续性轨迹浓缩！
        z_mamba = F.adaptive_avg_pool2d(x_mamba, 1).view(B, self.mamba_channels)

        # 提取 RGB 和 DEM 的基础全局语义
        z_rgb = F.adaptive_avg_pool2d(rgb, 1).view(B, C)
        z_dem = F.adaptive_avg_pool2d(dem, 1).view(B, C)

        # =========================================================
        # Step 3: Mamba 增强的跨模态通道共振激励
        # =========================================================
        # 让网络同时看到：光学特征 + 高程特征 + 宏观滑坡连通性
        z_joint = torch.cat([z_rgb, z_dem, z_mamba], dim=1)  #[B, 2C + mamba_C]
        weights = self.joint_excitation(z_joint)             # [B, 2C]

        weight_rgb = weights[:, :C].view(B, C, 1, 1)
        weight_dem = weights[:, C:].view(B, C, 1, 1)

        # =========================================================
        # Step 4: 绝对安全的残差调制 (The 80.1% Golden Rule)
        # =========================================================
        out_rgb = rgb * (1.0 + weight_rgb)
        out_dem = dem * (1.0 + weight_dem)

        # 纯粹的逐元素相加，0 空间溢出，完美保住所有边界，下限极高！
        out = out_rgb + out_dem

        return out

class BMIM(nn.Module):
    """
    Bilinear-Gradient Matrix Interaction Module (双线性梯度矩阵交互模块) BMIM
    TGRS 顶级创新：
    1. 使用二阶跨模态协方差矩阵 (Cross-Covariance Matrix) 捕获细粒度地貌-视觉耦合。 Bilinear Matrix Interaction Module
    2. 严格遵守深度标量缩放 (Depthwise Scalar Scaling)，数学上保证局部物理梯度的方向绝对不变。
    """
    def __init__(self, channels, reduction=4):
        super(BMIM, self).__init__()

        mip = max(8, channels // reduction)

        # 针对协方差矩阵进行降维提炼的 MLP (RGB 视角)
        self.rgb_weight_extractor = nn.Sequential(
            nn.Linear(channels, mip, bias=False),
            nn.BatchNorm1d(mip),
            nn.ReLU(inplace=True),
            nn.Linear(mip, channels, bias=False),
            nn.Sigmoid()
        )

        # 针对协方差矩阵进行降维提炼的 MLP (DEM 视角)
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

        # ---------------------------------------------------------------------
        # Step 1: 空间展平与中心化 (Flatten & L2 Normalization)
        # ---------------------------------------------------------------------
        # [B, C, N]
        rgb_flat = rgb.view(B, C, N)
        dem_flat = dem.view(B, C, N)

        # L2 归一化是计算协方差/相关性矩阵的标准操作，防止极端像素值主导
        rgb_norm = F.normalize(rgb_flat, dim=-1)
        dem_norm = F.normalize(dem_flat, dim=-1)

        # ---------------------------------------------------------------------
        # Step 2: 核心创新 - 构建二阶跨模态协方差矩阵 (Cross-Modal Bilinear Pooling)
        # ---------------------------------------------------------------------
        # [B, C, N] x[B, N, C] -> [B, C, C]
        # 矩阵的第 (i, j) 个元素代表：RGB的第i个通道与DEM的第j个通道在全局空间的共现强度！
        cross_covariance = torch.bmm(rgb_norm, dem_norm.transpose(1, 2))

        # ---------------------------------------------------------------------
        # Step 3: 提取梯度保向掩膜 (Gradient-Invariant Excitation Weights)
        # ---------------------------------------------------------------------
        # 对于 RGB：我们想知道它的每个通道与所有 DEM 通道的总体共振强度 (按行求和聚合)
        rgb_resonance = cross_covariance.sum(dim=2) # [B, C]

        # 对于 DEM：我们想知道它的每个通道与所有 RGB 通道的总体共振强度 (按列求和聚合)
        dem_resonance = cross_covariance.sum(dim=1) # [B, C]

        # 经过 MLP 获取最终的 0~1 标量权重
        weight_rgb = self.rgb_weight_extractor(rgb_resonance).view(B, C, 1, 1)
        weight_dem = self.dem_weight_extractor(dem_resonance).view(B, C, 1, 1)

        # ---------------------------------------------------------------------
        # Step 4: 梯度无损调制 (Non-destructive Depthwise Modulation)
        # ---------------------------------------------------------------------
        # 绝对不使用任何 1x1 或 3x3 卷积！
        # 直接按通道进行标量相乘。数学证明：w * grad(R) 的方向与 grad(R) 完全平行！
        # 残差相加进一步保证了原始特征底色不丢失。
        out_rgb = rgb + rgb * weight_rgb
        out_dem = dem + dem * weight_dem

        return out_rgb, out_dem

class VectorSobel(nn.Module):
    """
    深度可分离的矢量 Sobel 算子
    提取特征图在 X 和 Y 方向的空间梯度向量
    """
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
    """
    Cross-Gradient Directional Fusion (跨梯度方向融合模块)Cross-Gradient Directional Fusion
    TGRS 级别核心创新点：利用矢量场余弦相似度进行特征边界的动态调制。
    """
    def __init__(self, channels):
        super(CGDFusion, self).__init__()

        # 为了计算高效且提取核心语义边界，我们将特征压缩到统一的边界空间 (例如 16 通道)
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

        # 接收[RGB梯度幅值, DEM梯度幅值, 梯度余弦相似度] 共 3*bottleneck_dim 个通道
        # 生成通道数与原始输入一致的软性对齐门控 (Soft Alignment Gate)
        self.alignment_gate = nn.Sequential(
            nn.Conv2d(self.bottleneck_dim * 3, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

        # DEM 特征重塑层 (用于平滑注入)
        self.dem_injector = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

        # 最终聚合
        self.fusion_out = nn.Sequential(
            nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, rgb, dem):
        # --------------------------------------------------------------------
        # Step 1: 降维到统一语义空间，准备提取梯度
        # --------------------------------------------------------------------
        rgb_sub = self.rgb_proj(rgb)
        dem_sub = self.dem_proj(dem)

        # --------------------------------------------------------------------
        # Step 2: 提取 RGB 和 DEM 的空间梯度向量 (g_x, g_y)
        # --------------------------------------------------------------------
        r_gx, r_gy = self.vector_extractor(rgb_sub)
        d_gx, d_gy = self.vector_extractor(dem_sub)

        # --------------------------------------------------------------------
        # Step 3: 物理计算 - 梯度幅值 (Magnitude) 与 矢量点积 (Dot Product)
        # --------------------------------------------------------------------
        eps = 1e-6
        # 1. 计算幅值 (表示边界的强度)
        mag_rgb = torch.sqrt(r_gx**2 + r_gy**2 + eps)
        mag_dem = torch.sqrt(d_gx**2 + d_gy**2 + eps)

        # 2. 计算点积与绝对余弦相似度 (Absolute Cosine Similarity)
        # 绝对值是因为：不论 RGB/DEM 是由暗到亮还是由亮到暗，只要梯度平行，就是同一个物理边界
        dot_product = r_gx * d_gx + r_gy * d_gy
        cos_sim = torch.abs(dot_product / (mag_rgb * mag_dem + eps))

        # --------------------------------------------------------------------
        # Step 4: 跨梯度对齐掩膜生成 (Cross-Gradient Alignment Mask)
        # --------------------------------------------------------------------
        # 让网络同时看到：RGB边界强弱、DEM边界强弱、以及它们的对齐程度
        alignment_features = torch.cat([mag_rgb, mag_dem, cos_sim], dim=1)
        align_gate = self.alignment_gate(alignment_features) #[B, Channels, H, W]

        # --------------------------------------------------------------------
        # Step 5: 完美无损调制与融合 (Perfect Non-destructive Injection)
        # ---------------------------------------------------------
        # 核心机理：
        # 1. 大滑坡内部 (平坦区)：mag_rgb和mag_dem极小，align_gate自动趋近于0。
        #    注入项为 0，完美保留你深层 InteractionModule 提取的连通特征，彻底杜绝空洞！
        # 2. 真实边界区 (高度对齐区)：cos_sim趋近于1，align_gate激增。
        #    DEM 的精细高程特征被精确且平滑地注入到 RGB 中，边界锐度大幅提升！

        dem_injected = self.dem_injector(dem) * align_gate
        rgb_refined = rgb + dem_injected

        # 拼接后进行平滑降维，送入解码器
        out = self.fusion_out(torch.cat([rgb_refined, dem], dim=1))
        return out

class DualResNet34Unet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(DualResNet34Unet, self).__init__()

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

    model = DualResNet34Unet(num_classes=2,pretrained=False).to(device)

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
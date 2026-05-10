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

    def forward(self, x, skip):

        return out

class MCSF(nn.Module):

    def __init__(self, channels, reduction=4):
        super(MCSF, self).__init__()

    def forward(self, rgb, dem):


        return out

class BMIM(nn.Module):
    def __init__(self, channels, reduction=4):
        super(BMIM, self).__init__()

      
    def forward(self, rgb, dem):
       

        return out_rgb, out_dem

class VectorSobel(nn.Module)
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

    def forward(self, rgb, dem):
   
        return out

class GAMnet(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(GAMnet, self).__init__()
        rgb_backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
        self.rgb_initial = nn.Sequential(rgb_backbone.conv1, rgb_backbone.bn1, rgb_backbone.relu)
        self.rgb_maxpool = rgb_backbone.maxpool
        self.rgb_layer1 = rgb_backbone.layer1
        self.rgb_layer2 = rgb_backbone.layer2
        self.rgb_layer3 = rgb_backbone.layer3
        self.rgb_layer4 = rgb_backbone.layer4

        dem_backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)


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


import torch
import time
from thop import profile
from thop import clever_format

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_tensor = torch.rand(1, 4, 256, 256).to(device)

    model = GAMnet(num_classes=2,pretrained=False).to(device)

    model.eval()

    print("================ Model Info ================")

    with torch.no_grad():
        output = model(input_tensor)
    print(f"Input shape: {input_tensor.size()}")
    #print(f"Output shape: {output.size()}")

    print("\n================ Complexity ================")
    flops, params = profile(model, inputs=(input_tensor, ), verbose=False)
    flops_str, params_str = clever_format([flops, params], "%.3f")
    print(f"FLOPs:  {flops_str}")
    print(f"Params: {params_str}")

    print("\n================ Speed Test ================")

    print("Warming up GPU...")
    warmup_iters = 50
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(input_tensor)


    measure_iters = 200
    print(f"Measuring over {measure_iters} iterations...")

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start_time = time.time()

    with torch.no_grad():
        for _ in range(measure_iters):
            _ = model(input_tensor)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    end_time = time.time()

    total_time = end_time - start_time
    avg_inference_time_ms = (total_time / measure_iters) * 1000 
    fps = measure_iters / total_time                            

    print(f"Avg Inference Time: {avg_inference_time_ms:.2f} ms/image")
    print(f"FPS (Batch Size 1): {fps:.2f} frames/s")
    print("============================================")

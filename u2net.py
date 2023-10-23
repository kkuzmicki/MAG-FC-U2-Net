import torch
import torch.nn as nn
import torch.nn.functional as F

# despite of seed usage, program isn't deterministic:
# proof: https://discuss.pytorch.org/t/random-seed-with-external-gpu/102260
# ---------------
# RuntimeError: Deterministic behavior was enabled with either 
# `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, 
# but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. 
# To enable deterministic behavior in this case, 
# you must set an environment variable before running your PyTorch application: 
# CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, 
# go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility

class REBNCONV(nn.Module):
    def __init__(self,in_ch=3,out_ch=3,dirate=1):
        super(REBNCONV,self).__init__()

        # conv animations: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        self.conv = nn.Conv2d(in_ch,out_ch,3,padding=1*dirate,dilation=1*dirate) # https://www.youtube.com/watch?v=yb2tPt0QVPY
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x): # train.py (RSU7): torch.Size([12, 2, 256, 513])
        # test = self.conv(x) # train.py (RSU7): torch.Size([12, 64, 256, 513]) - probably works on 3 last dimensions
        x = self.relu(self.bn(self.conv(x)))

        return x # train.py: torch.Size([12, 64, 256, 513])

def _upsample_like(src,tar):

    src = F.interpolate(src,size=tar.shape[2:],mode='bilinear')

    return src

class FC(nn.Module):
    def __init__(self, channel, bins):
        super(FC,self).__init__()

        bn_unis = max(bins // 4, 16)

        self.fc = nn.Sequential(
            nn.Linear(bins, bn_unis, True),
            nn.BatchNorm2d(channel), # Batch normalization makes sure 
            # that the values of hidden units have standardized mean and variance
            # https://androidkt.com/use-the-batchnorm-layer-in-pytorch/
            nn.ReLU(inplace=True),
            nn.Linear(bn_unis, bins, True),
            nn.BatchNorm2d(channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fc(x)

### RSU-7 ### # de_2 de_4 attention
class RSU7(nn.Module):#UNet07DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,bins=64):
        super(RSU7,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool5 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv7 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv6d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)

        self.fc = FC(out_ch,bins)

    def forward(self,x):

        hx = x
        hxin = self.rebnconvin(hx) # REBNCONV IN=2 OUT=64

        hx1 = self.rebnconv1(hxin) # REBNCONV IN=64 OUT=16
        hx = self.pool1(hx1) # nn.AvgPool2d(2,stride=2,ceil_mode=True) # every avgpool divide last 2 dimension by 2 # train.py: torch.Size([12, 16, 128, 257])

        hx2 = self.rebnconv2(hx) # REBNCONV IN=16 OUT=16
        hx = self.pool2(hx2) # nn.AvgPool2d(2,stride=2,ceil_mode=True) # train.py: torch.Size([12, 16, 64, 129])

        hx3 = self.rebnconv3(hx) # REBNCONV IN=16 OUT=16
        hx = self.pool3(hx3) # nn.AvgPool2d(2,stride=2,ceil_mode=True) # train.py: torch.Size([12, 16, 32, 65])

        hx4 = self.rebnconv4(hx) # REBNCONV IN=16 OUT=16
        hx = self.pool4(hx4) # nn.AvgPool2d(2,stride=2,ceil_mode=True) # train.py: torch.Size([12, 16, 16, 33])

        hx5 = self.rebnconv5(hx) # REBNCONV IN=16 OUT=16
        hx = self.pool5(hx5) # nn.AvgPool2d(2,stride=2,ceil_mode=True) # train.py: torch.Size([12, 16, 8, 17])

        hx6 = self.rebnconv6(hx) # REBNCONV IN=16 OUT=16 # hx6 and hx7 have the same size because of no pooling

        hx7 = self.rebnconv7(hx6) # REBNCONV IN=16 OUT=16

        # train.py: hx7: torch.Size([12, 16, 8, 17]) hx6: torch.Size([12, 16, 8, 17])
        #test = torch.cat((hx7,hx6),1) # train.py: torch.Size([12, 32, 8, 17])
        hx6d =  self.rebnconv6d(torch.cat((hx7,hx6),1)) # REBNCONV IN=32 OUT=16 # train.py: torch.Size([12, 16, 8, 17])
        hx6dup = _upsample_like(hx6d,hx5) # train.py: torch.Size([12, 16, 16, 33])

        hx5d =  self.rebnconv5d(torch.cat((hx6dup,hx5),1)) # REBNCONV IN=32 OUT=16
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1)) # REBNCONV IN=32 OUT=16
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1)) # REBNCONV IN=32 OUT=16
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1)) # REBNCONV IN=32 OUT=16
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1)) # REBNCONV IN=32 OUT=64

        hx1d = self.fc(hx1d) # FC CHANNELS=64 BINS=64 # Part III from article

        return hx1d + hxin # Part I from article

### RSU-6 ###
class RSU6(nn.Module):#UNet06DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,bins=64):
        super(RSU6,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool4 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv6 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv5d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
        self.fc = FC(out_ch,bins)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)
        hx = self.pool4(hx4)

        hx5 = self.rebnconv5(hx)

        hx6 = self.rebnconv6(hx5)


        hx5d =  self.rebnconv5d(torch.cat((hx6,hx5),1))
        hx5dup = _upsample_like(hx5d,hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5dup,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        hx1d = self.fc(hx1d)

        return hx1d + hxin

### RSU-5 ###
class RSU5(nn.Module):#UNet05DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,bins=64):
        super(RSU5,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool3 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv5 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv4d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
        self.fc = FC(out_ch,bins)
        
    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)
        hx = self.pool3(hx3)

        hx4 = self.rebnconv4(hx)

        hx5 = self.rebnconv5(hx4)

        hx4d = self.rebnconv4d(torch.cat((hx5,hx4),1))
        hx4dup = _upsample_like(hx4d,hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4dup,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))

        # hx1d = self.fc(hx1d)

        return hx1d + hxin

### RSU-4 ###
class RSU4(nn.Module):#UNet04DRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,bins=64):
        super(RSU4,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.pool1 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=1)
        self.pool2 = nn.AvgPool2d(2,stride=2,ceil_mode=True)

        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=1)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=2)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=1)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
        self.fc = FC(out_ch,bins)

    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx = self.pool1(hx1)

        hx2 = self.rebnconv2(hx)
        hx = self.pool2(hx2)

        hx3 = self.rebnconv3(hx)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx3dup = _upsample_like(hx3d,hx2)

        hx2d = self.rebnconv2d(torch.cat((hx3dup,hx2),1))
        hx2dup = _upsample_like(hx2d,hx1)

        hx1d = self.rebnconv1d(torch.cat((hx2dup,hx1),1))
        hx1d = self.fc(hx1d)
        return hx1d + hxin

### RSU-4F ###
class RSU4F(nn.Module):#UNet04FRES(nn.Module):

    def __init__(self, in_ch=3, mid_ch=12, out_ch=3,bins=64):
        super(RSU4F,self).__init__()

        self.rebnconvin = REBNCONV(in_ch,out_ch,dirate=1)

        self.rebnconv1 = REBNCONV(out_ch,mid_ch,dirate=1)
        self.rebnconv2 = REBNCONV(mid_ch,mid_ch,dirate=2)
        self.rebnconv3 = REBNCONV(mid_ch,mid_ch,dirate=4)

        self.rebnconv4 = REBNCONV(mid_ch,mid_ch,dirate=8)

        self.rebnconv3d = REBNCONV(mid_ch*2,mid_ch,dirate=4)
        self.rebnconv2d = REBNCONV(mid_ch*2,mid_ch,dirate=2)
        self.rebnconv1d = REBNCONV(mid_ch*2,out_ch,dirate=1)
        self.fc = FC(out_ch,bins)
        
    def forward(self,x):

        hx = x

        hxin = self.rebnconvin(hx)

        hx1 = self.rebnconv1(hxin)
        hx2 = self.rebnconv2(hx1)
        hx3 = self.rebnconv3(hx2)

        hx4 = self.rebnconv4(hx3)

        hx3d = self.rebnconv3d(torch.cat((hx4,hx3),1))
        hx2d = self.rebnconv2d(torch.cat((hx3d,hx2),1))
        hx1d = self.rebnconv1d(torch.cat((hx2d,hx1),1))

        hx1d = self.fc(hx1d)

        return hx1d + hxin

### U^2-Net small ###
class u2net(nn.Module):

    def __init__(self,in_ch=2,out_ch=2,bins=64): # added '=64'
        super(u2net,self).__init__()

        self.stage1 = RSU7(in_ch,16,64,bins)
        self.pool12 = nn.AvgPool2d(2,stride=2)

        self.stage2 = RSU6(64,16,64,bins//2)
        self.pool23 = nn.AvgPool2d(2,stride=2)

        self.stage3 = RSU5(64,16,64,bins//4) # originally RSU5(64,16,64,bins//4)
        self.pool34 = nn.AvgPool2d(2,stride=2)

        #self.stage4 = RSU4(64,16,64,bins//8) # CHANGED originally RSU4(64,16,64,bins//8)
        #self.pool45 = nn.AvgPool2d(2,stride=2)

        #self.stage5 = RSU4F(64,16,64,bins//16) # CHANGED originally RSU4F(64,16,64,bins//16)
        #self.pool56 = nn.AvgPool2d(2,stride=2)

        self.stage6 = RSU4F(64,16,64,bins//8) # CHANGED originally RSU4F(64,16,64,bins//32)

        # decoder
        #self.stage5d = RSU4F(128,16,64,bins//16)
        #self.stage4d = RSU4(128,16,64,bins//8)
        self.stage3d = RSU5(128,16,64,bins//4)
        self.stage2d = RSU6(128,16,64,bins//2)
        self.stage1d = RSU7(128,16,64,bins)

        self.side1 = nn.Conv2d(64,out_ch,3,padding=1,bias = False) # 64 originally
        self.side2 = nn.Conv2d(64,out_ch,3,padding=1,bias = False) # 64 originally

    # hx - not used directly in output, after assignment to hx6 it is not used anymore
    # hx<1-...> - outputs from encoders
    # hx<1-...>d - 
    def forward(self,x):

        mix = x # train.py: torch.Size([12, 2, 513, 256]) | test.py: orch.Size([1, 2, 513, 256])

        x = x.permute(0, 1, 3, 2) # changes order # train.py: torch.Size([12, 2, 256, 513]) | test.py: torch.Size([1, 2, 256, 513])

        hx = x # train.py: torch.Size([12, 2, 256, 513])

        #stage 1
        hx1 = self.stage1(hx) # RSU7(in_ch=2,16,64,bins) # train.py: torch.Size([12, 64, 256, 513])
        hx = self.pool12(hx1) # nn.AvgPool2d(2,stride=2) # train.py: torch.Size([12, 64, 128, 256])

        #stage 2
        hx2 = self.stage2(hx) #   RSU6(64,16,64,bins//2) # train.py: torch.Size([12, 64, 128, 256])
        hx = self.pool23(hx2) # nn.AvgPool2d(2,stride=2) # train.py: torch.Size([12, 64, 64, 128])

        #stage 3
        hx3 = self.stage3(hx) #   RSU5(64,16,64,bins//4) # train.py: torch.Size([12, 64, 64, 128])
        hx = self.pool34(hx3) # nn.AvgPool2d(2,stride=2) # train.py: torch.Size([12, 64, 32, 64])

        # #stage 4
        # hx4 = self.stage4(hx) #   RSU4(64,16,64,bins//8) # train.py: torch.Size([12, 64, 32, 64])
        # hx = self.pool45(hx4) # nn.AvgPool2d(2,stride=2) # train.py: torch.Size([12, 64, 16, 32])

        # #stage 5
        # hx5 = self.stage5(hx) # RSU4F(64,16,64,bins//16) # train.py: torch.Size([12, 64, 16, 32])
        # hx = self.pool56(hx5) # nn.AvgPool2d(2,stride=2) # train.py: torch.Size([12, 64, 8, 16])

        #stage 6
        hx6 = self.stage6(hx) # train.py: torch.Size([12, 64, 8, 16])
        hx6up = _upsample_like(hx6,hx3) # train.py: torch.Size([12, 64, 16, 32])

        #decoder
        #test = torch.cat((hx6up, hx5), 1) # torch.Size([12, 128, 16, 32])
        # hx5d = self.stage5d(torch.cat((hx6up,hx5),1)) # train.py: torch.Size([12, 64, 16, 32])
        # hx5dup = _upsample_like(hx5d,hx4) # train.py: torch.Size([12, 64, 32, 64])

        # hx4d = self.stage4d(torch.cat((hx5dup,hx4),1)) # train.py: torch.Size([12, 64, 32, 64])
        # hx4dup = _upsample_like(hx4d,hx3) # train.py: torch.Size([12, 64, 64, 128])

        hx3d = self.stage3d(torch.cat((hx6up,hx3),1)) # train.py: torch.Size([12, 64, 64, 128])
        hx3dup = _upsample_like(hx3d,hx2) # train.py: torch.Size([12, 64, 128, 256])

        hx2d = self.stage2d(torch.cat((hx3dup,hx2),1)) # train.py: torch.Size([12, 64, 128, 256])
        hx2dup = _upsample_like(hx2d,hx1) # train.py: torch.Size([12, 64, 256, 513])

        hx1d = self.stage1d(torch.cat((hx2dup,hx1),1)) # train.py: torch.Size([12, 64, 256, 513])

        #side output
        d1 = self.side1(hx1d) # train.py: torch.Size([12, 2, 256, 513])
        d1 = d1.permute(0, 1, 3, 2) # train.py: torch.Size([12, 2, 513, 256])
        d2 = self.side2(hx1d) # train.py: torch.Size([12, 2, 256, 513])
        d2 = d2.permute(0, 1, 3, 2) # train.py: torch.Size([12, 2, 513, 256])
        return mix*F.relu(d1), d2

if __name__ == "__main__":

    net1 = u2net().cuda()

    x = torch.rand(10,2,513,256).cuda()
    y = net1(x)

    total_num = sum(p.numel() for p in net1.parameters())
    print(total_num)

    while True:
        pass    

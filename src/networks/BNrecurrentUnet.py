
import torch
import torch.nn as nn
import torch.nn.functional as F

# why can't import????
# give everything better names at some point
class BNUNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, dropout_rate):
        super().__init__()

        base = hidden_channels # ends up as hiddden channels
        self.base = base # to extract later
        
        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(input_channels + hidden_channels, base, 3, padding=1, bias = False) # but then with hidden_c you go from 65 to 64...
        self.bn_enc_conv0 = nn.BatchNorm2d(base)
        self.pool0 = nn.MaxPool2d(2, 2, padding=0) # 16 -> 8

        self.enc_conv1 = nn.Conv2d(base, base*2, 3, padding=1, bias = False)
        self.bn_enc_conv1 = nn.BatchNorm2d(base*2) 
        self.pool1 = nn.MaxPool2d(2, 2, padding=0) # 8 -> 4

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(base*2, base*4, 3, padding=1, bias = False)
        self.bn_bottleneck_conv = nn.BatchNorm2d(base*4) 
        
        # decoder_reg (upsampling)
        self.upsample0_reg = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_reg = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_reg = nn.BatchNorm2d(base*2) 

        self.upsample1_reg = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_reg = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_reg = nn.BatchNorm2d(base) 

        self.dec_conv2_reg = nn.Conv2d(base, output_channels, 3, padding=1) 
        #self.bn_dec_conv2_reg = nn.BatchNorm2d(output_channels)  # you sire you want one here?


        # decoder_class (upsampling)
        self.upsample0_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_class = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_class = nn.BatchNorm2d(base*2) 

        self.upsample1_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_class = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_class = nn.BatchNorm2d(base) 

        self.dec_conv2_class = nn.Conv2d(base, output_channels, 3, padding=1)
        #self.bn_dec_conv2_class = nn.BatchNorm2d(output_channels) 

        # misc
        self.dropout = nn.Dropout(p = dropout_rate)


    def forward(self, x, h):
        
        x = torch.cat([x, h], 1)

        # encoder
        e0s_ = F.relu(self.bn_enc_conv0(self.enc_conv0(x)))
        e0s = self.dropout(e0s_)
        e0 = self.pool0(e0s)
        
        e1s = self.dropout(F.relu(self.bn_enc_conv1(self.enc_conv1(e0))))
        e1 = self.pool1(e1s)
        

        # bottleneck
        b = F.relu(self.bn_bottleneck_conv(self.bottleneck_conv(e1)))
        b = self.dropout(b)

        # decoder_reg
        d0_reg = F.relu(self.bn_dec_conv0_reg(self.dec_conv0_reg(torch.cat([self.upsample0_reg(b),e1s],1))))
        d0_reg = self.dropout(d0_reg)
        
        d1_reg = F.relu(self.bn_dec_conv1_reg(self.dec_conv1_reg(torch.cat([self.upsample1_reg(d0_reg), e0s],1)))) # You did not have any activations before - why not?
        d1_reg = self.dropout(d1_reg)

        d2_reg = self.dec_conv2_reg(d1_reg) # test bc I don't want negative values...

        d2_reg = F.relu(d2_reg)

        # decoder_class
        d0_class = F.relu(self.bn_dec_conv0_class(self.dec_conv0_class(torch.cat([self.upsample0_class(b),e1s],1))))
        d0_class = self.dropout(d0_class)
        
        d1_class = F.relu(self.bn_dec_conv1_class(self.dec_conv1_class(torch.cat([self.upsample1_class(d0_class), e0s],1))))  # You did not have any activations before - why not?
        d1_class = self.dropout(d1_class)

        d2_class = self.dec_conv2_class(d1_class) # test bc I don't want negative values...

        d2_class = torch.sigmoid(d2_class) # could move sigmoid outta here...
        
        # Tyr without - you just had a Relu!
        #h = torch.tanh(e0s_) # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # return d2, e0s # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....
        #return d2_reg, d2_class, e0s # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....
        return d2_reg, d2_class, e0s_ # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....


    # YOU CAN MAKE HIDDEN STATE TAKE THE FIRST INPUT AND JUST BROADCAT IT OUT..

    def init_h(self, hidden_channels, dim, train_tensor): # could have x as input and then take x.shape

        # NEW -----------------------------------------------------------
        #hs = torch.zeros((1,hidden_channels,dim,dim), dtype= torch.float64) # + torch.exp(torch.tensor(-100)
        hs = torch.abs(torch.randn((1,hidden_channels, dim, dim), dtype= torch.float64) * torch.exp(torch.tensor(-100))) 
        #hs = torch.randn((1,hidden_channels,dim,dim), dtype= torch.float64)

        #torch.randn(2, 3, 20)
        #print(hs.shape)
        #print(train_tensor.shape)

        #hs_p = hs + train_tensor.detach().cpu()
        return hs  # the dims could just be infered... sp you donøt needd to funct or change if w siae changes.
        # NEW -----------------------------------------------------------

        #eturn torch.zeros((1,hidden_channels,dim,dim), dtype= torch.float64) # the dims could just be infered... sp you donøt needd to funct or change if w siae changes.

    def init_hTtime(self, hidden_channels, H, W, test_tensor):
        
        # NEW -----------------------------------------------------------
        #hs = torch.zeros((1,hidden_channels, H, W), dtype= torch.float64) # + torch.exp(torch.tensor(-100)
        hs = torch.abs(torch.randn((1,hidden_channels, H, W), dtype= torch.float64) * torch.exp(torch.tensor(-100))) 
        #hs = torch.randn((1,hidden_channels, H, W), dtype= torch.float64)   

        #hs_p = hs + test_tensor.detach().cpu() 
        return hs
        # NEW -----------------------------------------------------------
        
        #return torch.zeros((1,hidden_channels, H, W), dtype= torch.float64)
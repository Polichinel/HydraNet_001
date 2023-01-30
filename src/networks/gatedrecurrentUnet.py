
import torch
import torch.nn as nn
import torch.nn.functional as F

# give everything better names at some point
class UNet(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, dropout_rate):
        super().__init__()

        base = hidden_channels # ends up as hiddden channels
        self.base = base # to extract later
        
        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(input_channels + hidden_channels, base, 3, padding=1) # but then with hidden_c you go from 65 to 64...
        self.pool0 = nn.MaxPool2d(2, 2, padding=0) # 16 -> 8

        self.enc_conv1 = nn.Conv2d(base, base*2, 3, padding=1) 
        self.pool1 = nn.MaxPool2d(2, 2, padding=0) # 8 -> 4

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(base*2, base*4, 3, padding=1) 
        
        # decoder_reg (upsampling)
        self.upsample0_reg = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_reg = nn.Conv2d(base*4, base*2, 3, padding=1) # base+base=base*2 because of skip conneciton
        
        self.upsample1_reg = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_reg = nn.Conv2d(base*2, base, 3, padding=1) # base+base=base*2 because of skip connection

        self.dec_conv2_reg = nn.Conv2d(base, output_channels, 3, padding=1) 

        # decoder_class (upsampling)
        self.upsample0_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_class = nn.Conv2d(base*4, base*2, 3, padding=1) # base+base=base*2 because of skip conneciton
        
        self.upsample1_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_class = nn.Conv2d(base*2, base, 3, padding=1) # base+base=base*2 because of skip connection

        self.dec_conv2_class = nn.Conv2d(base, output_channels, 3, padding=1)

        # misc
        self.dropout = nn.Dropout(p = dropout_rate)

        #gated
        self.xz = nn.Conv2d( hidden_channels, base, 3, padding= 'same') # right now hidden_channels and base is the same so...
        self.hz = nn.Conv2d( hidden_channels, base, 3, padding= 'same')

        self.xr = nn.Conv2d( hidden_channels, base, 3, padding= 'same')
        self.hr = nn.Conv2d( hidden_channels, base, 3, padding= 'same')

        self.xh = nn.Conv2d( hidden_channels, base, 3, padding= 'same')
        self.hh = nn.Conv2d( hidden_channels, base, 3, padding= 'same')


    def forward(self, x, h):
        
        x = torch.cat([x, h], 1)

        # encoder
        e0s = self.dropout(F.relu(self.enc_conv0(x)))
        e0 = self.pool0(e0s)
        
        e1s = self.dropout(F.relu(self.enc_conv1(e0)))
        e1 = self.pool1(e1s)
        

        # bottleneck
        b = F.relu(self.bottleneck_conv(e1))
        b = self.dropout(b)

        # decoder_reg
        d0_reg = F.relu(self.dec_conv0_reg(torch.cat([self.upsample0_reg(b),e1s],1)))
        d0_reg = self.dropout(d0_reg)
        
        d1_reg = F.relu(self.dec_conv1_reg(torch.cat([self.upsample1_reg(d0_reg), e0s],1)))  # You did not have any activations before - why not?
        d1_reg = self.dropout(d1_reg)

        d2_reg = self.dec_conv2_reg(d1_reg) # test bc I don't want negative values...

        d2_reg = F.relu(d2_reg)

        # decoder_class
        d0_class = F.relu(self.dec_conv0_class(torch.cat([self.upsample0_class(b),e1s],1)))
        d0_class = self.dropout(d0_class)
        
        d1_class = F.relu(self.dec_conv1_class(torch.cat([self.upsample1_class(d0_class), e0s],1)))  # You did not have any activations before - why not?
        d1_class = self.dropout(d1_class)

        d2_class = self.dec_conv2_class(d1_class) # test bc I don't want negative values...

        d2_class = torch.sigmoid(d2_class)
        
        # return d2, e0s # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....
        # h_out = F.tanh(e0s)
        # return d2_reg, d2_class, e0s

        # GRU ----------------------------------------------------------------------------------------------------------------------
        Z = torch.sigmoid(self.xz(e0s) + self.hz(H))
        R = torch.sigmoid(self.xr(e0s) + self.hr(H))
        H_tilde = torch.tanh(self.xh(e0s) + self.hh(torch.mul(R,H)))
        H = torch.mul(torch.mul(Z,H) + (1 - Z), H_tilde)
        # --------------------------------------------------------------------------------------------------------------------------

        return d2_reg, d2_class, H # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....

        # return d2_reg, d2_class, e0s # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....


    # YOU CAN MAKE HIDDEN STATE TAKE THE FIRST INPUT AND JUST BROADCAT IT OUT..

    def init_h(self, hidden_channels, dim, train_tensor): # could have x as input and then take x.shape

        # NEW -----------------------------------------------------------
        #hs = torch.zeros((1,hidden_channels,dim,dim), dtype= torch.float64)
        hs = torch.randn((1,hidden_channels,dim,dim), dtype= torch.float64) # why you not using float32? Is that from the numpy imports? Then it should change there...

        #torch.randn(2, 3, 20)
        #print(hs.shape)
        #print(train_tensor.shape)

        #hs_p = hs + train_tensor.detach().cpu()
        return hs  # the dims could just be infered... sp you donøt needd to funct or change if w siae changes.
        # NEW -----------------------------------------------------------

        #eturn torch.zeros((1,hidden_channels,dim,dim), dtype= torch.float64) # the dims could just be infered... sp you donøt needd to funct or change if w siae changes.

    def init_hTtime(self, hidden_channels, H, W, test_tensor):
        
        # NEW -----------------------------------------------------------
#       hs = torch.zeros((1,hidden_channels, H, W), dtype= torch.float64)
        hs = torch.randn((1,hidden_channels, H, W), dtype= torch.float64)   
        #hs= torch.nn.init.kaiming_normal_(hs, mode='fan_out')

        #hs_p = hs + test_tensor.detach().cpu() 
        return hs
        # NEW -----------------------------------------------------------
        
        #return torch.zeros((1,hidden_channels, H, W), dtype= torch.float64)
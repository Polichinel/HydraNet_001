
import torch
import torch.nn as nn
import torch.nn.functional as F

# why can't import????
# give everything better names at some point
class HydraBNUNet05(nn.Module):
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
        

        # HEAD1 reg
        self.upsample0_head1_reg = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head1_reg = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head1_reg = nn.BatchNorm2d(base*2) 

        self.upsample1_head1_reg = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head1_reg = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head1_reg = nn.BatchNorm2d(base) 

        self.dec_conv2_head1_reg = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv2_head1_reg = nn.BatchNorm2d(base)

        self.dec_conv3_head1_reg = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv3_head1_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head1_reg = nn.Conv2d(base, 1, 3, padding=1) # 2 because reg and class
        #self.bn_dec_conv4_head1_reg = nn.BatchNorm2d(base)


        # HEAD1 class
        self.upsample0_head1_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head1_class = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head1_class = nn.BatchNorm2d(base*2) 

        self.upsample1_head1_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head1_class = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head1_class = nn.BatchNorm2d(base) 

        self.dec_conv2_head1_class = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv2_head1_class = nn.BatchNorm2d(base)

        self.dec_conv3_head1_class = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv3_head1_class = nn.BatchNorm2d(base)

        self.dec_conv4_head1_class = nn.Conv2d(base, 1, 3, padding=1)
        #self.bn_dec_conv4_head1_class = nn.BatchNorm2d(base)
        

        # HEAD2 reg
        self.upsample0_head2_reg = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head2_reg = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head2_reg = nn.BatchNorm2d(base*2) 

        self.upsample1_head2_reg = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head2_reg = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head2_reg = nn.BatchNorm2d(base) 

        self.dec_conv2_head2_reg = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv2_head2_reg = nn.BatchNorm2d(base)

        self.dec_conv3_head2_reg = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv3_head2_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head2_reg = nn.Conv2d(base, 1, 3, padding=1) # 2 because reg and class
        #self.bn_dec_conv4_head2_reg = nn.BatchNorm2d(base)


        # HEAD2 class
        self.upsample0_head2_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head2_class = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head2_class = nn.BatchNorm2d(base*2) 

        self.upsample1_head2_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head2_class = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head2_class = nn.BatchNorm2d(base) 

        self.dec_conv2_head2_class = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv2_head2_class = nn.BatchNorm2d(base)

        self.dec_conv3_head2_class = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv3_head2_class = nn.BatchNorm2d(base)

        self.dec_conv4_head2_class = nn.Conv2d(base, 1, 3, padding=1)
        #self.bn_dec_conv4_head2_class = nn.BatchNorm2d(base)


        # HEAD3 reg
        self.upsample0_head3_reg = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head3_reg = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head3_reg = nn.BatchNorm2d(base*2) 

        self.upsample1_head3_reg = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head3_reg = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head3_reg = nn.BatchNorm2d(base) 

        self.dec_conv2_head3_reg = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv2_head3_reg = nn.BatchNorm2d(base)

        self.dec_conv3_head3_reg = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv3_head3_reg = nn.BatchNorm2d(base)

        self.dec_conv4_head3_reg = nn.Conv2d(base, 1, 3, padding=1) # 2 because reg and class
        #self.bn_dec_conv4_head3_reg = nn.BatchNorm2d(base)


        # HEAD3 class
        self.upsample0_head3_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head3_class = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head3_class = nn.BatchNorm2d(base*2) 

        self.upsample1_head3_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head3_class = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head3_class = nn.BatchNorm2d(base) 

        self.dec_conv2_head3_class = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv2_head3_class = nn.BatchNorm2d(base)

        self.dec_conv3_head3_class = nn.Conv2d(base, base, 3, padding=1)
        self.bn_dec_conv3_head3_class = nn.BatchNorm2d(base)

        self.dec_conv4_head3_class = nn.Conv2d(base, 1, 3, padding=1)
        #self.bn_dec_conv4_head3_class = nn.BatchNorm2d(base)

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

        # decoders

        #H1 reg
        H1_d0 = F.relu(self.bn_dec_conv0_head1_reg(self.dec_conv0_head1_reg(torch.cat([self.upsample0_head1_reg(b),e1s],1))))
        H1_d0 = self.dropout(H1_d0)
        
        H1_d1 = F.relu(self.bn_dec_conv1_head1_reg(self.dec_conv1_head1_reg(torch.cat([self.upsample1_head1_reg(H1_d0), e0s],1)))) # You did not have any activations before - why not?
        H1_d1 = self.dropout(H1_d1)

        H1_d2 = F.relu(self.bn_dec_conv2_head1_reg(self.dec_conv2_head1_reg(H1_d1)))
        H1_d2 = self.dropout(H1_d2) # is this good?

        H1_reg = F.relu(self.bn_dec_conv3_head1_reg(self.dec_conv3_head1_reg(H1_d2)))
        H1_reg = self.dropout(H1_reg) # is this good?
        H1_reg = self.dec_conv4_head1_reg(H1_reg)
        
        out_reg1 = F.relu(H1_reg)

        #H1 class
        H1_d0 = F.relu(self.bn_dec_conv0_head1_class(self.dec_conv0_head1_class(torch.cat([self.upsample0_head1_class(b),e1s],1))))
        H1_d0 = self.dropout(H1_d0)
        
        H1_d1 = F.relu(self.bn_dec_conv1_head1_class(self.dec_conv1_head1_class(torch.cat([self.upsample1_head1_class(H1_d0), e0s],1)))) # You did not have any activations before - why not?
        H1_d1 = self.dropout(H1_d1)

        H1_d2 = F.relu(self.bn_dec_conv2_head1_class(self.dec_conv2_head1_class(H1_d1)))
        H1_d2 = self.dropout(H1_d2) # is this good?
        
        H1_class = F.relu(self.bn_dec_conv3_head1_class(self.dec_conv3_head1_class(H1_d2)))
        H1_class = self.dropout(H1_class) # is this good?
        H1_class = self.dec_conv4_head1_class(H1_class)

        out_class1 = H1_class # torch.sigmoid(H1_class) # could move sigmoid outta here...


        #H2 reg
        H2_d0 = F.relu(self.bn_dec_conv0_head2_reg(self.dec_conv0_head2_reg(torch.cat([self.upsample0_head2_reg(b),e1s],1))))
        H2_d0 = self.dropout(H1_d0)
        
        H2_d1 = F.relu(self.bn_dec_conv1_head2_reg(self.dec_conv1_head2_reg(torch.cat([self.upsample1_head2_reg(H2_d0), e0s],1)))) # You did not have any activations before - why not?
        H2_d1 = self.dropout(H1_d1)

        H2_d2 = F.relu(self.bn_dec_conv2_head2_reg(self.dec_conv2_head2_reg(H2_d1)))
        H2_d2 = self.dropout(H1_d2) # is this good?

        H2_reg = F.relu(self.bn_dec_conv3_head2_reg(self.dec_conv3_head2_reg(H2_d2)))
        H2_reg = self.dropout(H2_reg) # is this good?
        H2_reg = self.dec_conv4_head2_reg(H2_reg)
        
        out_reg2 = F.relu(H2_reg)

        #H2 class
        H2_d0 = F.relu(self.bn_dec_conv0_head2_class(self.dec_conv0_head2_class(torch.cat([self.upsample0_head2_class(b),e1s],1))))
        H2_d0 = self.dropout(H1_d0)
        
        H2_d1 = F.relu(self.bn_dec_conv1_head2_class(self.dec_conv1_head2_class(torch.cat([self.upsample1_head2_class(H2_d0), e0s],1)))) # You did not have any activations before - why not?
        H2_d1 = self.dropout(H2_d1)

        H2_d2 = F.relu(self.bn_dec_conv2_head2_class(self.dec_conv2_head2_class(H2_d1)))
        H2_d2 = self.dropout(H1_d2) # is this good?
        
        H2_class = F.relu(self.bn_dec_conv3_head2_class(self.dec_conv3_head2_class(H2_d2)))
        H2_class = self.dropout(H2_class) # is this good?
        H2_class = self.dec_conv4_head2_class(H2_class)

        out_class2 = H2_class # torch.sigmoid(H1_class) # could move sigmoid outta here...



        #H3 reg
        H3_d0 = F.relu(self.bn_dec_conv0_head3_reg(self.dec_conv0_head3_reg(torch.cat([self.upsample0_head3_reg(b),e1s],1))))
        H3_d0 = self.dropout(H3_d0)
        
        H3_d1 = F.relu(self.bn_dec_conv1_head3_reg(self.dec_conv1_head3_reg(torch.cat([self.upsample1_head3_reg(H3_d0), e0s],1)))) # You did not have any activations before - why not?
        H3_d1 = self.dropout(H3_d1)

        H3_d2 = F.relu(self.bn_dec_conv2_head3_reg(self.dec_conv2_head3_reg(H3_d1)))
        H3_d2 = self.dropout(H1_d2) # is this good?

        H3_reg = F.relu(self.bn_dec_conv3_head3_reg(self.dec_conv3_head3_reg(H3_d2)))
        H3_reg = self.dropout(H3_reg) # is this good?
        H3_reg = self.dec_conv4_head3_reg(H3_reg)
        
        out_reg3 = F.relu(H3_reg)

        #H3 class
        H3_d0 = F.relu(self.bn_dec_conv0_head3_class(self.dec_conv0_head3_class(torch.cat([self.upsample0_head3_class(b),e1s],1))))
        H3_d0 = self.dropout(H1_d0)
        
        H3_d1 = F.relu(self.bn_dec_conv1_head3_class(self.dec_conv1_head3_class(torch.cat([self.upsample1_head3_class(H3_d0), e0s],1)))) # You did not have any activations before - why not?
        H3_d1 = self.dropout(H3_d1)

        H3_d2 = F.relu(self.bn_dec_conv2_head3_class(self.dec_conv2_head3_class(H3_d1)))
        H3_d2 = self.dropout(H1_d2) # is this good?
        
        H3_class = F.relu(self.bn_dec_conv3_head3_class(self.dec_conv3_head3_class(H3_d2)))
        H3_class = self.dropout(H3_class) # is this good?
        H3_class = self.dec_conv4_head3_class(H3_class)

        out_class3 = H3_class # torch.sigmoid(H1_class) # could move sigmoid outta here...


        # RESTRUCTURE TO FIT "OLD" FORMAT. dim 1 should be depth
        out_reg = torch.concat([out_reg1, out_reg2, out_reg3], dim=1)        
        out_class = torch.concat([out_class1, out_class2, out_class3], dim=1)

        return out_reg, out_class, e0s_ # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....


    def init_h(self, hidden_channels, dim, train_tensor): # could have x as input and then take x.shape

        hs = torch.zeros((1,hidden_channels,dim,dim), dtype= torch.float64)
        
        return hs 

    def init_hTtime(self, hidden_channels, H, W, test_tensor):
        
        # works
        hs = torch.abs(torch.randn((1,hidden_channels, H, W), dtype= torch.float64) * torch.exp(torch.tensor(-100))) 
        hs = torch.zeros((1,hidden_channels, H, W), dtype= torch.float64)

        return hs

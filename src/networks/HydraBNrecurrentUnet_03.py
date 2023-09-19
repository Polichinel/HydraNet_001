
import torch
import torch.nn as nn
import torch.nn.functional as F

# why can't import????
# give everything better names at some point
class HydraBNUNet03(nn.Module):
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
        
        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0 = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0 = nn.BatchNorm2d(base*2) 

        self.upsample1 = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1 = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1 = nn.BatchNorm2d(base) 

        # self.dec_conv2_reg = nn.Conv2d(base, output_channels, 3, padding=1)  # THIS IS YOU OUT!!!!
        #self.bn_dec_conv2_reg = nn.BatchNorm2d(output_channels)  # you sire you want one here?

        # HEAD1
        self.dec_conv2_pre_head1 = nn.Conv2d(base, base, 3, padding=1)
        
        self.dec_conv2_reg1_head1 = nn.Conv2d(base, base, 3, padding=1)
        self.dec_conv2_reg2_head1 = nn.Conv2d(base, 1, 3, padding=1) # 2 because reg and class

        self.dec_conv2_class1_head1 = nn.Conv2d(base, base, 3, padding=1)
        self.dec_conv2_class2_head1 = nn.Conv2d(base, 1, 3, padding=1)

        self.bn1_head1 = nn.BatchNorm2d(base)
        self.bn2_head1 = nn.BatchNorm2d(base)
        self.bn3_head1 = nn.BatchNorm2d(base)



        # HEAD2
        self.dec_conv2_pre_head2 = nn.Conv2d(base, base, 3, padding=1)

        self.dec_conv2_reg1_head2 = nn.Conv2d(base, base, 3, padding=1)
        self.dec_conv2_reg2_head2 = nn.Conv2d(base, 1, 3, padding=1)

        self.dec_conv2_class1_head2 = nn.Conv2d(base, base, 3, padding=1)
        self.dec_conv2_class2_head2 = nn.Conv2d(base, 1, 3, padding=1)

        self.bn1_head2 = nn.BatchNorm2d(base)
        self.bn2_head2 = nn.BatchNorm2d(base)
        self.bn3_head2 = nn.BatchNorm2d(base)



        # HEAD3
        self.dec_conv2_pre_head3 = nn.Conv2d(base, base, 3, padding=1)

        self.dec_conv2_reg1_head3 = nn.Conv2d(base, base, 3, padding=1)
        self.dec_conv2_reg2_head3 = nn.Conv2d(base, 1, 3, padding=1)

        self.dec_conv2_class1_head3 = nn.Conv2d(base, base, 3, padding=1)
        self.dec_conv2_class2_head3 = nn.Conv2d(base, 1, 3, padding=1)
        
        self.bn1_head3 = nn.BatchNorm2d(base)
        self.bn2_head3 = nn.BatchNorm2d(base)
        self.bn3_head3 = nn.BatchNorm2d(base)


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

        # decoder
        d0 = F.relu(self.bn_dec_conv0(self.dec_conv0(torch.cat([self.upsample0(b),e1s],1))))
        d0 = self.dropout(d0)
        
        d1 = F.relu(self.bn_dec_conv1(self.dec_conv1(torch.cat([self.upsample1(d0), e0s],1)))) # You did not have any activations before - why not?
        d1 = self.dropout(d1)


        # HEADS 
        # Before the two two tasks did not learn enough. Now it a very (perhaps too) similar.
        # Maybe a smalle encoder-decoder between the prehead and the output layers. It is just one more layer really.. 

        #H1
        H1_pre = F.relu(self.bn1_head1(self.dec_conv2_pre_head1(d1)))
        H1_pre = self.dropout(H1_pre) # is this good?

        H1_reg = F.relu(self.bn2_head1(self.dec_conv2_reg1_head1(H1_pre)))
        H1_reg = self.dropout(H1_reg) # is this good?
        H1_reg = self.dec_conv2_reg2_head1(H1_reg)
        
        H1_class = F.relu(self.bn3_head1(self.dec_conv2_class1_head1(H1_pre)))
        H1_class = self.dropout(H1_class) # is this good?
        H1_class = self.dec_conv2_class2_head1(H1_class)

        out_reg1 = F.relu(H1_reg)
        out_class1 = torch.sigmoid(H1_class) # could move sigmoid outta here...

        #H2
        H2_pre = F.relu(self.bn1_head2(self.dec_conv2_pre_head2(d1)))
        H2_pre = self.dropout(H2_pre)

        H2_reg = F.relu(self.bn2_head2(self.dec_conv2_reg1_head2(H2_pre)))
        H2_reg = self.dropout(H2_reg)
        H2_reg = self.dec_conv2_reg2_head2(H2_reg)

        H2_class = F.relu(self.bn3_head2(self.dec_conv2_class1_head2(H2_pre)))
        H2_class = self.dropout(H2_class)
        H2_class = self.dec_conv2_class2_head2(H2_class)

        out_reg2 = F.relu(H2_reg)
        out_class2 = torch.sigmoid(H2_class)

        #H3
        H3_pre = F.relu(self.bn1_head3(self.dec_conv2_pre_head3(d1)))
        H3_pre = self.dropout(H3_pre)

        H3_reg = F.relu(self.bn2_head3(self.dec_conv2_reg1_head3(H3_pre)))
        H3_reg = self.dropout(H3_reg)
        H3_reg = self.dec_conv2_reg2_head3(H3_reg)

        H3_class = F.relu(self.bn3_head3(self.dec_conv2_class1_head3(H3_pre)))
        H3_class = self.dropout(H3_class)
        H3_class = self.dec_conv2_class2_head3(H3_class)

        out_reg3 = F.relu(H3_reg)
        out_class3 = torch.sigmoid(H3_class)

        
        # Hidden state
        # You could make a gate hare simply as a conv layer -> BN -> relu...
        #h = torch.tanh(e0s_) # --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

        # RESTRUCTURE TO FIT "OLD" FORMAT. dim 1 should be depth
        out_reg = torch.concat([out_reg1, out_reg2, out_reg3], dim=1)        
        out_class = torch.concat([out_class1, out_class2, out_class3], dim=1)
 
        # return d2, e0s # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....
        #return d2_reg, d2_class, e0s # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....

        return out_reg, out_class, e0s_ # e0s here also hidden state - should take tanh of self.enc_conv0(x) but it does not appear to make a big difference....


    # YOU CAN MAKE HIDDEN STATE TAKE THE FIRST INPUT AND JUST BROADCAT IT OUT..

    def init_h(self, hidden_channels, dim, train_tensor): # could have x as input and then take x.shape

        # NEW -----------------------------------------------------------
        #hs = torch.zeros((1,hidden_channels,dim,dim), dtype= torch.float64) # + torch.exp(torch.tensor(-100)
        
        # works
        #hs = torch.abs(torch.randn((1,hidden_channels, dim, dim), dtype= torch.float64) * torch.exp(torch.tensor(-100))) 
        hs = torch.zeros((1,hidden_channels,dim,dim), dtype= torch.float64)
        
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
        
        # works
        hs = torch.abs(torch.randn((1,hidden_channels, H, W), dtype= torch.float64) * torch.exp(torch.tensor(-100))) 
        
        #hs = torch.randn((1,hidden_channels, H, W), dtype= torch.float64)   

        #hs_p = hs + test_tensor.detach().cpu() 

        hs = torch.zeros((1,hidden_channels, H, W), dtype= torch.float64)

        return hs
        # NEW -----------------------------------------------------------
        
        #return torch.zeros((1,hidden_channels, H, W), dtype= torch.float64)
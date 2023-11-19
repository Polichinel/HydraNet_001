
import torch
import torch.nn as nn
import torch.nn.functional as F

# why can't import????
# give everything better names at some point
class HydraBNUNet06_LSTM4(nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, dropout_rate):
        super().__init__()

        base = hidden_channels # ends up as hiddden channels
        kernel_size = 3 # only use in the LSTM part but could be used through out.. 
        padding = kernel_size // 2 # only use in the LSTM part but could be used through out.. 
        hidden_channels_split = int(hidden_channels/4) # 4 for 2 LSTM layers /2) # For the LSTM part because we are splitting h into two tensors hs (short-term) and hl (long-term)


        self.base = base # to extract later
        
        # encoder (downsampling)
        #self.enc_conv0 = nn.Conv2d(input_channels + hidden_channels, base, 3, padding=1, bias = False) # but then with hidden_c you go from 65 to 64...
        self.enc_conv0 = nn.Conv2d(input_channels + hidden_channels_split*2, base, 3, padding=1, bias = False) # NOW (hidden_channels_split*2)+ input channels because of the LSTM2 - you only concat x with hs and not h

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

        self.dec_conv4_head1_reg = nn.Conv2d(base, 1, 3, padding=1) # 2 because reg and class


        # HEAD1 class
        self.upsample0_head1_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head1_class = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head1_class = nn.BatchNorm2d(base*2) 

        self.upsample1_head1_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head1_class = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head1_class = nn.BatchNorm2d(base) 

        self.dec_conv4_head1_class = nn.Conv2d(base, 1, 3, padding=1)
        

        # HEAD2 reg
        self.upsample0_head2_reg = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head2_reg = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head2_reg = nn.BatchNorm2d(base*2) 

        self.upsample1_head2_reg = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head2_reg = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head2_reg = nn.BatchNorm2d(base) 

        self.dec_conv4_head2_reg = nn.Conv2d(base, 1, 3, padding=1) # 2 because reg and class


        # HEAD2 class
        self.upsample0_head2_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head2_class = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head2_class = nn.BatchNorm2d(base*2) 

        self.upsample1_head2_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head2_class = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head2_class = nn.BatchNorm2d(base) 

        self.dec_conv4_head2_class = nn.Conv2d(base, 1, 3, padding=1)


        # HEAD3 reg
        self.upsample0_head3_reg = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head3_reg = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head3_reg = nn.BatchNorm2d(base*2) 

        self.upsample1_head3_reg = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head3_reg = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head3_reg = nn.BatchNorm2d(base) 

        self.dec_conv4_head3_reg = nn.Conv2d(base, 1, 3, padding=1) # 2 because reg and class


        # HEAD3 class
        self.upsample0_head3_class = nn.ConvTranspose2d(base*4, base*2, 2, stride= 2, padding= 0, output_padding= 0) # 4 -> 8
        self.dec_conv0_head3_class = nn.Conv2d(base*4, base*2, 3, padding=1, bias = False) # base+base=base*2 because of skip conneciton
        self.bn_dec_conv0_head3_class = nn.BatchNorm2d(base*2) 

        self.upsample1_head3_class = nn.ConvTranspose2d(base*2, base, 2, stride= 2, padding= 0, output_padding= 0) # 8 -> 16
        self.dec_conv1_head3_class = nn.Conv2d(base*2, base, 3, padding=1, bias = False) # base+base=base*2 because of skip connection
        self.bn_dec_conv1_head3_class = nn.BatchNorm2d(base) 

        self.dec_conv4_head3_class = nn.Conv2d(base, 1, 3, padding=1)

        # Dropout
        self.dropout = nn.Dropout(p = dropout_rate)


        # LSTM
        # kernel_size = 3 # could be specified in the init
        # padding = kernel_size // 2 # could be specified in the init
        # # /2 because we are splitting the hidden state into two tensors hs (short-term) and hl (long-term)
        # hidden_channels_split = int(hidden_channels/2) # could be specified in the init

        # LSTM 1
        self.Wxi_1 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True) # if it runs, try to remove bias - you are using batchnorm after all
        self.Whi_1 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxf_1 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whf_1 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxc_1 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whc_1 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxo_1 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Who_1 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)


        # LSTM 2
        self.Wxi_2 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True) # if it runs, try to remove bias - you are using batchnorm after all
        self.Whi_2 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxf_2 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whf_2 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxc_2 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whc_2 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxo_2 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Who_2 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)


        # LSTM 3
        self.Wxi_3 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True) # if it runs, try to remove bias - you are using batchnorm after all
        self.Whi_3 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxf_3 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whf_3 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxc_3 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whc_3 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxo_3 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Who_3 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)


        # LSTM 4
        self.Wxi_4 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True) # if it runs, try to remove bias - you are using batchnorm after all
        self.Whi_4 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxf_4 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whf_4 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxc_4 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Whc_4 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Wxo_4 = nn.Conv2d(input_channels, hidden_channels_split, kernel_size, padding=padding, bias=True)
        self.Who_4 = nn.Conv2d(hidden_channels_split, hidden_channels_split, kernel_size, padding=padding, bias=True)


    def forward(self, x, h):

        # The whole split than concat thing,  is just becaues the hidden state use to be one tensor, when it was just an RNN, but now as an LSTM it is two tensors. 
        split_hs = int(h.shape[1] / 8) 
        hs_1, hs_2, hs_3, hs_4, hl_1, hl_2, hl_3, hl_4 = torch.split(h, split_hs, dim=1) 



        #----------------- LSTM 1 -----------------
        # Input gate
        i_t_1 = torch.sigmoid(self.Wxi_1(x) + self.Whi_1(hs_1)) # Wxi changes to dims for x to the same as hs
        # Forget gate
        f_t_1 = torch.sigmoid(self.Wxf_1(x) + self.Whf_1(hs_1)) # Wxf changes to dims for x to the same as hs
        # Cell state
        hl_1_tilde = torch.tanh(self.Wxc_1(x) + self.Whc_1(hs_1)) # Wxc changes to dims for x to the same as hs
        hl_1 = f_t_1 * hl_1 + i_t_1 * hl_1_tilde
        # Output gate
        o_t_1 = torch.sigmoid(self.Wxo_1(x) + self.Who_1(hs_1)) # Wxo changes to dims for x to the same as hs
        
        hs_1 = o_t_1 * torch.tanh(hl_1) # The "input" that is used in the U-net below
        # I am unsure whether it is a good idea that the U-net now nevers sees the original x....
        # I possiple change is to concatenate x and hs before the U-net. It will just amount to a skip-connection.
        # I'll try with this more conservative solution first....


        #----------------- LSTM 2 -----------------
        # Input gate
        i_t_2 = torch.sigmoid(self.Wxi_2(x) + self.Whi_2(hs_2)) # Wxi changes to dims for x to the same as hs
        # Forget gate
        f_t_2 = torch.sigmoid(self.Wxf_2(x) + self.Whf_2(hs_2)) # Wxf changes to dims for x to the same as hs
        # Cell state
        hl_2_tilde = torch.tanh(self.Wxc_2(x) + self.Whc_2(hs_2)) # Wxc changes to dims for x to the same as hs
        hl_2 = f_t_2 * hl_2 + i_t_2 * hl_2_tilde
        # Output gate
        o_t_2 = torch.sigmoid(self.Wxo_2(x) + self.Who_2(hs_2)) # Wxo changes to dims for x to the same as hs
        
        hs_2 = o_t_2 * torch.tanh(hl_2) # The "input" that is used in the U-net below

        # -----------------


        #----------------- LSTM 3 -----------------
        # Input gate
        i_t_3 = torch.sigmoid(self.Wxi_3(x) + self.Whi_3(hs_3)) # Wxi changes to dims for x to the same as hs
        # Forget gate
        f_t_3 = torch.sigmoid(self.Wxf_3(x) + self.Whf_3(hs_3)) # Wxf changes to dims for x to the same as hs
        # Cell state
        hl_3_tilde = torch.tanh(self.Wxc_3(x) + self.Whc_3(hs_3)) # Wxc changes to dims for x to the same as hs
        hl_3 = f_t_3 * hl_3 + i_t_3 * hl_3_tilde
        # Output gate
        o_t_3 = torch.sigmoid(self.Wxo_3(x) + self.Who_3(hs_3)) # Wxo changes to dims for x to the same as hs

        hs_3 = o_t_3 * torch.tanh(hl_3) # The "input" that is used in the U-net below

        # -----------------

        #----------------- LSTM 4 -----------------
        # Input gate
        i_t_4 = torch.sigmoid(self.Wxi_4(x) + self.Whi_4(hs_4)) # Wxi changes to dims for x to the same as hs
        # Forget gate
        f_t_4 = torch.sigmoid(self.Wxf_4(x) + self.Whf_4(hs_4)) # Wxf changes to dims for x to the same as hs
        # Cell state
        hl_4_tilde = torch.tanh(self.Wxc_4(x) + self.Whc_4(hs_4)) # Wxc changes to dims for x to the same as hs
        hl_4 = f_t_4 * hl_4 + i_t_4 * hl_4_tilde
        # Output gate
        o_t_4 = torch.sigmoid(self.Wxo_4(x) + self.Who_4(hs_4)) # Wxo changes to dims for x to the same as hs

        hs_4 = o_t_4 * torch.tanh(hl_4) # The "input" that is used in the U-net below

        # -----------------

        h = torch.cat([hs_1, hs_2, hs_3, hs_4, hl_1, hl_2, hl_3, hl_4], 1) # concatenating short and long term memory along the channels. What is carried forward to the next timestep. The concat is just to keep it tight...
        # -----------------

        # THIS MIGHT BE BETTER. I.E. CONCATENATE X AND HS BEFORE THE U-NET AND USE THE NEW X... Start with this... 
        x = torch.cat([x, hs_1, hs_2, hs_3, hs_4], 1) # concatenating x and the new short term  memory along the channels

        # encoder
        e0s_ = F.relu(self.bn_enc_conv0(self.enc_conv0(x))) 
        #e0s_ = F.relu(self.bn_enc_conv0(self.enc_conv0(hs))) 

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
        H1_reg = self.dropout(H1_d1)

        # H1_d2 = F.relu(self.bn_dec_conv2_head1_reg(self.dec_conv2_head1_reg(H1_d1)))
        # H1_d2 = self.dropout(H1_d2) # is this good?

        # H1_reg = F.relu(self.bn_dec_conv3_head1_reg(self.dec_conv3_head1_reg(H1_d2)))
        # H1_reg = self.dropout(H1_reg) # is this good?

        H1_reg = self.dec_conv4_head1_reg(H1_reg)
        
        out_reg1 = F.relu(H1_reg)

        #H1 class
        H1_d0 = F.relu(self.bn_dec_conv0_head1_class(self.dec_conv0_head1_class(torch.cat([self.upsample0_head1_class(b),e1s],1))))
        H1_d0 = self.dropout(H1_d0)
        
        H1_d1 = F.relu(self.bn_dec_conv1_head1_class(self.dec_conv1_head1_class(torch.cat([self.upsample1_head1_class(H1_d0), e0s],1)))) # You did not have any activations before - why not?
        H1_class = self.dropout(H1_d1)

        # H1_d2 = F.relu(self.bn_dec_conv2_head1_class(self.dec_conv2_head1_class(H1_d1)))
        # H1_d2 = self.dropout(H1_d2) # is this good?
        
        # H1_class = F.relu(self.bn_dec_conv3_head1_class(self.dec_conv3_head1_class(H1_d2)))
        # H1_class = self.dropout(H1_class) # is this good?
        
        H1_class = self.dec_conv4_head1_class(H1_class)

        out_class1 = H1_class # torch.sigmoid(H1_class) # could move sigmoid outta here...


        #H2 reg
        H2_d0 = F.relu(self.bn_dec_conv0_head2_reg(self.dec_conv0_head2_reg(torch.cat([self.upsample0_head2_reg(b),e1s],1))))
        H2_d0 = self.dropout(H1_d0)
        
        H2_d1 = F.relu(self.bn_dec_conv1_head2_reg(self.dec_conv1_head2_reg(torch.cat([self.upsample1_head2_reg(H2_d0), e0s],1)))) # You did not have any activations before - why not?
        H2_reg = self.dropout(H1_d1)

        # H2_d2 = F.relu(self.bn_dec_conv2_head2_reg(self.dec_conv2_head2_reg(H2_d1)))
        # H2_d2 = self.dropout(H1_d2) # is this good?

        # H2_reg = F.relu(self.bn_dec_conv3_head2_reg(self.dec_conv3_head2_reg(H2_d2)))
        # H2_reg = self.dropout(H2_reg) # is this good?
        
        H2_reg = self.dec_conv4_head2_reg(H2_reg)
        
        out_reg2 = F.relu(H2_reg)

        #H2 class
        H2_d0 = F.relu(self.bn_dec_conv0_head2_class(self.dec_conv0_head2_class(torch.cat([self.upsample0_head2_class(b),e1s],1))))
        H2_d0 = self.dropout(H1_d0)
        
        H2_d1 = F.relu(self.bn_dec_conv1_head2_class(self.dec_conv1_head2_class(torch.cat([self.upsample1_head2_class(H2_d0), e0s],1)))) # You did not have any activations before - why not?
        H2_class = self.dropout(H2_d1)

        # H2_d2 = F.relu(self.bn_dec_conv2_head2_class(self.dec_conv2_head2_class(H2_d1)))
        # H2_d2 = self.dropout(H1_d2) # is this good?
        
        # H2_class = F.relu(self.bn_dec_conv3_head2_class(self.dec_conv3_head2_class(H2_d2)))
        # H2_class = self.dropout(H2_class) # is this good?
        
        
        H2_class = self.dec_conv4_head2_class(H2_class)

        out_class2 = H2_class # torch.sigmoid(H1_class) # could move sigmoid outta here...



        #H3 reg
        H3_d0 = F.relu(self.bn_dec_conv0_head3_reg(self.dec_conv0_head3_reg(torch.cat([self.upsample0_head3_reg(b),e1s],1))))
        H3_d0 = self.dropout(H3_d0)
        
        H3_d1 = F.relu(self.bn_dec_conv1_head3_reg(self.dec_conv1_head3_reg(torch.cat([self.upsample1_head3_reg(H3_d0), e0s],1)))) # You did not have any activations before - why not?
        H3_reg = self.dropout(H3_d1)

        # H3_d2 = F.relu(self.bn_dec_conv2_head3_reg(self.dec_conv2_head3_reg(H3_d1)))
        # H3_d2 = self.dropout(H1_d2) # is this good?

        # H3_reg = F.relu(self.bn_dec_conv3_head3_reg(self.dec_conv3_head3_reg(H3_d2)))
        # H3_reg = self.dropout(H3_reg) # is this good?
        
        
        H3_reg = self.dec_conv4_head3_reg(H3_reg)
        
        out_reg3 = F.relu(H3_reg)

        #H3 class
        H3_d0 = F.relu(self.bn_dec_conv0_head3_class(self.dec_conv0_head3_class(torch.cat([self.upsample0_head3_class(b),e1s],1))))
        H3_d0 = self.dropout(H1_d0)
        
        H3_d1 = F.relu(self.bn_dec_conv1_head3_class(self.dec_conv1_head3_class(torch.cat([self.upsample1_head3_class(H3_d0), e0s],1)))) # You did not have any activations before - why not?
        H3_class = self.dropout(H3_d1)

        # H3_d2 = F.relu(self.bn_dec_conv2_head3_class(self.dec_conv2_head3_class(H3_d1)))
        # H3_d2 = self.dropout(H1_d2) # is this good?
        
        # H3_class = F.relu(self.bn_dec_conv3_head3_class(self.dec_conv3_head3_class(H3_d2)))
        # H3_class = self.dropout(H3_class) # is this good?
        
        
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

"""
Adapted from: https://github.com/camlab-ethz/DSE-for-NeuralOperators

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


################################################################
# fourier layer
################################################################
class VFT:
    def __init__(self, x_positions, y_positions, modes):
        # it is important that positions are scaled between 0 and 2*pi
        #x_positions -= torch.min(x_positions)
        x_positions = x_positions - torch.min(x_positions)
        self.x_positions = x_positions * 6.28 / (torch.max(x_positions))
        y_positions = y_positions - torch.min(y_positions)
        #y_positions -= torch.min(y_positions)
        self.y_positions = y_positions * 6.28 / (torch.max(y_positions))
        self.number_points = x_positions.shape[1]
        self.batch_size = x_positions.shape[0]
        self.modes = modes

        self.X_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float()#.cuda()
        self.Y_ = torch.cat((torch.arange(modes), torch.arange(start=-(modes-1), end=0)), 0).repeat(self.batch_size, 1)[:,:,None].float()#.cuda()
        self.X_ = self.X_.to(x_positions.device)
        self.Y_ = self.Y_.to(x_positions.device)
        self.V_fwd, self.V_inv = self.make_matrix()

    def make_matrix(self):
        m = (self.modes*2)*(self.modes*2-1)
        X_mat = torch.bmm(self.X_, self.x_positions[:,None,:]).repeat(1, (self.modes*2-1), 1)
        Y_mat = (torch.bmm(self.Y_, self.y_positions[:,None,:]).repeat(1, 1, self.modes*2).reshape(self.batch_size,m,self.number_points))
        forward_mat = torch.exp(-1j* (X_mat+Y_mat)) 

        inverse_mat = torch.conj(forward_mat.clone()).permute(0,2,1)

        return forward_mat, inverse_mat

    def forward(self, data):
        data_fwd = torch.bmm(self.V_fwd, data)
        return data_fwd

    def inverse(self, data):
        data_inv = torch.bmm(self.V_inv, data)
        
        return data_inv

class SpectralConv2d_dse (nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d_dse, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))


    # Complex multiplication and complex batched multiplications
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x, transformer):
        batchsize = x.shape[0]
        num_pts = x.shape[-1]

        x = x.permute(0, 2, 1)
        
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = transformer.forward(x.cfloat()) #[4, 20, 32, 16]
        x_ft = x_ft.permute(0, 2, 1)
        # out_ft = self.compl_mul1d(x_ft, self.weights3)
        x_ft = torch.reshape(x_ft, (batchsize, self.out_channels, 2*self.modes1, 2*self.modes1-1))

        # # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, 2*self.modes1, self.modes1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes1] = self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes1], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes1] = self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes1], self.weights2)

        # #Return to physical space
        x_ft = torch.reshape(out_ft, (batchsize, self.out_channels, 2*self.modes1**2))
        x_ft2 = x_ft[..., 2*self.modes1:].flip(-1, -2).conj()
        x_ft = torch.cat([x_ft, x_ft2], dim=-1)

        x_ft = x_ft.permute(0, 2, 1)
        x = transformer.inverse(x_ft) # x [4, 20, 512, 512]
        x = x.permute(0, 2, 1)
        x = x / x.size(-1) * 2

        return x.real



class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2,  width, num_blocks, use_batch_norm=True):
        super(SimpleBlock2d, self).__init__()
        self.use_batch_norm = use_batch_norm

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.num_blocks = num_blocks
        self.fc0 = nn.Linear(3, self.width)

        self.convs = nn.ModuleList([SpectralConv2d_dse(self.width, self.width, self.modes1, self.modes2) for _ in range(self.num_blocks)])
        self.ws = nn.ModuleList([nn.Conv1d(self.width, self.width, 1) for _ in range(self.num_blocks)])
        
        if self.use_batch_norm:
            self.bns = nn.ModuleList([nn.BatchNorm1d(self.width) for _ in range(self.num_blocks)])

        self.fc1 = nn.Linear(self.width + 3, 128)
        self.fc2 = nn.Linear(128, 1)

        self.fc2.weight.data.fill_(0.0)
        self.fc2.bias.data.fill_(1.31)

    def forward(self, x):
        """
        x: [batch_size, num_points, dim]
        
        """
        
        transform = VFT(x[:,:,0], x[:,:,1], self.modes1) 

        h = self.fc0(x)
        h = h.permute(0, 2, 1)

        for i in range(self.num_blocks):

            h1 = self.convs[i](h, transform)
            h2 = self.ws[i](h)
            h = h1 + h2
            if self.use_batch_norm:
                h = self.bns[i](h)
            #print(x.shape)
            h = F.gelu(h)

        # x = x[..., :-self.padding, :-self.padding]
        h = h.permute(0, 2, 1)
        h = torch.concat([h, x], dim=-1)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)

        return h

class FNO_dse(nn.Module):
    def __init__(self, modes, width, num_blocks,use_batch_norm=True):
        super(FNO_dse, self).__init__()

        self.conv1 = SimpleBlock2d(modes, modes,  width, num_blocks, use_batch_norm)

    def forward(self, x):
        x = self.conv1(x)
        return x


if __name__ == "__main__":

    batch_size = 8
    num_points = 1600

    x = torch.randn(batch_size, num_points, 3)

    model = FNO_dse(modes=16, width=8, num_blocks=4)
    #model = model.to("cuda")
    #x = x.to("cuda")

    out = model(x)

    print(out.shape)

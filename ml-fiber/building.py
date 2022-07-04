import torch
from torch import nn
from layers import Conv, Hourglass, Pool, Residual, HeatmapLoss
from tensorboardX import SummaryWriter


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 256, 4, 4)


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class PoseNet(nn.Module):
    def __init__(self, nstack, inp_dim, oup_dim, bn=False, increase=0, **kwargs):
        super(PoseNet, self).__init__()

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(1, 32, 7, 1, bn=True, relu=True),
            Residual(32, 64),
            # Pool(2, 2),
            Residual(64, 64),
            Residual(64, inp_dim)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(2, inp_dim, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(inp_dim, inp_dim),
                Conv(inp_dim, inp_dim, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([Conv(inp_dim, oup_dim, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(inp_dim, inp_dim) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(oup_dim, inp_dim) for i in range(nstack - 1)])
        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()

    def forward(self, x):
        ## our posenet
        # x = imgs.permute(0, 3, 1, 2)  # x of size 1,3,inpdim,inpdim
        x = self.pre(x)
        combined_hm_preds = []
        for i in range(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)
        return torch.stack(combined_hm_preds, 1)

    def calc_loss(self, combined_hm_preds, heatmaps):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[:, i, :, :], heatmaps))
        combined_loss = torch.stack(combined_loss, dim=1)
        return combined_loss


if __name__ == '__main__':
    m = PoseNet(
        nstack=2,
        inp_dim=64,
        oup_dim=1
    )
    x = torch.randn(1, 1, 32, 32)
    y = m(x)
    print(y.shape)

    # with SummaryWriter("./log") as sw:
    #     sw.add_graph(m, x)

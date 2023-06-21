import torch
import utils.utils as utils
import modules.vision_transformer as vits

from torch import nn
from torchvision import models as torchvision_models


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)


class MultiheadClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, dim, attr):
        super(MultiheadClassifier, self).__init__()
        self.attr = attr
        self.mlps = nn.ModuleDict({k: LinearClassifier(dim=dim, num_labels=2) for k in attr})

    def forward(self, x, head_idx):
        # flatten
        x = x.view(x.size(0), -1)
        x = torch.cat([self.mlps[str(_k)](x[i : i + 1]) for i, _k in enumerate(head_idx)])
        return x


class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=True, batch_norm=True, bias=True):
        super(LinearBlock, self).__init__()
        modules = [nn.Linear(in_features=in_features, out_features=out_features, bias=bias)]

        if batch_norm:
            modules.append(nn.BatchNorm1d(out_features))

        if activation:
            modules.append(nn.ReLU(inplace=True))

        self.linear_block = nn.Sequential(*modules)

    def forward(self, x):
        x = self.linear_block(x)

        return x


class FeatureClassifier(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_targets):
        super(FeatureClassifier, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            LinearBlock(in_features=in_channels, out_features=hidden_channels, batch_norm=False),
            LinearBlock(in_features=hidden_channels, out_features=hidden_channels, batch_norm=False),
            LinearBlock(in_features=hidden_channels, out_features=num_targets, activation=False, batch_norm=False),
        )

    def forward(self, x):
        x = self.mlp(x)

        return x


class DINOMIMICClassification(nn.Module):
    def __init__(self, args, attr, num_targets=2, img_size=224):
        super(DINOMIMICClassification, self).__init__()
        assert args.feature_type in ["only_global", "both_local_global"]
        # ============ building network ... ============
        # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
        if args.arch in vits.__dict__.keys():
            model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
            embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
        # if the network is a XCiT
        elif "xcit" in args.arch:
            model = torch.hub.load("facebookresearch/xcit", args.arch, num_classes=0)
            embed_dim = model.embed_dim
        # otherwise, we check if the architecture is in torchvision models
        elif args.arch in torchvision_models.__dict__.keys():
            model = torchvision_models.__dict__[args.arch]()
            embed_dim = model.fc.weight.shape[1]
            model.fc = nn.Identity()
        else:
            print(f"Unknow architecture: {args.arch}")
            sys.exit(1)

        if not args.full_finetune:
            model.eval()

        # load weights to evaluate
        if not args.from_scratch:
            utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
            print(f"Model {args.arch} built.")
        else:
            print(f"Model {args.arch} built - from the scratch")

        self.args = args
        self.model = model
        self.img_size = img_size
        self.patch_size = args.patch_size
        self.model_embed_dim = embed_dim
        self.num_targets = num_targets

        # ============ building classification head ... ============
        if args.feature_type == "both_local_global":
            embed_dim = embed_dim * 2
        self.mlps = nn.ModuleDict({k: FeatureClassifier(in_channels=embed_dim, hidden_channels=embed_dim // 2, num_targets=self.num_targets) for k in attr})

    def feature_crop(self, intermediate_output, coord, batch_size, device):   
        n_patch = self.img_size // self.patch_size
        min_h = torch.floor(coord[:, 0] / self.patch_size).type(torch.int32)
        min_w = torch.floor(coord[:, 1] / self.patch_size).type(torch.int32)
        max_h = torch.ceil(coord[:, 2] / self.patch_size).type(torch.int32)
        max_w = torch.ceil(coord[:, 3] / self.patch_size).type(torch.int32)
        local_output = torch.FloatTensor([]).to(device)
        for _b in range(batch_size):
            _idx = []
            for _i in range(min_h[_b].item(), max_h[_b].item()):
                _idx += list(range(n_patch * _i + min_w[_b].item(), n_patch * _i + max_w[_b].item()))
            _output = torch.cat([torch.mean(x[_b, 1:][_idx], dim=0) for x in intermediate_output], dim=-1)
            local_output = torch.cat([local_output, _output.unsqueeze(0)], dim=0)

        return local_output

    def get_embedding(self, input, n_last_blocks, coord):
        if "vit" in self.args.arch:
            intermediate_output = self.model.get_intermediate_layers(input, n_last_blocks)
            if self.args.cropping_type in ["img_crop", "wo_crop"]:
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if self.args.avgpool_patchtokens:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            elif self.args.cropping_type == "feat_crop":
                local_output = self.feature_crop(intermediate_output, coord, input.shape[0], input.device)
                if self.args.feature_type == "only_local":
                    output = local_output
                elif self.args.feature_type == "both_local_global":
                    output = torch.cat([output, local_output], dim=1)
                else:
                    raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            output = self.model(input)

        return output

    def forward(self, data, n):
        head_idx = data["attribute"]
        if self.args.full_finetune:
            output = self.get_embedding(input=data["img"], n_last_blocks=n, coord=data["coord224"])
        else:
            with torch.no_grad():
                output = self.get_embedding(input=data["img"], n_last_blocks=n, coord=data["coord224"])

        output = output.view(output.size(0), -1)
        output = torch.cat([self.mlps[str(_k)](output[i : i + 1]) for i, _k in enumerate(head_idx)])
        return output


class DINOMIMICCOMPClassification(DINOMIMICClassification):
    def __init__(self, args, attr, img_size=224):
        super().__init__(args, attr)
        assert self.args.feature_type in ["only_global", "both_local_global"]
        # ============ building classification head ... ============
        # concat current emb & prev_emb
        self.num_targets = args.num_target_label if args.num_target_label != 1 else 2
        embed_dim = self.model_embed_dim * 2 
        if args.feature_type == "both_local_global":
            embed_dim = embed_dim * 2
        self.mlps = nn.ModuleDict({k: FeatureClassifier(in_channels=embed_dim, hidden_channels=embed_dim // 2, num_targets=self.num_targets) for k in attr})

    def forward(self, data, n):
        head_idx = data["attribute"]
        if self.args.full_finetune:
            output_cur = self.get_embedding(input=data["img"], n_last_blocks=n, coord=data["coord224"])
            output_prev = self.get_embedding(input=data["prev_img"], n_last_blocks=n, coord=data["coord224_prev"])
            output = torch.cat([output_cur, output_prev], dim=-1)
        else:
            with torch.no_grad():
                output_cur = self.get_embedding(input=data["img"], n_last_blocks=n, coord=data["coord224"])
                output_prev = self.get_embedding(input=data["prev_img"], n_last_blocks=n, coord=data["coord224_prev"])
                output = torch.cat([output_cur, output_prev], dim=-1)

        output = output.view(output.size(0), -1)
        output = torch.cat([self.mlps[str(_k)](output[i : i + 1]) for i, _k in enumerate(head_idx)])
        return output

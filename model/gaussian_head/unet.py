import torch
import torch.nn as nn   
from functools import partial
import spconv
if float(spconv.__version__[2:]) >= 2.2:
    spconv.constants.SPCONV_USE_DIRECT_TABLE = False
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv
from spconv.pytorch import functional as Fsp
from ..layers.spconv_unet import get_voxel_centers
from ..layers.spconv_unet import post_act_block, SparseBasicBlock, replace_feature

class UNet(nn.Module):
    def __init__(self, grid_size, voxel_size, point_cloud_range, gs_num):
        super().__init__()
        self.sparse_shape = grid_size[::-1]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range

        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(3+3+32, 32, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(32),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(128, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(256, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        # decoder
        self.conv_up_t4 = SparseBasicBlock(256, 256, indice_key='subm4', norm_fn=norm_fn)
        self.conv_up_m4 = block(256*2, 256, 3, norm_fn=norm_fn, padding=1, indice_key='subm4')
        self.inv_conv4 = block(256, 128, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        
        self.conv_up_t3 = SparseBasicBlock(128, 128, indice_key='subm3', norm_fn=norm_fn)
        self.conv_up_m3 = block(128*2, 128, 3, norm_fn=norm_fn, padding=1, indice_key='subm3')
        self.inv_conv3 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv3', conv_type='inverseconv')
        
        self.out_conv = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='out1',),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='out1',),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='out1')
        )
        
        self.fusion = spconv.SparseSequential(
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='fusion'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='fusion'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='fusion'),
        )
        self.to_gaussians = nn.Sequential(
            nn.Linear(64, 64),
            nn.GELU(),
            nn.Linear(64, 15*gs_num),
        )
        self.time_embedding = nn.Embedding(2, 64)
        self.pos_embedding = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, 64)
        )

    def UR_block_forward(self, x_lateral, x_bottom, conv_t, conv_m, conv_inv):
        x_trans = conv_t(x_lateral)
        x = x_trans
        x = replace_feature(x, torch.cat((x_bottom.features, x_trans.features), dim=1))
        x_m = conv_m(x)
        x = self.channel_reduction(x, x_m.features.shape[1])
        x = replace_feature(x, x_m.features + x.features)
        x = conv_inv(x)
        return x

    @staticmethod
    def channel_reduction(x, out_channels):
        """
        Args:
            x: x.features (N, C1)
            out_channels: C2

        Returns:

        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels == 0) and (in_channels >= out_channels)

        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def forward(self, voxel_features, voxel_coords, batch_size, history_infos):
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for segmentation head
        x_up4 = self.UR_block_forward(x_conv4, x_conv4, self.conv_up_t4, self.conv_up_m4, self.inv_conv4)
        x_up3 = self.UR_block_forward(x_up4, x_conv3, self.conv_up_t3, self.conv_up_m3, self.inv_conv3)
        
        x_up3_centers = get_voxel_centers(
            x_up3.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        x_up3_time = x_up3.features + self.time_embedding.weight[1][None, :] + self.pos_embedding(x_up3_centers)
        x_up3 = replace_feature(x_up3, x_up3_time)
    
        history_voxel_features, history_voxel_coords, history_voxel_pos, history_masks = history_infos
        
        # Check dimension consistency before creating sparse tensor
        if history_voxel_features.shape[0] != history_voxel_coords.shape[0]:
            print(f"[WARNING] UNet history dimension mismatch: features={history_voxel_features.shape}, coords={history_voxel_coords.shape}")
            min_history = min(history_voxel_features.shape[0], history_voxel_coords.shape[0])
            if min_history > 0:
                history_voxel_features = history_voxel_features[:min_history]
                history_voxel_coords = history_voxel_coords[:min_history]
                history_voxel_pos = history_voxel_pos[:min_history]
                history_masks = history_masks[:min_history]
            else:
                # Create dummy to avoid crash
                history_voxel_features = torch.zeros((1, history_voxel_features.shape[1]), dtype=history_voxel_features.dtype, device=history_voxel_features.device)
                history_voxel_coords = torch.zeros((1, 4), dtype=torch.int32, device=history_voxel_coords.device)
                history_voxel_pos = torch.zeros((1, 3), dtype=history_voxel_features.dtype, device=history_voxel_features.device)
                history_masks = torch.zeros((1,), dtype=torch.bool, device=history_voxel_features.device)
        
        history_voxel_features = history_voxel_features + self.time_embedding.weight[0][None, :]
        history_voxel_features = history_voxel_features + self.pos_embedding(history_voxel_pos)
        history_gate = history_masks[:, None].float()
        history_voxel_features = history_voxel_features * history_gate
        history_sp_tensor = spconv.SparseConvTensor(
            features=history_voxel_features,
            indices=history_voxel_coords.int(),
            spatial_shape=x_up3.spatial_shape,
            batch_size=batch_size
        )
        # history current fusion (scalar-gated: reliability modulates history before addition)
        x_up3 = Fsp.sparse_add(x_up3, history_sp_tensor)
        x_up3 = self.fusion(x_up3)
        
        save_features = x_up3.features
        save_coords = x_up3.indices[:, 1:]
        save_coords = get_voxel_centers(
            save_coords, downsample_times=2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        save_coords = torch.cat((x_up3.indices[:, 0:1].float(), save_coords), dim=1)
        
        x_up3 = self.out_conv(x_up3)
        gs_features = x_up3.features
        point_coords = get_voxel_centers(
            x_up3.indices[:, 1:], downsample_times=2, voxel_size=self.voxel_size,
            point_cloud_range=self.point_cloud_range
        )
        gaussians = self.to_gaussians(gs_features)
        point_coords = torch.cat((x_up3.indices[:, 0:1].float(), point_coords), dim=1)
        return gaussians, point_coords, save_features, save_coords 
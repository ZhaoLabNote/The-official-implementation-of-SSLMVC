import torch.nn as nn
from torch.nn.functional import normalize
import torch


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)


class FusionModule(nn.Module):

    def __init__(self, feature_dim, fusion_dim, view_num):
        super(FusionModule, self).__init__()
        
        self.projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(feature_dim, fusion_dim),
                nn.ReLU()
            ) for _ in range(view_num)
        ])

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(fusion_dim, fusion_dim),
                nn.Sigmoid()
            ) for _ in range(view_num)
        ])
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(view_num * fusion_dim, fusion_dim),
            nn.ReLU()
        )

    def forward(self, zs, return_gate_activations=False):
        gated_projected_features_list = []
        batch_avg_gate_activations = []

        for v_idx, z_v in enumerate(zs):
            projected_v = self.projections[v_idx](z_v)
            gate_weights_v = self.gates[v_idx](projected_v)
            gated_projected_v = projected_v * gate_weights_v
            gated_projected_features_list.append(gated_projected_v)
            
            if return_gate_activations:
                avg_activation = torch.mean(gate_weights_v)
                batch_avg_gate_activations.append(avg_activation)
        
        concatenated_features = torch.cat(gated_projected_features_list, dim=1)
        fused = self.fusion_mlp(concatenated_features)
        
        if return_gate_activations:
            return fused, batch_avg_gate_activations
        else:
            return fused


class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, fusion_dim, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

        self.fusion_dim = fusion_dim
        self.fusion_module = FusionModule(feature_dim, self.fusion_dim, view)
        
        self.fusion_decoders = nn.ModuleList([
            Decoder(input_size[v], self.fusion_dim) for v in range(view)
        ])
        
        self.fusion_contrastive = nn.Sequential(
            nn.Linear(self.fusion_dim, high_feature_dim),
        )
        
        self.fusion_cluster = nn.Sequential(
            nn.Linear(self.fusion_dim, class_num),
            nn.Softmax(dim=1)
        )

        self.feature_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, high_feature_dim),
        )
        self.feature_z = nn.Sequential(
            nn.Linear(high_feature_dim, class_num),
        )
        self.label_contrastive_module = nn.Sequential(
            nn.Linear(feature_dim, class_num),
            nn.Softmax(dim=1)
        )
        self.view = view

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        ps = []
        
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs.append(z)
            
            h = normalize(self.feature_contrastive_module(z), dim=1)
            q = self.label_contrastive_module(z)
            xr = self.decoders[v](z)
            p = self.feature_z(h)
            
            ps.append(p)
            hs.append(h)
            qs.append(q)
            xrs.append(xr)
        
        fused_z = self.fusion_module(zs)
        
        fusion_xrs = []
        for v in range(self.view):
            fusion_xr = self.fusion_decoders[v](fused_z)
            fusion_xrs.append(fusion_xr)
        
        fusion_h = normalize(self.fusion_contrastive(fused_z), dim=1)
        fusion_q = self.fusion_cluster(fused_z)
        fusion_p = self.feature_z(fusion_h)
        
        return hs, qs, xrs, zs, ps, fusion_h, fusion_q, fusion_xrs, fusion_p

    def forward_plot(self, xs):
        zs_views = []
        hs_views = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            zs_views.append(z)
            h = self.feature_contrastive_module(z)
            hs_views.append(h)
            
        fused_z, batch_avg_gate_activations = self.fusion_module(zs_views, return_gate_activations=True)
        fusion_h = self.fusion_contrastive(fused_z)
        
        all_zs_for_plotting = zs_views + [fused_z]
        all_hs_for_plotting = hs_views + [fusion_h]

        return all_zs_for_plotting, all_hs_for_plotting, batch_avg_gate_activations

    def forward_cluster(self, xs):
        qs = []
        preds = []
        
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            q = self.label_contrastive_module(z)
            pred = torch.argmax(q, dim=1)
            qs.append(q)
            preds.append(pred)
        
        zs_for_fusion = [self.encoders[v](xs[v]) for v in range(self.view)]
        fused_z = self.fusion_module(zs_for_fusion)
        fusion_q = self.fusion_cluster(fused_z)
        fusion_pred = torch.argmax(fusion_q, dim=1)
        
        qs.append(fusion_q)
        preds.append(fusion_pred)
        
        return qs, preds

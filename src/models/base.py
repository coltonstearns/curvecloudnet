import copy
import torch
import time
import torch.nn.functional as F

from torch_geometric.data.batch import Batch
from torch_geometric.nn import MLP

from src.models.modules.pointnet2 import SAModule, GlobalSAModule, FPModule, CurveSAModule, CurveFPModule
from src.models.modules.fast_conv1d import SymmetricCurve1DConvFastV1, SymmetricCurve1DConvV2
from src.models.modules.dgcnn import DGCNNLayer, SGCNNLayer, DGCNNLayerRadius
from src.models.modules.skip_connect import SkipConnect
from src.models.modules.mlp import SharedMLP


class ModelBase(torch.nn.Module):

    def __init__(self, in_dim, n_out, steps=("conv1d", "dgcnn", "conv1d", "sa", "sa", "sa-global"),
                 feat_dims=((32, 32, 64), (64, 128), (128, 128), (128, 128, 256), (256, 512, 1024), (1024,)),
                 out_mlp=dict(), **kwargs):
        """
        Args:
            steps (list[str]): A list of which DNN modules to sequentially execute for per-point feature extraction.
                    Currently supported options are ["conv1d", "conv2d", "sa", "fp", "sa-geo", "fp-geo", "dgcnn",
                    "pointnet", "sa-global", "skip-connect", "pos-enc"]. Moving forward, we hope to add in ["self-attention", "self-attention-geo"].
        """
        super(ModelBase, self).__init__()
        self.in_dim = in_dim
        self.n_out = n_out
        self.use_bias = kwargs.get('use_bias', False)
        self.version = kwargs.get('version', 2.0)
        self.mlp_func = MLP
        self.step_names = steps
        self.skip_connect_state_store = list() if 'skip_connect_state_store' not in kwargs else kwargs['skip_connect_state_store']

        # append each step
        self.steps = torch.nn.ModuleList()
        for i, step_name in enumerate(steps):
            step_kwargs = kwargs.copy()
            if isinstance(step_name, dict):
                step_kwargs = {**step_kwargs, **step_name}
                step_name = step_kwargs.pop("step_name")
                self.step_names[i] = step_name
            with_xyz = False if "with_xyz" not in step_kwargs else step_kwargs["with_xyz"]
            step_kwargs['with_xyz'] = with_xyz
            step_dims = self._get_input_dim(i, step_name, feat_dims, in_dim, with_xyz)
            self.steps.append(self.add_step(i, step_name, step_dims, **step_kwargs))

        # final-layer MLP
        final_mlp_kwargs = {"dropout": 0.5, "norm": "batch_norm", "plain_last": True}
        out_mlp_cp = copy.deepcopy(out_mlp)
        if isinstance(out_mlp_cp, dict):
            out_feat_dims = out_mlp_cp.pop("dims")
            final_mlp_kwargs = {**final_mlp_kwargs, **out_mlp_cp}
        else:
            out_feat_dims = out_mlp_cp
        step_dims = [feat_dims[-1][-1]] + out_feat_dims + [n_out]
        if "with_seg_category" in final_mlp_kwargs and final_mlp_kwargs['with_seg_category']:
            step_dims[0] += 64
            self.lin_categorical = self.mlp_func([16, 64, 64])
        if "identity" in final_mlp_kwargs and final_mlp_kwargs['identity']:
            self.mlp = torch.nn.Identity()
        else:
            self.mlp = self.mlp_func(step_dims, bias=self.use_bias, **final_mlp_kwargs)

    def _get_input_dim(self, step_idx, step_name, feat_dims, in_dim, with_xyz):
        if step_idx == 0 and step_name in ['dgcnn', 'sgcnn']:  # these modules use 2x the input dim
            input_dim = [in_dim*2]
        elif step_idx == 0 and step_name in ['sa', 'sa-global', 'sa-geo']:
            input_dim = [in_dim + 3*with_xyz]
        elif step_idx == 0 and step_name not in ['dgcnn', 'sgcnn', 'sa', 'sa-global', 'sa-geo']:  # standard
            input_dim = [in_dim]
        elif step_idx != 0 and step_name in ["sa", "sa-global", "sa-geo"]:  # SA modules append xyz feature
            input_dim = [feat_dims[step_idx-1][-1] + 3 + 3*with_xyz]
        elif step_idx != 0 and step_name in ["dgcnn", "sgcnn"]:  # takes double the input-size (local + global features)
            input_dim = [2*(feat_dims[step_idx-1][-1] + 3*with_xyz)]
        elif step_idx != 0 and step_name in ["skip-connect", "fp", "fp-geo"]:  # thesedo not automatically assign dims
            input_dim = []
        elif step_idx != 0 and step_name in ["mlp",  "conv1d-fast-v1", "conv1d-fast-v2"]:
            input_dim = [feat_dims[step_idx-1][-1] + 3*with_xyz]
        else:
            raise NotImplementedError("No Module Named >> %s" % step_name)
        dims = input_dim + feat_dims[step_idx]
        return dims

    def add_step(self, step_idx, step_name, dims, **kwargs):
        if step_name == "sa":  # PointNet++ Set Abstraction
            if 'aggr_type' in kwargs and (kwargs['aggr_type'] == 'attend' or kwargs['aggr_type'] == 'weighted-sum'):
                if self.version == 2.0:
                    attend_nn = self.mlp_func([dims[-1], dims[-1]//2, dims[-1]], act='leaky_relu', bias=self.use_bias)  # lowkey I wonder if this is it?
                elif self.version == 1.0:
                    attend_nn = self.mlp_func([dims[-1], dims[-1], dims[-1]], act='leaky_relu', bias=self.use_bias)
            else:
                attend_nn = None
            step = SAModule(kwargs['ratios'][step_idx], kwargs['radii'][step_idx], self.mlp_func(dims, bias=self.use_bias), attend_nn=attend_nn, k=kwargs['knn'][step_idx], **kwargs)
        elif step_name == "sa-global":
            step = GlobalSAModule(self.mlp_func(dims, bias=self.use_bias), **kwargs)
        elif step_name == "dgcnn":
            step = DGCNNLayer(self.mlp_func(dims, bias=self.use_bias), kwargs['knn'][step_idx], with_xyz=kwargs['with_xyz'])
        elif step_name == "dgcnn-rad":
            step = DGCNNLayerRadius(self.mlp_func(dims, bias=self.use_bias), kwargs['radii'][step_idx], with_xyz=kwargs['with_xyz'])
        elif step_name == "sgcnn":
            if 'aggr_type' in kwargs and (kwargs['aggr_type'] == 'attend' or kwargs['aggr_type'] == 'weighted-sum'):
                attend_nn = self.mlp_func([dims[-1], dims[-1], dims[-1]], act='leaky_relu', bias=self.use_bias)
            else:
                attend_nn = None
            step = SGCNNLayer(self.mlp_func(dims, bias=self.use_bias), kwargs['knn'][step_idx], r=kwargs['radii'][step_idx], attend_nn=attend_nn, **kwargs)
        elif step_name == "sa-geo":
            if 'aggr_type' in kwargs and (kwargs['aggr_type'] == 'attend' or kwargs['aggr_type'] == 'weighted-sum'):
                attend_nn = self.mlp_func([dims[-1], dims[-1], dims[-1]], act='leaky_relu', bias=self.use_bias)
            else:
                attend_nn = None
            step = CurveSAModule(kwargs['ratios'][step_idx], kwargs['radii'][step_idx], self.mlp_func(dims, act='leaky_relu', bias=self.use_bias), attend_nn=attend_nn, **kwargs)
        elif step_name == "conv1d-fast-v1":
            with_diff = False if 'with_diff' not in kwargs else kwargs['with_diff']
            step = SymmetricCurve1DConvFastV1(dims, kwargs['kernel_sizes'][step_idx], with_xyz=kwargs['with_xyz'], with_diff=with_diff)
        elif step_name == "conv1d-fast-v2":
            with_diff = False if 'with_diff' not in kwargs else kwargs['with_diff']
            step = SymmetricCurve1DConvV2(dims, kwargs['kernel_sizes'][step_idx], with_xyz=kwargs['with_xyz'], with_diff=with_diff)
        elif step_name == "skip-connect":
            step = SkipConnect(self.mlp_func(dims, act='leaky_relu', bias=self.use_bias), kwargs['num_skips'][step_idx])
        elif step_name == "fp":
            step = FPModule(kwargs['knn'][step_idx], self.mlp_func(dims, bias=self.use_bias), with_xyz=kwargs['with_xyz'])
        elif step_name == "fp-geo":
            step = CurveFPModule(kwargs['knn'][step_idx], self.mlp_func(dims, act='leaky_relu', bias=self.use_bias), with_xyz=kwargs['with_xyz'])
        elif step_name == "mlp":
            step = SharedMLP(dims, **kwargs)
        else:
            raise NotImplementedError("Have not implemented step %s yet!" % step_name)

        return step

    def forward(self, data, **kwargs):
        if isinstance(data, list):  # we're in data parallel mode
            data = Batch.from_data_list(data)

        out = (data.x, data.pos, data.batch, data.curve_idxs)
        state = {"x": [data.x], "pos": [data.pos], "batch": [data.batch], "point2curveidx": [data.curve_idxs.detach().clone()],
                 "skip-connections-proportional": [], "skip-connections-downsampled": [], "downsample-idxs": []}
        if hasattr(data, 'labels'):
            kwargs['shapenet-categories'] = data.labels
        return self._forward(out, state,  **kwargs)

    def _forward(self, out, state, **kwargs):
        batch = out[2].clone()
        out, state = self._apply_steps(out, state, **kwargs)
        out = out[0]

        # if shapenet segmentation, we concat class label at the end
        if 'shapenet-categories' in kwargs and hasattr(self, 'lin_categorical'):  # is size B x
            cats = F.one_hot(kwargs['shapenet-categories'], num_classes=16).float()
            cats = self.lin_categorical(cats)[batch]
            out = torch.cat([out, cats], dim=1)

        # regress final output
        out = self.mlp(out)
        return out

    def _apply_steps(self, out, state, **kwargs):
        for i, step_fn in enumerate(self.steps):
            if self.step_names[i] in ['fp', 'fp-geo']:  # skip-conv layer
                x_skip, pos_skip, batch_skip, curveidx_skip, down_idxs = self._get_upsample_skip_connect(state)
                if self.step_names[i] in ['fp']:
                    out = (out[0], out[1], out[2], x_skip, pos_skip, batch_skip, out[3], curveidx_skip)
                else:  # in fp-geo
                    assert torch.all(down_idxs[1:] - down_idxs[:-1] > 0)
                    out = (out[0], down_idxs, x_skip, pos_skip, batch_skip, curveidx_skip)
            elif self.step_names[i] in ['skip-connect']:
                x_skips = self._get_proportional_skip_connect(state, step_fn.num_skips)
                x_skips = [out[0]] + x_skips
                out = (x_skips, out[1], out[2], out[3])
            out = step_fn(*out, **kwargs)
            state = self._update_forward_pass_state(out, state, self.step_names[i], i, step_fn)
            out = out[:4]
        return out, state

    def _update_forward_pass_state(self, out, state, step_name, step_idx, step_model):
        # normal state
        state['x'].append(out[0].clone())
        state['pos'].append(out[1].clone())
        state['batch'].append(out[2].clone())
        state['point2curveidx'].append(out[3].clone())

        # for keeping track of which points were sampled along each curve
        if len(out) > 5:
            state['downsample-idxs'].append(out[5].clone())
        else:
            state['downsample-idxs'].append(None)

        # update skip-connection state
        if step_name in self.skip_connect_state_store:
            state['skip-connections-proportional'].append(step_idx)
        if step_name in ['sa', 'sa-geo', 'sa-global', 'pt-transition-down']:
            state['skip-connections-downsampled'].append(step_idx)
        elif step_name == "conv1d" and step_model.stride > 1:
            state['skip-connections-downsampled'].append(step_idx)
        return state

    def _get_upsample_skip_connect(self, state):
        skip_connect_idx = state['skip-connections-downsampled'][-1]
        x_skip = state['x'][skip_connect_idx] if state['x'][skip_connect_idx] is not None else state['pos'][skip_connect_idx]
        state['skip-connections-downsampled'] = state['skip-connections-downsampled'][:-1]
        return x_skip, state['pos'][skip_connect_idx], state['batch'][skip_connect_idx], state['point2curveidx'][skip_connect_idx], state['downsample-idxs'][skip_connect_idx]

    def _get_proportional_skip_connect(self, state, num_skips):
        x_skip_idxs = state['skip-connections-proportional'][-num_skips:]
        x_skips = [state['x'][idx] if state['x'][idx] is not None else state['pos'][idx] for idx in x_skip_idxs]
        state['skip-connections-proportional'] = state['skip-connections-proportional'][:-num_skips]
        return x_skips

    def to(self, device):
        out = super(ModelBase, self).to(device)
        for i in range(len(out.steps)):
            out.steps[i] = out.steps[i].to(device)
        return out

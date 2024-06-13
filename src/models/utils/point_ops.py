import torch
from pytorch3d.ops import sample_farthest_points, ball_query, knn_points
import torch
from torch_scatter import scatter_add
from torch_geometric.nn import knn
from torch_geometric.nn.glob import global_add_pool
import frnn
from torch_geometric.typing import OptTensor
import time


def old_curveidx_local2global(point2curveidx, batch):
    point2curveidx_glob = point2curveidx.clone()
    max_curves_per_sample = torch.max(point2curveidx_glob)
    point2curveidx_glob += batch.long() * max_curves_per_sample.long()
    _, point2curveidx_glob = torch.unique(point2curveidx_glob, return_inverse=True)
    return point2curveidx_glob


def curveidx_local2global(point2curveidx, batch):
    # get batch offset value
    ptr = batch2ptr(batch, with_ends=True)
    batch_size = ptr.size(0) - 1
    ptr = ptr[1:-1]

    # trivial check for batch-size 1 (speeds things up at inference)
    if batch_size == 1:
        return point2curveidx

    # otherwise, increase by appropriate offset
    assert is_sorted(point2curveidx)
    max_curveidxs = point2curveidx[ptr-1]  # get the curve index of the last-available curve idx
    max_curveidxs += 1  # we need to offset by 1 because this is index and not count
    cum_curveidxs = torch.cumsum(max_curveidxs, dim=0)
    cum_curveidxs = torch.cat([torch.zeros(1).to(cum_curveidxs), cum_curveidxs])
    offsets = torch.index_select(cum_curveidxs, dim=0, index=batch)  # vector of all the offsets
    point2curveidx_glob = point2curveidx + offsets

    # check with legacy code for batch size > 1!
    # old_point2curveidx_glob = old_curveidx_local2global(point2curveidx, batch)
    # bad = point2curveidx_glob != old_point2curveidx_glob
    # assert torch.all(point2curveidx_glob == old_point2curveidx_glob)

    return point2curveidx_glob


def batch2ptr(batch, with_ends=False):
    device, end_idx = batch.device, batch.size(0)+1
    assert torch.all((batch[1:] - batch[:-1]) >= 0)  # make sure is sorted
    ptr = torch.where(batch[1:] - batch[:-1] > 0)[0]
    ptr += 1
    if with_ends:
        ptr = torch.cat([torch.zeros(1).to(ptr), ptr, torch.tensor([batch.size(0)]).to(ptr)])
    return ptr


def fps_pytorch3d(pos, batch, ratio):
    # convert into batch-padded format
    pos_batched, mask = to_batch_padded(pos.clone(), batch.clone())

    # compute FPS in pytorch3D instead of torch_geometric (MUCH faster)
    lengths = mask.sum(dim=-1)
    K = torch.ceil(lengths * ratio)
    _, idx = sample_farthest_points(pos_batched, lengths, K=K, random_start_point=True)
    idx_mask = idx != -1
    offsets = torch.cumsum(lengths, dim=0)[:-1]
    idx[1:, :] += offsets.unsqueeze(-1)
    idx = idx[idx_mask].flatten()
    idx = idx.sort()[0]
    return idx


def knn_ball_group_pytorch3d(p1, p2, batch1, batch2, operation="ball-group", radius=None, knn=None, accel_knn=True, return_dense=False):
    # convert into batched form
    p1_batched, mask1, lengths1, offsets1 = dense2padded_pyg(p1, batch1)
    p2_batched, mask2, lengths2, offsets2 = dense2padded_pyg(p2, batch2)

    # Perform radial groupings
    if operation == "ball-group":
        assert radius is not None
        _, pnt_idx, _ = ball_query(p1_batched, p2_batched, lengths1, lengths2, radius=radius, K=128, return_nn=return_dense)
    elif operation == "knn":
        if accel_knn:
            if radius is None:
                print("Not setting radius for Fast-KNN!")
                radius = 0.25
            pnt_idx = fast_knn(p1_batched, p2_batched, lengths1, lengths2, K=knn, r=radius, return_nn=return_dense)
            if return_dense:
                return pnt_idx, lengths2, mask1
        else:
            _, pnt_idx, _ = knn_points(p1_batched, p2_batched, lengths1, lengths2, K=knn, return_nn=False)  # (N, P1, K)
            if return_dense:
                return pnt_idx, lengths2, mask1


    else:
        raise RuntimeError()
    B, N_p1, K_max = pnt_idx.size()

    # Convert point indices to flat form
    rad_idx_mask = pnt_idx != -1
    rad_idx_mask = rad_idx_mask * mask1.unsqueeze(-1)
    pnt_idx = pytorch3didx2pyg(pnt_idx, rad_idx_mask, offsets2)

    # generate corresponding query indices
    query_idxs = torch.arange(N_p1).view(1, N_p1, 1).repeat(B, 1, K_max).to(pnt_idx)  # B x N_down
    query_idxs = pytorch3didx2pyg(query_idxs, rad_idx_mask, offsets1)

    # return arrays
    row, col = query_idxs, pnt_idx
    return row, col


def knn_1d_group_subset(points, idxs, point2curveidx, batch, knn):
    BN, Q = points.size(0), idxs.size(0)
    point2curveidx_glob = curveidx_local2global(point2curveidx, batch)

    # precompute offset indices
    offset_idxs = (torch.tensor([[-1, 1]]) * torch.arange(1, knn+1)[:, None]).flatten()
    offset_idxs = torch.cat([torch.zeros(1), offset_idxs]).to(points.device)

    # get KNN idxs for each query point
    knn_idxs = idxs[:, None] + offset_idxs[None, :]  # q_pnts x offsets
    valid_mask = (knn_idxs < points.size(0)) & (knn_idxs >= 0)
    knn_idxs[~valid_mask] = 0
    knn_idxs = knn_idxs.long()

    # check if the idx is on the same curve
    same_curve_mask = point2curveidx_glob[idxs].view(Q, 1) == point2curveidx_glob[knn_idxs].view(Q, -1)
    valid_mask = valid_mask & same_curve_mask

    # remove extra neighbors
    extra_neighbor_mask = torch.cumsum(valid_mask, dim=1) <= knn
    valid_mask = valid_mask & extra_neighbor_mask

    # finally, obtain pairings
    query_idxs = torch.arange(Q).to(points.device)[:, None].repeat(1, 2*knn+1)
    row = query_idxs[valid_mask]
    col = knn_idxs[valid_mask]
    return row, col


def radius_1d_group_subset(points, idxs, point2curveidx, batch, radius):
    BN, Q = points.size(0), idxs.size(0)
    point2curveidx_glob = curveidx_local2global(point2curveidx, batch)

    # computer avg edge length of each curve
    point2curveidx_glob = curveidx_local2global(point2curveidx, batch)
    edges, edge_validity = points[1:] - points[:-1], (point2curveidx_glob[1:] - point2curveidx_glob[:-1]) == 0
    edge_norms = torch.linalg.norm(edges, dim=-1)
    edge_norms[~edge_validity] = 0

    # compute average edge length
    # assert(torch.all(point2curveidx_glob[1:] - point2curveidx_glob[:-1] >= 0))
    edge2curveidx_glob = point2curveidx_glob[1:]  # distance from starting point to next point
    curve_lens = global_add_pool(edge_norms.unsqueeze(-1), batch=edge2curveidx_glob).squeeze(-1)
    pnts_per_curve = global_add_pool(torch.ones((point2curveidx_glob.size(0), 1), device=points.device), batch=point2curveidx_glob).squeeze(-1)
    avg_edge_len = curve_lens / pnts_per_curve.float()
    per_curve_knn = torch.ceil(radius / avg_edge_len)
    # assert(torch.all(pnts_per_curve[torch.isinf(per_curve_knn)] == 1))

    per_curve_knn[torch.isinf(per_curve_knn)] = 1

    # compute maximum KNN for padding reasons
    max_knn = torch.max(per_curve_knn)
    max_pnts = torch.max(pnts_per_curve)
    knn = min(max_knn.item(), max_pnts.item())

    # precompute offset indices
    offset_idxs = (torch.tensor([[-1, 1]]) * torch.arange(1, knn+1)[:, None]).flatten()
    offset_idxs = torch.cat([torch.zeros(1), offset_idxs]).to(points.device)

    # get KNN idxs for each query point
    knn_idxs = idxs[:, None] + offset_idxs[None, :]  # q_pnts x offsets
    valid_mask = (knn_idxs < points.size(0)) & (knn_idxs >= 0)
    knn_idxs[~valid_mask] = 0
    knn_idxs = knn_idxs.long()

    # check if the idx is on the same curve
    same_curve_mask = point2curveidx_glob[idxs].view(Q, 1) == point2curveidx_glob[knn_idxs].view(Q, -1)
    valid_mask = valid_mask & same_curve_mask

    # remove neighbors that go past our radius
    # print(point2curveidx.dtype)
    per_query_knn = per_curve_knn[point2curveidx[idxs].long()]
    extra_neighbor_mask = torch.cumsum(valid_mask, dim=1) <= per_query_knn[:, None]
    valid_mask = valid_mask & extra_neighbor_mask

    # finally, obtain pairings
    query_idxs = torch.arange(Q).to(points.device)[:, None].repeat(1, 2*int(knn)+1)
    row = query_idxs[valid_mask]
    col = knn_idxs[valid_mask]
    return row, col


def knn_1d_group_superset(points, idxs, point2curveidx, batch, knn):
    BN, Q = points.size(0), idxs.size(0)
    point2curveidx_glob = curveidx_local2global(point2curveidx, batch)
    assert torch.all(point2curveidx_glob[1:] - point2curveidx_glob[:-1]) >= 0

    assignments = torch.zeros(points.size(0), dtype=torch.bool).to(points.device)
    assignments[idxs] = True
    assignments = torch.cumsum(assignments, dim=0)  # assigns the next-closest lookup (sequentially)
    assert torch.all(assignments < points.size(0))

    # precompute offset indices
    offset_idxs = (torch.tensor([[-1, 1]]) * torch.arange(1, knn+2)[:, None]).flatten()
    offset_idxs = torch.cat([torch.zeros(1), offset_idxs]).to(points.device)

    # get KNN idxs for each input point (which is mapped to query points)
    knn_idxs = assignments[:, None] + offset_idxs[None, :]  # q_pnts x offsets
    valid_mask = (knn_idxs < idxs.size(0)) & (knn_idxs >= 0)
    knn_idxs[~valid_mask] = 0
    knn_idxs = knn_idxs.long()

    # check if the idx is on the same curve
    knn_idxs_glob = torch.arange(points.size(0)).to(points.device)[idxs]
    knn_idxs_glob = knn_idxs_glob[knn_idxs].view(-1, 2*knn+3)
    same_curve_mask = point2curveidx_glob.view(BN, 1) == point2curveidx_glob[knn_idxs_glob].view(BN, -1)
    valid_mask = valid_mask & same_curve_mask

    # compute geodesic lengths
    ptr_glob = torch.cat([torch.tensor([0]).to(points.device), batch2ptr(point2curveidx_glob)])
    point2curvestartidx_glob = ptr_glob[point2curveidx_glob]
    polyline_edge_lens = (points[1:] - points[:-1]).norm(dim=-1)
    geodesic_lengths = torch.cat([torch.tensor([0]).to(points.device), torch.cumsum(polyline_edge_lens, dim=0)])
    geodesic_lengths -= geodesic_lengths[point2curvestartidx_glob]

    # re-order candidates by closest to farthest geodesic
    dists = (points[knn_idxs_glob].view(-1, knn*2+3, 3) - points.view(-1, 1, 3)).norm(dim=-1)
    dists[~valid_mask] = 100
    reorder_idx = torch.argsort(dists, dim=1)
    # print(reorder_idx[:15])
    knn_idxs = torch.gather(knn_idxs, dim=1, index=reorder_idx)
    valid_mask = torch.gather(valid_mask, dim=1, index=reorder_idx)

    # remove extra neighbors
    extra_neighbor_mask = torch.cumsum(valid_mask, dim=1) <= knn
    valid_mask = valid_mask & extra_neighbor_mask
    assert torch.all(valid_mask.sum(dim=1) > 0)

    # finally, obtain mapping from all-points to query-idx-points
    query_idxs = torch.arange(BN).to(points.device)[:, None].repeat(1, 2*knn+3)

    # finally, obtain pairings
    row = query_idxs[valid_mask]
    col = knn_idxs[valid_mask]

    # visualize
    # knn_idxs_glob = torch.gather(knn_idxs_glob, dim=1, index=reorder_idx)
    # col_glob = knn_idxs_glob[valid_mask]
    # import networkx as nx
    # import torch_geometric
    # import matplotlib.pyplot as plt
    # data = torch_geometric.data.Data(x=points[:50], edge_index=torch.stack([row[:100], col_glob[:100]]))
    # g = torch_geometric.utils.to_networkx(data, to_undirected=True)
    # nx.draw(g)
    # plt.show()

    return row, col



def dense2padded_pyg(x, batch):
    """
    Converts a `dense` array of size (N', feats) with its batch-assignments to a zero-padded array of size (B, N, feats)

    Args:
        x (torch.FloatTensor): Tensor of size (N', feats) which stores the dense per-point features.
        batch (torch.LongTensor): Tensor of size (N',) which stores the batch assignment of x. For instance, if batch
            is [0,0,1,1,1,1,2]), then the first 2 points belong to the first batch, the next 4 points belong to the
            second batch, and the final point belongs to the third batch.

    Return:
        x_padded (torch.FloatTensor): Tensor of size (B, N, feats) which stores zero-padded per-point features
        mask (torch.BoolTensor): Tensor of size (B, N). True indicates a valid feature in x_padded. False indicates padding.
        lengths (torch.LongTensor): Tensor of size (B,) containing the number of non-padded values in each batch
        offsets (torch.LongTensor): Necessary when x is an array of indices. This is the cumulative-sum of <lengths>. When converting
                                    indices from padded back to a dense representation, you must shift indices by this cumulative offset.
    """
    x_padded, mask = to_batch_padded(x.clone(), batch.clone())
    lengths = mask.sum(dim=-1)
    offsets = torch.cumsum(lengths, dim=0)[:-1]
    return x_padded, mask, lengths, offsets


def pytorch3didx2pyg(idx, mask, offsets):
    idx[1:, :] += offsets.view(-1, 1, 1)
    idx = idx[mask].flatten()  # flattens into 1D tensor
    return idx


def knn_interpolate_pytorch3d(x: torch.Tensor, pos_x: torch.Tensor, pos_y: torch.Tensor,
                    batch_x: OptTensor = None, batch_y: OptTensor = None,
                    k: int = 3):
    r"""The k-NN interpolation from the `"PointNet++: Deep Hierarchical
    Feature Learning on Point Sets in a Metric Space"
    <https://arxiv.org/abs/1706.02413>`_ paper.
    For each point :math:`y` with position :math:`\mathbf{p}(y)`, its
    interpolated features :math:`\mathbf{f}(y)` are given by

    .. math::
        \mathbf{f}(y) = \frac{\sum_{i=1}^k w(x_i) \mathbf{f}(x_i)}{\sum_{i=1}^k
        w(x_i)} \textrm{, where } w(x_i) = \frac{1}{d(\mathbf{p}(y),
        \mathbf{p}(x_i))^2}

    and :math:`\{ x_1, \ldots, x_k \}` denoting the :math:`k` nearest points
    to :math:`y`.

    Args:
        x (Tensor): Node feature matrix
            :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
        pos_x (Tensor): Node position matrix
            :math:`\in \mathbb{R}^{N \times d}`.
        pos_y (Tensor): Upsampled node position matrix
            :math:`\in \mathbb{R}^{M \times d}`.
        batch_x (LongTensor, optional): Batch vector
            :math:`\mathbf{b_x} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{X}` to a specific example.
            (default: :obj:`None`)
        batch_y (LongTensor, optional): Batch vector
            :math:`\mathbf{b_y} \in {\{ 0, \ldots, B-1\}}^N`, which assigns
            each node from :math:`\mathbf{Y}` to a specific example.
            (default: :obj:`None`)
        k (int, optional): Number of neighbors. (default: :obj:`3`)
        num_workers (int): Number of workers to use for computation. Has no
            effect in case :obj:`batch_x` or :obj:`batch_y` is not
            :obj:`None`, or the input lies on the GPU. (default: :obj:`1`)
    """
    with torch.no_grad():
        y_idx, x_idx = knn_ball_group_pytorch3d(pos_y, pos_x, batch_y, batch_x, operation="knn", knn=k, accel_knn=False)
        # assign_index = knn(pos_x, pos_y, k, batch_x=batch_x, batch_y=batch_y, num_workers=1)
        # y_idx, x_idx = assign_index[0], assign_index[1]
        diff = pos_x[x_idx] - pos_y[y_idx]
        squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

    y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))

    return y


def knn_interpolate_1D_pytorch3d(x, idx, pos_y, batch_y, point2curveidx_y, k):
    pos_x = pos_y[idx].clone()

    with torch.no_grad():
        y_idx, x_idx = knn_1d_group_superset(pos_y, idx, point2curveidx_y, batch_y, k)  # (points, idxs, point2curveidx, batch, knn):
        dist = (pos_x[x_idx] - pos_y[y_idx]).norm(dim=-1, keepdim=True)**2
        weights = 1.0 / torch.clamp(dist, min=1e-16)

    y = scatter_add(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0))
    y = y / scatter_add(weights, y_idx, dim=0, dim_size=pos_y.size(0))

    return y


def to_batch_padded(tensor, val2batchidx, pad_size=None, batch_size=None):
    # preliminaries
    ptr = batch2ptr(val2batchidx, with_ends=True)
    counts = ptr[1:] - ptr[:-1]
    if batch_size is None:
        batch_size = ptr.size(0) - 1
    if pad_size is None:
        pad_size = torch.max(counts).item()

    # in trivial case of batch_size=1, simply return the same tensor
    if batch_size == 1:
        return tensor.unsqueeze(0), torch.ones((1, tensor.size(0)), dtype=torch.bool, device=tensor.device)

    # create indices for sparse tensor
    val2ptr = torch.index_select(ptr, dim=0, index=val2batchidx)  # gets ptr offset for each value in batch
    local_idx = torch.arange(val2batchidx.size(0)).to(val2ptr) - val2ptr
    sparse_idx = torch.stack([val2batchidx, local_idx], dim=0)  # 2 x N set of sparse indices

    # convert to sparse tensor, then rapidly convert to dense
    tensor_coo = torch.sparse_coo_tensor(sparse_idx, tensor, dtype=tensor.dtype, device=tensor.device)
    mask = torch.zeros((batch_size, pad_size), dtype=torch.bool, device=tensor.device)
    mask[sparse_idx[0], sparse_idx[1]] = True
    out = tensor_coo.to_dense()
    return out, mask


def group_first_k_values(values, batch, k):
    """
    Given a set of values and their batch-assignments, this operation selects and groups the first K values for
    each batch. If there are not K values in a batch, 0's are padded.
    Args:
        values (torch.tensor): 1D or 2D tensor of values
        batch (torch.LongTensor): 1D tensor of batch-idx assignments
        k (int or torch.tensor): number of values to group per-batch. If a tensor is given, the number of values is k.max(),
                                 and each batch will have a different limit of values (reflected via padding)

    Returns:
        grouped_values (torch.tensor): 2D tensor of size (num_batch, k)
        mask (torch.tensor): 2D tensor indicating which values are valid
    """
    # prelim attributes
    device, dims = values.device, len(values.size())
    num_batches = torch.unique(batch).size(0)
    if torch.is_tensor(k):
        k_max = torch.max(k).item()
    else:
        k_max = k

    # get first occurrence values in each batch
    batch_sorted, batch_sort_mapping = torch.sort(batch, stable=True)
    batch_initvals_bool = torch.cat([torch.ones(1).to(device).long(), (batch_sorted[1:] - batch_sorted[: -1]).long()])
    batch_initvals_bool = batch_initvals_bool > 0
    batch_initvals = torch.where(batch_initvals_bool)[0]
    # batch_initvals = torch.where(torch.cat([torch.ones(1).to(device), batch_sorted[1:] - batch_sorted[: -1]]))[0]

    # get the number of values assigned to each batch
    _, values_per_batch = torch.unique(batch_sorted, return_counts=True)
    values_per_batch = torch.clamp(values_per_batch, max=k).unsqueeze(1)

    # given first occurrence and values-per-batch, generate 2D array of indices
    inds = torch.arange(k_max).unsqueeze(0).expand(num_batches, k_max).to(device)
    mask = inds < values_per_batch
    inds = inds.clone() + batch_initvals.unsqueeze(1)
    inds[~mask] = 0

    # reformat to original, unsorted values
    values = values[batch_sort_mapping[inds.flatten()]].view(num_batches, k_max, -1)
    values[~mask] = 0
    if dims == 1:
        values = values.squeeze(-1)
    return values, mask


def fast_knn(points1, points2, lengths1, lengths2, K, r, return_nn=False):
    if points1.shape[0] != points2.shape[0]:
        raise ValueError(
            "points1 and points2 must have the same batch  dimension")
    if points1.shape[2] != points2.shape[2]:
        raise ValueError(
            f"dimension mismatch: points1 of dimension {points1.shape[2]} while points2 of dimension {points2.shape[2]}"
        )
    if not points1.is_cuda or not points2.is_cuda:
        raise TypeError("for now only cuda version is supported")

    points1 = points1.contiguous()
    points2 = points2.contiguous()

    N = points1.shape[0]
    if isinstance(r, float) or isinstance(r, int):
        r = torch.ones((N,), dtype=torch.float32) * r
    if isinstance(r, torch.Tensor):
        assert (len(r.shape) == 1 and (r.shape[0] == 1 or r.shape[0] == N))
        if r.shape[0] == 1:
            r = r * torch.ones((N,), dtype=r.dtype, device=r.device)
    r = r.type(torch.float32)
    if not r.is_cuda:
        r = r.cuda()

    # idxs, dists, sorted_points2, pc2_grid_off, sorted_points2_idxs, grid_params_cuda = frnn._frnn_grid_points.apply(
    #     points1, points2, lengths1, lengths2, K, r, None, None, None, None, True, 2)

    dists, idxs, points2_nn, grid = frnn.frnn_grid_points(points1, points2, lengths1, lengths2, K, r)

    return idxs


def is_sorted(t):
    return torch.all(t[1:] - t[:-1]) >= 0

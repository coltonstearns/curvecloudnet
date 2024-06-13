import torch
from torch_geometric.nn import knn
from src.models.utils.point_ops import batch2ptr, curveidx_local2global, to_batch_padded
EPS = 1e-5


def estimate_curvature_grads_batch(points, X, batch, point2curveidx, k, kernel_width=2.5, hinge_reg=1e-4, return_curvature=True, return_grads=False):
    point2curveidx_glob = curveidx_local2global(point2curveidx, batch)
    return estimate_curvature_and_grads(points, X, point2curveidx_glob, k, kernel_width, hinge_reg, return_curvature, return_grads)


def estimate_curvature_and_grads(points, X, point2curveidx, k, kernel_width=2.5, hinge_reg=1e-4,
                                 return_curvature=True, return_grads=True):
    """
    Does not support batching right now!
    points: (N, 3) tensor
    point2curveidx: (N,) tensor
    k: int
    """
    # compute geodesic curve-position
    ptr_glob = torch.cat([torch.tensor([0]).to(points.device), batch2ptr(point2curveidx)])
    point2curvestartidx = ptr_glob[point2curveidx]
    polyline_edge_lens = (points[1:] - points[:-1]).norm(dim=-1)
    geodesic_lengths = torch.cat([torch.tensor([0]).to(points.device), torch.cumsum(polyline_edge_lens, dim=0)])
    geodesic_lengths -= geodesic_lengths[point2curvestartidx]

    # compute KNN for each neighbor
    N = points.size(0)
    assign_index = knn(x=points, y=points, k=k, batch_x=point2curveidx, batch_y=point2curveidx)
    orig_idxs, assigned_idxs = assign_index  # note: more query-idxs than orig-idxs
    orig_sort_idxs = torch.argsort(orig_idxs)
    assigned_idxs, orig_idxs = assigned_idxs[orig_sort_idxs], orig_idxs[orig_sort_idxs]

    # convert to batch-padded format
    assert(torch.unique(orig_idxs).size(0) == N)
    knn_point_idxs, withpoint_mask = to_batch_padded(assigned_idxs, orig_idxs, pad_size=k, batch_size=N)
    knn_points = points[knn_point_idxs.flatten()].view(N, k, 3)
    knn_points_geo = geodesic_lengths[knn_point_idxs.flatten()].view(N, k)
    knn_points_centered = knn_points - points.view(N, 1, 3)
    knn_points_geo_centered = knn_points_geo - geodesic_lengths.view(N, 1)
    if X is not None:
        knn_X = X[knn_point_idxs.flatten()].view(N, k, -1) if X is not None else None
        knn_X_centered = knn_X - X.view(N, 1, -1)
    else:
        knn_X_centered = None

    # get gaussian weights
    valid_edge_lens = (points[1:] - points[:-1]).norm(dim=-1)[point2curveidx[1:] - point2curveidx[:-1] == 0]
    mean_edge_len = valid_edge_lens.mean()
    gauss_weights = torch.exp(-knn_points_geo_centered.pow(2) / (kernel_width * mean_edge_len).pow(2))  # (N, k)
    gauss_weights = gauss_weights / gauss_weights.sum(dim=1, keepdim=True).clamp(EPS)

    # compute curvature --> size N
    min_pnts = max(5, int(k/2.5))
    curvature = weighted_regression_curvature(knn_points_geo_centered, knn_X_centered, knn_points_centered, gauss_weights, withpoint_mask,
                                              alpha=hinge_reg, min_pnts=min_pnts, return_curvature=return_curvature,
                                              return_grads=return_grads)
    return curvature


def weighted_regression_curvature(X_geo, X, pos, weights, mask, alpha=1e-3, min_pnts=5, return_curvature=True, return_grads=False):
    """
    Args:
        X_geo (torch.tensor): size (B, K) of geodesic distances along the "curve" surrounding the point
        X (torch.tensor): size (B, K, 3) tensor of relative position surrounding each point of interest
        weights( torch.tensor): size (B, K) tensor of weights of points
        mask (torch.tensor): size (B, K) boolean tensor with True indicating to use the value in regression
    Returns:

    """
    B, K = X_geo.size()

    # first, solve for physical curve tangents
    pos_parametrics = solve_weighted_parametric(X_geo, pos, weights, mask, alpha, min_pnts)
    velocities = pos_parametrics[:, 1, :].clone()
    tangents_ = velocities / velocities.norm(dim=-1, keepdim=True).clone()
    tangents = remove_nans(tangents_)

    # second, solve for curve normals
    if return_curvature:
        accelerations = 2 * torch.round(pos_parametrics[:, 2, :].clone(), decimals=6)
        tangent_acceleration = tangents * (accelerations * tangents).sum(dim=-1, keepdim=True)
        normal_accelerations = accelerations - tangent_acceleration
        normals_ = normal_accelerations / normal_accelerations.norm(dim=-1, keepdim=True).clone()
        normals = remove_nans(normals_)  # should be for ill-defined least-squares from not enough points

        # third, compute curvatures
        curvature__ = torch.cross(velocities, accelerations, dim=-1).norm(dim=-1)
        curvature_ = (curvature__.log() - 3 * velocities.norm(dim=-1).log()).exp()
        curvature = remove_nans(curvature_)  # should be for ill-defined least-squares from not enough points
        curvature = normals * curvature.unsqueeze(-1)
        # curvature *= 1e2  # todo: for some reason we need to rescale curvature?
    else:
        curvature = None

    # fourth, compute feature gradients
    # todo: use larger alpha value for velocities in feature gradients?
    if return_grads:
        feat_parametrics = solve_weighted_parametric(X_geo, X, weights, mask, alpha, min_pnts)
        gradient_1d = feat_parametrics[:, 1, :].unsqueeze(-1).clone()  # B x feats x 1
        gradients_ = gradient_1d * tangents.view(B, 1, 3)  # B x feats x 3
        gradients = remove_nans(gradients_)
    else:
        gradients = None

    # format output
    out = []
    if return_curvature:
        out.append(curvature)
    if return_grads:
        out.append(gradients)
    if len(out) == 1:
        out = out[0]
    return out


def solve_weighted_parametric(T, X, weights, mask, alpha=1e-3, min_pnts=5):
    B, K = T.size()
    weights = torch.diag_embed(weights)  # now is (B, K, K)

    # initialize solving matrices
    A = torch.stack([torch.ones(B, K).to(X), T, T ** 2], dim=2)  # B x K x 3

    # pad out invalid values (set everything to 0)
    A[~mask] = 0
    X[~mask] = 0

    # solve best-fit 1st order parametrics
    lI = float(alpha) * torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], device=X.device).unsqueeze(0)
    # lI = alpha * torch.eye(3, 3, device=X.device).unsqueeze(0)  # regularizer
    # lI[:, :2, :2] *= 0  # we only want to push position to 0, as we need regularization of our tangent and acceleration
    lI = lI + EPS * torch.eye(3, 3, device=X.device).unsqueeze(0)  # for numerical stability
    # todo: should our numerical stability be diagonal?

    left_side = (A.transpose(1, 2) @ weights @ A).double()  # B x interp-order x interp-order
    left_side += lI
    right_side = (A.transpose(1, 2) @ weights @ X).double()

    # solve for parametrics (pos, vel, acc)
    parametrics_ = torch.linalg.solve(left_side, right_side)  # (B, interp-order, 3)
    # Note: keep as separate tmp variable for auto-grad issue

    # remove parametrics which were solved with too few points
    enoughpnts = mask.sum(dim=-1) >= min_pnts
    enoughpnts_mask = torch.zeros(B).to(parametrics_)
    enoughpnts_mask[enoughpnts.clone()] = 1.0
    parametrics = parametrics_ * enoughpnts_mask.view(B, 1, 1)
    return parametrics


def remove_nans(tnsr):
    mask = torch.ones(tnsr.size(), dtype=torch.bool, device=tnsr.device)
    mask[torch.isnan(tnsr.detach().clone())] = False

    # construct new tensor without any in-place ops
    tnsr_nonan = tnsr.clone()[mask].contiguous().clone()
    out = torch.zeros(tnsr.size()).to(tnsr)
    out[mask] = tnsr_nonan
    return out


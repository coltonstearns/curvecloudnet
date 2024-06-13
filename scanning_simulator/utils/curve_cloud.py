import torch
import torch.nn.functional as F
from torch_scatter import scatter_min
from scanning_simulator.utils.curvature_approx import estimate_curvature_and_grads


class CurveClouds:
    CURVE_SPLIT_THRESH = 0.01
    INTERSECTION_DIST_THRESH = 0.01
    ANTI_ALIAS_KERNEL = [1, 2, 3, 3, 3, 2, 2]


    def __init__(self, points, normals, uv, point2uvcurve_idxs, anti_alias=True, with_intersections=False, line_density=1.0, curvature_knn=10):
        """

        :param points (torch.tensor): size (B, N, 3) of points per object
        :param normals (torch.tensor): size (B, N, 3) of normals per object
        :param uv (torch.tensor): size (B, N, 2) of UV-image-plane coordinates per object
        :param uv_curve_idxs (torch.tensor): size (B, N) of point-to-uv-curve assignments
        """
        self.points = points.double()
        self.normals = normals.double()
        self.uv = uv
        self.point2uvcurve_idxs = point2uvcurve_idxs
        self.device = points.device
        self.point2curve_idxs = self.compute_3d_curve_idxs()

        # initialize other fields empty
        if anti_alias:
            self.points, self.normals = self.anti_alias()

        self.curvature_knn = curvature_knn
        self.curvature, self.gradients = self.compute_curvature_new()
        self.intersections = None
        self.intersection2batch_idxs = None
        if with_intersections:
            self.intersections, self.intersection2batch_idxs = self.compute_intersections()
            # intersections will be size (2 x num_intersections) of longs

    def compute_3d_curve_idxs(self):
        B = self.points.size(0)
        edges = self.points[:, 1:].double() - self.points[:, :-1].double()
        edge_norms = torch.linalg.norm(edges, dim=2)
        curve_splits = edge_norms > self.CURVE_SPLIT_THRESH
        curve_idxs = torch.cumsum(curve_splits, dim=1)
        curve_idxs = torch.cat([torch.zeros(B, 1).to(self.device), curve_idxs], dim=1)  # first point belongs to idx=0
        return curve_idxs

    def compute_curvature_new(self):
        B, N_orig, dim = self.points.size()
        curvature, gradients = [], []
        for i in range(B):
            this_curvature, this_grads = estimate_curvature_and_grads(self.points[i], self.points[i], self.point2curve_idxs[i].long(), k=self.curvature_knn,
                                                                       return_curvature=True, return_grads=True)
            curvature.append(this_curvature)
            gradients.append(this_grads)
        curvature = torch.stack(curvature, dim=0)
        gradients = torch.stack(gradients, dim=0)

        return curvature, gradients

    def compute_curvature(self):
        B, N_orig, dim = self.points.size()
        curves, mask = batchdense2curvepadded(self.points, self.point2curve_idxs)  # M x Nmax x 3; M x Nmax
        M, N_max = mask.size()
        if N_max < 3:  # cannot compute curvature info
            return torch.zeros((B, N_orig, 1), dtype=torch.double, device=self.device)

        # todo: compute an approximate 3rd order polynomial on M points, then take the second derivative...? messy!

        # get turning angles along each curve
        edges = curves[:, 1:, :].double() - curves[:, :-1, :].double()  # M x Nmax-1 x 3
        edge_norms = torch.linalg.norm(edges, dim=2)
        edge_dots = torch.sum(edges[:, 1:] * edges[:, :-1], dim=2) / (edge_norms[:, 1:] * edge_norms[:, :-1])
        joint_lens = (edge_norms[:, 1:] * edge_norms[:, :-1]) / 2

        # find duplicated points (ie "valid" interior edges of norm 0)
        # batch_idxs = torch.arange(B).repeat_interleave(N_orig).view(B, N_orig, 1).to(self.device)
        # find_repeats = torch.cat([self.points, self.point2uvcurve_idxs.unsqueeze(-1), batch_idxs], dim=2)
        # uniques, counts = torch.unique(find_repeats.view(B*N_orig, -1), return_counts=True, dim=0)
        # if torch.any(counts > 1):
        #     print(torch.max(counts))
        #     print("We have a repeat!!!")
        #     print("##############")

        turning_angles = torch.acos(torch.clip(edge_dots, -1, 1))
        turning_angles_mask = mask[:, 2:]

        # compute curvature from turning angles
        # joints_curvatures = 2 * torch.sin(turning_angles / 2)
        joints_curvatures = turning_angles / joint_lens
        last_joint_idxs = torch.sum(turning_angles_mask, dim=1) - 1

        # broadcast per-joint curvature to per-point (set endpoints as KNN curvature)
        per_point_curvatures = torch.zeros((M, N_max), dtype=torch.double, device=self.device)
        per_point_curvatures[:, 1:-1] = joints_curvatures

        # Weird pytorch bugs make the following not work properly, so we're just going to set to 0 for now...
        # per_point_curvatures[:, 0] = joints_curvatures[:, 0]
        per_point_curvatures[torch.arange(M), last_joint_idxs+2] = 0
        # per_point_curvatures[torch.arange(M), last_joint_idxs+2] = joints_curvatures[torch.arange(M), last_joint_idxs]

        # format into batch dense representation
        curvatures = per_point_curvatures[mask].view(B, N_orig, 1)

        # satifies edge-case where we may have two extremely-close (or repeated points)...
        if torch.any(torch.isnan(curvatures)):
            print("Setting %s NAN to 0" % torch.sum(torch.isnan(curvatures)).item())
            curvatures[torch.isnan(curvatures)] = 0
        # assert not torch.any(torch.isnan(curvatures))

        return curvatures

    def compute_intersections(self):
        """
        Note: current limitation is that we only allow 1 intersection per pair of curves!
        """
        # first compute distances between all pairs of line segments
        # self.points.size --> [1, 2048, 3]
        B, N = self.points.size(0), self.points.size(1)

        # compute distance matrix between all polyline edges
        all_edges = torch.stack([self.points[:, :-1, :], self.points[:, 1:, :]], dim=2)  # (B, N-1, 2, 3)
        edge_dists, edge_intersect_ts = line_seg_dists(all_edges)  # intersect_ts are (row, col)
        # edge_dists is (B, N-1, N-1)

        # get end-of-curve edges --> these don't actually exist
        assert torch.all((self.point2curve_idxs[:, 1:] - self.point2curve_idxs[:, :-1]) >= 0)
        invalid_edge_mask = self.point2curve_idxs[:, 1:] - self.point2curve_idxs[:, :-1] > 0  # (B, N-1)
        invalid_edge_mask = (invalid_edge_mask.view(B, N-1, 1) + invalid_edge_mask.view(B, 1, N-1)) > 0
        edge_dists[invalid_edge_mask] = self.INTERSECTION_DIST_THRESH*100

        # get self-curve edges --> we do not allow self-intersection
        edge_curveidxs = self.point2curve_idxs[:, :-1]
        same_curve_mask = (edge_curveidxs.view(B, N-1, 1) - edge_curveidxs.view(B, 1, N-1)) == 0
        edge_dists[same_curve_mask] = self.INTERSECTION_DIST_THRESH*100

        # format into pytorch-geometric batch
        edge2curve_idxs = self.point2curve_idxs[:, :-1]
        batch = torch.arange(B).repeat_interleave(N-1).to(self.device)
        max_curve_idx = torch.max(self.point2curve_idxs)
        edge2curveidx_glob = torch.unique(edge2curve_idxs.flatten() + batch*max_curve_idx, return_inverse=True)[1]
        edge_dists_glob = edge_dists.view(B*(N-1), N-1)
        min_edge_dists, min_edge_idxs = scatter_min(edge_dists_glob, edge2curveidx_glob, dim=0)
        min_edge_dists2, min_edge_idxs2 = scatter_min(min_edge_dists, edge2curveidx_glob, dim=1)
        # edge row represent a curve, the index represents the edge on the curve closest to the global col-edge

        # condense to a matrix of k-curves x k-curves to find which pairs of curves intersect
        intersection_mask = min_edge_dists2 < self.INTERSECTION_DIST_THRESH
        intersection_edge2_idxs = min_edge_idxs2[torch.where(intersection_mask)]
        intersection_edge1_idxs = min_edge_idxs[torch.where(intersection_mask)[0], intersection_edge2_idxs]  # 1D array of all edges of intersection
        intersection_batch = batch[intersection_edge1_idxs]

        # map intersecting edge to intersecting point
        edge1_times = edge_intersect_ts.view(B*(N-1), (N-1))[intersection_edge1_idxs, intersection_edge2_idxs]
        point1_offset = (edge1_times > 0.5).long()  # we add 0 if it's at starting point of edge, else add 1
        intersection_edge1_idxs += point1_offset

        edge2_times = edge_intersect_ts.view(B*(N-1), (N-1))[intersection_edge2_idxs, intersection_edge1_idxs]
        point2_offset = (edge2_times > 0.5).long()  # we add 0 if it's at starting point of edge, else add 1
        intersection_edge2_idxs += point2_offset

        # Finally, combine intersection 1 and intersection 2 indices!
        intersection_adjacency_list = torch.stack([intersection_edge1_idxs, intersection_edge2_idxs], dim=0)
        return intersection_adjacency_list, intersection_batch


    def anti_alias(self):
        aliased_points = self._anti_alias(self.points)
        aliased_normals = self._anti_alias(self.normals)
        # we shouldn't do UV pixel coordinates?
        return aliased_points, aliased_normals

    def _anti_alias(self, vals):
        dim = vals.size(2)
        curves, mask = batchdense2curvepadded(vals, self.point2curve_idxs)  # M x Nmax x 3; M x Nmax
        M, N_max = mask.size()
        curves = curves.permute(0, 2, 1).contiguous().view(M*dim, 1, N_max)  # M*3 x Nmax
        filter = torch.tensor(self.ANTI_ALIAS_KERNEL, dtype=torch.double).to(self.device).view(1, 1, -1)
        normalizer = mask.view(M, 1, N_max).repeat(1, dim, 1).double().view(M*dim, 1, N_max)

        # run 1D filter over curves
        curves = F.conv1d(curves, filter, padding='same')  # (M*3 x Nmax)
        normalization = F.conv1d(normalizer, filter, padding='same')  # (M*3 x Nmax), resolves endpoints
        curves /= normalization  # note: will create nans

        # format back into batch-dense
        B, N_orig = vals.size(0), vals.size(1)
        curves = curves.view(M, dim, N_max).permute(0, 2, 1).contiguous()
        new_vals = curves[mask].view(B, N_orig, dim)
        return new_vals

    def __len__(self):
        return self.points.size(0)

    def to(self, device):
        self.points = self.points.to(device)
        self.normals = self.normals.to(device)
        self.uv = self.uv.to(device)
        self.point2uvcurve_idxs = self.point2uvcurve_idxs.to(device)
        self.point2curve_idxs = self.point2curve_idxs.to(device)
        self.curvature = self.curvature.to(device)

        if self.intersections is not None:
            self.intersections = self.intersections.to(device)
            self.intersection2batch_idxs = self.intersection2batch_idxs.to(device)


def batchdense2curvepadded(values, index):
    """

    :param values: (B, N, dim) tensor
    :param index: (B, N) tensor, each batch contains indices
    :return: (n-curves, n-points, dim) tensor
    """
    B, N = values.size(0), values.size(1)
    ptr = torch.cumsum(torch.max(index, dim=1)[0] + 1, dim=0)  # size B tensor of num-curves-per-batch
    index_flat = index + ptr.unsqueeze(-1)
    values_flat, index_flat = values.view(B*N, -1), index_flat.flatten()

    # get per-curve padded version
    max_num_pnts = torch.max(torch.unique(index_flat, return_counts=True)[1])
    padded = group_first_k_values(values_flat, index_flat, max_num_pnts)
    return padded


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
    batch_initvals = torch.where(torch.cat([torch.ones(1).to(device), batch_sorted[1:] - batch_sorted[: -1]]))[0]

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


def variational_curvature(turning_angles, mask):
    """

    :param turning_angles: size (M, n-angles)
    :param mask: size (M, n-angles) with False indicating that it is not a valid angle
    :return:
    """
    curvatures = 2 * torch.sin(turning_angles/2)
    zeros = torch.zeros(turning_angles.size(0), 1).to(turning_angles)
    curvatures = torch.cat([zeros, curvatures, zeros], dim=1)
    # todo: fix me with fill-in curvature
    return curvatures


def line_seg_dists(line_segs):
    """
    From http://paulbourke.net/geometry/pointlineplane/
    Args:
        line_segs (torch.tensor): size (B, N, 2, 3) where we have N line segments, each with start-end points of dim xyz
    """
    B, N = line_segs.size(0), line_segs.size(1)

    # Set up line seg endpoints
    P1 = line_segs[:, :, 0, :]
    P2 = line_segs[:, :, 1, :]
    P3, P4 = P1.clone(), P2.clone()

    P2_minus_P1 = (P2 - P1).view(B, N, 1, 3).repeat(1, 1, N, 1).double()
    P4_minus_P3 = (P4 - P3).view(B, 1, N, 3).repeat(1, N, 1, 1).double()
    P1_minus_P3 = (P1.view(B, N, 1, 3) - P3.view(B, 1, N, 3)).double()

    # Set up intermediates
    D1343 = (P1_minus_P3 * P4_minus_P3).sum(dim=-1)
    D4321 = (P2_minus_P1 * P4_minus_P3).sum(dim=-1)
    D1321 = (P1_minus_P3 * P2_minus_P1).sum(dim=-1)
    D4343 = (P4_minus_P3 * P4_minus_P3).sum(dim=-1)
    D2121 = (P2_minus_P1 * P2_minus_P1).sum(dim=-1)
    denom = (D2121 * D4343 - D4321 * D4321)
    num = (D1343 * D4321 - D1321 * D4343)

    # find parallel lines (time of "intersection" can be any time, because any point is closest distance)
    parallel_mask = denom < 1e-8
    mua = num / denom
    mua[parallel_mask] = 0.0
    assert torch.all(torch.isfinite(mua))
    mua = torch.clip(mua, min=0, max=1)

    # compute time of intersection on line seg 1 (i.e. for rows)
    # mua = (D1343 * D4321 - D1321 * D4343) / (D2121 * D4343 - D4321 * D4321)  # B x N x N  # todo: denom is always zero due to P4=P2!
    P1_rep, P2_rep = P1.view(B, N, 1, 3).repeat(1, 1, N, 1), P2.view(B, N, 1, 3).repeat(1, 1, N, 1)
    closest_pnts = P1_rep + mua.unsqueeze(-1) * (P2_rep - P1_rep)
    dists = (closest_pnts - closest_pnts.transpose(1, 2)).norm(dim=-1)  # BxNxNx3 --> BxNxN

    return dists, mua








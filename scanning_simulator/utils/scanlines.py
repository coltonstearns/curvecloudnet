import numpy as np
import torch

class ScanLineGenerator:
    UV_BASIS = np.array([[1, 0], [0, 1]])
    NLINES = 100
    MAX_WHILE_ATTEMPTS = 400

    def __init__(self, resolution, device, style='linear', direction='random', line_density=1.0):
        self.style = style
        self.direction = direction
        self.resolution = resolution
        self.line_density = line_density
        self.device = device
        self.borders = np.array([[0, 0], [0, 0], [0, resolution - 1], [resolution - 1, 0]])
        self.ts = int(2**0.5 * self.resolution * self.line_density)

    def generate_scan_lines(self, B, N, masks):
        if self.style == 'linear':
            return self.generate_linear_scan_lines(B, N, masks)
        elif self.style == 'sine':
            raise NotImplementedError("Have not implemented sine scanning")

    def generate_linear_scan_lines(self, B, N, masks):
        uv_idxs, npoints_per_scan = torch.zeros(B, 0, self.ts, 2).to(self.device), torch.zeros(B).to(self.device)
        idxs_mask = torch.zeros((B, 0, self.ts), dtype=torch.bool).to(self.device)
        num_attempts = 0
        while torch.any(npoints_per_scan < N).item():  # fix to get exactly N lines with points
            # sample line endpoints along the image border
            if self.direction == "random":
                border_points = self._sample_linear_border_points(self.NLINES, B)
            elif self.direction == "parallel":
                border_points = self._sample_parallel_border_points(num_attempts, self.NLINES, B)
            elif self.direction == "grid":
                border_points = self._sample_grid_border_points(num_attempts, self.NLINES, B)
            else:
                raise RuntimeError("Sampling direction must be parallel, grid, or random!")

            # get line indices
            pixel_locs = self.sample_line_points(border_points)  # (N*B, ts, 2)

            # remove duplicate point samples
            duplicates_mask = torch.zeros(B*self.NLINES*self.ts).bool().to(self.device)
            batch_idxs = torch.arange(B).reshape(-1, 1).to(self.device) * torch.ones((1, self.NLINES*self.ts)).to(self.device)
            batch_idxs = batch_idxs.view(B*self.NLINES, self.ts, 1)
            curve_idxs = torch.arange(B * self.NLINES).view(B, -1, 1).to(self.device) * torch.ones(1, 1, self.ts).to(self.device)
            curve_idxs = curve_idxs.view(B*self.NLINES, self.ts, 1)
            pixel_locs_wbatch = torch.cat([pixel_locs, batch_idxs, curve_idxs], dim=2)
            _, unique_idxs = np.unique(pixel_locs_wbatch.cpu().data.numpy().reshape(-1, 4), return_index=True, axis=0)
            duplicates_mask[unique_idxs] = True

            # find out which 'scan lines' have points on them
            pixel_locs = pixel_locs.view(B, -1, 2)
            pixel_locs_1d = pixel_locs[:, :, 0] * self.resolution + pixel_locs[:, :, 1]
            tmp_idxs = torch.arange(B).repeat_interleave(self.NLINES * self.ts)
            pixel_locs_mask = masks.view(B, -1)[tmp_idxs, pixel_locs_1d.flatten()].view(B, self.NLINES*self.ts)
            pixel_locs_mask &= duplicates_mask.view(B, self.NLINES*self.ts)

            # perform check on if we've reached enough points
            num_additional_pts = torch.sum(pixel_locs_mask, dim=1)
            new_pnts_per_scan = npoints_per_scan + num_additional_pts
            enough_idxs = torch.where(new_pnts_per_scan > N)[0]
            for i in enough_idxs:
                delta_n = int((N - npoints_per_scan[i]).item())
                remove_idxs = torch.where(pixel_locs_mask[i])[0][delta_n:]
                pixel_locs_mask[i, remove_idxs] = False

            # log our pixel locations
            npoints_per_scan += torch.sum(pixel_locs_mask, dim=1)
            uv_idxs = torch.cat([uv_idxs, pixel_locs.view(B, self.NLINES, self.ts, 2)], dim=1)
            idxs_mask = torch.cat([idxs_mask, pixel_locs_mask.view(B, self.NLINES, self.ts)], dim=1)

            num_attempts += 1
            if num_attempts > self.MAX_WHILE_ATTEMPTS:
                print("Reached maximum line-sampling attempts of %s" % self.MAX_WHILE_ATTEMPTS)
                print("Total number of possible points: %s" % torch.sum(masks.view(B, -1), dim=1))
                return False, None

        # initialize curve and batch indices
        curve_idxs = torch.arange(B * uv_idxs.size(1)).view(B, -1, 1).to(self.device) * torch.ones(1, 1, self.ts).to(self.device)

        # remove all redundant points indices
        uv_idxs = uv_idxs[idxs_mask]
        curve_idxs = curve_idxs[idxs_mask].unsqueeze(-1)

        # compress curve_idxs into range [0,n-curves]
        out = torch.cat([uv_idxs, curve_idxs], dim=1).view(B, N, 3)
        for i in range(B):
            out[i, :, 2] = torch.unique(out[i, :, 2], return_inverse=True)[1]

        return True, out.long()


    def _sample_linear_border_points(self, N, B):
        border_idxs = np.array([np.random.choice(np.arange(4), size=2, replace=False) for i in range(N * B)])  # n_curves x 2
        border_directions = self.UV_BASIS[border_idxs.flatten() % 2].reshape(-1, 2, 2)
        border_points = self.borders[border_idxs.flatten()].reshape(-1, 2, 2)  # creates n_curves x 2-endpoints x 2-dims
        border_offsets = np.random.randint(0, high=self.resolution, size=border_points.shape)
        border_points += border_directions * border_offsets
        border_points = torch.tensor(border_points).to(self.device)
        return border_points

    def _sample_parallel_border_points(self, num_attempts, N, B):
        # compute binary-search locations
        num_bins = int(np.ceil(np.log2((num_attempts+1)*N)))
        N_full = 2**num_bins
        idxs = np.arange(num_attempts*N_full, (num_attempts+1)*N_full) + 1
        bin_nums = ((idxs.reshape(-1, 1) & (2 ** np.arange(num_bins)).astype(np.int)) != 0).astype(int)
        factors = np.ones(num_bins) * 2**(-np.arange(1, num_bins+1).astype(float))
        locs = np.sum(bin_nums * factors.reshape(1, -1), axis=1)
        locs = locs[num_attempts*N: (num_attempts+1)*N]
        locs *= self.resolution
        border_pnts = np.stack([locs, np.zeros(N), locs, np.ones(N) * self.resolution - 1], axis=1).reshape((N, 2, 2))
        border_pnts = np.repeat(border_pnts, B, axis=0)
        return torch.tensor(border_pnts).to(self.device)

    def _sample_grid_border_points(self, num_attempts, N, B):
        assert (N % 2) == 0
        N = N // 2
        # compute binary-search locations
        num_bins = int(np.ceil(np.log2((num_attempts+1)*N)))
        N_full = 2**num_bins
        idxs = np.arange(num_attempts*N_full, (num_attempts+1)*N_full) + 1
        bin_nums = ((idxs.reshape(-1, 1) & (2 ** np.arange(num_bins)).astype(np.int)) != 0).astype(int)
        factors = np.ones(num_bins) * 2**(-np.arange(1, num_bins+1).astype(float))
        locs = np.sum(bin_nums * factors.reshape(1, -1), axis=1)
        locs = locs[num_attempts*N: (num_attempts+1)*N]
        locs *= self.resolution
        border_pnts_horizontal = np.stack([locs, np.zeros(N), locs, np.ones(N) * self.resolution - 1], axis=1).reshape((N, 2, 2))
        border_pnts_vertical = np.stack([np.zeros(N), locs, np.ones(N) * self.resolution - 1, locs], axis=1).reshape((N, 2, 2))
        border_pnts = np.stack([border_pnts_horizontal, border_pnts_vertical], axis=1).reshape((N*2, 2, 2))
        border_pnts = np.repeat(border_pnts, B, axis=0)
        return torch.tensor(border_pnts).to(self.device)

    def sample_line_points(self, border_points):
        """
        Performs brute-force sampling on a batch of lines defined by their endpoints.
        :param border_points (np.ndarray): Size (n, 2, 2) array of n end-point pairs, each with an (x,y) location
        :return:
        """
        ts = (torch.arange(self.ts) / int(self.ts)).to(self.device)
        rand_offset = torch.rand(1).to(self.device) / self.ts
        ts += rand_offset
        points = (border_points[:, 1:2, :] - border_points[:, 0:1, :]) * ts.reshape(1, -1, 1) + border_points[:, 0:1, :]
        return points.long()




    def generate_sine_scan_lines(self):
        pass
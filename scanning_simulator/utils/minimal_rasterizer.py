import minimal_pytorch_rasterizer as mpr
import torch
import cv2
import numpy as np
import time
import plotly.graph_objects as go
import os.path as osp

dtype = torch.float32
device = torch.device('cuda:0')
I = 0


def rasterize(vertices, faces, R, t, res=1024, viz_outdir=None):
    vertices = vertices[:, [0, 1, 2]] @ R.T
    vertices += t.view(1, 3)

    pinhole2d = mpr.Pinhole2D(
        fx=res, fy=res,
        cx=res//2, cy=res//2,
        w=res, h=res,
    )

    # z_buffer = mpr.project_mesh(
    #     vertices=vertices.float(),
    #     faces=faces.int(),
    #     vertice_values=vertices[:, [2]].float(),  # take z coordinate as values
    #     pinhole=pinhole2d
    # )
    # vis_z_buffer_cpu = mpr.vis_z_buffer(z_buffer)
    # cv2.imwrite('./out/depth_%s.png' % I, vis_z_buffer_cpu)
    # I += 1

    coords, normals = mpr.estimate_normals(
        vertices=vertices.float(),
        faces=faces.int(),
        pinhole=pinhole2d
    )

    if viz_outdir is not None:
        global I
        vis_normals_cpu = mpr.vis_normals(coords, normals)
        cv2.imwrite(osp.join(viz_outdir, 'normals_%s.png' % I), vis_normals_cpu)
        I += 1

    # transform back into scene reference frame
    mask = coords[:, :, 2] > 0
    coords = (coords.view(-1, 3) - t.view(1, 3)) @ R
    normals = normals.view(-1, 3) @ R
    coords, normals = coords.view(res, res, 3), normals.view(res, res, 3)

    return coords, normals, mask
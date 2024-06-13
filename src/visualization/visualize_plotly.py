from matplotlib import pyplot as plt
import plotly.graph_objects as go
import numpy as np
import wandb
import torch
import plotly

UNIT_BOX_CORNERS_1 = np.array([[0,0,0],[0,1,0],[1,1,0], [1,0,0], [0,0,0]]) - 0.5
UNIT_BOX_CORNERS_2 = np.array([[0,0,1],[0,1,1],[1,1,1], [1,0,1], [0,0,1]]) - 0.5
UNIT_BOX_CORNERS_3 = np.array([[0,0,0],[0,1,0],[0,1,1], [0,0,1], [0,0,0]]) - 0.5
UNIT_BOX_CORNERS_4 = np.array([[1,0,0],[1,1,0],[1,1,1], [1,0,1], [1,0,0]]) - 0.5
UNIT_BOX_CORNERS = [UNIT_BOX_CORNERS_1, UNIT_BOX_CORNERS_2, UNIT_BOX_CORNERS_3, UNIT_BOX_CORNERS_4]


# ==========================================================
# ===================== PLOTLY HELPERS =====================
# ==========================================================

def generate_plotly_cone_fig(pc, vecs, title, size=[-0.5, 0.5]):
    fig = go.Figure(data =[go.Cone(x=pc[:, 0], y=pc[:, 1], z=pc[:, 2], u=vecs[:, 0], v=vecs[:, 1], w=vecs[:, 2],
                                   sizemode="scaled", sizeref=10.0)])

    fig.update_layout(title_text=title,
                      scene=dict(
                          xaxis=dict(range=size,
                                 backgroundcolor="rgba(0, 0, 0,0)",
                                 gridcolor="white",
                                 showbackground=True,
                                 zerolinecolor="white",
                                 visible=False),
                          yaxis=dict(range=size,
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     ),
                          zaxis=dict(range=size,
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     )
                      ),
                      scene_aspectmode='cube',
                      )
    return fig


def generate_plotly_pc_fig(pc, clrs, title, size=[-0.5, 0.5], fig=None, row=None, col=None, markersize=None):
    """
    pc: numpy array of size Nx3
    clrs: numpy array or list of size N x 3
    """
    clrs = [f'rgb({int(clrs[i][0])}, {int(clrs[i][1])}, {int(clrs[i][2])})' for i in range(clrs.shape[0])]

    # create plotly visualization of curves
    fig = go.Figure(data =[go.Scatter3d(x = pc[:, 0],
                                        y = pc[:, 1],
                                        z = pc[:, 2],
                                        mode ='markers',
                                        marker=dict(color=clrs, size=2 if markersize is None else markersize, opacity=1, line=dict(width=0)))])
    fig.update_layout(title_text=title,
                      scene=dict(
                          xaxis=dict(range=size,
                                 backgroundcolor="rgba(0, 0, 0,0)",
                                 gridcolor="white",
                                 showbackground=True,
                                 zerolinecolor="white",
                                 visible=False),
                          yaxis=dict(range=size,
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     ),
                          zaxis=dict(range=size,
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     )
                      ),
                      scene_aspectmode='cube',
                      )

    return fig


def append_plotly_pc_fig(pc, clrs, fig, row, col, markersize=None):
    clrs = [f'rgb({int(clrs[i][0])}, {int(clrs[i][1])}, {int(clrs[i][2])})' for i in range(clrs.shape[0])]
    fig.add_trace(
        go.Scatter3d(x=pc[:, 0],
                     y=pc[:, 1],
                     z=pc[:, 2],
                     mode='markers',
                     marker=dict(color=clrs, size=2 if markersize is None else markersize, opacity=1, line=dict(width=0))),
        row=row, col=col
    )
    fig.update_layout(scene_aspectmode='cube')


def generate_plotly_nocs_fig(pc, clrs, title):
    # create plotly visualization of curves
    scatter = go.Scatter3d(x = pc[:, 0],
                           y = pc[:, 1],
                           z = pc[:, 2],
                           mode ='markers',
                           marker=dict(color=clrs, size=2))

    # add in NOCS-box lines
    Xe, Ye, Ze = [], [], []
    for b in UNIT_BOX_CORNERS:
        Xe.extend([b[k][0] for k in range(5)] + [None])
        Ye.extend([b[k][1] for k in range(5)] + [None])
        Ze.extend([b[k][2] for k in range(5)] + [None])
        Xe.extend([b[k][0] + 1.2 for k in range(5)] + [None])
        Ye.extend([b[k][1] for k in range(5)] + [None])
        Ze.extend([b[k][2] for k in range(5)] + [None])
    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color='rgb(70,70,70)', width=1))

    # Update layout
    layout = go.Layout(title=title,
                      scene=dict(
                          xaxis=dict(range=[-2.5, 2.5],
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False),
                          yaxis=dict(range=[-2.5, 2.5],
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     ),
                          zaxis=dict(range=[-2.5, 2.5],
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     ),
                          aspectmode="cube"
                          ),
                       )

    fig = go.Figure(data=[scatter, lines], layout=layout)
    return fig


def viz_points_plotly(pc, clr_gradient, title, cmap_name="plasma", size=[-0.5, 0.5], log_wandb=False):
    pc, clr_gradient = pc.to('cpu').data.numpy(), clr_gradient.to('cpu').data.numpy()
    clr_gradient -= np.min(clr_gradient)
    clr_gradient /= np.max(clr_gradient)
    cmap = plt.get_cmap(cmap_name)
    clrs = cmap(clr_gradient)[:, :3] * 255
    if not log_wandb:
        fig = generate_plotly_pc_fig(pc, clrs, title, size=size)
        return fig
    else:
        wandb_pc = np.concatenate([pc, clrs], axis=1)
        wandb.log(
            {
                title: wandb.Object3D(
                    {
                        "type": "lidar/beta",
                        "points": wandb_pc,
                        "boxes": np.array([])
                    }
                )
            })
        return None

# ==========================================================
# ==========================================================


# ==========================================================
# ========= PLOTLY TASK SPECIFIC VISUALIZATIONS ============
# ==========================================================


def visualize_normals(pc, normals, errs, max_err=0.2):
    # Plot actual normal vectors as cones
    pc, normals, errs = pc.to('cpu').data.numpy(), normals.to('cpu').data.numpy(), errs.to('cpu').data.numpy()
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    cone_fig = generate_plotly_cone_fig(pc, normals, "Predicted Normal Vectors")
    wandb.log({"Predicted Normal Vectors": cone_fig})

    # Plot normal errors
    err_clrs = np.clip(errs / max_err, 0, 1)
    cmap = plt.get_cmap("plasma")
    clrs = cmap(err_clrs)[:, :3] * 255
    error_fig = generate_plotly_pc_fig(pc, clrs, "Normal Estimation Error")
    wandb.log({"Normal Estimation Error": error_fig})


def visualize_keypoints(pc, keypoint_mask, title, size=[-0.5, 0.5]):
    if torch.is_tensor(pc):
        pc = pc.to('cpu').data.numpy()
    if torch.is_tensor(keypoint_mask):
        keypoint_mask = keypoint_mask.to('cpu').data.numpy()

    # get per-cone colors based on error
    N = pc.shape[0]
    clrs = np.array([[155, 155, 155]]).repeat(N, axis=0)
    clrs[keypoint_mask] = np.array([255, 0, 0])
    keypoint_sizes = np.ones((pc.shape[0], 1)) * 4
    keypoint_sizes[keypoint_mask] = 12

    # get plotly graph
    fig = generate_plotly_pc_fig(pc, clrs, title, markersize=keypoint_sizes, size=size)
    wandb.log({"Keypoint Visualization": fig})


def visualize_intersect_adjacency(pc, intersections, title):
    intersections = intersections.to('cpu').data.numpy()

    # get per-cone colors based on error
    N = pc.shape[0]
    clrs = np.array([[155, 155, 155]]).repeat(N, axis=0)

    # get plotly graph
    clrs = [f'rgb({int(clrs[i][0])}, {int(clrs[i][1])}, {int(clrs[i][2])})' for i in range(clrs.shape[0])]
    scatter = go.Scatter3d(x = pc[:, 0], y = pc[:, 1], z = pc[:, 2], mode ='markers', marker=dict(color=clrs, size=2, opacity=1, line=dict(width=0)))

    # add intesection lines
    Xe, Ye, Ze = [], [], []
    for l in range(intersections.shape[1]):
        Xe.extend([pc[intersections[0, l], 0], pc[intersections[1, l], 0], None])
        Ye.extend([pc[intersections[0, l], 1], pc[intersections[1, l], 1], None])
        Ze.extend([pc[intersections[0, l], 2], pc[intersections[1, l], 2], None])

    lines = go.Scatter3d(
        x=Xe,
        y=Ye,
        z=Ze,
        mode='lines',
        name='',
        line=dict(color='rgb(255,0,0)', width=10))

    # Update layout
    layout = go.Layout(title=title,
                      scene=dict(
                          xaxis=dict(range=[-2.5, 2.5],
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False),
                          yaxis=dict(range=[-2.5, 2.5],
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     ),
                          zaxis=dict(range=[-2.5, 2.5],
                                     backgroundcolor="rgba(0, 0, 0,0)",
                                     gridcolor="white",
                                     showbackground=True,
                                     zerolinecolor="white",
                                     visible=False
                                     ),
                          aspectmode="cube"
                          ),
                       )

    fig = go.Figure(data=[scatter, lines], layout=layout)
    wandb.log({"Keypoint Line Visualization": fig})


def visualize_pointnet2_groupings(row, col, pos, batch, size=[-0.5, 0.5], log_wandb=False):
    viz_points = pos.cpu()[col]
    viz_batch = batch.cpu()[col]
    colors = row.clone().cpu()
    for i in range(torch.max(batch.cpu())+1):
        pnts = viz_points.cpu()[viz_batch == i]
        clrs = colors[viz_batch == i]
        clrs -= torch.min(clrs)
        _, clrs = torch.unique(clrs, return_inverse=True)
        clrs = (clrs / torch.max(clrs)).flatten()
        clrs = (clrs * 1117) % 1  # adds randomness to color scheme

        # get unique indices
        this_viz_points, viz_idxs = np.unique(pnts.numpy(), return_index=True, axis=0)
        viz_clrs = clrs[viz_idxs]
        this_viz_points = torch.from_numpy(this_viz_points)
        # viz_clrs = torch.from_numpy(viz_clrs)

        # NOTE: if there are too many points, this will fail!
        fig = viz_points_plotly(this_viz_points, viz_clrs, "PointNet++ Groupings", cmap_name="tab20", size=size, log_wandb=log_wandb)
        if not log_wandb:
            wandb.log({"Curve Groupings": fig})


def visualize_nocs(pc, pred_nocs, gt_nocs, errs, max_err=0.1):
    pc, pred_nocs, gt_nocs, errs = pc.to('cpu').data.numpy(), pred_nocs.to('cpu').data.numpy(), gt_nocs.to('cpu').data.numpy(), errs.to('cpu').data.numpy()

    # get PC with everything
    pc += np.array([[-1.2, 0, 0]])
    gt_nocs += np.array([[1.2, 0, 0]])
    pc_viz = np.concatenate([pc, pred_nocs, gt_nocs], axis=0)

    # get colors for each pc
    err_cmap = plt.get_cmap('cool')
    errs /= max_err
    clrs = err_cmap(errs)[:, :3] * 255
    clrs_pc = np.ones((pc.shape[0], 3)) * 155
    clrs_nocs = (gt_nocs[:, :3] + np.array([[-1.2 + 0.5, 0.5, 0.5]])) * 255
    clrs_nocs = np.clip(clrs_nocs, 0, 255)
    clrs_viz = np.concatenate([clrs_pc, clrs, clrs_nocs])

    # generate figure
    fig = generate_plotly_nocs_fig(pc_viz, clrs_viz, "NOCS Prediction Error")
    wandb.log({"NOCS L1 Error Vis": fig})
    # fig.show()


def visualize_seg(pc, pred, gt, errs, correct_mask, title="Segmentation Vis", max_err=1.0, max_classes=5, range=1):
    pc, pred_labels, gt_labels, errs, correct_mask = pc.to('cpu').data.numpy(), pred.to('cpu').data.numpy(), gt.to('cpu').data.numpy(), errs.to('cpu').data.numpy(), correct_mask.to('cpu').data.numpy()

    # get colors
    cmap = plt.get_cmap("tab20")
    pred_clrs = cmap(pred_labels / max_classes)[:, :3] * 255
    gt_clrs = cmap(gt_labels / max_classes)[:, :3] * 255
    mask_clrs = np.ones((pc.shape[0], 3)) * 0.55 * 255
    mask_clrs[~correct_mask] = np.array([255, 0, 0])
    err_cmap = plt.get_cmap('coolwarm')
    errs /= max_err
    err_clrs = err_cmap(errs)[:, :3] * 255

    # Plot 4 segmentation color schemes
    # fig = go.Figure()

    fig = plotly.tools.make_subplots(rows=2, cols=2,
                                     subplot_titles=("Predicted Seg.", "GT Seg.", "Prediction Err", "Err Mask"),
                                     specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                                            [{'type': 'scatter3d'}, {'type': 'scatter3d'}]]
                                     )
    append_plotly_pc_fig(pc, pred_clrs, fig, 1, 1)
    append_plotly_pc_fig(pc, gt_clrs, fig, 1, 2)
    append_plotly_pc_fig(pc, err_clrs, fig, 2, 1)
    append_plotly_pc_fig(pc, mask_clrs, fig, 2, 2)

    # Update layout
    fig.update_scenes(xaxis_range=(-range, range),
                      yaxis_range=(-range, range),
                      zaxis_range=(-range, range))
    fig.update_layout(title_text=title,
                      # autosize = False,
                      height=1200,
                      width=1200,
                      margin=go.layout.Margin(
                          l=0,  # left margin
                          r=0,  # right margin
                          b=0,  # bottom margin
                          t=0,  # top margin
                      ),
                      scene_aspectmode='cube',
                      )
    wandb.log({"Segmentation Results": fig})

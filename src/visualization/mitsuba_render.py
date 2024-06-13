# Import the library using the alias "mi"
import mitsuba as mi
import numpy as np
import matplotlib.pyplot as plt
import time

# Set default CPU variant for scene loading
# mi.set_variant('cuda_ad_rgb', 'scalar_rgb')
mi.set_variant('scalar_rgb')

# Set up global transformation matrices
CAMERA_AFFINE = mi.Transform4f.look_at(origin=(2.2, 2.2, 2.2), target=(0, 0, 0), up=(0, 0, 1))
LIGHT_AFFINE = np.array(mi.Transform4f.look_at(origin=(-1.5, 1.5, 5), target=(0, 0, 0), up=(0, 0, 1)).matrix)
LIGHT_AFFINE[0, :3] *= 3.2
LIGHT_AFFINE[1, :3] *= 3.2
LIGHT_AFFINE[2, :3] *= 1
LIGHT_AFFINE = mi.Transform4f(LIGHT_AFFINE.tolist())
BACKGROUND_AFFINE = mi.Transform4f([[30, 0, 0, 0],
                                     [0, 30, 0, 0],
                                     [0, 0, 1, 0.0],
                                     [0, 0, 0, 1]])
LIGHT_AFFINE_2 = np.array(mi.Transform4f.look_at(origin=(-4, -4, 20), target=(0, 0, 0), up=(0, 0, 1)).matrix)
LIGHT_AFFINE_2[0, :3] *= 5
LIGHT_AFFINE_2[1, :3] *= 5
LIGHT_AFFINE_2[2, :3] *= 1
LIGHT_AFFINE_2 = mi.Transform4f(LIGHT_AFFINE_2.tolist())


# Set up scene without point cloud in it
SCENE_BASE = \
{
    'type': 'scene',
    'integrator': {'type': 'path', 'max_depth': -1},
    'sensor': {
        'type': 'perspective',
        'near_clip': 0.001,
        'far_clip': 100.0,
        'fov': 25,
        'to_world':  CAMERA_AFFINE,
        'sampler': {'type': 'ldsampler', 'sample_count': 256},
        'film': {
            'type': 'hdrfilm',
            # "crop_offset_x": 512,
            # "crop_offset_y": 512,
            # "crop_width": 512,
            # "crop_height": 512,
            'width': 2000,
            'height': 1700,
            'rfilter': {'type': 'gaussian'},
            'pixel_format': 'rgb',
            'component_format': 'float32'}
    },

    'white': {
        'type': 'diffuse',
        'reflectance': {'type': 'rgb', 'value': [0.885809, 0.698859, 0.666422]}
    },

    'bsdf': {
        'type': 'roughplastic',
        'id': 'surfaceMaterial',
        'distribution': 'ggx',
        'alpha': 0.05,
        # 'reflectance':  {'type': 'rgb', 'value': [1, 1, 1]}
    },

    "background": {
        "type": "rectangle",
        "to_world": BACKGROUND_AFFINE,
        'bsdf': {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [0.95, 0.95, 0.95]}
        }
    },

}

SCENE_TAIL = {
    'light': {
        'type': 'rectangle',
        'to_world': LIGHT_AFFINE,
        'bsdf': {'type': 'ref', 'id': 'white'},
        'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [4.0, 4.0, 4.0]}}
    },

    # 'light2': {
    #     'type': 'rectangle',
    #     'to_world': LIGHT_AFFINE_2,
    #     'bsdf': {'type': 'ref', 'id': 'white'},
    #     'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [4, 4, 4]}}
    # },

}

NOCS_BOX_COORDS = [[(0, 0, 0), (0, 1, 0)],
                   [(0, 1, 0), (1, 1, 0)],
                   [(1, 1, 0), (1, 0, 0)],
                   [(1, 0, 0), (0, 0, 0)],
                   [(0, 0, 1), (0, 1, 1)],
                   [(0, 1, 1), (1, 1, 1)],
                   [(1, 1, 1), (1, 0, 1)],
                   [(1, 0, 1), (0, 0, 1)],
                   [(0, 0, 0), (0, 0, 1)],
                   [(1, 0, 0), (1, 0, 1)],
                   [(0, 1, 0), (0, 1, 1)],
                   [(1, 1, 0), (1, 1, 1)]]


def get_mitsuba_cylinder(p0, p1, radius=0.012):
    return \
    {
        'type': 'cylinder',
        'p0': p0,
        'p1': p1,
        'radius': radius,
        # 'bsdf': {
        #     'type': 'diffuse',
        #     'reflectance': {'type': 'rgb', 'value': [0.8, 0.8, 0.8]}
        # },
        'bsdf': {
            'type': 'dielectric',
        }

    }


def get_mitsuba_point(x, y, z, r, g, b, radius=0.005):
    return \
    {
        "type": "sphere",
        "radius": radius,
        "to_world":  mi.Transform4f([[1, 0, 0, x],
                                     [0, 1, 0, y],
                                     [0, 0, 1, z],
                                     [0, 0, 0, 1]]),
        "bsdf": {
            'type': 'diffuse',
            'reflectance': {'type': 'rgb', 'value': [r, g, b]}
        }
    }


def render_pc_kitti(pc, clrs, ax=None, scene_dict=None, point_radius=0.005):
    # rescale point cloud into rough unit cube
    pc = pc / 40
    pc = pc + np.array([[0.4, 0.35, 0.3]])

    SCENE_BASE['background']['to_world'] = mi.Transform4f([[30, 0, 0, 0],
                                                            [0, 30, 0, 0],
                                                            [0, 0, 1, 0.725],
                                                            [0, 0, 0, 1]])

    # shapenet scene params
    kitti_ego_camera = mi.Transform4f.look_at(origin=(0.1, 0.35, 0.2), target=(1.4, 0.35, 0.0), up=(0, 0, 1))
    sample_count = 16
    im_width_res = 5000
    im_height_res = 4000
    fov = 50

    # render standard scene and egocentric scene
    im_scene = render_pc_shapenet(pc, clrs, ax, scene_dict, point_radius, width=2500, height=2500, sample_count=sample_count, camera_origin=(1.8, 1.8, 2.2))
    # im_egocentric = render_pc(pc, clrs, ax, scene_dict, point_radius, kitti_ego_camera, sample_count, im_width_res, im_height_res, fov)
    return im_scene


def render_pc_audi(pc, clrs, ax=None, scene_dict=None, point_radius=0.005):
    # reset global Lighting to better suit audi data
    global LIGHT_AFFINE
    LIGHT_AFFINE = np.array(mi.Transform4f.look_at(origin=(-4, 4, 20), target=(0, 0, 0), up=(0, 0, 1)).matrix)
    LIGHT_AFFINE[0, :3] *= 5
    LIGHT_AFFINE[1, :3] *= 5
    LIGHT_AFFINE[2, :3] *= 1
    LIGHT_AFFINE = mi.Transform4f(LIGHT_AFFINE.tolist())
    global SCENE_TAIL
    SCENE_TAIL = {
        'light': {
            'type': 'rectangle',
            'to_world': LIGHT_AFFINE,
            'bsdf': {'type': 'ref', 'id': 'white'},
            'emitter': {'type': 'area', 'radiance': {'type': 'rgb', 'value': [17, 17, 17]}}
        }
    }

    # rescale point cloud into rough unit cube
    pc = pc / 30
    angle = -2.6*np.pi/4.6
    angle2 = 0.03
    rot_z = np.array([[np.cos(angle), np.sin(angle), 0],
                      [-np.sin(angle), np.cos(angle), 0],
                      [0, 0, 1]])
    rot_x = np.array([[1, 0, 0],
                      [0, np.cos(angle2), np.sin(angle2)],
                      [0, -np.sin(angle2), np.cos(angle2)]])
    pc = pc @ rot_z @ rot_x
    pc = pc + np.array([[0.3, 0.9, 0.14]])
    # clrs *= 0
    # clrs += 0.5

    # shapenet scene params
    # kitti_ego_camera = mi.Transform4f.look_at(origin=(0.1, 0.35, 0.45), target=(1.4, 0.35, 0.35), up=(0, 0, 1))
    kitti_ego_camera = mi.Transform4f.look_at(origin=(2.2, 2.2, 2.2), target=(0, 0, 0), up=(0, 0, 1))
    sample_count = 64
    im_width_res = 2000
    im_height_res = 2000
    fov = 20

    # render standard scene and egocentric scene
    # im_scene = render_pc_shapenet(pc, clrs, ax, scene_dict, point_radius, width=1000, height=800, sample_count=sample_count)
    im_egocentric = render_pc(pc, clrs, ax, scene_dict, point_radius, kitti_ego_camera, sample_count, im_width_res, im_height_res, fov)
    return im_egocentric


def render_pc_shapenet(pc, clrs, ax=None, scene_dict=None, point_radius=0.005, width=1200, height=1000, sample_count=256, camera_origin=(2.2, 2.2, 2.2), is_kortx=False):
    # shapenet scene params
    if is_kortx:
        shapenet_camera = mi.Transform4f.look_at(origin=(2.2, 2.2, 2.2), target=(0, 0, 0), up=(0, 0, 1))
    else:
        shapenet_camera = mi.Transform4f.look_at(origin=camera_origin, target=(0, 0, 0.5), up=(0, 0, 1))
    im_width_res = width
    im_height_res = height

    return render_pc(pc, clrs, ax, scene_dict, point_radius, shapenet_camera, sample_count, im_width_res, im_height_res)


def render_pc(pc, clrs, ax=None, scene_dict=None, point_radius=0.005, camera_transform=None, sample_count=256, width=2000, height=1700, fov=25):
    # build scene with ShapeNet params
    if scene_dict is None:
        scene_dict = SCENE_BASE.copy()
    scene_dict['sensor']['to_world'] = camera_transform
    scene_dict['sensor']['sampler']['sample_count'] = sample_count
    scene_dict['sensor']['film']['width'] = width
    scene_dict['sensor']['film']['height'] = height
    scene_dict['sensor']['fov'] = fov

    # Add in point cloud
    N = pc.shape[0]
    for i in range(N):
        pnt_dict = get_mitsuba_point(pc[i, 0], pc[i, 1], pc[i, 2], clrs[i, 0], clrs[i, 1], clrs[i, 2], radius=point_radius)
        scene_dict.update({"point-{0}".format(i): pnt_dict})

    # update with scene tail
    scene_dict.update(SCENE_TAIL.copy())

    # render scene
    # mi.set_variant('cuda_ad_rgb', 'scalar_rgb')  # UNCOMMENT TO USE CUDA ACCELERATION!
    pc_scene = mi.load_dict(scene_dict)
    img = mi.render(pc_scene)

    if ax is not None:
        ax.imshow(img ** (1.0 / 2.2), interpolation='none')
    return img


def render_nocs(pc, clrs, ax=None):
    scene_dict = SCENE_BASE.copy()
    for i in range(len(NOCS_BOX_COORDS)):
        cylinder_dict = get_mitsuba_cylinder(NOCS_BOX_COORDS[i][0], NOCS_BOX_COORDS[i][1])
        scene_dict.update({"cylinder-{0}".format(i): cylinder_dict})
    return render_pc_shapenet(pc, clrs, ax, scene_dict)


def render_sphere_example():
    scene_dict = SCENE_BASE.copy()
    for i in range(len(NOCS_BOX_COORDS)):
        cylinder_dict = get_mitsuba_cylinder(NOCS_BOX_COORDS[i][0], NOCS_BOX_COORDS[i][1])
        scene_dict.update({"cylinder-{0}".format(i): cylinder_dict})

    pnts = np.random.randn(1000, 3)
    pnts = pnts / np.linalg.norm(pnts, axis=1).reshape(-1, 1)
    pnts = np.abs(pnts)
    clrs = np.random.rand(1000, 3)
    for i in range(1000):
        pnt_dict = get_mitsuba_point(pnts[i, 0], pnts[i, 1], pnts[i, 2], clrs[i, 0], clrs[i, 1], clrs[i, 2])
        scene_dict.update({"point-{0}".format(i): pnt_dict})

    # update with scene tail
    scene_dict.update(SCENE_TAIL.copy())

    t0 = time.time()
    # try:
    #     mi.set_variant('cuda_ad_rgb')  # todo: figure out cuda version
    #     pc_scene = mi.cuda_ad_rgb.load_dict(scene_dict)
    #     img = mi.cuda_ad_rgb.render(pc_scene)
    #     mi.set_variant('scalar_rgb')  # todo: figure out cuda version
    # except:
    pc_scene = mi.scalar_rgb.load_dict(scene_dict)
    img = mi.scalar_rgb.render(pc_scene)
    t1 = time.time()
    print("Timing:")
    print(t1 - t0)

    # visualize in matplotlib
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.imshow(img ** (1.0 / 2.2), interpolation='none')
    plt.axis('off')
    plt.tight_layout()
    # plt.imshow(img ** (1.0 / 2.2), interpolation='none') # approximate sRGB tonemapping
    plt.show()
    # mi.Bitmap(img).write('points.exr')


if __name__ == "__main__":
    render_sphere_example()






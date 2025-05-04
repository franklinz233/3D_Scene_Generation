from .orbit import Orbit
from .spiral import Spiral
from .wobble import Wobble


def build_camera_path(cfg, scene, nframes=None):
    method = scene.traj_type.lower()
    nframe = cfg.scene.traj.n_sample*6 if nframes is None else nframes
    if method == 'rot':
        runner = Orbit(scene,nframe)
    elif method == 'wobble':
        runner = Wobble(scene,nframe)
    elif method == 'spiral':
        runner = Spiral(scene,nframe)
    else:
        raise ValueError('traj_type must be one of: rot, spiral, wobble')
    return runner()

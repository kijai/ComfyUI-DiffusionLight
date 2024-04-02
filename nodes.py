import torch
import numpy as np
import torch
import torch.nn.functional as F
from .relighting.tonemapper import TonemapHDR

def create_envmap_grid(size: int):
    """
    BLENDER CONVENSION
    Create the grid of environment map that contain the position in sperical coordinate
    Top left is (0,0) and bottom right is (pi/2, 2pi)
    """    
    theta = torch.linspace(0, np.pi * 2, size * 2)
    phi = torch.linspace(0, np.pi, size)
    
    #use indexing 'xy' torch match vision's homework 3
    theta, phi = torch.meshgrid(theta, phi ,indexing='xy') 
    
    theta_phi = torch.cat([theta[..., None], phi[..., None]], dim=-1)
    theta_phi = theta_phi.numpy()
    return theta_phi

def get_normal_vector(incoming_vector: np.ndarray, reflect_vector: np.ndarray):
    """
    BLENDER CONVENSION
    incoming_vector: the vector from the point to the camera
    reflect_vector: the vector from the point to the light source
    """
    #N = 2(R ⋅ I)R - I
    N = (incoming_vector + reflect_vector) / np.linalg.norm(incoming_vector + reflect_vector, axis=-1, keepdims=True)
    return N

def get_cartesian_from_spherical(theta: np.array, phi: np.array, r = 1.0):
    """
    BLENDER CONVENSION
    theta: vertical angle
    phi: horizontal angle
    r: radius
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.concatenate([x[...,None],y[...,None],z[...,None]], axis=-1)

class chrome_ball_to_envmap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "ball_images": ("IMAGE", ),
                "envmap_height": ("INT", {"default": 256, "min": 1, "max": 2048, "step": 1}, ),
                "scale": ("INT", {"default": 4, "min": 1, "max": 30, "step": 1}, ),
            },
        }
        
    CATEGORY = "DiffusionLight"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",  )

    FUNCTION = "process"

   
    def process(self, ball_images, envmap_height, scale):     
        I = np.array([1, 0, 0])

        # compute  normal map that create from reflect vector
        env_grid = create_envmap_grid(envmap_height * scale)   
        reflect_vec = get_cartesian_from_spherical(env_grid[...,1], env_grid[...,0])
        normal = get_normal_vector(I[None,None], reflect_vec)
        
        # turn from normal map to position to lookup [Range: 0,1]
        pos = (normal + 1.0) / 2
        pos  = 1.0 - pos
        pos = pos[...,1:]
        
        env_map = None
        # convert position to pytorch grid look up
        grid = torch.from_numpy(pos)[None].float()
        grid = grid * 2 - 1 # convert to range [-1,1]
        print(grid.shape)
        ball_images = ball_images.permute(0,3,1,2) # [1,3,H,W]

        env_maps_list = []
        for ball in ball_images:
            env_map = F.grid_sample(ball.unsqueeze(0), grid, mode='bilinear', padding_mode='border', align_corners=True)
            env_map_default = F.interpolate(env_map, size=(envmap_height, envmap_height*2), mode='bilinear', align_corners=True)
            env_map_default = env_map_default.permute(0,2,3,1).cpu().to(torch.float32)
            env_maps_list.append(env_map_default)
        env_maps_out = torch.cat(env_maps_list, dim=0)

        
 
        return env_maps_out,

class exposure_to_hdr:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                #"EV": ("FLOAT", {"default": 0, "min": 1, "max": 30, "step": 1}, ),
                "gamma": ("FLOAT", {"default": 2.4, "min": 1, "max": 30, "step": 0.01}, ),
            },
        }
        
    CATEGORY = "DiffusionLight"
    RETURN_TYPES = ("IMAGE", "IMAGE",)
    RETURN_NAMES = ("hrd_image", "ldr_image", )

    FUNCTION = "exposuretohdr"

    def exposuretohdr(self, images, gamma):     
        first_image = torch.pow(images[0], gamma)
        evs = [0.0, -2.5, -5.0]
        hdr2ldr = TonemapHDR(gamma=gamma, percentile=99, max_mapping=0.9)
        scaler = torch.tensor([0.212671, 0.715160, 0.072169])
        
        # read luminace for every image
        luminances = []
        for i in range(len(evs)):
            linear_img = torch.pow(images[i], gamma)
            linear_img = linear_img * 1 / (2** evs[i])
            # compute luminace
            lumi = linear_img @ scaler
            luminances.append(lumi)

        # start from darkest image
        out_luminace = luminances[len(evs) - 1]
        for i in range(len(evs) - 1, 0, -1):
            # compute mask
            maxval = 1 / (2 ** evs[i-1])
            p1 = torch.clip((luminances[i-1] - 0.9 * maxval) / (0.1 * maxval), 0, 1)
            p2 = out_luminace > luminances[i-1]
            mask = (p1 * p2)
            out_luminace = luminances[i-1] * (1-mask) + out_luminace * mask
        
        hdr_rgb = first_image * (out_luminace / (luminances[0] + 1e-10)).unsqueeze(-1)
        ldr_rgb, _, _ = hdr2ldr(hdr_rgb)

        hrd_rgb = hdr_rgb.unsqueeze(0).cpu().to(torch.float32)
        ldr_rgb = ldr_rgb.unsqueeze(0).cpu().to(torch.float32)

        return (hrd_rgb, ldr_rgb,)

NODE_CLASS_MAPPINGS = {
    "chrome_ball_to_envmap": chrome_ball_to_envmap,
    "exposure_to_hdr": exposure_to_hdr,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "chrome_ball_to_envmap": "Chrome Ball to Envmap",    
    "exposure_to_hdr": "Exposure to HDR",
}

import torch
import numpy as np
from PIL import Image
from typing import Any # Added import for Any


# Import from the local renderer directory
from .renderer.mesh_render import MeshRender
# Need comfy.model_management for device handling
import comfy.model_management as mm

class Hy3DTools_RenderSpecificView:
    @classmethod
    def INPUT_TYPES(s):
        # Changed input back to TRIMESH
        return {
            "required": {
                "trimesh": ("TRIMESH",), # Changed back from MESHRENDERA
                "render_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                # camera_type, ortho_scale, etc. are needed again to configure the renderer
                # but keeping them for now allows overriding the view settings.
                "camera_type": (["orth", "perspective"], {"default": "orth"}),
                "camera_distance": ("FLOAT", {"default": 1.45, "min": 0.1, "max": 10.0, "step": 0.001}),
                "pan_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "pan_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.001}),
                "azimuth": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "elevation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "bg_color_rgb": ("STRING", {"default": "128, 128, 255", "tooltip": "Background Color (RGB 0-255). Used for textured, world normal, and position views."}), # Corrected tooltip
            },
            "optional": {
                "camera_params_override": ("CAMERA_PARAMS",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE", "MASK", "TRIMESH", "CAMERA_PARAMS", "STRING",)
    RETURN_NAMES = ("textured_view", "tangent_normal", "world_normal", "position", "depth", "mask", "original_mesh_out", "camera_params", "detected_texture_size_WH",)
    FUNCTION = "render_view"
    CATEGORY = "Hunyuan3DTools" # New category for our tools

    def render_view(self, trimesh: Any, render_size: int, camera_type: str, ortho_scale: float, camera_distance: float, pan_x: float, pan_y: float, azimuth: float, elevation: float, bg_color_rgb: str, camera_params_override: dict = None):

        # Use override camera params if provided
        if camera_params_override is not None:
            camera_type = camera_params_override.get("camera_type", camera_type)
            ortho_scale = camera_params_override.get("ortho_scale", ortho_scale)
            camera_distance = camera_params_override.get("camera_distance", camera_distance)
            pan_x = camera_params_override.get("pan_x", pan_x)
            pan_y = camera_params_override.get("pan_y", pan_y)
            azimuth = camera_params_override.get("azimuth", azimuth)
            elevation = camera_params_override.get("elevation", elevation)

            
        # Parse background color again
        try:
            bg_color = [int(x.strip())/255.0 for x in bg_color_rgb.split(",")]
            if len(bg_color) != 3:
                raise ValueError("Background color must have 3 components (R, G, B)")
        except Exception as e:
            print(f"Error parsing background color '{bg_color_rgb}': {e}. Using default [0.5, 0.5, 1.0]")
            bg_color = [0.5, 0.5, 1.0] # Default blueish background

        # --- Get Original Texture Size ---
        # Use a copy to avoid modifying the input object
        mesh_copy_for_size_check = trimesh.copy()
        original_texture_size = (512, 512) # Default if no texture found
        device_for_check = mm.get_torch_device() # Need device for temp renderer
        try:
            # Attempt to access texture directly from the trimesh object
            if hasattr(mesh_copy_for_size_check, 'visual') and \
               hasattr(mesh_copy_for_size_check.visual, 'material') and \
               hasattr(mesh_copy_for_size_check.visual.material, 'image') and \
               mesh_copy_for_size_check.visual.material.image is not None and \
               isinstance(mesh_copy_for_size_check.visual.material.image, Image.Image): # Check if it's a PIL Image
                # Assuming the image is a PIL Image
                original_texture_pil = mesh_copy_for_size_check.visual.material.image
                original_texture_size = original_texture_pil.size # (width, height)
                print(f"### DEBUG RenderView: Found original texture size directly: {original_texture_size}")
            else:
                # Fallback: Try loading into a temporary renderer (might be slow)
                print("WARN RenderView: Could not access texture directly or it's not a PIL Image, trying temporary renderer.")
                temp_renderer = MeshRender(device=device_for_check)
                temp_renderer.load_mesh(mesh_copy_for_size_check)
                if temp_renderer.tex is not None:
                    # Texture tensor is HWC, size needs W, H
                    original_texture_size = (temp_renderer.tex.shape[1], temp_renderer.tex.shape[0])
                    print(f"### DEBUG RenderView: Found original texture size via temp renderer: {original_texture_size}")
                else:
                     print(f"WARN RenderView: Original mesh has no texture loaded by temp renderer. Using default size {original_texture_size}.")
                del temp_renderer # Clean up
        except Exception as e:
            print(f"WARN RenderView: Error getting original texture size: {e}. Using default {original_texture_size}.")
        # Ensure it's a tuple/list of two positive integers
        if not (isinstance(original_texture_size, (tuple, list)) and len(original_texture_size) == 2 and all(isinstance(x, int) and x > 0 for x in original_texture_size)):
             print(f"WARN RenderView: Determined original_texture_size {original_texture_size} is invalid. Falling back to 512x512.")
             original_texture_size = (512, 512)

        # Instantiate the renderer using the local copy
        renderer = MeshRender(
            default_resolution=render_size, # Use render_size for output view resolution
            texture_size=original_texture_size, # Use ACTUAL texture size for internal texture handling
            camera_distance=camera_distance,
            camera_type=camera_type,
            ortho_scale=ortho_scale,
            filter_mode='linear' # Default from wrapper's single view node
            )

        # Load the mesh into the newly created renderer instance
        # This should also load the texture if available on the trimesh object
        renderer.load_mesh(trimesh)

        # --- Render Textured View ---
        # Use the render method which applies the texture (self.tex)
        textured_view = renderer.render(
            elev=elevation,
            azim=azimuth,
            camera_distance=camera_distance,
            center=None,
            resolution=render_size,
            keep_alpha=False, # Don't need alpha for preview
            bgcolor=bg_color, # Use parsed background color
            return_type='th',
            pan_x=pan_x,
            pan_y=pan_y
        )
        # Output shape should be (H, W, 3)

        # --- Render Tangent Space Normals ---
        # Use a temporary variable name to avoid UnboundLocalError
        raw_tangent_normals, tangent_mask = renderer.render_normal(
            elevation,
            azimuth,
            camera_distance=camera_distance,
            center=None, # Use default centering
            resolution=render_size,
            bg_color=[0.5, 0.5, 1.0], # Use standard normal map background color (128, 128, 255)
            use_abs_coor=False, # Tangent space normals
            pan_x=pan_x,
            pan_y=pan_y,
            return_type='th' # Ensure tensor output
        )
        # Refactored processing steps for tangent normals with unique variable names
        temp_normals_scaled = 2.0 * raw_tangent_normals - 1.0  # Map [0,1] to [-1,1]
        normalized_normals = temp_normals_scaled / (torch.norm(temp_normals_scaled, dim=-1, keepdim=True) + 1e-6) # Normalize
        remapped_normals = torch.zeros_like(normalized_normals)
        remapped_normals[..., 0] = normalized_normals[..., 0]  # View right -> R
        remapped_normals[..., 1] = normalized_normals[..., 1]  # View up -> G
        remapped_normals[..., 2] = -normalized_normals[..., 2] # View forward (negated) -> B
        scaled_normals = (remapped_normals + 1) * 0.5 # Map [-1,1] back to [0,1] for viewing
        # Apply background color using the mask
        tangent_image = scaled_normals * tangent_mask.float() + torch.tensor([0.5, 0.5, 1.0], device=scaled_normals.device) * (1.0 - tangent_mask.float())


        # --- Render World Space Normals ---
        world_normals, world_mask = renderer.render_normal(
            elevation, azimuth, camera_distance=camera_distance, center=None,
            resolution=render_size, bg_color=[0, 0, 0], # Render on black
            use_abs_coor=True, # World space
            pan_x=pan_x, pan_y=pan_y, return_type='th'
        )
        # Reinstate normalization to [0, 1] range
        world_image = (world_normals + 1) * 0.5
        # Apply mask manually and blend with background color
        world_image = world_image * world_mask.float() + torch.tensor(bg_color, device=world_image.device) * (1.0 - world_mask.float())


        # --- Render Position Map ---
        # Capture the mask returned by render_position
        position_map, position_mask = renderer.render_position(
            elevation, azimuth, camera_distance=camera_distance, center=None,
            resolution=render_size, bg_color=[0, 0, 0], # Render on black
            return_type='th',
            pan_x=pan_x, pan_y=pan_y # Pass pan values
        )
        # Apply mask manually and blend with background color
        position_map = position_map * position_mask.float() + torch.tensor(bg_color, device=position_map.device) * (1.0 - position_mask.float())


        # --- Render Depth Map ---
        try:
            depth_map, depth_mask = renderer.render_depth(
                elevation, azimuth, camera_distance=camera_distance, center=None,
                resolution=render_size, pan_x=pan_x, pan_y=pan_y, return_type='th'
            )
            # If mesh is not visible, create empty depth map
            if depth_mask.sum() == 0:
                depth_map = torch.zeros((render_size, render_size, 1), device=depth_map.device)
                depth_mask = torch.zeros((render_size, render_size, 1), device=depth_mask.device)
                print("WARN: Mesh is not visible in this view, creating empty depth map.")

            # Depth map is single channel (1, H, W, 1), repeat to 3 channels for IMAGE type
            depth_image = depth_map.repeat(1, 1, 1, 3)
            # Apply mask (Depth map doesn't have bg_color option, so mask manually)
            depth_image = depth_image * depth_mask.float()
        except Exception as e:
            print(f"Error rendering depth map: {e}. Creating empty depth map.")
            depth_image = torch.zeros((1, render_size, render_size, 3))
            depth_mask = torch.zeros((1, render_size, render_size, 1))

        # --- Final Formatting ---
        # Ensure outputs are B H W C float tensors on CPU (Batch=1)
        textured_image_out = textured_view.unsqueeze(0).cpu().float()
        tangent_image_out = tangent_image.cpu().float() # Already (1, H, W, 3) after broadcasting
        world_image_out = world_image.cpu().float() # Already (1, H, W, 3) after broadcasting
        position_map_out = position_map.cpu().float() # Already (1, H, W, 3) after broadcasting
        depth_image_out = depth_image.cpu().float() # Already (1, H, W, 3)

        # Ensure mask is B H W float tensor on CPU (use the most reliable mask, e.g., depth_mask)
        mask_out = depth_mask.squeeze(-1).cpu().float() # Shape (1, H, W)

        # --- Prepare Camera Params Output ---
        camera_params = {
            "render_size": render_size,
            "camera_type": camera_type,
            "camera_distance": camera_distance,
            "pan_x": pan_x,
            "pan_y": pan_y,
            "ortho_scale": ortho_scale,
            "azimuth": azimuth,
            "elevation": elevation,
            # Pass ACTUAL texture size used by the renderer
            "texture_size": original_texture_size[0] # Assuming square, pass width
        }

        # Format detected texture size for debug output
        detected_size_str = f"{original_texture_size[0]}, {original_texture_size[1]}"

        # print(f"### DEBUG RenderView: Final shapes: textured={textured_image_out.shape}, tangent={tangent_image_out.shape}, world={world_image_out.shape}, pos={position_map_out.shape}, depth={depth_image_out.shape}, mask={mask_out.shape}")

        # Return original mesh (trimesh input), camera params dict, and detected size string
        return (textured_image_out, tangent_image_out, world_image_out, position_map_out, depth_image_out, mask_out, trimesh, camera_params, detected_size_str,)


class Hy3DTools_BackProjectInpaint:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "original_mesh": ("TRIMESH",),
                "inpainted_view": ("IMAGE",),
                "camera_params": ("CAMERA_PARAMS",), # Combined camera parameters from render node
                "blend_threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Confidence threshold (from cos map) to use inpainted pixels."}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("updated_mesh",)
    FUNCTION = "back_project_inpaint"
    CATEGORY = "Hunyuan3DTools"

    def back_project_inpaint(self, original_mesh: Any, inpainted_view: torch.Tensor,
                             camera_params: dict, blend_threshold: float):

        device = mm.get_torch_device()

        # 1. Unpack camera parameters
        render_size = camera_params.get("render_size", 512)
        camera_type = camera_params.get("camera_type", "orth")
        camera_distance = camera_params.get("camera_distance", 1.45)
        pan_x = camera_params.get("pan_x", 0.0)
        pan_y = camera_params.get("pan_y", 0.0)
        ortho_scale = camera_params.get("ortho_scale", 1.2)
        azimuth = camera_params.get("azimuth", 0.0)
        elevation = camera_params.get("elevation", 0.0)
        # Get texture size from camera_params (should be correct now)
        texture_size_w = camera_params.get("texture_size", 512) # Assuming width is passed
        # Attempt to get height if available, otherwise assume square
        # Note: Render node currently only passes width assuming square.
        # If non-square textures are possible, Render node needs update too.
        texture_size_h = camera_params.get("texture_height", texture_size_w)
        original_texture_size = (texture_size_w, texture_size_h)
        print(f"### DEBUG BackProject: Using texture size from camera_params: {original_texture_size}")


        # 2. Convert inpainted_view tensor (B, H, W, C) to format for back_project (numpy HWC 0-1)
        if inpainted_view.shape[0] != 1:
             print(f"WARN: Expected batch size 1 for inpainted_view, got {inpainted_view.shape[0]}. Using first image.")
        # Ensure correct shape HWC and range 0-1
        inpainted_view_np = inpainted_view[0].cpu().numpy()
        if inpainted_view_np.max() > 1.1: # Check if it's 0-255 (allow for slight float variance)
             print("WARN: inpainted_view seems to be in 0-255 range, converting to 0-1.")
             inpainted_view_np = inpainted_view_np / 255.0
        inpainted_view_np = np.clip(inpainted_view_np, 0.0, 1.0) # Ensure range

        # Check if render_size matches inpainted_view dimensions
        if inpainted_view_np.shape[0] != render_size or inpainted_view_np.shape[1] != render_size:
             print(f"WARN: inpainted_view dimensions ({inpainted_view_np.shape[0]}x{inpainted_view_np.shape[1]}) do not match render_size ({render_size}). Back-projection might be inaccurate.")


        # 3. Instantiate Renderer
        renderer = MeshRender(
            default_resolution=render_size,
            texture_size=original_texture_size, # Use ACTUAL original texture size
            camera_distance=camera_distance,
            camera_type=camera_type,
            ortho_scale=ortho_scale,
            device=device # Ensure renderer uses the correct device
        )

        # 4. Load Original Mesh & Store Original Texture
        # Important: Use a copy to avoid modifying the input object if it's reused elsewhere
        mesh_to_modify = original_mesh.copy()
        renderer.load_mesh(mesh_to_modify)
        if renderer.tex is None:
            raise ValueError("Original mesh does not have a texture loaded by the renderer.")
        original_texture = renderer.tex.clone() # HWC, 0-1 float tensor on device
        # No longer need to set texture resolution here, it was set during init


        # 5. Back-Project Inpainted View
        # back_project expects HWC, 0-1 float numpy array or PIL image
        print(f"### DEBUG BackProject: Calling back_project with elev={elevation}, azim={azimuth}")
        project_texture, project_cos_map, project_boundary_map = renderer.back_project(
            inpainted_view_np, # Pass the numpy array
            elevation,
            azimuth,
            camera_distance=camera_distance,
            center=None, # Use default centering from loaded mesh
            pan_x=pan_x, # Pass pan_x
            pan_y=pan_y, # Pass pan_y
            # method='linear' # Default method
        )
        # project_texture and project_cos_map are tensors on the renderer's device (H, W, C) and (H, W, 1)

        # 6. Blend Textures
        # Create a mask based on the cosine map threshold
        # Ensure project_cos_map is on the correct device
        project_cos_map = project_cos_map.to(device)
        blend_mask = (project_cos_map > blend_threshold).float() # Shape (H, W, 1)

        # Ensure textures are on the correct device
        project_texture = project_texture.to(device)
        original_texture = original_texture.to(device)

        # Use torch.where for blending: condition ? value_if_true : value_if_false
        # Ensure channel dimensions match if original is RGB and projected is RGBA or vice versa (unlikely here)
        if project_texture.shape[-1] != original_texture.shape[-1]:
             print(f"WARN: Channel mismatch between projected ({project_texture.shape[-1]}) and original ({original_texture.shape[-1]}) textures. Using projected.")
             # This case shouldn't happen if inpainted_view is RGB, but handle defensively
             blended_texture = torch.where(
                 blend_mask.bool(),
                 project_texture[..., :original_texture.shape[-1]], # Slice projected to match original
                 original_texture
             )
        else:
             blended_texture = torch.where(
                 blend_mask.bool(), # Condition
                 project_texture,   # Value if true (use projected inpainted texture)
                 original_texture   # Value if false (use original texture)
             )

        # 7. Update Renderer Texture
        renderer.tex = blended_texture

        # 8. Save and Output Mesh
        # save_mesh uses the mesh stored internally during load_mesh
        updated_mesh = renderer.save_mesh()

        return (updated_mesh,)

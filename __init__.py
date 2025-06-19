# ComfyUI-Hunyuan3DTools init file

# Import nodes here as they are created
# Make sure both nodes are imported
from .nodes import *

NODE_CLASS_MAPPINGS = {
    "Hy3DTools_RenderSpecificView": Hy3DTools_RenderSpecificView,
    "Hy3DTools_BackProjectInpaint": Hy3DTools_BackProjectInpaint,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DTools_RenderSpecificView": "Hy3DTools Render Specific View",
    "Hy3DTools_BackProjectInpaint": "Hy3DTools Back-Project Inpaint",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

print("### Loading: ComfyUI-Hunyuan3DTools")

# Open Source Model Licensed under the Apache License Version 2.0
# and Other Licenses of the Third-Party Components therein:
# The below Model in this distribution may have been modified by THL A29 Limited
# ("Tencent Modifications"). All Tencent Modifications are Copyright (C) 2024 THL A29 Limited.

# Copyright (C) 2024 THL A29 Limited, a Tencent company.  All rights reserved.
# The below software and/or models in this distribution may have been
# modified by THL A29 Limited ("Tencent Modifications").
# All Tencent Modifications are Copyright (C) THL A29 Limited.

# Hunyuan 3D is licensed under the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT
# except for the third-party components listed below.
# Hunyuan 3D does not impose any additional limitations beyond what is outlined
# in the repsective licenses of these third-party components.
# Users must comply with all terms and conditions of original licenses of these third-party
# components and must ensure that the usage of the third party components adheres to
# all relevant laws and regulations.

# For avoidance of doubts, Hunyuan 3D means the large language models and
# their software and algorithms, including trained model weights, parameters (including
# optimizer states), machine-learning model code, inference-enabling code, training-enabling code,
# fine-tuning enabling code and other elements of the foregoing made publicly available
# by Tencent in accordance with TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.

import trimesh


def load_mesh(mesh):
    vtx_pos = mesh.vertices if hasattr(mesh, 'vertices') else None
    pos_idx = mesh.faces if hasattr(mesh, 'faces') else None

    vtx_uv = mesh.visual.uv if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'uv') else None
    uv_idx = mesh.faces if hasattr(mesh, 'faces') else None

    texture_data = None
    if hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material'):
        if hasattr(mesh.visual.material, 'image') and mesh.visual.material.image is not None:
            texture_data = mesh.visual.material.image
            print("### DEBUG (mesh_utils): Found texture in material.image")
        elif hasattr(mesh.visual.material, 'baseColorTexture') and mesh.visual.material.baseColorTexture is not None:
            texture_data = mesh.visual.material.baseColorTexture
            print("### DEBUG (mesh_utils): Found texture in material.baseColorTexture")
        else:
            print("### DEBUG (mesh_utils): Material found, but no image or baseColorTexture.")
    else:
        print("### DEBUG (mesh_utils): No visual or material found on mesh.")


    return vtx_pos, pos_idx, vtx_uv, uv_idx, texture_data


def save_mesh(mesh, texture_data):
    material = trimesh.visual.texture.SimpleMaterial(image=texture_data, diffuse=(255, 255, 255))
    texture_visuals = trimesh.visual.TextureVisuals(uv=mesh.visual.uv, image=texture_data, material=material)
    mesh.visual = texture_visuals
    return mesh

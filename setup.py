from setuptools import setup, find_packages

setup(
    name="manifold4d",
    version="0.1.0",
    description="Manifold4D: Denoising on Point Cloud Rendered Manifolds "
                "for Video Re-shooting (inference)",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        # The Wan2.1 source tree (wan.* modules) is cloned to external/;
        # see README — no pip install needed (paths resolve via configs).
        "torch>=2.0",
        "torchvision",
        "diffusers",
        "numpy",
        "einops",
        "pyyaml",
        "opencv-python",
        "pillow",
        "scipy",
        "tqdm",
        "imageio",
        "transformers",  # Qwen2.5-VL captioning (scripts/preprocess)
    ],
)

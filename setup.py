from setuptools import setup, find_packages

setup(
    name="FGIR_SSL",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10",
        "torchvision>=0.11",
        "timm>=0.5",
        "PyYAML>=5.4",
        "numpy",
        "Pillow",
    ],
    entry_points={
        "console_scripts": [
            "fgir_train=scripts.train:main",
            "fgir_eval=scripts.evaluate:main",
        ]
    },
)

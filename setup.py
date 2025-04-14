from setuptools import setup, find_packages

setup(
    name="ReProver",
    version="0.1",
    packages=find_packages(
        include=["ReProver", "ReProver.*"]
    ),  # Include only 'ReProver'
    install_requires=[
        "torch==2.4.0",
        "tqdm",
        "loguru",
        "pytorch-lightning[extra]==2.3.3",
        "jsonlines",
        "numpy==1.24.2",
        "deepspeed==0.15.0",
        "accelerate==0.34.2",
        "transformers==4.44.2",
        "openai",
        "lean-dojo==2.1.2",
        "wandb",
        "sentencepiece",
        "vllm==0.6.0",
        "datasets==3.1.0",
        "loguru",
    ],  # list of dependencies
)

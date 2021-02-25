from setuptools import setup, find_namespace_packages

# pip install "nvidia-pyindex~=1.0.5"

requirements = [
    "torch~=1.6.0",
    "torchvision~=0.7.0",
    "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@09e3ba4dfe333f86d6c5c1048e07210924294be9#egg=nemo_toolkit",
    "fairseq @ git+https://github.com/pytorch/fairseq.git@94a1b924f3adec25c8c508ac112410d02b400d1e#egg=fairseq",
    # "google-cloud-texttospeech~=1.0.1",
    "tqdm~=4.54.0",
    # "pydub~=0.24.0",
    # "scikit_learn~=0.22.1",
    # "pandas~=1.0.3",
    # "boto3~=1.12.35",
    # "ruamel.yaml~=0.16.10",
    # "pymongo==3.10.1",
    # "matplotlib==3.2.1",
    # "tabulate==0.8.7",
    # "natural==0.2.0",
    # "num2words==0.5.10",
    "typer[all]~=0.3.2",
    # "python-slugify==4.0.0",
    # "websockets==8.1",
    # "lenses @ git+https://github.com/ingolemo/python-lenses.git@b2a2a9aa5b61540992d70b2cf36008d0121e8948#egg=lenses",
    "rpyc~=4.1.4",
    # "streamlit~=0.61.0",
    # "librosa~=0.7.2",
    # "tritonclient[http]~=2.6.0",
    "numba~=0.48.0",
]

extra_requirements = {
    "data": [
        "pydub~=0.24.0",
        "google-cloud-texttospeech~=1.0.1",
        "scikit_learn~=0.22.1",
        "pandas~=1.0.3",
        "boto3~=1.12.35",
        "ruamel.yaml~=0.16.10",
        "pymongo~=3.10.1",
        "librosa~=0.7.2",
        "matplotlib~=3.2.1",
        "pandas~=1.0.3",
        "tabulate~=0.8.7",
        "natural~=0.2.0",
        "num2words~=0.5.10",
        "python-slugify~=4.0.0",
        "rpyc~=4.1.4",
        # "lenses @ git+https://github.com/ingolemo/python-lenses.git@b2a2a9aa5b61540992d70b2cf36008d0121e8948#egg=lenses",
    ],
    "validation": [
        "pymongo~=3.10.1",
        "matplotlib~=3.2.1",
        "pydub~=0.24.0",
        "streamlit~=0.58.0",
        "natural~=0.2.0",
        "stringcase~=1.2.0",
        "google-cloud-speech~=1.3.1",
    ],
    "train": ["torchaudio~=0.6.0", "torch-stft~=0.1.4"],
}

extra_requirements["all"] = list({d for l in extra_requirements.values() for d in l})
packages = find_namespace_packages()

setup(
    name="plume-asr",
    version="0.2.0",
    description="Multi model ASR base package",
    url="http://github.com/malarinv/plume-asr",
    author="Malar Kannan",
    author_email="malarkannan.invention@gmail.com",
    license="MIT",
    install_requires=requirements,
    extras_require=extra_requirements,
    packages=packages,
    entry_points={"console_scripts": ["plume = plume.cli:main"]},
    zip_safe=False,
)

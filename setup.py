from setuptools import setup, find_namespace_packages

# pip install "nvidia-pyindex~=1.0.5"

requirements = [
    # "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@09e3ba4dfe333f86d6c5c1048e07210924294be9#egg=nemo_toolkit",
    # "fairseq @ git+https://github.com/pytorch/fairseq.git@94a1b924f3adec25c8c508ac112410d02b400d1e#egg=fairseq",
    # "google-cloud-texttospeech~=1.0.1",
    "six~=1.16.0",
    "tqdm~=4.49.0",
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
        "webrtcvad~=2.0.10",
        # "datasets"
        # "lenses @ git+https://github.com/ingolemo/python-lenses.git@b2a2a9aa5b61540992d70b2cf36008d0121e8948#egg=lenses",
    ],
    "models": [
        # "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@09e3ba4dfe333f86d6c5c1048e07210924294be9#egg=nemo_toolkit",
        "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@v1.0.0#egg=nemo_toolkit",
        "fairseq @ git+https://github.com/pytorch/fairseq.git@94a1b924f3adec25c8c508ac112410d02b400d1e#egg=fairseq",
        "transformers~=4.5.0",
        "torch~=1.7.0",
        "torchvision~=0.8.2",
        "torchaudio~=0.7.2",
    ],
    "infer": [
        "jiwer~=2.2.0",
        "pydub~=0.24.0",
        "tritonclient[grpc]~=2.9.0",
        "pyspellchecker~=0.6.2",
        "num2words~=0.5.10",
        "pydub~=0.24.0",
    ],
    "infer_min": [
        "pyspellchecker~=0.6.2",
        "num2words~=0.5.10",
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
    "ui": [
        "rangehttpserver~=1.2.0",
    ],
    "crypto": ["cryptography~=3.4.7"],
    "train": ["torchaudio~=0.6.0", "torch-stft~=0.1.4"],
}
extra_requirements["deploy"] = (
    extra_requirements["models"] + extra_requirements["infer_min"]
)
extra_requirements["all"] = list(
    {d for r in extra_requirements.values() for d in r}
)
packages = find_namespace_packages("src")

setup(
    name="plume-asr",
    version="0.2.1",
    description="Multi model ASR base package",
    url="http://github.com/malarinv/plume-asr",
    author="Malar Kannan",
    author_email="malarkannan.invention@gmail.com",
    license="MIT",
    install_requires=requirements,
    extras_require=extra_requirements,
    packages=packages,
    package_dir={"": "src"},
    entry_points={"console_scripts": ["plume = plume.cli:main"]},
    zip_safe=False,
)

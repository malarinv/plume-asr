from collections import namedtuple
from io import BytesIO
from pathlib import Path

# from cryptography.fernet import Fernet

import typer
from tqdm import tqdm

from . import asr_manifest_writer
from .extended_path import ExtendedPath
from .audio import audio_seg_to_wav_bytes, audio_wav_bytes_to_seg
from .parallel import parallel_apply
from .lazy_import import lazy_module

cryptography = lazy_module("cryptography.fernet", level='base')
# cryptography.fernet = lazy_module("cryptography.fernet")
pydub = lazy_module("pydub")

app = typer.Typer()


@app.callback()
def encrypt():
    """
    encrypt sub commands
    """


def wav_cryptor(key=""):
    WavCryptor = namedtuple(
        "WavCryptor",
        (
            "keygen",
            "encrypt_wav_path_to",
            "decrypt_wav_path_to",
            "decrypt_wav_path",
        ),
    )
    _enc_key = key
    _crypto_f = cryptography.fernet.Fernet(_enc_key)

    def encrypt_wav_bytes(f, dec_wav_bytes):
        b = BytesIO(dec_wav_bytes)
        audio_seg = pydub.AudioSegment.from_file(b)
        # audio_seg.raw_data
        enc_wav_bytes = f.encrypt(audio_seg.raw_data)
        encrypted_seg = pydub.AudioSegment(
            enc_wav_bytes,
            frame_rate=audio_seg.frame_rate,
            channels=audio_seg.channels,
            sample_width=audio_seg.sample_width,
        )
        return audio_seg_to_wav_bytes(encrypted_seg)

    def decrypt_wav_bytes(f, enc_wav_bytes):
        b = BytesIO(enc_wav_bytes)
        audio_seg = pydub.AudioSegment.from_file(b)
        dec_wav_bytes = f.decrypt(audio_seg.raw_data)
        decrypted_seg = pydub.AudioSegment(
            dec_wav_bytes,
            frame_rate=audio_seg.frame_rate,
            channels=audio_seg.channels,
            sample_width=audio_seg.sample_width,
        )
        return audio_seg_to_wav_bytes(decrypted_seg)

    def encrypt_wav_path_to(dec_audio_path: Path, enc_audio_path: Path):
        dec_wav_bytes = dec_audio_path.read_bytes()
        enc_audio_path.write_bytes(encrypt_wav_bytes(_crypto_f, dec_wav_bytes))

    def decrypt_wav_path_to(enc_audio_path: Path, dec_audio_path: Path):
        enc_wav_bytes = enc_audio_path.read_bytes()
        dec_audio_path.write_bytes(decrypt_wav_bytes(_crypto_f, enc_wav_bytes))

    def decrypt_wav_path(enc_audio_path: Path):
        enc_wav_bytes = enc_audio_path.read_bytes()
        return decrypt_wav_bytes(_crypto_f, enc_wav_bytes)

    return WavCryptor(
        cryptography.fernet.Fernet.generate_key,
        encrypt_wav_path_to,
        decrypt_wav_path_to,
        decrypt_wav_path,
    )


def text_cryptor(key=""):
    TextCryptor = namedtuple(
        "TextCryptor",
        ("keygen", "encrypt_text", "decrypt_text"),
    )
    _enc_key = key
    _crypto_f = cryptography.fernet.Fernet(_enc_key)

    def encrypt_text(text: str):
        return _crypto_f.encrypt(text.encode("utf-8"))

    def decrypt_text(text: str):
        return _crypto_f.decrypt(text).decode("utf-8")

    return TextCryptor(
        cryptography.fernet.Fernet.generate_key, encrypt_text, decrypt_text
    )


def encrypted_asr_manifest_reader(
    data_manifest_path: Path, encryption_key: str, verbose=True, parallel=True
):
    print(f"reading encrypted manifest from {data_manifest_path}")
    asr_data = list(ExtendedPath(data_manifest_path).read_jsonl())
    enc_key_bytes = encryption_key.encode("utf-8")
    wc = wav_cryptor(enc_key_bytes)
    tc = text_cryptor(enc_key_bytes)

    def decrypt_fn(p):
        d = {
            "audio_seg": audio_wav_bytes_to_seg(
                wc.decrypt_wav_path(
                    data_manifest_path.parent / Path(p["audio_filepath"])
                )
            ),
            "text": tc.decrypt_text(p["text"].encode("utf-8")),
        }
        return d

    if parallel:
        for d in parallel_apply(decrypt_fn, asr_data, verbose=verbose):
            yield d
    else:
        for p in tqdm.tqdm(asr_data) if verbose else asr_data:
            yield decrypt_fn(d)


def decrypt_asr_dataset(
    src_dataset_dir: Path,
    dest_dataset_dir: Path,
    encryption_key: str,
    verbose=True,
    parallel=True,
):
    data_manifest_path = src_dataset_dir / "manifest.json"
    (dest_dataset_dir / "wavs").mkdir(exist_ok=True, parents=True)
    dest_manifest_path = dest_dataset_dir / "manifest.json"
    print(f"reading encrypted manifest from {data_manifest_path}")
    asr_data = list(ExtendedPath(data_manifest_path).read_jsonl())
    enc_key_bytes = encryption_key.encode("utf-8")
    wc = wav_cryptor(enc_key_bytes)
    tc = text_cryptor(enc_key_bytes)

    def decrypt_fn(p):
        dest_path = dest_dataset_dir / Path(p["audio_filepath"])
        wc.decrypt_wav_path_to(
            src_dataset_dir / Path(p["audio_filepath"]), dest_path
        )
        d = {
            "audio_filepath": dest_path,
            "duration": p["duration"],
            "text": tc.decrypt_text(p["text"].encode("utf-8")),
        }
        return d

    def datagen():
        if parallel:
            for d in parallel_apply(decrypt_fn, asr_data, verbose=verbose):
                yield d
        else:
            for p in tqdm.tqdm(asr_data) if verbose else asr_data:
                yield decrypt_fn(d)

    asr_manifest_writer(dest_manifest_path, datagen)


@app.command()
def keygen():
    gen_key = cryptography.fernet.Fernet.generate_key()
    typer.echo(f"KEY: {gen_key}")


@app.command()
def encrypt_text(
    text_to_encrypt: str,
    encryption_key: str = typer.Option(..., prompt=True, hide_input=True),
):
    enc_key_bytes = encryption_key.encode("utf-8")
    tc = text_cryptor(enc_key_bytes)
    cryptext = tc.encrypt_text(text_to_encrypt)
    typer.echo(cryptext)

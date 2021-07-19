from pathlib import Path

# from tqdm import tqdm
import json

# from .extended_path import ExtendedPath
# from .parallel import parallel_apply
# from .encrypt import wav_cryptor, text_cryptor


def manifest_str(path, dur, text):
    k = {"audio_filepath": path, "duration": round(dur, 1), "text": text}
    return json.dumps(k) + "\n"


def asr_manifest_reader(data_manifest_path: Path):
    print(f"reading manifest from {data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        data_jsonl = pf.readlines()
    data_data = [json.loads(v) for v in data_jsonl]
    for p in data_data:
        p["audio_path"] = data_manifest_path.parent / Path(p["audio_filepath"])
        p["text"] = p["text"].strip()
        yield p


def asr_manifest_writer(
    asr_manifest_path: Path, manifest_str_source, verbose=False
):
    with asr_manifest_path.open("w") as mf:
        if verbose:
            print(f"writing asr manifest to {asr_manifest_path}")
        for mani_dict in manifest_str_source:
            manifest = manifest_str(
                mani_dict["audio_filepath"],
                mani_dict["duration"],
                mani_dict["text"],
            )
            mf.write(manifest)


#
# def decrypt(
#     src_dataset_dir: Path,
#     dest_dataset_dir: Path,
#     encryption_key: str,
#     verbose=True,
#     parallel=True,
# ):
#     data_manifest_path = src_dataset_dir / "manifest.json"
#     (dest_dataset_dir / "wavs").mkdir(exist_ok=True, parents=True)
#     dest_manifest_path = dest_dataset_dir / "manifest.json"
#     print(f"reading encrypted manifest from {data_manifest_path}")
#     asr_data = list(ExtendedPath(data_manifest_path).read_jsonl())
#     enc_key_bytes = encryption_key.encode("utf-8")
#     wc = wav_cryptor(enc_key_bytes)
#     tc = text_cryptor(enc_key_bytes)
#
#     def decrypt_fn(p):
#         dest_path = dest_dataset_dir / Path(p["audio_filepath"])
#         wc.decrypt_wav_path_to(
#             src_dataset_dir / Path(p["audio_filepath"]), dest_path
#         )
#         d = {
#             "audio_filepath": dest_path,
#             "duration": p["duration"],
#             "text": tc.decrypt_text(p["text"].encode("utf-8")),
#         }
#         return d
#
#     def datagen():
#         if parallel:
#             for d in parallel_apply(decrypt_fn, asr_data, verbose=verbose):
#                 yield d
#         else:
#             for p in tqdm.tqdm(asr_data) if verbose else asr_data:
#                 yield decrypt_fn(d)
#
#     asr_manifest_writer(dest_manifest_path, datagen)

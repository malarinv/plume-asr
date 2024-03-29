import json
from pathlib import Path
from random import shuffle
from typing import List
from itertools import chain

# from sklearn.model_selection import train_test_split
from tqdm import tqdm
import shutil
import typer

from plume.utils import (
    asr_manifest_reader,
    asr_manifest_writer,
    ExtendedPath,
    duration_str,
    generate_filter_map,
    get_mongo_conn,
    tscript_uuid_fname,
    lazy_callable,
    lazy_module,
    wav_cryptor,
    text_cryptor,
    parallel_apply,
)

from ...models.wav2vec2.data import app as wav2vec2_app
from .generate import app as generate_app
from .process import app as process_app

soundfile = lazy_module("soundfile")
pydub = lazy_module("pydub")
train_test_split = lazy_callable("sklearn.model_selection.train_test_split")

app = typer.Typer()
app.add_typer(generate_app)
app.add_typer(process_app)
app.add_typer(wav2vec2_app, name="wav2vec2")


@app.callback()
def data():
    """
    data sub commands
    """


@app.command()
def fix_path(dataset_path: Path, force: bool = False):
    manifest_path = dataset_path / Path("manifest.json")
    real_manifest_path = dataset_path / Path("abs_manifest.json")

    def fix_real_path():
        for i in asr_manifest_reader(manifest_path):
            i["audio_filepath"] = str(
                (dataset_path / Path(i["audio_filepath"])).absolute()
            )
            yield i

    def fix_rel_path():
        for i in asr_manifest_reader(real_manifest_path):
            i["audio_filepath"] = str(
                Path(i["audio_filepath"]).relative_to(dataset_path)
            )
            yield i

    if not manifest_path.exists() and not real_manifest_path.exists():
        typer.echo("Invalid dataset directory")
    if not real_manifest_path.exists() or force:
        asr_manifest_writer(real_manifest_path, fix_real_path())
    if not manifest_path.exists():
        asr_manifest_writer(manifest_path, fix_rel_path())


@app.command()
def merge(
    src_dataset_paths: List[Path],
    dest_dataset_path: Path,
    unlink: bool = False,
    verbose: bool = True,
):
    reader_list = []
    abs_manifest_path = Path("abs_manifest.json")
    for dataset_path in src_dataset_paths:
        manifest_path = dataset_path / abs_manifest_path
        reader_list.append(asr_manifest_reader(manifest_path))
    dest_dataset_path.mkdir(parents=True, exist_ok=True)
    dest_manifest_path = dest_dataset_path / abs_manifest_path
    asr_manifest_writer(
        dest_manifest_path, chain(*reader_list), verbose=verbose
    )
    if unlink:
        eject(dest_dataset_path, verbose=verbose)


def eject(dest_dataset_path: Path, verbose: bool = False):
    wav_dir = dest_dataset_path / Path("wavs")
    wav_dir.mkdir(exist_ok=True, parents=True)
    abs_manifest_path = ExtendedPath(
        dest_dataset_path / Path("abs_manifest.json")
    )
    backup_abs_manifest_path = abs_manifest_path.with_suffix(".json.orig")
    shutil.copy(abs_manifest_path, backup_abs_manifest_path)
    manifest_data = list(abs_manifest_path.read_jsonl())
    for md in tqdm(manifest_data) if verbose else manifest_data:
        orig_path = Path(md["audio_filepath"])
        new_path = wav_dir / Path(orig_path.name)
        shutil.copy(orig_path, new_path)
        md["audio_filepath"] = str(new_path)
    abs_manifest_path.write_jsonl(manifest_data)
    fix_path(dest_dataset_path)


@app.command()
def training_split(dataset_path: Path, test_size: float = 0.03):
    manifest_path = dataset_path / Path("abs_manifest.json")
    if not manifest_path.exists():
        fix_path(dataset_path)
    asr_data = list(asr_manifest_reader(manifest_path))
    train_pnr, test_pnr = train_test_split(asr_data, test_size=test_size)
    asr_manifest_writer(
        manifest_path.with_name("train_manifest.json"), train_pnr
    )
    asr_manifest_writer(
        manifest_path.with_name("test_manifest.json"), test_pnr
    )


@app.command()
def parts_split_by_size(
    dataset_path: Path,
    test_size: float = 0.03,
    split_prefix_names: List[str] = ["train", "test"],
):
    manifest_path = dataset_path / Path("abs_manifest.json")
    if not manifest_path.exists():
        fix_path(dataset_path)
    asr_data = list(asr_manifest_reader(manifest_path))
    train_pnr, test_pnr = train_test_split(asr_data, test_size=test_size)
    dest_paths = [
        (dataset_path.parent / (dataset_path.name + "_" + spn), sd)
        for (spn, sd) in zip(split_prefix_names, [train_pnr, test_pnr])
    ]
    for dest_path, manifest_data in dest_paths:
        wav_dir = dest_path / Path("wavs")
        wav_dir.mkdir(exist_ok=True, parents=True)
        abs_manifest_path = ExtendedPath(dest_path / Path("abs_manifest.json"))
        for md in manifest_data:
            orig_path = Path(md["audio_filepath"])
            new_path = wav_dir / Path(orig_path.name)
            shutil.copy(orig_path, new_path)
            md["audio_filepath"] = str(new_path)
            md.pop("audio_path")
        abs_manifest_path.write_jsonl(manifest_data)
        fix_path(dest_path)


@app.command()
def parts_split_by_dur(
    dataset_path: Path,
    dur_sec: int = 7200,
    suffix_name: List[str] = ["train", "test"],
):
    manifest_path = dataset_path / Path("abs_manifest.json")
    if not manifest_path.exists():
        fix_path(dataset_path)
    asr_data = list(asr_manifest_reader(manifest_path))

    def dur_split(dataset, dur_seconds):
        shuffle(dataset)
        counter_dur = 0
        train_set, test_set = [], []
        for d in dataset:
            if counter_dur <= dur_seconds:
                test_set.append(d)
            else:
                train_set.append(d)
            counter_dur += d["duration"]
        return train_set, test_set

    train_pnr, test_pnr = dur_split(asr_data, dur_sec)
    dest_paths = [
        (dataset_path.parent / (dataset_path.name + "_" + spn), sd)
        for (spn, sd) in zip(suffix_name, [train_pnr, test_pnr])
    ]
    for dest_path, manifest_data in dest_paths:
        wav_dir = dest_path / Path("wavs")
        wav_dir.mkdir(exist_ok=True, parents=True)
        abs_manifest_path = ExtendedPath(dest_path / Path("abs_manifest.json"))
        for md in manifest_data:
            orig_path = Path(md["audio_filepath"])
            new_path = wav_dir / Path(orig_path.name)
            shutil.copy(orig_path, new_path)
            md["audio_filepath"] = str(new_path.absolute())
            md.pop("audio_path")
        abs_manifest_path.write_jsonl(manifest_data)
        fix_path(dest_path.absolute())


@app.command()
def validate(dataset_path: Path):
    from natural.date import compress
    from datetime import timedelta

    for mf_type in ["train_manifest.json", "test_manifest.json"]:
        data_file = dataset_path / Path(mf_type)
        print(f"validating {data_file}.")
        with Path(data_file).open("r") as pf:
            pnr_jsonl = pf.readlines()
        duration = 0
        for (i, s) in enumerate(pnr_jsonl):
            try:
                d = json.loads(s)
                duration += d["duration"]
                audio_file = data_file.parent / Path(d["audio_filepath"])
                if not audio_file.exists():
                    raise OSError(f"File {audio_file} not found")
            except BaseException as e:
                print(f'failed on {i} with "{e}"')
        duration_str = compress(timedelta(seconds=duration), pad=" ")
        print(
            f"no errors found. seems like a valid {mf_type}. contains {duration_str} of audio"
        )


@app.command()
def filter(
    src_dataset_path: Path,
    dest_dataset_path: Path,
    kind: str = "",
):
    dest_manifest = dest_dataset_path / Path("manifest.json")
    data_file = src_dataset_path / Path("manifest.json")
    dest_wav_dir = dest_dataset_path / Path("wavs")
    dest_wav_dir.mkdir(exist_ok=True, parents=True)
    filter_kind_map = generate_filter_map(
        src_dataset_path, dest_dataset_path, data_file
    )

    selected_filter = filter_kind_map.get(kind, None)
    if selected_filter:
        asr_manifest_writer(dest_manifest, selected_filter())
    else:
        typer.echo(f"filter kind - {kind} not implemented")
        typer.echo(f"select one of {', '.join(filter_kind_map.keys())}")


@app.command()
def info(dataset_path: Path):
    for k in ["", "abs_", "train_", "test_"]:
        mf_wav_duration = (
            real_duration
        ) = max_duration = empty_duration = empty_count = total_count = 0
        data_file = dataset_path / Path(f"{k}manifest.json")
        if data_file.exists():
            print(f"stats on {data_file}")
            for s in ExtendedPath(data_file).read_jsonl():
                total_count += 1
                mf_wav_duration += s["duration"]
                if s["text"] == "":
                    empty_count += 1
                    empty_duration += s["duration"]
                wav_path = str(dataset_path / Path(s["audio_filepath"]))
                if max_duration < soundfile.info(wav_path).duration:
                    max_duration = soundfile.info(wav_path).duration
                real_duration += soundfile.info(wav_path).duration

            # frame_count = soundfile.info(audio_fname).frames
            print(
                f"max audio duration : {duration_str(max_duration, show_hours=True)}"
            )
            print(
                f"total audio duration : {duration_str(mf_wav_duration, show_hours=True)}"
            )
            print(
                f"total real audio duration : {duration_str(real_duration, show_hours=True)}"
            )
            print(
                f"total content duration : {duration_str(mf_wav_duration-empty_duration, show_hours=True)}"
            )
            print(
                f"total empty duration : {duration_str(empty_duration, show_hours=True)}"
            )
            print(
                f"total empty samples : {empty_count}/{total_count} ({empty_count*100/total_count:.2f}%)"
            )


@app.command()
def audio_duration(dataset_path: Path):
    wav_duration = 0
    for audio_rel_fname in dataset_path.absolute().glob("**/*.wav"):
        audio_fname = str(audio_rel_fname)
        wav_duration += soundfile.info(audio_fname).duration
    typer.echo(
        f"duration of wav files @ {dataset_path}: {duration_str(wav_duration)}"
    )


@app.command()
def encrypt(
    src_dataset_path: Path,
    dest_dataset_path: Path,
    encryption_key: str = typer.Option(..., prompt=True, hide_input=True),
    verbose: bool = True,
):
    dest_manifest = dest_dataset_path / Path("manifest.json")
    src_manifest = src_dataset_path / Path("manifest.json")
    dest_wav_dir = dest_dataset_path / Path("wavs")
    dest_wav_dir.mkdir(exist_ok=True, parents=True)
    wav_crypt = wav_cryptor(encryption_key)
    text_crypt = text_cryptor(encryption_key)
    # warmup
    _ = pydub.AudioSegment.from_file

    def encrypt_item(s):
        crypt_text = text_crypt.encrypt_text(s["text"])
        src_wav_path = src_dataset_path / s["audio_filepath"]
        dst_wav_path = dest_dataset_path / s["audio_filepath"]
        wav_crypt.encrypt_wav_path_to(src_wav_path, dst_wav_path)
        s["text"] = crypt_text.decode("utf-8")
        return s

    def encryted_gen():
        data = list(ExtendedPath(src_manifest).read_jsonl())
        iter_data = tqdm(data) if verbose else data
        encrypted_iter_data = parallel_apply(
            encrypt_item, iter_data, verbose=verbose, workers=64
        )
        for s in encrypted_iter_data:
            yield s

    asr_manifest_writer(dest_manifest, encryted_gen(), verbose=verbose)


@app.command()
def decrypt(
    src_dataset_path: Path,
    dest_dataset_path: Path,
    encryption_key: str = typer.Option(..., prompt=True, hide_input=True),
    verbose: bool = True,
):
    dest_manifest = dest_dataset_path / Path("manifest.json")
    src_manifest = src_dataset_path / Path("manifest.json")
    dest_wav_dir = dest_dataset_path / Path("wavs")
    dest_wav_dir.mkdir(exist_ok=True, parents=True)
    wav_crypt = wav_cryptor(encryption_key)
    text_crypt = text_cryptor(encryption_key)
    # warmup
    _ = pydub.AudioSegment.from_file

    def decrypt_item(s):
        crypt_text = text_crypt.decrypt_text(s["text"].encode("utf-8"))
        src_wav_path = src_dataset_path / s["audio_filepath"]
        dst_wav_path = dest_dataset_path / s["audio_filepath"]
        wav_crypt.decrypt_wav_path_to(src_wav_path, dst_wav_path)
        s["text"] = crypt_text
        return s

    def decryted_gen():
        data = list(ExtendedPath(src_manifest).read_jsonl())
        iter_data = tqdm(data) if verbose else data
        decrypted_iter_data = parallel_apply(
            decrypt_item, iter_data, verbose=verbose, workers=64
        )
        for s in decrypted_iter_data:
            yield s

    asr_manifest_writer(dest_manifest, decryted_gen(), verbose=verbose)


@app.command()
def migrate(src_path: Path, dest_path: Path):
    shutil.copytree(str(src_path), str(dest_path))
    wav_dir = dest_path / Path("wavs")
    wav_dir.mkdir(exist_ok=True, parents=True)
    abs_manifest_path = ExtendedPath(dest_path / Path("abs_manifest.json"))
    backup_abs_manifest_path = abs_manifest_path.with_suffix(".json.orig")
    shutil.copy(abs_manifest_path, backup_abs_manifest_path)
    manifest_data = list(abs_manifest_path.read_jsonl())
    for md in manifest_data:
        orig_path = Path(md["audio_filepath"])
        new_path = wav_dir / Path(orig_path.name)
        shutil.copy(orig_path, new_path)
        md["audio_filepath"] = str(new_path)
    abs_manifest_path.write_jsonl(manifest_data)
    fix_path(dest_path)


@app.command()
def task_split(
    data_dir: Path,
    dump_file: Path = Path("ui_dump.json"),
    task_count: int = typer.Option(2, show_default=True),
    task_file: str = "task_dump",
    sort: bool = True,
):
    """
    split ui_dump.json to `task_count` tasks
    """
    import pandas as pd
    import numpy as np

    processed_data_path = data_dir / dump_file
    processed_data = ExtendedPath(processed_data_path).read_json()
    df = (
        pd.DataFrame(processed_data["data"])
        .sample(frac=1)
        .reset_index(drop=True)
    )
    for t_idx, task_f in enumerate(np.array_split(df, task_count)):
        task_f = task_f.reset_index(drop=True)
        task_f["real_idx"] = task_f.index
        task_data = task_f.to_dict("records")
        if sort:
            task_data = sorted(
                task_data, key=lambda x: x["asr_wer"], reverse=True
            )
        processed_data["data"] = task_data
        task_path = data_dir / Path(task_file + f"-{t_idx}.json")
        ExtendedPath(task_path).write_json(processed_data)


def get_corrections(task_uid):
    col = get_mongo_conn(col="asr_validation")
    task_id = [
        c
        for c in col.distinct("task_id")
        if c.rsplit("-", 1)[1] == task_uid or c == task_uid
    ][0]
    corrections = list(
        col.find({"type": "correction"}, projection={"_id": False})
    )
    cursor_obj = col.find(
        {"type": "correction", "task_id": task_id}, projection={"_id": False}
    )
    corrections = [c for c in cursor_obj]
    return corrections


@app.command()
def dump_task_corrections(data_dir: Path, task_uid: str):
    dump_fname: Path = Path(f"corrections-{task_uid}.json")
    dump_path = data_dir / dump_fname
    corrections = get_corrections(task_uid)
    ExtendedPath(dump_path).write_json(corrections)


@app.command()
def dump_all_corrections(data_dir: Path):
    for task_lcks in data_dir.glob("task-*.lck"):
        task_uid = task_lcks.stem.replace("task-", "")
        dump_task_corrections(data_dir, task_uid)


@app.command()
def update_corrections(
    data_dir: Path,
    skip_incorrect: bool = typer.Option(
        False, show_default=True, help="treats incorrect as invalid"
    ),
    skip_inaudible: bool = typer.Option(
        False, show_default=True, help="include invalid as blank target"
    ),
):
    """
    applies the corrections-*.json
    backup the original dataset
    """
    manifest_file: Path = Path("manifest.json")
    renames_file: Path = Path("rename_map.json")
    ui_dump_file: Path = Path("ui_dump.json")
    data_manifest_path = data_dir / manifest_file
    renames_path = data_dir / renames_file

    def correct_ui_dump(data_dir, rename_result):
        ui_dump_path = data_dir / ui_dump_file
        # corrections_path = data_dir / Path("corrections.json")
        corrections = [
            t
            for p in data_dir.glob("corrections-*.json")
            for t in ExtendedPath(p).read_json()
        ]
        ui_data = ExtendedPath(ui_dump_path).read_json()["data"]
        correct_set = {
            c["code"] for c in corrections if c["value"]["status"] == "Correct"
        }
        correction_map = {
            c["code"]: c["value"]["correction"]
            for c in corrections
            if c["value"]["status"] == "Incorrect"
        }
        for d in ui_data:
            orig_audio_path = (data_dir / Path(d["audio_path"])).absolute()
            if d["utterance_id"] in correct_set:
                d["corrected_from"] = d["text"]
                yield d
            elif d["utterance_id"] in correction_map:
                correct_text = correction_map[d["utterance_id"]]
                if skip_incorrect:
                    ap = d["audio_path"]
                    print(
                        f"skipping incorrect {ap} corrected to {correct_text}"
                    )
                    orig_audio_path.unlink()
                else:
                    new_fname = tscript_uuid_fname(correct_text)
                    rename_result[new_fname] = {
                        "orig_text": d["text"],
                        "correct_text": correct_text,
                        "orig_id": d["utterance_id"],
                    }
                    new_name = str(Path(new_fname).with_suffix(".wav"))
                    new_audio_path = orig_audio_path.with_name(new_name)
                    orig_audio_path.replace(new_audio_path)
                    new_filepath = str(
                        Path(d["audio_path"]).with_name(new_name)
                    )
                    d["corrected_from"] = d["text"]
                    d["text"] = correct_text
                    d["audio_path"] = new_filepath
                    yield d
            else:
                if skip_inaudible:
                    orig_audio_path.unlink()
                else:
                    d["corrected_from"] = d["text"]
                    d["text"] = ""
                    yield d

    dataset_dir = data_manifest_path.parent
    dataset_name = dataset_dir.name
    backup_dir = dataset_dir.with_name(dataset_name + ".bkp")
    if not backup_dir.exists():
        typer.echo(f"backing up to {backup_dir}")
        shutil.copytree(str(dataset_dir), str(backup_dir))
    renames = {}
    corrected_ui_dump = list(correct_ui_dump(data_dir, renames))
    ExtendedPath(data_dir / ui_dump_file).write_json(
        {"data": corrected_ui_dump}
    )
    corrected_manifest = (
        {
            "audio_filepath": d["audio_path"],
            "duration": d["duration"],
            "text": d["text"],
        }
        for d in corrected_ui_dump
    )
    asr_manifest_writer(data_manifest_path, corrected_manifest)
    ExtendedPath(renames_path).write_json(renames)


def main():
    app()


if __name__ == "__main__":
    main()

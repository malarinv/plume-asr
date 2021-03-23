import io
import os
import re
import json
import wave
import logging
from pathlib import Path
from functools import partial
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import subprocess
import shutil
from urllib.parse import urlsplit

# from .lazy_loader import LazyLoader
from .lazy_import import lazy_callable, lazy_module

# from ruamel.yaml import YAML
# import boto3
import typer

# import pymongo
# from slugify import slugify
# import pydub
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display as audio_display
# from natural.date import compress
# from num2words import num2words
from tqdm import tqdm
from datetime import timedelta

# from .transcribe import triton_transcribe_grpc_gen
# from .eval import app as eval_app
from .tts import app as tts_app
from .transcribe import app as transcribe_app
from .align import app as align_app

boto3 = lazy_module("boto3")
pymongo = lazy_module("pymongo")
pydub = lazy_module("pydub")
audio_display = lazy_module("librosa.display")
plt = lazy_module("matplotlib.pyplot")
librosa = lazy_module("librosa")
YAML = lazy_callable("ruamel.yaml.YAML")
num2words = lazy_callable("num2words.num2words")
slugify = lazy_callable("slugify.slugify")
compress = lazy_callable("natural.date.compress")

app = typer.Typer()
app.add_typer(tts_app, name="tts")
app.add_typer(align_app, name="align")
app.add_typer(transcribe_app, name="transcribe")


@app.callback()
def utils():
    """
    utils sub commands
    """


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def manifest_str(path, dur, text):
    return (
        json.dumps({"audio_filepath": path, "duration": round(dur, 1), "text": text})
        + "\n"
    )


def duration_str(seconds):
    return compress(timedelta(seconds=seconds), pad=" ")


def replace_digit_symbol(w2v_out):
    num_int_map = {num2words(i): str(i) for i in range(10)}
    out = w2v_out.lower()
    for (k, v) in num_int_map.items():
        out = re.sub(k, v, out)
    return out


def discard_except_digits(inp):
    return re.sub("[^0-9]", "", inp)


def digits_to_chars(text):
    num_tokens = [num2words(c) + " " if "0" <= c <= "9" else c for c in text]
    return ("".join(num_tokens)).lower()


def replace_redundant_spaces_with(text, sub):
    return re.sub(" +", sub, text)


def space_out(text):
    letters = " ".join(list(text))
    return letters


def wav_bytes(audio_bytes, frame_rate=24000):
    wf_b = io.BytesIO()
    with wave.open(wf_b, mode="w") as wf:
        wf.setnchannels(1)
        wf.setframerate(frame_rate)
        wf.setsampwidth(2)
        wf.writeframesraw(audio_bytes)
    return wf_b.getvalue()


def tscript_uuid_fname(transcript):
    return str(uuid4()) + "_" + slugify(transcript, max_length=8)


def run_shell(cmd_str, work_dir="."):
    cwd_path = Path(work_dir).absolute()
    p = subprocess.Popen(
        cmd_str,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        shell=True,
        cwd=cwd_path,
    )
    for line in p.stdout:
        print(line.replace(b"\n", b"").decode("utf-8"))


def upload_s3(dataset_path, s3_path):
    run_shell(f"aws s3 sync {dataset_path} {s3_path}")


def copy_s3(dataset_path, s3_path):
    run_shell(f"aws s3 cp {dataset_path} {s3_path}")


def get_download_path(s3_uri, output_path):
    s3_uri_p = urlsplit(s3_uri)
    download_path = output_path / Path(s3_uri_p.path[1:])
    download_path.parent.mkdir(exist_ok=True, parents=True)
    return download_path


def s3_downloader():
    s3 = boto3.client("s3")

    def download_s3(s3_uri, download_path, verbose=False):
        s3_uri_p = urlsplit(s3_uri)
        download_path.parent.mkdir(exist_ok=True, parents=True)
        if not download_path.exists():
            if verbose:
                print(f"downloading {s3_uri} to {download_path}")
            s3.download_file(s3_uri_p.netloc, s3_uri_p.path[1:], str(download_path))

    return download_s3


def asr_data_writer(dataset_dir, asr_data_source, verbose=False):
    (dataset_dir / Path("wavs")).mkdir(parents=True, exist_ok=True)
    asr_manifest = dataset_dir / Path("manifest.json")
    num_datapoints = 0
    with asr_manifest.open("w") as mf:
        print(f"writing manifest to {asr_manifest}")
        for transcript, audio_dur, wav_data in asr_data_source:
            fname = tscript_uuid_fname(transcript)
            audio_file = dataset_dir / Path("wavs") / Path(fname).with_suffix(".wav")
            audio_file.write_bytes(wav_data)
            rel_data_path = audio_file.relative_to(dataset_dir)
            manifest = manifest_str(str(rel_data_path), audio_dur, transcript)
            mf.write(manifest)
            if verbose:
                print(f"writing '{transcript}' of duration {audio_dur}")
            num_datapoints += 1
    return num_datapoints


def ui_data_generator(dataset_dir, asr_data_source, verbose=False):
    (dataset_dir / Path("wavs")).mkdir(parents=True, exist_ok=True)
    (dataset_dir / Path("wav_plots")).mkdir(parents=True, exist_ok=True)

    def data_fn(
        transcript,
        audio_dur,
        wav_data,
        caller_name,
        aud_seg,
        fname,
        audio_file,
        num_datapoints,
        rel_data_path,
    ):
        png_path = Path(fname).with_suffix(".png")
        rel_plot_path = Path("wav_plots") / png_path
        wav_plot_path = dataset_dir / rel_plot_path
        if not wav_plot_path.exists():
            plot_seg(wav_plot_path.absolute(), audio_file)
        return {
            "audio_path": str(rel_data_path),
            "audio_filepath": str(rel_data_path),
            "duration": round(audio_dur, 1),
            "text": transcript,
            "real_idx": num_datapoints,
            "caller": caller_name,
            "utterance_id": fname,
            "plot_path": str(rel_plot_path),
        }

    num_datapoints = 0
    data_funcs = []
    for transcript, audio_dur, wav_data, caller_name, aud_seg in asr_data_source:
        fname = str(uuid4()) + "_" + slugify(transcript, max_length=8)
        audio_file = (
            dataset_dir / Path("wavs") / Path(fname).with_suffix(".wav")
        ).absolute()
        audio_file.write_bytes(wav_data)
        # audio_path = str(audio_file)
        rel_data_path = audio_file.relative_to(dataset_dir.absolute())
        data_funcs.append(
            partial(
                data_fn,
                transcript,
                audio_dur,
                wav_data,
                caller_name,
                aud_seg,
                fname,
                audio_file,
                num_datapoints,
                rel_data_path,
            )
        )
        num_datapoints += 1
    ui_data = parallel_apply(lambda x: x(), data_funcs)
    return ui_data, num_datapoints


def ui_dump_manifest_writer(dataset_dir, asr_data_source, verbose=False):
    dump_data, num_datapoints = ui_data_generator(
        dataset_dir, asr_data_source, verbose=verbose
    )

    asr_manifest = dataset_dir / Path("manifest.json")
    asr_manifest_writer(asr_manifest, dump_data, verbose=verbose)
    # with asr_manifest.open("w") as mf:
    #     print(f"writing manifest to {asr_manifest}")
    #     for d in dump_data:
    #         rel_data_path = d["audio_path"]
    #         audio_dur = d["duration"]
    #         transcript = d["text"]
    #         manifest = manifest_str(str(rel_data_path), audio_dur, transcript)
    #         mf.write(manifest)
    ui_dump_file = dataset_dir / Path("ui_dump.json")
    ExtendedPath(ui_dump_file).write_json({"data": dump_data}, verbose=verbose)
    return num_datapoints


def asr_manifest_reader(data_manifest_path: Path):
    print(f"reading manifest from {data_manifest_path}")
    with data_manifest_path.open("r") as pf:
        data_jsonl = pf.readlines()
    data_data = [json.loads(v) for v in data_jsonl]
    for p in data_data:
        p["audio_path"] = data_manifest_path.parent / Path(p["audio_filepath"])
        p["text"] = p["text"].strip()
        yield p


def asr_manifest_writer(asr_manifest_path: Path, manifest_str_source, verbose=False):
    with asr_manifest_path.open("w") as mf:
        if verbose:
            print(f"writing asr manifest to {asr_manifest_path}")
        for mani_dict in manifest_str_source:
            manifest = manifest_str(
                mani_dict["audio_filepath"], mani_dict["duration"], mani_dict["text"]
            )
            mf.write(manifest)


def asr_test_writer(out_file_path: Path, source):
    def dd_str(dd, idx):
        path = dd["audio_filepath"]
        # dur = dd["duration"]
        # return f"SAY {idx}\nPAUSE 3\nPLAY {path}\nPAUSE 3\n\n"
        return f"PAUSE 2\nPLAY {path}\nPAUSE 60\n\n"

    res_file = out_file_path.with_suffix(".result.json")
    with out_file_path.open("w") as of:
        print(f"opening {out_file_path} for writing test")
        results = []
        idx = 0
        for ui_dd in source:
            results.append(ui_dd)
            out_str = dd_str(ui_dd, idx)
            of.write(out_str)
            idx += 1
        of.write("DO_HANGUP\n")
        ExtendedPath(res_file).write_json(results)


def batch(iterable, n=1):
    ls = len(iterable)
    return [iterable[ndx : min(ndx + n, ls)] for ndx in range(0, ls, n)]


class ExtendedPath(type(Path())):
    """docstring for ExtendedPath."""

    def read_json(self, verbose=False):
        if verbose:
            print(f"reading json from {self}")
        with self.open("r") as jf:
            return json.load(jf)

    def read_yaml(self, verbose=False):
        yaml = YAML(typ="safe", pure=True)
        if verbose:
            print(f"reading yaml from {self}")
        with self.open("r") as yf:
            return yaml.load(yf)

    def read_jsonl(self, verbose=False):
        if verbose:
            print(f"reading jsonl from {self}")
        with self.open("r") as jf:
            for ln in jf.readlines():
                yield json.loads(ln)

    def write_json(self, data, verbose=False):
        if verbose:
            print(f"writing json to {self}")
        self.parent.mkdir(parents=True, exist_ok=True)
        with self.open("w") as jf:
            json.dump(data, jf, indent=2)

    def write_yaml(self, data, verbose=False):
        yaml = YAML()
        if verbose:
            print(f"writing yaml to {self}")
        with self.open("w") as yf:
            yaml.dump(data, yf)

    def write_jsonl(self, data, verbose=False):
        if verbose:
            print(f"writing jsonl to {self}")
        self.parent.mkdir(parents=True, exist_ok=True)
        with self.open("w") as jf:
            for d in data:
                jf.write(json.dumps(d) + "\n")


def get_mongo_coll(uri):
    ud = pymongo.uri_parser.parse_uri(uri)
    conn = pymongo.MongoClient(uri)
    return conn[ud["database"]][ud["collection"]]


def get_mongo_conn(host="", port=27017, db="db", col="collection"):
    mongo_host = host if host else os.environ.get("MONGO_HOST", "localhost")
    mongo_uri = f"mongodb://{mongo_host}:{port}/"
    return pymongo.MongoClient(mongo_uri)[db][col]


def strip_silence(sound):
    from pydub.silence import detect_leading_silence

    start_trim = detect_leading_silence(sound)
    end_trim = detect_leading_silence(sound.reverse())
    duration = len(sound)
    return sound[start_trim : duration - end_trim]


def plot_seg(wav_plot_path, audio_path):
    fig = plt.Figure()
    ax = fig.add_subplot()
    (y, sr) = librosa.load(str(audio_path))
    audio_display.waveplot(y=y, sr=sr, ax=ax)
    with wav_plot_path.open("wb") as wav_plot_f:
        fig.set_tight_layout(True)
        fig.savefig(wav_plot_f, format="png", dpi=50)


def parallel_apply(fn, iterable, workers=8, pool="thread"):
    if pool == "thread":
        with ThreadPoolExecutor(max_workers=workers) as exe:
            print(f"parallelly applying {fn}")
            return [
                res
                for res in tqdm(
                    exe.map(fn, iterable), position=0, leave=True, total=len(iterable)
                )
            ]
    elif pool == "process":
        with ProcessPoolExecutor(max_workers=workers) as exe:
            print(f"parallelly applying {fn}")
            return [
                res
                for res in tqdm(
                    exe.map(fn, iterable), position=0, leave=True, total=len(iterable)
                )
            ]
    else:
        raise Exception(f"unsupported pool type - {pool}")


def generate_filter_map(src_dataset_path, dest_dataset_path, data_file):
    min_nums = 3
    max_duration = 1 * 60 * 60
    skip_duration = 1 * 60 * 60

    def filtered_max_dur():
        wav_duration = 0
        for s in ExtendedPath(data_file).read_jsonl():
            nums = re.sub(" ", "", s["text"])
            if len(nums) >= min_nums:
                wav_duration += s["duration"]
                shutil.copy(
                    src_dataset_path / Path(s["audio_filepath"]),
                    dest_dataset_path / Path(s["audio_filepath"]),
                )
                yield s
            if wav_duration > max_duration:
                break
        typer.echo(f"filtered only {duration_str(wav_duration)} of audio")

    def filtered_skip_dur():
        wav_duration = 0
        for s in ExtendedPath(data_file).read_jsonl():
            nums = re.sub(" ", "", s["text"])
            if len(nums) >= min_nums:
                wav_duration += s["duration"]
            if wav_duration <= skip_duration:
                continue
            elif len(nums) >= min_nums:
                yield s
                shutil.copy(
                    src_dataset_path / Path(s["audio_filepath"]),
                    dest_dataset_path / Path(s["audio_filepath"]),
                )
        typer.echo(f"skipped {duration_str(skip_duration)} of audio")

    def filtered_blanks():
        blank_count = 0
        for s in ExtendedPath(data_file).read_jsonl():
            nums = re.sub(" ", "", s["text"])
            if nums != "":
                blank_count += 1
                shutil.copy(
                    src_dataset_path / Path(s["audio_filepath"]),
                    dest_dataset_path / Path(s["audio_filepath"]),
                )
                yield s
        typer.echo(f"filtered {blank_count} blank samples")

    def filtered_transform_digits():
        count = 0
        for s in ExtendedPath(data_file).read_jsonl():
            count += 1
            digit_text = replace_digit_symbol(s["text"])
            only_digits = discard_except_digits(digit_text)
            char_text = digits_to_chars(only_digits)
            shutil.copy(
                src_dataset_path / Path(s["audio_filepath"]),
                dest_dataset_path / Path(s["audio_filepath"]),
            )
            s["text"] = char_text
            yield s
        typer.echo(f"transformed {count} samples")

    def filtered_extract_chars():
        count = 0
        for s in ExtendedPath(data_file).read_jsonl():
            count += 1
            no_digits = digits_to_chars(s["text"]).upper()
            only_chars = re.sub("[^A-Z'\b]", " ", no_digits)
            filter_text = replace_redundant_spaces_with(only_chars, " ").strip()
            shutil.copy(
                src_dataset_path / Path(s["audio_filepath"]),
                dest_dataset_path / Path(s["audio_filepath"]),
            )
            s["text"] = filter_text
            yield s
        typer.echo(f"transformed {count} samples")

    def filtered_resample():
        count = 0
        for s in ExtendedPath(data_file).read_jsonl():
            count += 1
            src_aud = pydub.AudioSegment.from_file(
                src_dataset_path / Path(s["audio_filepath"])
            )
            dst_aud = src_aud.set_channels(1).set_sample_width(1).set_frame_rate(24000)
            dst_aud.export(dest_dataset_path / Path(s["audio_filepath"]), format="wav")
            yield s
        typer.echo(f"transformed {count} samples")

    filter_kind_map = {
        "max_dur_1hr_min3num": filtered_max_dur,
        "skip_dur_1hr_min3num": filtered_skip_dur,
        "blanks": filtered_blanks,
        "transform_digits": filtered_transform_digits,
        "extract_chars": filtered_extract_chars,
        "resample_ulaw24kmono": filtered_resample,
    }
    return filter_kind_map

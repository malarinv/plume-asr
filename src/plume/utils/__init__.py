import io
import os
import re
import wave
import logging
import subprocess
import shutil
import random
from pathlib import Path
from functools import partial
from uuid import uuid4
from urllib.parse import urlsplit

# from .lazy_loader import LazyLoader

# from ruamel.yaml import YAML
# import boto3
import typer
from tqdm import tqdm

# import pymongo
# from slugify import slugify
# import pydub
# import matplotlib.pyplot as plt
# import librosa
# import librosa.display as audio_display
# from natural.date import compress
# from num2words import num2words
import datetime
import six

# from .transcribe import triton_transcribe_grpc_gen
# from .eval import app as eval_app
from .manifest import (
    asr_manifest_writer,
    asr_manifest_reader,
    manifest_str,
)  # noqa
from .lazy_import import lazy_callable, lazy_module
from .parallel import parallel_apply
from .extended_path import ExtendedPath
from .tts import app as tts_app
from .transcribe import app as transcribe_app
from .align import app as align_app
from .encrypt import app as encrypt_app, wav_cryptor, text_cryptor  # noqa
from .regentity import (  # noqa
    num_replacer,
    alnum_replacer,
    num_keeper,
    alnum_keeper,
    default_num_rules,
    default_num_only_rules,
    default_alnum_rules,
    entity_replacer_keeper,
    vocab_corrector_gen,
)

boto3 = lazy_module("boto3")
pymongo = lazy_module("pymongo")
pydub = lazy_module("pydub")
audio_display = lazy_module("librosa.display")
plt = lazy_module("matplotlib.pyplot")
librosa = lazy_module("librosa")
YAML = lazy_callable("ruamel.yaml.YAML")
num2words = lazy_callable("num2words.num2words")
slugify = lazy_callable("slugify.slugify")

app = typer.Typer()
app.add_typer(encrypt_app)
app.add_typer(tts_app, name="tts")
app.add_typer(align_app, name="align")
app.add_typer(transcribe_app, name="transcribe")


@app.callback()
def utils():
    """
    utils sub commands
    """


log_fmt_str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=log_fmt_str)
logger = logging.getLogger(__name__)


# Precalculated timestamps
TIME_MINUTE = 60
TIME_HOUR = 3600
TIME_DAY = 86400
TIME_WEEK = 604800


def compress(t, show_hours=False, sign=False, pad=""):
    """
    Convert the input to compressed format, works with a
    :class:`datetime.timedelta` object or a number that represents the number
    of seconds you want to compress.  If you supply a timestamp or a
    :class:`datetime.datetime` object, it will give the delta relative to the
    current time.
    You can enable showing a sign in front of the compressed format with the
    ``sign`` parameter, the default is not to show signs.
    Optionally, you can chose to pad the output. If you wish your values to be
    separated by spaces, set ``pad`` to ``' '``.
    :param t: seconds or :class:`datetime.timedelta` object
    :param sign: default ``False``
    :param pad: default ``''``
    >>> print(compress(0))
    0s
    >>> print(compress(1))
    1s
    >>> print(compress(12))
    12s
    >>> print(compress(123))
    2m3s
    >>> print(compress(1234))
    20m34s
    >>> print(compress(12345))
    3h25m45s
    >>> print(compress(123456))
    1d10h17m36s
    ==============
    src: https://github.com/tehmaze/natural/blob/master/natural/date.py
    """

    if isinstance(t, datetime.timedelta):
        seconds = t.seconds + (t.days * 86400)
    elif isinstance(t, six.integer_types + (float,)):
        return compress(datetime.timedelta(seconds=t), sign, pad)
    else:
        raise Exception("Invalid time format")

    parts = []
    if sign:
        parts.append("-" if t.days < 0 else "+")

    if not show_hours:
        weeks, seconds = divmod(seconds, TIME_WEEK)
        days, seconds = divmod(seconds, TIME_DAY)
    hours, seconds = divmod(seconds, TIME_HOUR)
    minutes, seconds = divmod(seconds, TIME_MINUTE)

    if not show_hours:
        if weeks:
            parts.append(("%dw") % (weeks,))
        if days:
            parts.append(("%dd") % (days,))
    if hours:
        parts.append(("%dh") % (hours,))
    if minutes:
        parts.append(("%dm") % (minutes,))
    if seconds or len(parts) == 0:
        parts.append(("%ds") % (seconds,))

    return pad.join(parts)


def duration_str(seconds, show_hours=False):
    t = datetime.timedelta(seconds=seconds)
    return compress(t, show_hours=show_hours, pad=" ")


def replace_digit_symbol(w2v_out, num_range=10):
    def rep_i(i):
        return (num2words(i).replace("-", " "), str(i))

    num_int_map = [rep_i(i) for i in reversed(range(num_range))]
    out = w2v_out.lower()
    for (k, v) in num_int_map:
        out = re.sub(k, v, out)
    return out


def num_keeper_orig(num_range=10, extra_rules=[]):
    num_int_map_ty = [
        (
            r"\b" + num2words(i) + r"\b",
            " " + str(i) + " ",
        )
        for i in reversed(range(num_range))
    ]
    re_rules = [
        (re.compile(k, re.IGNORECASE), v)
        for (k, v) in [
            # (r"[ ;,.]", " "),
            (r"\bdouble(?: |-)(\w+)\b", "\\1 \\1"),
            (r"\btriple(?: |-)(\w+)\b", "\\1 \\1 \\1"),
            (r"hundred", "00"),
            (r"\boh\b", " 0 "),
            (r"\bo\b", " 0 "),
        ]
        + num_int_map_ty
    ] + [(re.compile(k), v) for (k, v) in extra_rules]

    def merge_intervals(intervals):
        # https://codereview.stackexchange.com/a/69249
        sorted_by_lower_bound = sorted(intervals, key=lambda tup: tup[0])
        merged = []

        for higher in sorted_by_lower_bound:
            if not merged:
                merged.append(higher)
            else:
                lower = merged[-1]
                # test for intersection between lower and higher:
                # we know via sorting that lower[0] <= higher[0]
                if higher[0] <= lower[1]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1] = (
                        lower[0],
                        upper_bound,
                    )  # replace by merged interval
                else:
                    merged.append(higher)
        return merged

    # merging interval tree for optimal # https://www.geeksforgeeks.org/interval-tree/

    def keep_numeric_literals(w2v_out):
        # out = w2v_out.lower()
        out = re.sub(r"[ ;,.]", " ", w2v_out).strip()
        # out = " " + out.strip() + " "
        # out = re.sub(r"double (\w+)", "\\1 \\1", out)
        # out = re.sub(r"triple (\w+)", "\\1 \\1 \\1", out)
        num_spans = []
        for (k, v) in re_rules:  # [94:]:
            matches = k.finditer(out)
            for m in matches:
                # num_spans.append((k, m.span()))
                num_spans.append(m.span())
            # out = re.sub(k, v, out)
        merged = merge_intervals(num_spans)
        num_ents = len(merged)
        keep_out = " ".join((out[s[0] : s[1]] for s in merged))
        return keep_out, num_ents

    return keep_numeric_literals


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


def random_segs(total, min_val, max_val):
    out_list = []
    rand_total = prev_start = 0
    while True:
        if total < rand_total + min_val or total < rand_total:
            break
        sample = random.randint(min_val, max_val)
        if total - rand_total < max_val:
            break
        if total - rand_total < max_val + min_val:
            sample = random.randint(min_val, max_val - min_val)
        prev_start = rand_total
        if 0 < rand_total + sample - total < max_val:
            break
        rand_total += sample
        out_list.append((prev_start, rand_total))
    out_list.append((rand_total, total))
    return out_list


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


def run_shell(cmd_str, work_dir=".", verbose=True):
    cwd_path = Path(work_dir).absolute()
    if verbose:
        with subprocess.Popen(
            cmd_str,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=True,
            cwd=cwd_path,
        ) as p:
            for line in p.stdout:
                print(line.replace(b"\n", b"").decode("utf-8"))
    else:
        subprocess.run(cmd_str, shell=True, cwd=cwd_path, capture_output=True)


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
            dp_s = str(download_path)
            s3.download_file(s3_uri_p.netloc, s3_uri_p.path[1:], dp_s)

    return download_s3


def asr_data_writer(dataset_dir, asr_data_source, verbose=False):
    (dataset_dir / Path("wavs")).mkdir(parents=True, exist_ok=True)
    asr_manifest = dataset_dir / Path("manifest.json")
    num_datapoints = 0
    with asr_manifest.open("w") as mf:
        print(f"writing manifest to {asr_manifest}")
        for transcript, audio_dur, wav_data in asr_data_source:
            fname = tscript_uuid_fname(transcript)
            wav_fname = Path(fname).with_suffix(".wav")
            audio_file = dataset_dir / Path("wavs") / wav_fname
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
    for (
        transcript,
        audio_dur,
        wav_data,
        caller_name,
        aud_seg,
    ) in asr_data_source:
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
    ui_dump_file = dataset_dir / Path("ui_dump.json")
    ExtendedPath(ui_dump_file).write_json({"data": dump_data}, verbose=verbose)
    return num_datapoints


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


def generate_filter_map(src_dataset_path, dest_dataset_path, data_file):
    min_nums = 3
    max_duration = 1 * 60 * 60
    skip_duration = 1 * 60 * 60
    max_sample_dur = 20
    min_sample_dur = 2
    verbose = True

    src_data_enum = (
        tqdm(list(ExtendedPath(data_file).read_jsonl()))
        if verbose
        else ExtendedPath(data_file).read_jsonl()
    )

    def filtered_max_dur():
        wav_duration = 0
        for s in src_data_enum:
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
        for s in src_data_enum:
            nums = re.sub(" ", "", s["text"])
            if len(nums) >= min_nums:
                wav_duration += s["duration"]
            if wav_duration <= skip_duration:
                continue
            elif len(nums) >= min_nums:
                shutil.copy(
                    src_dataset_path / Path(s["audio_filepath"]),
                    dest_dataset_path / Path(s["audio_filepath"]),
                )
                yield s
        typer.echo(f"skipped {duration_str(skip_duration)} of audio")

    def filtered_blanks():
        blank_count = total_count = 0
        for s in src_data_enum:
            total_count += 1
            nums = re.sub(" ", "", s["text"])
            if nums != "":
                shutil.copy(
                    src_dataset_path / Path(s["audio_filepath"]),
                    dest_dataset_path / Path(s["audio_filepath"]),
                )
                yield s
            else:
                blank_count += 1
        typer.echo(f"filtered {blank_count} of {total_count} blank samples")

    def filtered_maxmin_sample_dur():
        import soundfile

        max_dur_count = 0
        for s in src_data_enum:
            wav_real_duration = soundfile.info(
                src_dataset_path / Path(s["audio_filepath"])
            ).duration
            wav_duration = min(wav_real_duration, s["duration"])
            if (
                wav_duration <= max_sample_dur
                and wav_duration > min_sample_dur
            ):
                shutil.copy(
                    src_dataset_path / Path(s["audio_filepath"]),
                    dest_dataset_path / Path(s["audio_filepath"]),
                )
                yield s
            else:
                max_dur_count += 1
        typer.echo(
            f"filtered {max_dur_count} samples longer thans {max_sample_dur}s and shorter than {min_sample_dur}s"
        )

    def filtered_transform_digits():
        count = 0
        for s in src_data_enum:
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
        for s in src_data_enum:
            count += 1
            no_digits = digits_to_chars(s["text"]).upper()
            only_chars = re.sub("[^A-Z'\b]", " ", no_digits)
            filter_text = replace_redundant_spaces_with(
                only_chars, " "
            ).strip()
            shutil.copy(
                src_dataset_path / Path(s["audio_filepath"]),
                dest_dataset_path / Path(s["audio_filepath"]),
            )
            s["text"] = filter_text
            yield s
        typer.echo(f"transformed {count} samples")

    def filtered_resample():
        count = 0
        for s in src_data_enum:
            count += 1
            src_aud = pydub.AudioSegment.from_file(
                src_dataset_path / Path(s["audio_filepath"])
            )
            dst_aud = (
                src_aud.set_channels(1)
                .set_sample_width(1)
                .set_frame_rate(24000)
            )
            dst_aud.export(
                dest_dataset_path / Path(s["audio_filepath"]), format="wav"
            )
            yield s
        typer.echo(f"transformed {count} samples")

    def filtered_msec_to_sec():
        count = 0
        for s in src_data_enum:
            count += 1
            s["duration"] = s["duration"] / 1000
            shutil.copy(
                src_dataset_path / Path(s["audio_filepath"]),
                dest_dataset_path / Path(s["audio_filepath"]),
            )
            yield s
        typer.echo(f"transformed {count} samples")

    def filtered_blank_hr_max_dur():
        max_duration = 3 * 60 * 60
        wav_duration = 0
        for s in src_data_enum:
            # nums = re.sub(" ", "", s["text"])
            s[
                "text"
            ] = "gAAAAABgq2FR6ajbhMsDmWRQBzX6gIzyAG5sMwFihGeV7E_6eVJqqF78yzmtTJPsJAOJEEXhJ9Z45MrYNgE1sq7VUdsBVGh2cw=="
            if (
                s["duration"] >= min_sample_dur
                and s["duration"] <= max_sample_dur
            ):
                wav_duration += s["duration"]
                shutil.copy(
                    src_dataset_path / Path(s["audio_filepath"]),
                    dest_dataset_path / Path(s["audio_filepath"]),
                )
                yield s
            if wav_duration > max_duration:
                break
        typer.echo(f"filtered only {duration_str(wav_duration)} of audio")

    filter_kind_map = {
        "max_dur_1hr_min3num": filtered_max_dur,
        "skip_dur_1hr_min3num": filtered_skip_dur,
        "blanks": filtered_blanks,
        "transform_digits": filtered_transform_digits,
        "extract_chars": filtered_extract_chars,
        "resample_ulaw24kmono": filtered_resample,
        "maxmin_sample_dur": filtered_maxmin_sample_dur,
        "msec_to_sec": filtered_msec_to_sec,
        "blank_3hr_max_dur": filtered_blank_hr_max_dur,
    }
    return filter_kind_map

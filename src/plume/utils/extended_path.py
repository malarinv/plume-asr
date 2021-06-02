from pathlib import Path
import json

from .lazy_import import lazy_module

yaml = lazy_module("ruamel.yaml")
pydub = lazy_module("pydub")


class ExtendedPath(type(Path())):
    """docstring for ExtendedPath."""

    def read_json(self, verbose=False):
        if verbose:
            print(f"reading json from {self}")
        with self.open("r") as jf:
            return json.load(jf)

    def read_yaml(self, verbose=False):
        yaml_o = yaml.YAML(typ="safe", pure=True)
        if verbose:
            print(f"reading yaml from {self}")
        with self.open("r") as yf:
            return yaml_o.load(yf)

    def read_jsonl(self, verbose=False):
        if verbose:
            print(f"reading jsonl from {self}")
        with self.open("r") as jf:
            for ln in jf.readlines():
                yield json.loads(ln)

    def read_audio_segment(self):
        return pydub.AudioSegment.from_file(self)

    def write_json(self, data, verbose=False):
        if verbose:
            print(f"writing json to {self}")
        self.parent.mkdir(parents=True, exist_ok=True)
        with self.open("w") as jf:
            json.dump(data, jf, indent=2)

    def write_yaml(self, data, verbose=False):
        yaml_o = yaml.YAML()
        if verbose:
            print(f"writing yaml to {self}")
        with self.open("w") as yf:
            yaml_o.dump(data, yf)

    def write_jsonl(self, data, verbose=False):
        if verbose:
            print(f"writing jsonl to {self}")
        self.parent.mkdir(parents=True, exist_ok=True)
        with self.open("w") as jf:
            for d in data:
                jf.write(json.dumps(d) + "\n")

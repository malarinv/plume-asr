from plume.utils import lazy_module
import typer

rpyc = lazy_module("rpyc")

app = typer.Typer()


class ASRService(rpyc.Service):
    def __init__(self, asr_recognizer):
        self.asr = asr_recognizer

    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        pass

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def exposed_transcribe(
        self, utterance: bytes
    ):  # this is an exposed method
        speech_audio = self.asr.transcribe(utterance)
        return speech_audio

    def exposed_transcribe_cb(
        self, utterance: bytes, respond
    ):  # this is an exposed method
        speech_audio = self.asr.transcribe(utterance)
        respond(speech_audio)

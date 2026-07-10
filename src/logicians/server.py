"""FastAPI server for the live improvisation frontend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .groove import MarkovDrumModel
from .live_session import LiveSessionConfig, SessionState, get_session
from .midi_io import list_midi_ports

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="The Logicians", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_ports_cache: dict[str, list[str]] | None = None


class SessionConfigBody(BaseModel):
    midi_input: str = "IAC Driver Bus 1"
    midi_output: str = "IAC Driver Bus 2"
    drum_output: str | None = "IAC Driver Bus 2"
    drum_channel: int = 1
    drum_variation: float = 0.3
    drum_style: str | None = None
    bass_output: str | None = "IAC Driver Bus 2"
    bass_channel: int = 2
    bass_variation: float = 0.0
    tempo: float = 120.0
    bars: int = 4
    time_signature: str = "4/4"
    sync: str = "first-note"
    seed: int | None = None
    density: str = "medium"
    count_in: int = 0
    playback_loop: int = 3
    key: str | None = None
    continuous: bool = True


def _status_dict() -> dict[str, Any]:
    try:
        status = get_session().get_status()
        return {
            "state": status.state.value,
            "loop": status.loop,
            "bar": status.bar,
            "beat": round(status.beat, 2),
            "key_label": status.key_label,
            "chord_progression": status.chord_progression,
            "message": status.message,
            "error": status.error,
            "captured_notes": status.captured_notes,
            "harmony_notes": status.harmony_notes,
            "capture_warning": status.capture_warning,
        }
    except Exception as exc:
        return {"state": "error", "error": str(exc), "loop": 0, "bar": 0, "beat": 0.0,
                "key_label": None, "chord_progression": [], "message": None,
                "captured_notes": 0, "harmony_notes": 0, "capture_warning": None}


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": str(exc)})


def _to_config(body: SessionConfigBody) -> LiveSessionConfig:
    return LiveSessionConfig(
        midi_input=body.midi_input,
        midi_output=body.midi_output,
        drum_output=body.drum_output or None,
        drum_channel=body.drum_channel,
        drum_variation=body.drum_variation,
        drum_style=body.drum_style,
        bass_output=body.bass_output or None,
        bass_channel=body.bass_channel,
        bass_variation=body.bass_variation,
        tempo=body.tempo,
        bars=body.bars,
        time_signature=body.time_signature,
        sync=body.sync,
        seed=body.seed,
        density=body.density,
        count_in=body.count_in,
        playback_loop=body.playback_loop,
        key=body.key,
        continuous=body.continuous,
    )


@app.get("/api/ports")
def api_ports() -> dict[str, list[str]]:
    global _ports_cache
    if _ports_cache is None:
        inputs, outputs = list_midi_ports()
        _ports_cache = {"inputs": inputs, "outputs": outputs}
    return _ports_cache


@app.get("/api/drum-styles")
def api_drum_styles() -> dict[str, list[str]]:
    model = MarkovDrumModel.load()
    if model is None:
        return {"styles": []}
    return {"styles": model.styles()}


@app.get("/api/session/status")
def api_session_status() -> dict[str, Any]:
    return _status_dict()


@app.post("/api/session/start")
def api_session_start(body: SessionConfigBody) -> dict[str, Any]:
    session = get_session()
    if session.state not in (SessionState.IDLE, SessionState.STOPPED):
        raise HTTPException(status_code=409, detail=f"Cannot start from state {session.state.value}")
    try:
        session.start_session(_to_config(body))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _status_dict()


@app.post("/api/session/improvise")
def api_session_improvise() -> dict[str, Any]:
    session = get_session()
    if session.state != SessionState.SYNCED:
        raise HTTPException(status_code=409, detail=f"Cannot improvise from state {session.state.value}")
    try:
        session.improvise(wait_for_boundary=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _status_dict()


@app.post("/api/session/stop")
def api_session_stop() -> dict[str, Any]:
    get_session().stop()
    return _status_dict()


@app.post("/api/session/reset")
def api_session_reset() -> dict[str, Any]:
    get_session().reset()
    return _status_dict()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

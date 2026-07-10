# The Logicians — System Architecture

AI-assisted live improvisation for Logic Pro.

```mermaid
flowchart LR
    subgraph LogicIn["Logic Pro (your loop)"]
        LP1[Loop playback]
    end

    subgraph Interfaces["Interfaces"]
        CLI[CLI\ncli.py]
        APP[Live App\ndesktop · server · UI]
        SESSION[Live Session\nlive_session.py]
    end

    subgraph Capture["Capture & parse"]
        MIDI_IN[MIDI Input\nmidi_io.py]
        MIDI_FILE[MIDI File\nparse · ingest]
        QUANT[Quantizer\nquantize.py]
    end

    subgraph Analysis["Analysis"]
        ANALYZER[Loop Analyzer\nkey · chords · density]
        CTX[LoopContext\nmodels.py]
    end

    subgraph Generation["Generation"]
        MELODY[Melody\ngenerator.py]
        DRUMS[Drums\ndrums · groove]
        BASS[Bass\nbass.py]
    end

    subgraph Output["Output"]
        SCHED[Schedulers\nscheduler.py]
        EXPORT[MIDI Export\nexport.py]
    end

    subgraph LogicOut["Logic Pro (improvised parts)"]
        LP2[Melody · drums · bass tracks]
    end

    CLI --> SESSION
    APP --> SESSION
    SESSION --> MIDI_IN

    LP1 -->|IAC MIDI out| MIDI_IN
    MIDI_IN --> MIDI_FILE
    MIDI_FILE --> QUANT
    QUANT --> ANALYZER
    ANALYZER --> CTX

    CTX --> MELODY
    CTX --> DRUMS
    CTX --> BASS

    MELODY --> SCHED
    DRUMS --> SCHED
    BASS --> SCHED
    SCHED --> EXPORT
    EXPORT -->|IAC MIDI in| LP2
    SCHED -->|live playback| LP2
```

## Pipeline (left to right)

| Stage | What happens |
|-------|----------------|
| **1. Input** | Logic plays your loop. MIDI is captured live (IAC bus) or read from a file. |
| **2. Quantize** | Raw notes are snapped to bars, beats, and symbolic positions. |
| **3. Analyze** | The system infers key, chord progression, bass roots, and rhythm density into a `LoopContext`. |
| **4. Generate** | Three generators run in parallel: melody, drums, and bass. |
| **5. Output** | Schedulers send MIDI back to Logic in time with your loop; optionally export to a `.mid` file. |

## Live performance flow

```mermaid
sequenceDiagram
    participant User
    participant App as Live App
    participant Session as live_session.py
    participant Logic as Logic Pro

    User->>App: Start Session
    App->>Session: Open MIDI ports, wait for sync
    User->>Logic: Press Play
    Logic->>Session: First MIDI note (sync clock)
    Session->>App: State: Synced

    User->>App: Improvise
    Session->>Logic: Capture 1 loop via IAC input
    Session->>Session: Analyze → generate melody, drums, bass
    Session->>Logic: Play generated parts on loop boundary
    Note over Session,Logic: Repeats each loop until Stop
```

## Module map

```
the-logicians/
├── cli.py              Command-line entry (analyze, generate, live, serve)
├── desktop.py          Native app window (pywebview)
├── server.py           FastAPI backend for the live UI
├── live_session.py     Sync, capture, improvise state machine
├── midi_io.py          MIDI ports, live capture, file parsing
├── quantize.py         Symbolic quantization
├── analysis.py         Key, chords, density inference
├── generator.py        Rule-based melody generator
├── drums.py            Reactive drum variation
├── bass.py             Bass line generator
├── groove.py           Optional drum styles (Groove MIDI Dataset)
├── scheduler.py        Timed MIDI playback
└── export.py           Write melody to MIDI file
```

## Design note

The `MelodyGenerator` interface is swappable — the rule-based generator can later be replaced by a neural model without changing the MIDI capture, analysis, or scheduling pipeline.

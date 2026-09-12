Voice samples for zero-shot voice cloning.

Add a voice: drop an audio file here (WAV, MP3, FLAC; anything else works
via ffmpeg). Use the filename without extension as the voice name:

  omatts -v YourVoice "Hello world"

Format doesn't matter — any sample rate, stereo, volume are handled
automatically. What matters is the recording:

- one speaker, clean speech, little background noise
- roughly 5-30 seconds is plenty
- good sources: podcast clip, voice memo, interview

List available voices:  omatts voices
Open this folder in the file manager:  omatts voices open

Deleting a file here removes the voice. Cached embeddings live in
.cache/ and are invalidated automatically; delete .cache/ to reclaim
space or force a fresh encode.

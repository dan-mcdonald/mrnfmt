from faster_whisper import WhisperModel
import srt
from sys import argv

if len(argv) < 3:
    print("Usage: python transcribe.py <media_input_file> <srt_output_file>")
    exit(1)

media_input_file = argv[1]
srt_output_file = argv[2]

print("Loading model...")
model = WhisperModel("large-v3", device="cpu", compute_type="float32", download_root="/home/ubuntu/dev/faster-whisper")

print("Loading media file...")
segments, info = model.transcribe(media_input_file, beam_size=5, language="en")

print("Transcribing...")
subs = []
for segment in segments:
    print("[%.02f - %.02f] %s" % (segment.start, segment.end, segment.text))
    subs.append(srt.Subtitle(
        segment.id,
        start=srt.timedelta(seconds=segment.start),
        end=srt.timedelta(seconds=segment.end),
        content=segment.text
    ))

print("Writing srt...")
with open(srt_output_file, "w") as f:
    f.write(srt.compose(subs))

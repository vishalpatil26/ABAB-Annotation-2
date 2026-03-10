import tempfile
import traceback
from pathlib import Path
import gradio as gr
from pipeline import get_pipeline, to_json, to_csv, to_abab_text

UPLOAD_DIR = Path(tempfile.gettempdir()) / "speech_annotation"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

_last_segments = []

def process_audio(audio_path):
    global _last_segments
    if audio_path is None:
        return "⚠️ Please upload an audio file first.", "", ""
    try:
        pipeline = get_pipeline()
        segments = pipeline.process(audio_path)
    except Exception as e:
        return f"❌ Error: {e}\n{traceback.format_exc()}", "", ""

    if not segments:
        return "⚠️ No speech detected.", "", ""

    _last_segments = segments
    abab = to_abab_text(segments)

    rows = "".join(
        f"<tr style='background:{'#f0f4ff' if i%2==0 else 'white'}'>"
        f"<td style='padding:6px 8px;font-weight:bold;color:#1e3a5f'>{s.speaker}</td>"
        f"<td style='padding:6px 8px'>{s.start_fmt}</td>"
        f"<td style='padding:6px 8px'>{s.end_fmt}</td>"
        f"<td style='padding:6px 8px'>{s.text}</td></tr>"
        for i, s in enumerate(segments)
    )
    table = f"""<table style='width:100%;border-collapse:collapse;font-size:14px'>
      <thead style='background:#1e3a5f;color:white'>
        <tr><th style='padding:8px'>Speaker</th><th>Start</th><th>End</th><th style='text-align:left;padding:8px'>Transcript</th></tr>
      </thead><tbody>{rows}</tbody></table>"""

    return f"✅ Done — {len(segments)} segments", abab, table


def export_json():
    if not _last_segments:
        return None
    out = str(UPLOAD_DIR / "annotation.json")
    to_json(_last_segments, out)
    return out

def export_csv():
    if not _last_segments:
        return None
    out = str(UPLOAD_DIR / "annotation.csv")
    to_csv(_last_segments, out)
    return out


with gr.Blocks(title="Speech Annotation Pipeline") as demo:
    gr.Markdown("# 🎙️ Speech Annotation Pipeline")
    gr.Markdown("Upload audio → get speaker-labeled transcripts automatically")

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(label="Upload Audio (.wav/.mp3/.flac)", type="filepath")
            run_btn = gr.Button("▶ Run Annotation", variant="primary")
            status_box = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                json_btn = gr.Button("Export JSON")
                csv_btn  = gr.Button("Export CSV")
            json_file = gr.File(label="JSON")
            csv_file  = gr.File(label="CSV")

        with gr.Column():
            abab_box   = gr.Textbox(label="Editable Transcript (ABAB)", lines=20,
                                     placeholder="Annotations will appear here...")
            table_html = gr.HTML(label="Segment Table")

    run_btn.click(fn=process_audio, inputs=[audio_input],
                  outputs=[status_box, abab_box, table_html])
    json_btn.click(fn=export_json, inputs=[], outputs=[json_file])
    csv_btn.click(fn=export_csv,  inputs=[], outputs=[csv_file])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)

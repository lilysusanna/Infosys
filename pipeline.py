# pipeline.py
import threading
import queue
from realtime_stt import audio_callback  # Your STT function (callback)
from diarization_module import diarize_audio  # Your pyannote diarization function
from summarization_module import summarize_text  # Your HuggingFace/Groq summarizer

# Use whisper-based transcriber wrapper from the project's stt module
from stt_whisper import transcribe_whisper
import json
import os
import time


task_queue = queue.Queue()
result_queue = queue.Queue()


def transcribe_audio(audio_file):
    """Call the whisper transcriber and return the full result dict (or None on error).

    The caller is responsible for writing any JSON/text artifacts to disk.
    """
    try:
        # honor device env if set
        import os
        device = os.environ.get('MODEL_DEVICE')
        res = transcribe_whisper(audio_file, out_json=None, device=device)
        if isinstance(res, dict):
            return res
        # If whisper wrapper returned a string, wrap it into a dict for compatibility
        return {"text": str(res)}
    except Exception:
        return None


def worker():
    while True:
        item = task_queue.get()
        if item is None:
            break
        audio_file = item
        # create a small status file next to the audio file to communicate progress
        status_file = f"{audio_file}.status"
        # persist last job info for UI recovery across reloads
        try:
            import json as _json
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'last_job.json'), 'w', encoding='utf-8') as _lf:
                _json.dump({"audio_file": audio_file, "status": "processing"}, _lf)
        except Exception:
            pass
        # helper to tick progress asynchronously while a stage runs
        class ProgressTicker:
            def __init__(self, path, start_pct=0, end_pct=100, phase=''):
                self.path = path
                self.start = start_pct
                self.end = end_pct
                self.phase = phase
                self._stop = False
                self._thread = None

            def _run(self):
                try:
                    cur = self.start
                    # write initial
                    import json as _json
                    _json_stats = {'phase': self.phase, 'percent': int(cur)}
                    open(self.path, 'w', encoding='utf-8').write(_json.dumps(_json_stats))
                    steps = max(1, (self.end - self.start) // 2)
                    while not self._stop and cur < self.end:
                        time.sleep(1.0)
                        if self._stop:
                            break
                        # small increment
                        cur = min(self.end, cur + max(1, (self.end - self.start) // 8))
                        try:
                            _json_stats = {'phase': self.phase, 'percent': int(cur)}
                            open(self.path, 'w', encoding='utf-8').write(_json.dumps(_json_stats))
                        except Exception:
                            pass
                except Exception:
                    pass

            def start(self):
                self._stop = False
                self._thread = threading.Thread(target=self._run, daemon=True)
                self._thread.start()

            def stop(self, final_pct=None, final_phase=None):
                self._stop = True
                try:
                    if self._thread:
                        self._thread.join(timeout=0.1)
                except Exception:
                    pass
                try:
                    import json as _json
                    _json_stats = {'phase': final_phase or self.phase, 'percent': int(final_pct) if final_pct is not None else self.end}
                    open(self.path, 'w', encoding='utf-8').write(_json.dumps(_json_stats))
                except Exception:
                    try:
                        open(self.path, 'w').write(final_phase or self.phase)
                    except Exception:
                        pass
        try:
            import json as _json
            _json_stats = {'phase': 'transcribing', 'percent': 5}
            open(status_file, 'w', encoding='utf-8').write(_json.dumps(_json_stats))
        except Exception:
            try:
                open(status_file, "w").write("transcribing")
            except Exception:
                pass

        # start a small ticker while transcribing (will be stopped after transcribe_audio)
        trans_ticker = ProgressTicker(status_file, start_pct=5, end_pct=20, phase='transcribing')
        try:
            trans_ticker.start()
        except Exception:
            pass

        transcript = transcribe_audio(audio_file)

        # stop transcribe ticker and write a solid 20%
        try:
            trans_ticker.stop(final_pct=20, final_phase='transcribed')
        except Exception:
            try:
                import json as _json
                _json_stats = {'phase': 'transcribed', 'percent': 20}
                open(status_file, 'w', encoding='utf-8').write(_json.dumps(_json_stats))
            except Exception:
                pass

        # write transcript artifacts for downstream modules (diarization expects a JSON file)
        base = os.path.basename(audio_file).replace('.wav', '')
        transcripts_dir = os.path.join('results', 'transcripts')
        os.makedirs(transcripts_dir, exist_ok=True)
        stt_json_file = os.path.join(transcripts_dir, f"{base}_stt.json")
        stt_txt_file = os.path.join(transcripts_dir, f"{base}_stt.txt")
        transcript_text = ""
        try:
            if transcript is not None:
                with open(stt_json_file, 'w', encoding='utf-8') as jf:
                    json.dump(transcript, jf, ensure_ascii=False, indent=2)
                transcript_text = transcript.get('text', '') if isinstance(transcript, dict) else str(transcript)
                with open(stt_txt_file, 'w', encoding='utf-8') as tf:
                    tf.write(transcript_text)
        except Exception:
            # continue even if writing fails
            try:
                transcript_text = transcript.get('text', '') if isinstance(transcript, dict) else (str(transcript) if transcript else "")
            except Exception:
                transcript_text = ""

        # start diarization ticker
        diar_ticker = ProgressTicker(status_file, start_pct=20, end_pct=70, phase='diarizing')
        try:
            diar_ticker.start()
        except Exception:
            pass

        # resolve diarize_audio fresh each time (helps pick up file edits during dev)
        import importlib
        diarize_audio = importlib.import_module('diarization_module').diarize_audio
        diarized_transcript = diarize_audio(audio_file)

        # stop diarization ticker and mark 70%
        try:
            diar_ticker.stop(final_pct=70, final_phase='diarized')
        except Exception:
            try:
                open(status_file, "w").write("diarized")
            except Exception:
                pass


        # start summarization ticker
        sum_ticker = ProgressTicker(status_file, start_pct=70, end_pct=95, phase='summarizing')
        try:
            sum_ticker.start()
        except Exception:
            pass

        # summarizer expects a raw text string (or filepath). Join diarized lines.
        summary_input = "\n".join(diarized_transcript) if isinstance(diarized_transcript, (list, tuple)) else str(diarized_transcript)
        summary = summarize_text(summary_input)

        # stop summarizer ticker and mark near-complete
        try:
            sum_ticker.stop(final_pct=95, final_phase='summarized')
        except Exception:
            try:
                import json as _json
                _json_stats = {'phase': 'summarized', 'percent': 95}
                open(status_file, 'w', encoding='utf-8').write(_json.dumps(_json_stats))
            except Exception:
                pass

        try:
            import json as _json
            _json_stats = {'phase': 'done', 'percent': 100}
            open(status_file, 'w', encoding='utf-8').write(_json.dumps(_json_stats))
        except Exception:
            try:
                open(status_file, "w").write("done")
            except Exception:
                pass

        result_queue.put((transcript_text, diarized_transcript, summary))

        # Structured session logging (JSON; Parquet optional)
        try:
            import time as _time
            base_name = os.path.basename(audio_file).replace('.wav', '')
            session = {
                'audio_file': audio_file,
                'timestamp': _time.strftime('%Y-%m-%dT%H:%M:%S'),
                'transcript_text': transcript_text,
                'diarized_transcript': diarized_transcript,
                'summary': summary,
            }
            logs_dir = os.path.join('results', 'transcripts')
            os.makedirs(logs_dir, exist_ok=True)
            with open(os.path.join(logs_dir, f"{base_name}_session.json"), 'w', encoding='utf-8') as sf:
                json.dump(session, sf, ensure_ascii=False, indent=2)
            # Optional: write parquet if pandas+pyarrow available
            try:
                import pandas as _pd
                df = _pd.DataFrame([
                    {
                        'audio_file': audio_file,
                        'timestamp': session['timestamp'],
                        'transcript_text': transcript_text,
                        'speaker_line': line,
                        'summary': summary,
                    }
                    for line in (diarized_transcript if isinstance(diarized_transcript, (list, tuple)) else [str(diarized_transcript)])
                ])
                df.to_parquet(os.path.join(logs_dir, f"{base_name}_session.parquet"), index=False)
            except Exception:
                pass
        except Exception:
            pass
        # update last job as done
        try:
            import json as _json
            with open(os.path.join('results', 'last_job.json'), 'w', encoding='utf-8') as _lf:
                _json.dump({"audio_file": audio_file, "status": "done"}, _lf)
        except Exception:
            pass

        # Auto-send email if configured in environment
        try:
            recipient = os.environ.get('EMAIL_RECIPIENT')
            smtp_server = os.environ.get('SMTP_SERVER')
            smtp_port = int(os.environ.get('SMTP_PORT', '587') or 587)
            username = os.environ.get('SMTP_USERNAME')
            password = os.environ.get('SMTP_PASSWORD')
            use_ssl = str(os.environ.get('SMTP_USE_SSL', 'false')).lower() in ('1','true','yes','y')
            auto = os.environ.get('EMAIL_AUTO_SEND', 'false').lower() in ('1','true','yes','y')
            if auto and recipient and smtp_server:
                from email.message import EmailMessage
                import smtplib
                subject = f"Meeting Summary – {audio_file}"
                body = f"Summary for {audio_file}:\n\n{summary}\n\n---\nTranscript:\n{transcript_text}"
                msg = EmailMessage()
                msg['From'] = username or 'noreply@example.com'
                msg['To'] = recipient
                msg['Subject'] = subject
                msg.set_content(body)
                if use_ssl:
                    with smtplib.SMTP_SSL(smtp_server, smtp_port, timeout=30) as s:
                        if username and password:
                            s.login(username, password)
                        s.send_message(msg)
                else:
                    with smtplib.SMTP(smtp_server, smtp_port, timeout=30) as s:
                        s.starttls()
                        if username and password:
                            s.login(username, password)
                        s.send_message(msg)
        except Exception:
            pass

        try:
            os.remove(status_file)
        except Exception:
            pass

        task_queue.task_done()


# Start the worker thread
thread = threading.Thread(target=worker, daemon=True)
thread.start()


def add_to_pipeline(audio_file):
    task_queue.put(audio_file)


def get_result():
    if not result_queue.empty():
        return result_queue.get()
    return None

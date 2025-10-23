import os, json, tempfile, wave
import whisper
from transformers import pipeline
from rouge_score import rouge_scorer
from jiwer import wer
from vosk import Model, KaldiRecognizer
from pyannote.audio import Pipeline as DiarizationPipeline
from pyannote.metrics.diarization import DiarizationErrorRate
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Meeting Summarization Demo", layout="wide")

st.title("🎙️ Meeting Summarization Demo")
st.markdown("Pipeline: **Audio → Transcript (Whisper/Vosk) → Diarization (Auto RTTM + Visualization) → Summary (T5 & BART) → Evaluation (WER + ROUGE + DER)**")

# ---- Step 1: Upload audio ----
uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name
    
    st.audio(audio_path)

    # ---- Step 2: Transcription ----
    st.subheader("Step 2: Transcription")

    method = st.radio("Choose Transcription Model", ["Whisper (small)", "Vosk (offline)"])

    if method == "Whisper (small)":
        with st.spinner("Transcribing with Whisper..."):
            model = whisper.load_model("small")
            result = model.transcribe(audio_path)
            transcript = result["text"]
            segments = result.get("segments", None)
    else:
        with st.spinner("Transcribing with Vosk..."):
            vosk_model = Model(lang="en-us")
            wf = wave.open(audio_path, "rb")
            rec = KaldiRecognizer(vosk_model, wf.getframerate())
            rec.SetWords(True)
            text_result = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    res = json.loads(rec.Result())
                    text_result += res.get("text", "") + " "
            final_res = json.loads(rec.FinalResult())
            text_result += final_res.get("text", "")
            transcript = text_result.strip()
            segments = None

    st.text_area("Transcript Preview", transcript[:1000] + "...", height=200)
    st.download_button("Download Transcript", transcript, file_name="meeting_transcript.txt")

    # ---- Step 2.5: Auto Diarization (RTTM Generation + Visualization) ----
    st.subheader("Step 2.5: Speaker Diarization (Auto RTTM + Visualization)")
    auto_rttm_path = None
    diarization_result = None

    if st.checkbox("Run automatic speaker diarization (generate RTTM and visualize)"):
        with st.spinner("Running Pyannote speaker diarization..."):
            # Fixed: use revision keyword instead of @2.1
            diarization_pipeline = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization",
                revision="2.1"
            )
            diarization_result = diarization_pipeline(audio_path)

            # Convert to RTTM string
            rttm_str = diarization_result.to_rttm()

            # Save RTTM
            with tempfile.NamedTemporaryFile(delete=False, suffix=".rttm") as tmp_rttm:
                tmp_rttm.write(rttm_str.encode("utf-8"))
                auto_rttm_path = tmp_rttm.name

            st.success("✅ RTTM file generated automatically.")
            st.download_button("Download Auto RTTM", rttm_str, file_name="auto_generated.rttm")

        # ---- Visualization: Speaker timeline ----
        st.subheader("🎧 Speaker Timeline")
        diarization_df = pd.DataFrame(
            [(segment.start, segment.end, label) for segment, _, label in diarization_result.itertracks(yield_label=True)],
            columns=["Start", "End", "Speaker"]
        )

        fig, ax = plt.subplots(figsize=(10, 2 + len(diarization_df["Speaker"].unique()) * 0.4))
        speakers = diarization_df["Speaker"].unique()
        colors = plt.cm.tab10.colors

        for i, spk in enumerate(speakers):
            spk_data = diarization_df[diarization_df["Speaker"] == spk]
            for _, row in spk_data.iterrows():
                ax.barh(spk, row["End"] - row["Start"], left=row["Start"], color=colors[i % len(colors)], edgecolor="black")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Speaker")
        ax.set_title("Speaker Segmentation Timeline")
        st.pyplot(fig)

        # ---- Visualization: Speaker-labeled Transcript ----
        st.subheader("🗣️ Speaker-Labeled Transcript")
        labeled_transcript = ""
        for segment, _, speaker in diarization_result.itertracks(yield_label=True):
            labeled_transcript += f"{speaker}: [From {segment.start:.1f}s to {segment.end:.1f}s]\n"
        st.text_area("Speaker Segments", labeled_transcript, height=250)

    # ---- Step 3: Summarization ----
    st.subheader("Step 3: Summarization")

    summarizer_t5 = pipeline("summarization", model="t5-small")
    summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn")

    st.info("Summarizing (first 2000 chars only for demo)...")
    t5_summary = summarizer_t5(transcript[:2000], max_length=120, min_length=30, do_sample=False)[0]["summary_text"]
    bart_summary = summarizer_bart(transcript[:2000], max_length=120, min_length=30, do_sample=False)[0]["summary_text"]

    st.write("**T5 Summary:**")
    st.success(t5_summary)
    st.write("**BART Summary:**")
    st.info(bart_summary)

    # ---- Step 4: Evaluation ----
    st.subheader("Step 4: Evaluation")

    ref_summary = st.text_area("Paste Reference Summary (required for evaluation)", "")
    ref_diarization_file = st.file_uploader("Upload Reference Diarization (RTTM format, optional for DER)", type=["rttm"])

    if st.button("Run Full Evaluation"):
        results = {}

        with st.spinner("Running Evaluation (WER + ROUGE + DER)..."):
            # ---- WER ----
            wer_score = wer(ref_summary, transcript)
            st.write(f"🎯 **WER (Transcript vs Reference): {wer_score:.2f}**")
            results["WER"] = wer_score

            # ---- ROUGE ----
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            rouge_t5 = scorer.score(ref_summary, t5_summary)
            rouge_bart = scorer.score(ref_summary, bart_summary)

            st.write("📊 **ROUGE (T5 vs Reference):**", rouge_t5)
            st.write("📊 **ROUGE (BART vs Reference):**", rouge_bart)

            results["T5_ROUGE"] = rouge_t5
            results["BART_ROUGE"] = rouge_bart

            # ---- DER ----
            ref_path = None
            if ref_diarization_file is not None:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".rttm") as tmp_ref:
                    tmp_ref.write(ref_diarization_file.read())
                    ref_path = tmp_ref.name
                st.write("📁 Using uploaded RTTM for DER.")
            elif auto_rttm_path is not None:
                ref_path = auto_rttm_path
                st.write("📁 Using auto-generated RTTM for DER.")
            else:
                st.warning("No RTTM file available — DER skipped.")

            if ref_path:
                st.write("🧩 Calculating **Diarization Error Rate (DER)**...")
                # Fixed: use revision keyword
                diarization_pipeline = DiarizationPipeline.from_pretrained(
                    "pyannote/speaker-diarization",
                    revision="2.1"
                )
                hyp = diarization_pipeline(audio_path)

                with open(ref_path, "r") as f:
                    ref_rttm = f.read()

                der_metric = DiarizationErrorRate()
                ref_annotation = der_metric._load_rttm_string(ref_rttm)
                der_score = der_metric(ref_annotation, hyp)

                st.write(f"🗣️ **DER: {der_score:.3f}**")
                results["DER"] = der_score

        # ---- Save Results ----
        st.download_button("Download Evaluation Results", json.dumps(results, indent=2),
                           file_name="evaluation_results.json")

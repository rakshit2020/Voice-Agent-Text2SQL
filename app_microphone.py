import streamlit as st
import riva.client
import riva.client.audio_io
from copy import deepcopy

st.title("🎤 NVIDIA Riva LIVE ASR Interface")

# Initialize Riva service
uri = "localhost:50051"
auth = riva.client.Auth(uri=uri)
asr_service = riva.client.ASRService(auth)

offline_config = riva.client.RecognitionConfig(
    encoding=riva.client.AudioEncoding.LINEAR_PCM,
    max_alternatives=1,
    enable_automatic_punctuation=True,
    language_code="en-US",
    verbatim_transcripts=False,
)
streaming_config = riva.client.StreamingRecognitionConfig(config=deepcopy(offline_config), interim_results=True)

# Add audio specs
my_wav_file = r"C:\Users\raksh\Downloads\en-US_sample.wav"
riva.client.add_audio_file_specs_to_config(offline_config, my_wav_file)
riva.client.add_audio_file_specs_to_config(streaming_config, my_wav_file)

if st.button("Start Recording"):
    st.write("Recording... Speak into your microphone")

    # Create placeholder for transcription
    transcription_placeholder = st.empty()
    accumulated_text = ""

    input_device = None
    with riva.client.audio_io.MicrophoneStream(
            rate=streaming_config.config.sample_rate_hertz,
            chunk=streaming_config.config.sample_rate_hertz // 10,
            device=input_device,
    ) as audio_chunk_iterator:

        responses = asr_service.streaming_response_generator(
            audio_chunks=audio_chunk_iterator,
            streaming_config=streaming_config,
        )

        for response in responses:
            if response.results:
                for result in response.results:
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        if result.is_final:
                            accumulated_text += transcript + " "
                            transcription_placeholder.write(f"**Transcription:** {accumulated_text}")
                        else:
                            # Show interim results without overwriting
                            current_display = accumulated_text + f"[{transcript}]"
                            transcription_placeholder.write(f"**Transcription:** {current_display}")
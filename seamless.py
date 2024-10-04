import os
import torch
import torchaudio
from pydub import AudioSegment
from seamless_communication.inference import Translator
from pydub import silence

audio_folder_path = "IndicConformer2/audio"  #audio folder path
output_transcription_file = "output/transcriptions_seamless_lahaja.txt"

model_name = "seamlessM4T_large"
vocoder_name = "vocoder_v2"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

translator = Translator(
    model_name,
    vocoder_name,
    device=device,
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
)

def split_large_chunk(chunk, max_length_ms=5000):
    sub_chunks = []
    for i in range(0, len(chunk), max_length_ms):
        sub_chunk = chunk[i:i + max_length_ms]
        sub_chunks.append(sub_chunk)
    return sub_chunks

os.makedirs(os.path.dirname(output_transcription_file), exist_ok=True)
with open(output_transcription_file, "w", encoding="utf-8") as transcription_file:
    for audio_file in os.listdir(audio_folder_path):
        if audio_file.endswith(".wav"):
            audio_file_path = os.path.join(audio_folder_path, audio_file)

            print(f"Processing file: {audio_file_path}")
            combined_transcription = ""

            try:
                audio, sample_rate = torchaudio.load(audio_file_path)
                if sample_rate != 16000:
                    audio = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio)
                audio_segment = AudioSegment.from_wav(audio_file_path)
                chunks = silence.split_on_silence(audio_segment, min_silence_len=500, silence_thresh=-40)

                for i, chunk in enumerate(chunks):
                    if len(chunk) > 5000:
                        sub_chunks = split_large_chunk(chunk)
                    else:
                        sub_chunks = [chunk]

                    for j, sub_chunk in enumerate(sub_chunks):
                        sub_chunk_path = f"/tmp/chunk_{i}_subchunk_{j}.wav"
                        sub_chunk.export(sub_chunk_path, format="wav")

                        try:
                            text_output, _ = translator.predict(
                                input=sub_chunk_path,
                                task_str="asr",
                                tgt_lang="hin"
                            )
                            transcription_text = str(text_output[0]) if text_output else ""
                            combined_transcription += transcription_text + " "
                        except Exception as e:
                            print(f"Error transcribing {sub_chunk_path}: {e}")

                transcription_line = f"{audio_file}\t[{combined_transcription.strip()}]\n"
                transcription_file.write(transcription_line)
                print(f"Combined transcription for {audio_file_path} added to transcription file")

            except Exception as e:
                print(f"Error processing {audio_file_path}: {e}")

print(f"All transcriptions saved to {output_transcription_file}")
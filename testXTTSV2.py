from huggingface_hub import snapshot_download

from nltk.tokenize import sent_tokenize
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import soundfile as sf
import os
from pydub import AudioSegment

config = XttsConfig()
config.load_json("config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=".\\XTTS-v2\\", eval=True)
model.cuda()

reference_audio = ".\\XTTS-v2\\samples\\male_it.wav"

def process_tts(input_text, speaker_audio, lang, output_file):
    outputs = model.synthesize(
      input_text,
      config,
      speaker_wav=speaker_audio,
      gpt_cond_len=3,
      language=lang,
    )
    audio_data = outputs["wav"]
    sample_rate = 24000
    sf.write(output_file, audio_data, sample_rate)
    print("Audio saved successfully!")
    
with open('chapter_1.txt', 'r', encoding="utf-8") as file:
    text = file.read()
    sentences = sent_tokenize(text)

audio_files = []
for i, sentence in enumerate(sentences):
    output_file = f"audio_sentence_{i}.wav"
    process_tts(sentence, reference_audio, "it", output_file)
    audio_files.append(output_file)


combined = AudioSegment.empty()
silence = AudioSegment.silent(duration=500)  

for audio_file in audio_files:
    audio = AudioSegment.from_wav(audio_file)
    combined += audio + silence

combined.export("combined_audio.wav", format="wav")

for audio_file in audio_files:
    os.remove(audio_file)
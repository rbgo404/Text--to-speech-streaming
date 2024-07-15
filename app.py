import io
import math
from queue import Queue
from threading import Thread

import numpy as np
import torch
import base64

from parler_tts import ParlerTTSForConditionalGeneration
from pydub import AudioSegment
from transformers import AutoTokenizer, AutoFeatureExtractor, set_seed
from transformers.generation.streamers import BaseStreamer

SEED = 42
class ParlerTTSStreamer(BaseStreamer):
    def __init__(self):
        device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch_dtype = torch.float16 if device != "cpu" else torch.float32

        repo_id = "parler-tts/parler_tts_mini_v0.1"
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(repo_id)

        self.SAMPLE_RATE = self.feature_extractor.sampling_rate

        self.model = ParlerTTSForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True).to(device)
        self.decoder = self.model.decoder
        self.audio_encoder = self.model.audio_encoder
        self.generation_config = self.model.generation_config
        self.device = device if device is not None else self.model.device

        self.sampling_rate = self.model.audio_encoder.config.sampling_rate
        frame_rate = self.model.audio_encoder.config.frame_rate


        play_steps_in_s=2.0
        play_steps = int(frame_rate * play_steps_in_s)

        # variables used in the streaming process
        self.play_steps = play_steps
        # if stride is not None:
        #     self.stride = stride

        hop_length = math.floor(self.audio_encoder.config.sampling_rate / self.audio_encoder.config.frame_rate)
        self.stride = hop_length * (play_steps - self.decoder.num_codebooks) // 6
        self.token_cache = None
        self.to_yield = 0

        # varibles used in the thread process
        self.audio_queue = Queue()
        self.stop_signal = None
        self.timeout = None
    def apply_delay_pattern_mask(self, input_ids):
        # build the delay pattern mask for offsetting each codebook prediction by 1 (this behaviour is specific to Parler)
        _, delay_pattern_mask = self.decoder.build_delay_pattern_mask(
            input_ids[:, :1],
            bos_token_id=self.generation_config.bos_token_id,
            pad_token_id=self.generation_config.decoder_start_token_id,
            max_length=input_ids.shape[-1],
        )
        # apply the pattern mask to the input ids
        input_ids = self.decoder.apply_delay_pattern_mask(input_ids, delay_pattern_mask)

        # revert the pattern delay mask by filtering the pad token id
        mask = (delay_pattern_mask != self.generation_config.bos_token_id) & (delay_pattern_mask != self.generation_config.pad_token_id)
        input_ids = input_ids[mask].reshape(1, self.decoder.num_codebooks, -1)
        # append the frame dimension back to the audio codes
        input_ids = input_ids[None, ...]

        # send the input_ids to the correct device
        input_ids = input_ids.to(self.audio_encoder.device)

        decode_sequentially = (
            self.generation_config.bos_token_id in input_ids
            or self.generation_config.pad_token_id in input_ids
            or self.generation_config.eos_token_id in input_ids
        )
        if not decode_sequentially:
            output_values = self.audio_encoder.decode(
                input_ids,
                audio_scales=[None],
            )
        else:
            sample = input_ids[:, 0]
            sample_mask = (sample >= self.audio_encoder.config.codebook_size).sum(dim=(0, 1)) == 0
            sample = sample[:, :, sample_mask]
            output_values = self.audio_encoder.decode(sample[None, ...], [None])

        audio_values = output_values.audio_values[0, 0]
        return audio_values.cpu().float().numpy()

    def put(self, value):
        batch_size = value.shape[0] // self.decoder.num_codebooks
        if batch_size > 1:
            raise ValueError("ParlerTTSStreamer only supports batch size 1")

        if self.token_cache is None:
            self.token_cache = value
        else:
            self.token_cache = torch.concatenate([self.token_cache, value[:, None]], dim=-1)

        if self.token_cache.shape[-1] % self.play_steps == 0:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
            self.on_finalized_audio(audio_values[self.to_yield : -self.stride])
            self.to_yield += len(audio_values) - self.to_yield - self.stride

    def end(self):
        """Flushes any remaining cache and appends the stop symbol."""
        if self.token_cache is not None:
            audio_values = self.apply_delay_pattern_mask(self.token_cache)
        else:
            audio_values = np.zeros(self.to_yield)

        self.on_finalized_audio(audio_values[self.to_yield :], stream_end=True)

    def on_finalized_audio(self, audio: np.ndarray, stream_end: bool = False):
        """Put the new audio in the queue. If the stream is ending, also put a stop signal in the queue."""
        self.audio_queue.put(audio, timeout=self.timeout)
        if stream_end:
            self.audio_queue.put(self.stop_signal, timeout=self.timeout)

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get(timeout=self.timeout)
        if not isinstance(value, np.ndarray) and value == self.stop_signal:
            raise StopIteration()
        else:
            return value



class InferlessPythonModel:

    def initialize(self):
        self.streamer = ParlerTTSStreamer()

    def numpy_to_mp3(self, audio_array, sampling_rate):
        # Normalize audio_array if it's floating-point
        if np.issubdtype(audio_array.dtype, np.floating):
            max_val = np.max(np.abs(audio_array))
            audio_array = (audio_array / max_val) * 32767  # Normalize to 16-bit range
            audio_array = audio_array.astype(np.int16)

        # Create an audio segment from the numpy array
        audio_segment = AudioSegment(
            audio_array.tobytes(),
            frame_rate=sampling_rate,
            sample_width=audio_array.dtype.itemsize,
            channels=1
        )

        # Export the audio segment to MP3 bytes - use a high bitrate to maximise quality
        mp3_io = io.BytesIO()
        audio_segment.export(mp3_io, format="mp3", bitrate="320k")

        # Get the MP3 bytes
        mp3_bytes = mp3_io.getvalue()
        mp3_io.close()

        return mp3_bytes

    def infer(self, inputs, stream_output_handler):
        self.streamer.token_cache = None
        self.streamer.to_yield = 0

        input_value = inputs["input_value"]
        prompt_value = inputs["prompt_value"]
        inputs_ = self.streamer.tokenizer(input_value, return_tensors="pt").to(self.streamer.device)
        prompt = self.streamer.tokenizer(prompt_value, return_tensors="pt").to(self.streamer.device)

        generation_kwargs = dict(
            input_ids=inputs_.input_ids,
            prompt_input_ids=prompt.input_ids,
            streamer=self.streamer,
            do_sample=True,
            temperature=1.0,
            min_new_tokens=10)

        set_seed(SEED)
        thread = Thread(target=self.streamer.model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_audio in self.streamer:
            mp3_bytes =  self.numpy_to_mp3(new_audio, sampling_rate=self.streamer.sampling_rate)
            mp3_str = base64.b64encode(mp3_bytes).decode('utf-8')
            output_dict = {}
            output_dict["OUT"] = mp3_str
            stream_output_handler.send_streamed_output(output_dict)
        thread.join()

        stream_output_handler.finalise_streamed_output()

    # perform any cleanup activity here
    def finalize(self,args):
        self.streamer = None

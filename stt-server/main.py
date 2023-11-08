import torch
import torch.multiprocessing as mp
import torchaudio
from torchaudio.io import StreamReader
 
ITERATIONS = 100
def stream(queue: mp.Queue(),
   format: str,
   src: str,
   frames_per_chunk: int,
   sample_rate: int):
   '''Streams audio data
  
   Parameters:
       queue: Queue of data chunks
       format: Format
       src: Source
       frames_per_chunk: How many frames are in each data chunk
       sample_rate: Sample rate
 
   Returns:
       None'''
   print("Initializing Audio Stream")
   streamer = StreamReader(src, format=format)
   streamer.add_basic_audio_stream(frames_per_chunk=frames_per_chunk,
       sample_rate=sample_rate)
   print("Streaming\n")
   stream_iterator = streamer.stream(timeout=-1, backoff=1.0)
   for _ in range(ITERATIONS):
       (chunk,) = next(stream_iterator)
       queue.put(chunk)
 
class InferencePipeline:
   '''Creates an inference pipeline for streaming audio data'''
   def __init__(self,
       pipeline: torchaudio.pipelines.RNNTBundle,
       beam_width: int=10):
       '''Initializes TorchAudio RNNT Pipeline
      
       Parameters:
           pipeline: TorchAudio Pipeline to use
           beam_width: Beam width
 
       Returns:
           None'''
 
       self.pipeline = pipeline
       self.feature_extractor = pipeline.get_streaming_feature_extractor()
       self.decoder = pipeline.get_decoder()
       self.token_processor = pipeline.get_token_processor()
       self.beam_width = beam_width
       self.state = None
       self.hypothesis = None
  
   def infer(self, segment: torch.Tensor) -> str:
       '''Runs inference using the initialized pipeline
      
       Parameters:
           segment: Torch tensor with features to extract
      
       Returns:
           Transcript as string type'''
 
       features, length = self.feature_extractor(segment)
       predictions, self.state = self.decoder.infer(
           features, length, self.beam_width, state=self.state,
           hypothesis=self.hypothesis
       )
       self.hypothesis = predictions[0]
       transcript = self.token_processor(self.hypothesis[0], lstrip=False)
       return transcript
 
class ContextCacher:
   def __init__(self, segment_length: int, context_length: int):
       '''Creates initial context cache
      
       Parameters:
           segment_length: length of one audio segment
           context_length: length of the context
 
       Returns:
           None'''
       self.segment_length = segment_length
       self.context_length = context_length
       self.context = torch.zeros([context_length])
  
   def __call__(self, chunk: torch.Tensor):
       '''Adds chunk to context and returns it
      
       Parameters:
           chunk: chunk of audio data to process
      
       Returns:
           Tensor'''
       if chunk.size(0) < self.segment_length:
           chunk = torch.nn.functional.pad(chunk,
               (0, self.segment_length - chunk.size(0)))
       chunk_with_context = torch.cat((self.context, chunk))
       self.context = chunk[-self.context_length :]
       return chunk_with_context
 
def main(device: str, src: str, bundle: torchaudio.pipelines):
   '''Transcribed audio data from the mic
  
   Parameters:
       device: Input device name
       src: Source from input
       bundle: TorchAudio pipeline
  
   Returns:
       None'''
   pipeline = InferencePipeline(bundle)
  
   sample_rate = bundle.sample_rate
   segment_length = bundle.segment_length * bundle.hop_length
   context_length = bundle.right_context_length * bundle.hop_length
  
   cacher = ContextCacher(segment_length, context_length)
  
   @torch.inference_mode()
   def infer():
       for _ in range(ITERATIONS):
           chunk = q.get()
           segment = cacher(chunk[:,0])
           transcript = pipeline.infer(segment)
           print(transcript, end="", flush=True)
  
   ctx = mp.get_context("spawn")
   q = ctx.Queue()
   p = ctx.Process(target=stream, args=(q, device, src, segment_length, sample_rate))
   p.start()
   infer()
   p.join()
if __name__ == "__main__":
   main(
       device="avfoundation",
       src=":1",
       bundle=torchaudio.pipelines.EMFORMER_RNNT_BASE_LIBRISPEECH
   )

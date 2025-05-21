import os
import torch
import yaml
import numpy as np
import torchaudio
import time
import re
import soundfile as sf


# Import Gradio
import gradio as gr

# Khởi tạo biến model và char_dict trước khi import
model = None
char_dict = None
chunkformer_imports_ok = False # Cờ kiểm tra import thành công

try:
    # Thử import các module Chunkformer cần thiết
    from chunkformer.model.utils.init_model import init_model
    from chunkformer.model.utils.checkpoint import load_checkpoint
    from chunkformer.model.utils.file_utils import read_symbol_table
    from chunkformer.model.utils.ctc_utils import get_output
    import torchaudio.compliance.kaldi as kaldi # Cần cho tính Fbank

    print("Successfully imported Chunkformer modules.")
    chunkformer_imports_ok = True
except ImportError as e:
    print(f"Error importing Chunkformer modules: {e}")
    print("Please ensure the 'chunkformer' directory is accessible and contains the necessary model and utility files.")
    print("You might need to add the root directory of 'chunkformer' to your PYTHONPATH.")
    print("Example: export PYTHONPATH=/path/to/your/chunkformer/root:$PYTHONPATH")
    chunkformer_imports_ok = False


# Import Pydub và check FFmpeg (cho chế độ upload file)
from pydub import AudioSegment
from pydub.utils import which
from pydub.exceptions import CouldntDecodeError # Thêm để bắt lỗi giải mã cụ thể

# Kiểm tra xem ffmpeg có sẵn không (cần cho pydub - chế độ upload file)
FFMPEG_AVAILABLE = which("ffmpeg") is not None and which("ffprobe") is not None
if not FFMPEG_AVAILABLE:
    print("CẢNH BÁO: FFmpeg/FFprobe không được tìm thấy trong PATH.")
    print("Chế độ Tải File có thể không hoạt động với một số định dạng âm thanh.")
    print("Pydub cần FFmpeg/FFprobe để xử lý audio file upload. Vui lòng cài đặt FFmpeg.")
    print("Hướng dẫn: https://ffmpeg.org/download.html")


# Đường dẫn model
MODEL_DIR = "chunkformer/chunkformer-large-vie" # Thay đổi nếu model ở đường dẫn khác
CONFIG_PATH = os.path.join(MODEL_DIR, "config.yaml")
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "pytorch_model.bin")
VOCAB_PATH = os.path.join(MODEL_DIR, "vocab.txt")

# Các tham số Chunkformer (mặc định)
CHUNK_SIZE = 16
LEFT_CONTEXT_SIZE = 32
RIGHT_CONTEXT_SIZE = 8
TRUNCATED_CONTEXT_SIZE = 16
CNN_MODULE_KERNEL = 31
NUM_BLOCKS = 12
ATTENTION_HEADS = 8
OUTPUT_SIZE = 256

# Cố gắng đọc từ config nếu file tồn tại và imports thành công
if chunkformer_imports_ok and os.path.exists(CONFIG_PATH):
    try:
        with open(CONFIG_PATH, 'r') as fin:
            model_config = yaml.load(fin, Loader=yaml.FullLoader)
        # Đọc các tham số từ config nếu có sẵn tên thuộc tính tương ứng
        # Sử dụng .get() với giá trị mặc định an toàn
        CHUNK_SIZE = model_config.get('encoder_conf', {}).get('chunk_size', CHUNK_SIZE)
        LEFT_CONTEXT_SIZE = model_config.get('encoder_conf', {}).get('left_context_size', LEFT_CONTEXT_SIZE)
        RIGHT_CONTEXT_SIZE = model_config.get('encoder_context_conf', {}).get('right_context_size', model_config.get('encoder_conf', {}).get('right_context_size', RIGHT_CONTEXT_SIZE))
        TRUNCATED_CONTEXT_SIZE = model_config.get('encoder_context_conf', {}).get('truncated_context_size', model_config.get('encoder_conf', {}).get('truncated_context_size', TRUNCATED_CONTEXT_SIZE))
        CNN_MODULE_KERNEL = model_config.get('encoder_conf', {}).get('cnn_module_kernel', CNN_MODULE_KERNEL)
        NUM_BLOCKS = model_config.get('encoder_conf', {}).get('num_blocks', NUM_BLOCKS)
        ATTENTION_HEADS = model_config.get('encoder_conf', {}).get('attention_heads', ATTENTION_HEADS)
        OUTPUT_SIZE = model_config.get('encoder_conf', {}).get('output_size', OUTPUT_SIZE)

        print(f"Model parameters from config: CHUNK_SIZE={CHUNK_SIZE}, LEFT_CONTEXT_SIZE={LEFT_CONTEXT_SIZE}, RIGHT_CONTEXT_SIZE={RIGHT_CONTEXT_SIZE}, TRUNCATED_CONTEXT_SIZE={TRUNCATED_CONTEXT_SIZE}, CNN_MODULE_KERNEL={CNN_MODULE_KERNEL}, NUM_BLOCKS={NUM_BLOCKS}, ATTENTION_HEADS={ATTENTION_HEADS}, OUTPUT_SIZE={OUTPUT_SIZE}")

    except Exception as e:
         print(f"CẢNH BẢO: Lỗi đọc config file tại {CONFIG_PATH}: {e}. Sử dụng giá trị mặc định.")
else:
     if chunkformer_imports_ok: # Imports ok, but config file missing
          print(f"CẢNH BÁO: Không tìm thấy file config tại {CONFIG_PATH}. Sử dụng giá trị mặc định.")
     # Else: Imports failed, message already printed, using defaults


# Cấu hình cho xử lý theo segment trong chế độ streaming
SEGMENT_DURATION_S = 3
TARGET_SAMPLE_RATE = 16000 # Sample rate mà model mong đợi (đã resample trước khi tính fbank)
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION_S * TARGET_SAMPLE_RATE) # Số mẫu cần để tạo 1 segment

# Load model (chỉ cần load 1 lần khi script bắt đầu)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def load_model(device="cpu"):
    """Loads the model, character dictionary."""
    # Check if necessary files exist AND imports were successful
    if not chunkformer_imports_ok:
        print("Lỗi: Không thể tải model. Các module Chunkformer chưa được import thành công.")
        return None, None

    if not all([os.path.exists(CONFIG_PATH), os.path.exists(CHECKPOINT_PATH), os.path.exists(VOCAB_PATH)]):
        missing_files = [f for f in [CONFIG_PATH, CHECKPOINT_PATH, VOCAB_PATH] if not os.path.exists(f)] if chunkformer_imports_ok else [] # Only list if imports were ok
        import_error_msg = "Chunkformer modules failed to import." if not chunkformer_imports_ok else ""
        print(f"Lỗi: Không thể tải model. {import_error_msg} Missing files: {missing_files}")
        return None, None

    try:
        with open(CONFIG_PATH, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)

        # Sử dụng các hàm đã import
        model = init_model(config, CONFIG_PATH)
        model.eval()
        load_checkpoint(model, CHECKPOINT_PATH)
        model.to(device)
        symbol_table = read_symbol_table(VOCAB_PATH)
        char_dict = {v: k for k, v in symbol_table.items()}
        # print("Model and char_dict loaded successfully.") # Logged by caller
        return model, char_dict
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Vui lòng kiểm tra đường dẫn model, file config, checkpoint và vocab.")
        return None, None

# Load model và char_dict globally ONLY if imports were successful
if chunkformer_imports_ok:
    print("Attempting to load model...")
    # Hàm load_model sẽ xử lý các lỗi FileNotFoundError và lỗi tải khác
    model, char_dict = load_model(device=device)
    if model is not None and char_dict is not None:
        print("Global model and char_dict are set.")
    else:
        print("Failed to load model, global model and char_dict remain None.")
else:
    print("Skipping model loading due to failed imports.")
    # model và char_dict đã được khởi tạo là None


# Hàm tiền xử lý audio segment (numpy array) thành features
def preprocess_numpy_audio(audio_np: np.ndarray, sample_rate: int, device: torch.device):
    """
    Preprocesses a numpy audio array into fbank features.
    Expects input numpy array to be float32, typically in range [-1, 1].

    Args:
        audio_np: Input audio as a numpy array (float32).
        sample_rate: Sample rate of the input audio.
        device: The torch device.

    Returns:
        A tuple (feats, x_len) or (None, None).
    """
    if audio_np is None or audio_np.size == 0:
        # print("preprocess_numpy_audio: Input is None or empty.")
        return None, None

    try:
        # Chuyển numpy float32 sang torch tensor float32 VÀ đưa lên device ngay lập tức
        waveform = torch.tensor(audio_np, dtype=torch.float32, device=device).unsqueeze(0)
    except Exception as e:
        print(f"preprocess_numpy_audio: Error converting numpy to torch tensor or moving to device: {e}")
        return None, None

    # Resample nếu sample rate không khớp
    if sample_rate != TARGET_SAMPLE_RATE:
        # print(f"preprocess_numpy_audio: Resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz.")
        try:
            # Tạo resampler mới cho sample rate đầu vào hiện tại và đưa lên device
            current_resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE).to(device)
            waveform = current_resampler(waveform) # waveform đã trên device sau resample
        except Exception as resample_e:
            print(f"preprocess_numpy_audio: Error during resampling from {sample_rate}Hz to {TARGET_SAMPLE_RATE}Hz: {resample_e}")
            return None, None

    # Kiểm tra waveform size sau resampling
    if waveform is None or waveform.numel() == 0:
         print("preprocess_numpy_audio: Waveform is None or empty after resampling.")
         return None, None

    # Tính đặc trưng fbank (kaldi cần sample rate là số nguyên, sử dụng TARGET_SAMPLE_RATE)
    # Input waveform cho kaldi.fbank phải ở trên device nếu device là GPU
    try:
        feats = kaldi.fbank(waveform,
            num_mel_bins=80, # Số bin mel-frequency
            frame_length=25, # Độ dài frame (ms)
            frame_shift=10,  # Bước dịch frame (ms)
            dither=0.0,      # Không thêm nhiễu
            energy_floor=1e-6, # Ngưỡng năng lượng tối thiểu
            sample_frequency=TARGET_SAMPLE_RATE).unsqueeze(0) # Sử dụng sample rate ĐÍCH

        # feats có shape (1, num_frames, num_mel_bins)
        x_len = torch.tensor([feats.shape[1]], dtype=torch.int, device=device) # feature length trên device

        # Loại bỏ các đoạn feature quá ngắn nếu có (ví dụ < 3 frames)
        if x_len.item() < 3:
            print(f"preprocess_numpy_audio: Generated feature is too short ({x_len.item()} frames), skipping.")
            return None, None

        # print(f"preprocess_numpy_audio: Successfully generated feats shape={feats.shape}, x_len={x_len.item()}")
        return feats, x_len

    except Exception as fbank_e:
         print(f"preprocess_numpy_audio: Error during fbank computation: {fbank_e}")
         return None, None


def transcribe_features(feats, x_len):
    """
    Runs inference on precomputed features.
    Expects feats and x_len to be on the correct device.

    Args:
        feats (torch.Tensor): Input features (batch_size=1, num_frames, feature_dim) on device.
        x_len (torch.Tensor): Lengths of features (batch_size=1) on device.

    Returns:
        str: The recognized text, or an error message string starting with "[Lỗi:".
             Returns empty string "" if model predicts only blank tokens for valid input.
    """
    if model is None or char_dict is None:
        print("DEBUG (INFER): Model or char_dict is None.")
        return "[Lỗi: Model chưa tải hoặc import thất bại]"
    if feats is None or x_len is None or feats.size(0) == 0 or x_len.item() == 0:
        print("DEBUG (INFER): Received None or empty input features.")
        return "" # Return empty string if no input features

    # Ensure feats and x_len are on the correct device (should be from preprocess_numpy_audio)
    # Sửa lỗi: So sánh device.type với device (string)
    if feats.device.type != device or x_len.device.type != device:
         print(f"DEBUG (INFER): Warning: Input features not on expected device {device}. Moving.")
         feats = feats.to(device)
         x_len = x_len.to(device)


    try:
        # Tính toán kích thước cache dựa trên các tham số đã load/mặc định
        att_cache_shape = (NUM_BLOCKS, LEFT_CONTEXT_SIZE, ATTENTION_HEADS, OUTPUT_SIZE * 2 // ATTENTION_HEADS)
        cnn_cache_shape = (NUM_BLOCKS, OUTPUT_SIZE, CNN_MODULE_KERNEL // 2) # Giả định cnn_module_kernel/2 cho right padding

        # Khởi tạo cache trên device. Luôn khởi tạo mới cho mỗi inference đoạn độc lập.
        att_cache = torch.zeros(att_cache_shape, device=device)
        cnn_cache = torch.zeros(cnn_cache_shape, device=device)
        # offset là chỉ số thời gian, luôn bắt đầu từ 0 cho mỗi segment mới
        offset = torch.zeros(1, dtype=torch.int, device=device)

        print(f"DEBUG (INFER): Running encoder on feats shape {feats.shape}, x_len {x_len.item()}...")

        # Chạy qua Encoder
        # xs_origin_lens cần là list hoặc tensor 1D trên CPU
        # Đảm bảo x_len được đưa về CPU trước khi pass vào xs_origin_lens
        encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
            xs=feats,
            xs_origin_lens=x_len.cpu(),
            chunk_size=CHUNK_SIZE,
            left_context_size=LEFT_CONTEXT_SIZE,
            right_context_size=RIGHT_CONTEXT_SIZE,
            att_cache=att_cache, # Sử dụng cache đã khởi tạo
            cnn_cache=cnn_cache, # Sử dụng cache đã khởi tạo
            truncated_context_size=TRUNCATED_CONTEXT_SIZE,
            offset=offset # Bắt đầu từ offset 0
        )

        print(f"DEBUG (INFER): Encoder output shape: {encoder_outs.shape}, Encoder lens: {encoder_lens}, n_chunks: {n_chunks}")

        # --- CTC Decoding (Greedy Search) ---
        hyps = model.encoder.ctc_forward(encoder_outs, encoder_lens, n_chunks)

        # In ra output thô từ decoder để kiểm tra
        print(f"DEBUG (INFER): Raw output from ctc_forward (hyps): {hyps}")

        # --- Chuẩn bị output cho get_output ---
        # get_output cần một list các numpy array chứa token ID
        hyps_cpu_np_list = []

        if isinstance(hyps, torch.Tensor):
             # Nếu ctc_forward trả về một tensor duy nhất (ví dụ: batch size 1)
             print("DEBUG (INFER): ctc_forward returned a single Tensor.")
             if hyps.numel() > 0: # Kiểm tra nếu tensor không rỗng
                  hyps_cpu_np_list.append(hyps.cpu().numpy())
                  # print(f"DEBUG (INFER): Converted single Tensor to numpy array with shape {hyps_cpu_np_list[-1].shape}.")
             else:
                 print("DEBUG (INFER): ctc_forward returned an empty Tensor.")

        elif isinstance(hyps, list):
             # Nếu ctc_forward trả về một list
             print(f"DEBUG (INFER): ctc_forward returned a list of length {len(hyps)}.")
             for i, h in enumerate(hyps):
                 if isinstance(h, torch.Tensor):
                     if h.numel() > 0:
                         hyps_cpu_np_list.append(h.cpu().numpy())
                         # print(f"DEBUG (INFER): Converted Tensor at index {i} to numpy array with shape {hyps_cpu_np_list[-1].shape}.")
                     else:
                         print(f"DEBUG (INFER): Warning: Tensor at index {i} in hyps list is empty.")
                 else:
                      print(f"DEBUG (INFER): Warning: Found non-Tensor element at index {i} in hyps list: {type(h)}")

        else:
             # Xử lý trường hợp ctc_forward trả về định dạng không mong đợi
             print(f"DEBUG (INFER): Unexpected type for hyps from ctc_forward: {type(hyps)}")
             return "[Lỗi: Định dạng output model không mong muốn]"

        # In ra list numpy array đã chuẩn bị
        print(f"DEBUG (INFER): Prepared hyps_cpu_np_list for get_output: {hyps_cpu_np_list}")

        # --- Chuyển token ID thành văn bản ---
        # get_output lấy list các numpy array token ID và dictionary
        # It handles removing blanks and collapses repeated non-blanks
        # If hyps_cpu_np_list contains only arrays of 0s (blank ID), get_output will return a list of empty strings ['']
        # get_output always returns a list.
        text_list = get_output(hyps_cpu_np_list, char_dict)

        # In ra output từ get_output để kiểm tra
        print(f"DEBUG (INFER): Output from get_output (list): {text_list}")

        # Extract the text for batch_size=1.
        if text_list and isinstance(text_list, list) and len(text_list) > 0:
            text = text_list[0]
            print(f"DEBUG (INFER): Final raw recognized text extracted from get_output: '{text}'")
            return text
        else:
             # This might happen if hyps_cpu_np_list was empty or get_output returned an empty list
             print(f"DEBUG (INFER): get_output returned empty list or unexpected format after processing hypotheses: {text_list}")
             return "" # Trả về chuỗi rỗng nếu không có văn bản nào được trích xuất


    except Exception as e:
        print(f"transcribe_features: Error during model inference pipeline: {e}")
        # import traceback
        # traceback.print_exc() # Uncomment for full traceback
        return "[Lỗi xử lý segment trong inference]"


# --- Hàm xử lý cho chế độ Tải file (sử dụng pydub) ---
def transcribe_uploaded_file(audio_filepath):
    if model is None:
        return "Lỗi: Không thể tải model."

    if audio_filepath is None:
        return "Vui lòng tải lên một file âm thanh."

    if not FFMPEG_AVAILABLE:
         return "Lỗi: FFmpeg/FFprobe không có sẵn. Không thể xử lý file âm thanh."

    try:
        print(f"Processing uploaded file: {audio_filepath}")
        # Sử dụng pydub để đọc toàn bộ file
        audio = AudioSegment.from_file(audio_filepath)
        # Chuyển đổi sang định dạng yêu cầu của model (16kHz, 16-bit, mono)
        audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        audio = audio.set_sample_width(2) # 16-bit
        audio = audio.set_channels(1) # Mono

        # Chuyển AudioSegment sang numpy array (float32)
        waveform_np = np.array(audio.get_array_of_samples()).astype(np.float32)

        # Tiền xử lý và tính feature cho toàn bộ audio (dùng hàm chung)
        feats, x_len = preprocess_numpy_audio(waveform_np, TARGET_SAMPLE_RATE, device) # Đã resample bởi pydub

        # Chạy inference trên feature đầy đủ
        text = transcribe_features(feats, x_len)
        print("Transcription complete.")
        return text
    except FileNotFoundError:
        return "Lỗi: File không tồn tại."
    except Exception as e:
        print(f"Error during transcription (upload mode): {e}")
        return f"Đã xảy ra lỗi trong quá trình xử lý file: {e}"


# --- Hàm xử lý cho chế độ Ghi âm trực tiếp (Streaming) ---

# State structure for streaming
STREAMING_STATE = {
    'current_session_buffer_np': np.array([], dtype=np.float32), # Buffer cho audio của phiên ghi âm HIỆN TẠI
    'processed_samples_in_session': 0, # Số mẫu đã xử lý trong phiên hiện tại
    'sample_rate': None, # Sample rate của input audio (từ microphone)
    'is_finalizing': False, # Flag đang xử lý đoạn cuối sau khi Stop
    'cumulative_transcript': '', # Transcript TÍCH LŨY qua các phiên ghi âm
    'current_session_draft': '', # Transcript nháp cho phiên ghi âm HIỆN TẠY
    'session_audio_to_save': None, # Lưu trữ numpy array của phiên vừa kết thúc để lưu file
    'session_counter': 0, # Đếm số phiên ghi âm đã hoàn thành
    'status_message': 'Sẵn sàng ghi âm.', # Thông báo trạng thái cho người dùng
    'clear_pending': False, # Flag chờ clear state trước khi bắt đầu phiên mới
}

def transcribe_streaming_segments(new_chunk, state):
    """
    Processes incoming audio chunks from streaming recording.
    Handles state management for multiple sessions, cumulative transcript,
    and preparing audio for saving.

    Args:
        new_chunk: Tuple (sample_rate, audio_array_numpy) or (None, None) on stop.
                   audio_array_numpy is float32 and normalized [-1, 1] by Gradio.
        state: The current state dictionary maintained by Gradio (modified by yield).

    Yields:
        A tuple (display_text, status_text, updated_state).
        display_text: The transcript shown to the user (cumulative + draft).
        status_text: Messages about recording/processing status.
        updated_state: The modified state dictionary.
    """
    # print(f"\n--- transcribe_streaming_segments: Start ---")
    # print(f"State on entry: is_finalizing={state.get('is_finalizing')}, clear_pending={state.get('clear_pending')}, cumulative_transcript len={len(state.get('cumulative_transcript',''))}, current_session_draft len={len(state.get('current_session_draft',''))}, session_counter={state.get('session_counter')}")
    # if new_chunk and new_chunk[1] is not None:
    #     print(f"DEBUG (STREAM): Received chunk: sample_rate={new_chunk[0]}, audio_chunk_np is {'None' if new_chunk[1] is None else f'shape {new_chunk[1].shape}, dtype {new_chunk[1].dtype}'}")
    # else:
    #     print("DEBUG (STREAM): Received chunk: Stop signal (None, None)")


    # Ensure state is initialized (safety check, should not be None with gr.State(value=...))
    if state is None:
        print("DEBUG (STREAM): Initial state was None, initializing.")
        state = STREAMING_STATE.copy()
        # Yield initial state and messages immediately
        yield state['cumulative_transcript'], state['status_message'], state
        return

    # If model isn't loaded, yield an error message and return
    if model is None or char_dict is None:
        error_msg = "Lỗi: Model chưa tải hoặc import thất bại."
        # Update status message only if it's different
        if state.get('status_message') != error_msg:
             state['status_message'] = error_msg
             print(error_msg)
        # Yield current state (which contains the error message) and stop
        # Yielding the cumulative transcript + draft
        display_text = state['cumulative_transcript']
        if state['current_session_draft']:
             if display_text: display_text += "\n"
             display_text += "[Phiên hiện tại] " + state['current_session_draft'].strip()
        yield display_text, state['status_message'], state
        return # Exit generator

    sample_rate, audio_chunk_np = new_chunk

    # --- Handle Clear Pending ---
    # Check clear_pending at the start of processing a new chunk or stop signal
    if state.get('clear_pending', False):
        print("DEBUG (STREAM): Clear pending detected. Resetting state.")
        # Perform a full reset, but keep sample rate if known (as it's likely hardware default)
        initial_sample_rate = state.get('sample_rate')
        state = STREAMING_STATE.copy()
        state['sample_rate'] = initial_sample_rate
        state['status_message'] = 'Đã xóa. Sẵn sàng ghi âm mới.'
        state['clear_pending'] = False # Reset the flag

        # Yield immediate clear feedback
        # Note: If audio_chunk_np is None (Stop signal concurrent with clear button),
        # this yield happens, and the function exits below.
        # If audio_chunk_np is a valid chunk, this yields the cleared UI,
        # and then processing continues for the chunk in the new state.
        yield state['cumulative_transcript'], state['status_message'], state

        # If the clear was triggered by a Stop signal (audio_chunk_np is None), exit after clearing state
        if audio_chunk_np is None:
            # print("DEBUG (STREAM): Clear concurrent with Stop. Cleared state.")
            return # Exit generator


    # --- Handle Stop Recording (audio_chunk_np is None) ---
    if audio_chunk_np is None:
        # Process the very end of the current session's audio buffer
        # Only do this if not already finalizing (handles multiple rapid Stop signals)
        if not state.get('is_finalizing', False):
            state['is_finalizing'] = True
            state['status_message'] = 'Đang xử lý đoạn cuối...'
            print("DEBUG (STREAM): Stop signal received. Starting finalization.")
            # Yield current state to update status immediately
            display_text = state['cumulative_transcript']
            if state['current_session_draft']:
                 if display_text: display_text += "\n"
                 display_text += "[Phiên hiện tại] " + state['current_session_draft'].strip()
            yield display_text, state['status_message'], state # Update UI with final status before processing

            remaining_audio_np = state['current_session_buffer_np'][state['processed_samples_in_session']:]

            if remaining_audio_np.size > 0:
                print(f"DEBUG (STREAM): Processing remaining audio buffer ({remaining_audio_np.size} samples).")
                try:
                    # Use sample_rate from state (it's the rate of the buffer)
                    current_sample_rate = state.get('sample_rate', TARGET_SAMPLE_RATE)
                    if current_sample_rate is None:
                         print("DEBUG (STREAM): Error: sample_rate is None during finalization.")
                         segment_text = "[Lỗi: Thiếu sample rate]"
                    else:
                         # preprocess_numpy_audio handles resampling to TARGET_SAMPLE_RATE internally
                         feats, x_len = preprocess_numpy_audio(remaining_audio_np, current_sample_rate, device)

                         segment_text = "" # Default segment text
                         if feats is not None: # Only process if preprocess successful
                             segment_text = transcribe_features(feats, x_len)
                             # Check if model returned empty text for valid audio
                             if not segment_text.strip() and not segment_text.startswith("[Lỗi:"):
                                  segment_text = "[...]" # Indicate processed but no speech detected
                         else:
                             # This case implies the remaining audio was too short after resampling
                             segment_text = "[Đoạn cuối quá ngắn hoặc lỗi tiền xử lý]" # Informative message

                    # Append segment result to current session draft
                    state['current_session_draft'] += " " + segment_text.strip() if state['current_session_draft'] else segment_text.strip()

                    # Mark as processed (the entire remaining buffer)
                    state['processed_samples_in_session'] = len(state['current_session_buffer_np'])
                    print(f"DEBUG (STREAM): Processed final chunk. Raw text: '{segment_text.strip()}'")

                except Exception as e:
                    print(f"DEBUG (STREAM): Error processing final chunk: {e}")
                    # Add error message to current session draft
                    state['current_session_draft'] += " [Lỗi xử lý đoạn cuối]" if state['current_session_draft'] else "[Lỗi xử lý đoạn cuối]"
                    state['processed_samples_in_session'] = len(state['current_session_buffer_np']) # Move index forward

            # --- Finalize Session ---
            state['session_counter'] += 1
            # Clean up the current session draft and append to cumulative transcript
            cleaned_session_transcript = re.sub(r'\s+', ' ', state['current_session_draft']).strip()

            if cleaned_session_transcript: # Only add to cumulative if there's actual text
                 if state['cumulative_transcript']:
                     state['cumulative_transcript'] += "\n\n" # Add separator between sessions
                 # Add session header and the cleaned transcript
                 state['cumulative_transcript'] += f"[Phiên {state['session_counter']}] " + cleaned_session_transcript

            # Store the full audio of this session for saving
            state['session_audio_to_save'] = state['current_session_buffer_np']
            # Clear current session state variables to prepare for a new session
            state['current_session_buffer_np'] = np.array([], dtype=np.float32)
            state['processed_samples_in_session'] = 0
            state['current_session_draft'] = ''
            state['is_finalizing'] = False # Ready for next session

            state['status_message'] = f'Phiên {state["session_counter"]} đã hoàn thành. Có thể Lưu audio hoặc Ghi âm tiếp.'
            print(f"DEBUG (STREAM): Session {state['session_counter']} finalized. Cumulative length: {len(state['cumulative_transcript'])}")

            # Yield the final cumulative transcript and the updated state
            # This is the final yield triggered by the Stop signal
            yield state['cumulative_transcript'], state['status_message'], state

        # After processing the stop signal, this generator call is finished.
        # Subsequent calls with None will just hit the `is_finalizing` check and return.
        # print("DEBUG (STREAM): Stop handling complete.")
        return # Exit generator

    # --- Handle Incoming Audio Chunk ---
    # If we are in a finalizing state but receive audio, something is wrong, ignore chunk
    if state.get('is_finalizing', False):
         # print("DEBUG (STREAM): Warning: Received chunk while finalizing. Ignoring.")
         # Still yield current state to keep UI responsive, but don't process chunk
         display_text = state['cumulative_transcript']
         if state['current_session_draft']:
              if display_text: display_text += "\n"
              display_text += "[Phiên hiện tại] " + state['current_session_draft'].strip()
         yield display_text, state['status_message'], state
         return

    # Cập nhật sample rate trong state nếu đây là chunk đầu tiên của một phiên mới
    # hoặc nếu sample rate thay đổi (không mong đợi từ mic)
    if state.get('sample_rate') is None or state['sample_rate'] != sample_rate:
         print(f"DEBUG (STREAM): Sample rate updated to {sample_rate} Hz.")
         state['sample_rate'] = sample_rate


    # Chuyển chunk numpy sang float32 (nếu chưa) và thêm vào buffer phiên hiện tại
    # Gradio type="numpy" audio is usually float32 and normalized [-1, 1]
    chunk_float32 = audio_chunk_np.astype(np.float32)
    state['current_session_buffer_np'] = np.concatenate((state['current_session_buffer_np'], chunk_float32))

    total_samples_in_session_buffer = len(state['current_session_buffer_np'])
    # print(f"DEBUG (STREAM): Chunk added. Total samples in current session buffer: {total_samples_in_session_buffer}. Processed samples in session: {state['processed_samples_in_session']}.")

    # Update status message with current recording duration
    # Use state.get('sample_rate', TARGET_SAMPLE_RATE) for safety, though sample_rate should be set by now
    current_duration_s = total_samples_in_session_buffer / state.get('sample_rate', TARGET_SAMPLE_RATE)
    state['status_message'] = f'Đang ghi âm... ({current_duration_s:.1f} s)'


    # Process buffer into full segments and update draft transcript
    segments_processed_in_this_call = 0
    # Check if we have enough new samples for at least one segment
    while total_samples_in_session_buffer - state['processed_samples_in_session'] >= SAMPLES_PER_SEGMENT:
        segment_start_idx = state['processed_samples_in_session']
        segment_end_idx = state['processed_samples_in_session'] + SAMPLES_PER_SEGMENT
        current_segment_np = state['current_session_buffer_np'][segment_start_idx : segment_end_idx]

        # print(f"DEBUG (STREAM): Processing segment from index {segment_start_idx} to {segment_end_idx} ({SAMPLES_PER_SEGMENT} samples).")

        try:
            # Use sample rate from state (rate of the buffer). Preprocessor resamples.
            current_sample_rate = state.get('sample_rate', TARGET_SAMPLE_RATE)
            if current_sample_rate is None:
                 print("DEBUG (STREAM): Error: sample_rate is None during segment processing.")
                 segment_text = "[Lỗi: Thiếu sample rate]"
            else:
                 # preprocess_numpy_audio handles resampling to TARGET_SAMPLE_RATE internally
                 feats, x_len = preprocess_numpy_audio(current_segment_np, current_sample_rate, device)

                 segment_text = "" # Default segment text
                 if feats is not None: # Only process if preprocess successful
                      segment_text = transcribe_features(feats, x_len)
                      # Check if model returned empty text for valid audio chunk
                      if not segment_text.strip() and not segment_text.startswith("[Lỗi:"):
                           segment_text = "[...]" # Indicate processed but no speech detected in this segment chunk
                 else:
                      # This case happens if a segment chunk, after resampling, is too short for features
                      segment_text = "[Đoạn quá ngắn hoặc lỗi tiền xử lý]" # Informative message


            # Append segment result to the current session draft
            state['current_session_draft'] += " " + segment_text.strip() if state['current_session_draft'] else segment_text.strip()

            # Update processed samples index for the current session buffer
            state['processed_samples_in_session'] = segment_end_idx
            segments_processed_in_this_call += 1

            # print(f"DEBUG (STREAM): Processed segment up to {state['processed_samples_in_session']} samples. Raw text: '{segment_text.strip()}'")

            # Yield the current display text (cumulative + current session draft) and state
            # Only strip the current session draft for display, cumulative should retain newlines
            display_text = state['cumulative_transcript']
            if state['current_session_draft']:
                 if display_text:
                      display_text += "\n" # Add newline before the current session draft
                 # Add a temporary header for the current session draft for clarity in UI
                 display_text += "[Phiên hiện tại] " + state['current_session_draft'].strip()


            yield display_text, state['status_message'], state

        except Exception as segment_e:
            print(f"DEBUG (STREAM): Error processing segment {segment_start_idx}-{segment_end_idx}: {segment_e}")
            # Add error message to current session draft
            state['current_session_draft'] += " [Lỗi xử lý segment]" if state['current_session_draft'] else "[Lỗi xử lý segment]"
            state['processed_samples_in_session'] = segment_end_idx # Move index forward
            segments_processed_in_this_call += 1 # Count as processed (with error)

            # Yield the transcript including the error
            display_text = state['cumulative_transcript']
            if state['current_session_draft']:
                 if display_text: display_text += "\n"
                 display_text += "[Phiên hiện tại] " + state['current_session_draft'].strip()

            yield display_text, state['status_message'], state # Yield both transcript and status

        # Re-check total samples in buffer in case buffer was cleaned up below
        total_samples_in_session_buffer = len(state['current_session_buffer_np'])

        # Add a small delay to avoid high CPU/GPU usage if segments are processed very fast
        # Tùy chỉnh giá trị này (ví dụ 0.01s - 0.1s)
        time.sleep(0.05)


    # --- Buffer Cleanup (Garbage Collection for current session buffer) ---
    # Clean up processed samples from the start of the current session buffer
    # Use a threshold relative to segment size
    BUFFER_CLEANUP_THRESHOLD_SAMPLES = SAMPLES_PER_SEGMENT * 5 # e.g., clean when at least 5 segments processed

    # Only perform cleanup if there's something processed AND the threshold is met
    if state['processed_samples_in_session'] > 0 and state['processed_samples_in_session'] >= BUFFER_CLEANUP_THRESHOLD_SAMPLES:
        # print(f"DEBUG (STREAM): Cleaning current session buffer. Removing {state['processed_samples_in_session']} processed samples.")
        # Keep only the unprocessed part
        state['current_session_buffer_np'] = state['current_session_buffer_np'][state['processed_samples_in_session']:]
        state['processed_samples_in_session'] = 0 # Reset processed index relative to the new buffer start
        # print(f"DEBUG (STREAM): Buffer cleaned. New buffer length: {len(state['current_session_buffer_np'])}. Processed samples index reset to 0.")


    # If no segments were processed in this call (e.g., chunk too small),
    # or after processing all available segments, yield the current state
    # to update the status message (e.g., duration) and save the state for the next chunk.
    # Only yield if not finalizing (finalizing yields differently).
    if not state.get('is_finalizing', False):
         # print("DEBUG (STREAM): Yielding intermediate state after processing chunk.")
         display_text = state['cumulative_transcript']
         if state['current_session_draft']:
              if display_text: display_text += "\n"
              display_text += "[Phiên hiện tại] " + state['current_session_draft'].strip()

         yield display_text, state['status_message'], state

    # print(f"--- transcribe_streaming_segments: End ---")


# --- Helper functions for saving ---
# (Giữ nguyên save_audio_file và save_transcript_file)

def save_audio_file(audio_np: np.ndarray, sample_rate: int, filename: str):
    """Saves a numpy audio array (float32, [-1, 1]) to a WAV file."""
    if audio_np is None or audio_np.size == 0:
        return None, "Không có dữ liệu âm thanh để lưu."

    try:
        # Convert float32 [-1, 1] to int16 [-32768, 32767] for WAV
        # Scale and clip to prevent overflow
        # np.int16 has range -32768 to 32767
        audio_int16 = np.int16(audio_np * 32767.0)

        # Use soundfile to write the WAV file
        sf.write(filename, audio_int16, sample_rate)

        print(f"Audio saved successfully to {filename}")
        return filename, "Lưu audio thành công!"

    except Exception as e:
        print(f"Error saving audio file {filename}: {e}")
        return None, f"Lỗi khi lưu file audio: {e}"


def save_transcript_file(transcript_text: str, filename: str):
    """Saves a transcript string to a TXT file."""
    if not transcript_text or not transcript_text.strip():
        return None, "Không có nội dung bản nhận dạng để lưu."

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(transcript_text.strip())

        print(f"Transcript saved successfully to {filename}")
        return filename, "Lưu bản nhận dạng thành công!"

    except Exception as e:
        print(f"Error saving transcript file {filename}: {e}")
        return None, f"Lỗi khi lưu file bản nhận dạng: {e}"


# --- Gradio event handlers for buttons ---
# (Giữ nguyên handle_save_audio_wrapper, handle_save_transcript_wrapper, handle_clear_history)

# Wrapper functions for button handlers to use the directory saving
def handle_save_audio_wrapper(state):
     """Gradio handler for the Save Audio button."""
     audio_to_save = state.get('session_audio_to_save')
     session_num = state.get('session_counter', 0)
     file_output_path = None # Clear the gr.File component before potentially setting it

     if audio_to_save is None or audio_to_save.size == 0 or session_num <= 0:
         message = 'Không có audio của phiên cuối cùng để lưu.'
         print("handle_save_audio_wrapper: No audio data to save.")
         # Clear the gr.File component by returning None for the file path
         return message, file_output_path, state

     filename = f"session_{session_num}_audio.wav"
     sample_rate = state.get('sample_rate', TARGET_SAMPLE_RATE) # Use last known sample rate or default

     output_dir = "saved_audio"
     os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
     filepath = os.path.join(output_dir, filename)

     saved_filepath, message = save_audio_file(audio_to_save, sample_rate, filepath)

     state['session_audio_to_save'] = None # Clear after saving
     state['status_message'] = message
     file_output_path = saved_filepath
     return state['status_message'], file_output_path, state


def handle_save_transcript_wrapper(state):
    """Gradio handler for the Save Transcript button."""
    transcript_to_save = state.get('cumulative_transcript', '')
    file_output_path = None

    if not transcript_to_save or not transcript_to_save.strip():
        message = 'Không có bản nhận dạng để lưu.'
        print("handle_save_transcript_wrapper: No transcript data to save.")
        return message, file_output_path, state # Clear file component

    filename = "full_transcript.txt"
    output_dir = "saved_transcripts"
    os.makedirs(output_dir, exist_ok=True) # Ensure directory exists
    filepath = os.path.join(output_dir, filename)

    saved_filepath, message = save_transcript_file(transcript_to_save, filepath)

    state['status_message'] = message
    file_output_path = saved_filepath
    return state['status_message'], file_output_path, state


def handle_clear_history(state):
    """Gradio handler for the Clear button."""
    print("handle_clear_history: Clearing state and UI.")
    # Set the flag for the streaming function to handle the actual state reset
    state['clear_pending'] = True
    # Immediately clear UI elements for responsiveness
    cleared_transcript_output = ''
    cleared_status_output = 'Đã xóa. Sẵn sàng ghi âm mới.'
    cleared_audio_file_output = None
    cleared_transcript_file_output = None
    # Return UI updates and the current state (reset happens on next streaming call)
    return cleared_transcript_output, cleared_status_output, cleared_audio_file_output, cleared_transcript_file_output, state


# Tạo Gradio Interface
# Check if model was successfully loaded globally
if model is not None and char_dict is not None:
    print("Model loaded successfully. Setting up Gradio interface.")

    # Interface cho chế độ Tải file
upload_interface = gr.Interface(
    fn=transcribe_uploaded_file,
    inputs=gr.Audio(type="filepath", label="Tải lên file âm thanh (.wav, .mp3, .flac, ...)"),
    outputs=gr.Text(label="Kết quả nhận dạng"),
    title="Nhận dạng Giọng nói - Tải File",
    description="Tải lên một file âm thanh để nhận dạng toàn bộ.",
    allow_flagging="never"
)

# Interface cho chế độ Ghi âm trực tiếp (Streaming)
with gr.Blocks() as record_interface_block:
    gr.Markdown(
        """
        # Nhận dạng Giọng nói - Ghi Âm Trực Tiếp (Streaming)
        Nhấn Record để bắt đầu ghi âm. Âm thanh sẽ được xử lý tự động theo các đoạn khoảng {SEGMENT_DURATION_S} giây khi bạn ghi.
        Kết quả hiển thị trong lúc ghi là bản nháp thô (có thể có khoảng trắng thừa/lỗi nối) của phiên hiện tại, bên dưới là bản tích lũy.
        Nhấn Stop khi bạn xong để xử lý đoạn cuối của phiên và thêm vào bản tích lũy. Bạn có thể ghi âm nhiều phiên liên tục.
        Sử dụng các nút Lưu để tải về audio của phiên cuối cùng hoặc toàn bộ bản nhận dạng tích lũy. Sử dụng Xóa để bắt đầu lại.
        """.format(SEGMENT_DURATION_S=SEGMENT_DURATION_S)
    )

    # Audio input component (Microphone)
    audio_input = gr.Audio(
        type="numpy",
        streaming=True,
        # Gradio provides sample rate along with numpy array, so no fixed sample_rate needed here
        label="Ghi âm trực tiếp"
    )

    # Output components
    transcript_output = gr.Textbox(label="Kết quả nhận dạng (Tích lũy + Nháp phiên hiện tại)", interactive=False, lines=10)
    status_output = gr.Textbox(label="Trạng thái", interactive=False)

    # Buttons for saving and clearing
    with gr.Row():
        save_audio_btn = gr.Button("Lưu Audio Phiên Cuối")
        save_transcript_btn = gr.Button("Lưu Bản Nhận Dạng Đầy Đủ")
        clear_btn = gr.Button("Xóa Lịch Sử")

    # File download components
    audio_file_output = gr.File(label="Tải xuống Audio Phiên Cuối", visible=True, file_count="single")
    transcript_file_output = gr.File(label="Tải xuống Bản Nhận Dạng Đầy Đủ", visible=True, file_count="single")

    # Define the state variable *INSIDE* the Blocks context
    # This state variable will be passed to and updated by the streaming and button handler functions
    state_variable = gr.State(value=STREAMING_STATE.copy())


    # --- Connect event handlers ---

    # Connect the audio stream to the processing function
    audio_input.stream(
        transcribe_streaming_segments,
        inputs=[audio_input, state_variable], # Pass audio chunk and current state
        outputs=[transcript_output, status_output, state_variable] # Update transcript, status, and state
    )

    # Connect button clicks to their respective handler functions
    save_audio_btn.click(
         handle_save_audio_wrapper,
         inputs=[state_variable], # Pass the current state
         outputs=[status_output, audio_file_output, state_variable] # Update status, audio file output, and state
    )

    save_transcript_btn.click(
         handle_save_transcript_wrapper,
         inputs=[state_variable], # Pass the current state
         outputs=[status_output, transcript_file_output, state_variable] # Update status, text file output, and state
    )

    clear_btn.click(
        handle_clear_history,
        inputs=[state_variable], # Pass the current state
        outputs=[transcript_output, status_output, audio_file_output, transcript_file_output, state_variable] # Clear text, status, file outputs, and update state (setting clear_pending)
    )

# Wrap the Blocks interface for TabbedInterface
record_interface = record_interface_block

# Combine the two interfaces into a tabbed interface
full_app = gr.TabbedInterface(
    [upload_interface, record_interface],
    ["Tải File Âm Thanh", "Ghi Âm Trực Tiếp"]
)

# --- Launch the Gradio app ---
if __name__ == "__main__":
    print("Starting Gradio app...")
    # Create output directories for saved files if they don't exist
    os.makedirs("saved_transcripts", exist_ok=True)
    os.makedirs("saved_audio", exist_ok=True)
    # Launch the app
    full_app.launch(debug=True, share=False) # Set share=True to get a public link
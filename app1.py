import os
import torch
import yaml
import numpy as np
import torchaudio # Cần cho Resample
import time

# Import các module Chunkformer
from chunkformer.model.utils.init_model import init_model
from chunkformer.model.utils.checkpoint import load_checkpoint
from chunkformer.model.utils.file_utils import read_symbol_table
from chunkformer.model.utils.ctc_utils import get_output
import torchaudio.compliance.kaldi as kaldi # Để tính Fbank

# Import Pydub và check FFmpeg (cho chế độ upload file)
from pydub import AudioSegment
from pydub.utils import which

# Import Gradio
import gradio as gr

# Kiểm tra xem ffmpeg có sẵn không (cần cho pydub - chế độ upload file)
# Chế độ streaming dùng numpy/torchaudio nên không cần ffmpeg
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

# Các tham số Chunkformer (có thể cần lấy từ config chính xác)
# Đây là các giá trị ví dụ dựa trên mã gốc, điều chỉnh nếu cần
# Tốt nhất là đọc từ config nếu có
try:
    with open(CONFIG_PATH, 'r') as fin:
        model_config = yaml.load(fin, Loader=yaml.FullLoader)
    # Đọc các tham số từ config nếu có sẵn tên thuộc tính tương ứng
    CHUNK_SIZE = model_config.get('encoder_conf', {}).get('chunk_size', 16)
    LEFT_CONTEXT_SIZE = model_config.get('encoder_conf', {}).get('left_context_size', 32)
    RIGHT_CONTEXT_SIZE = model_config.get('encoder_context_conf', {}).get('right_context_size', 8) # Tên thuộc tính có thể khác
    TRUNCATED_CONTEXT_SIZE = model_config.get('encoder_context_conf', {}).get('truncated_context_size', 16) # Tên thuộc tính có thể khác
    # Lấy kernel size CNN để tính cache shape
    CNN_MODULE_KERNEL = model_config.get('encoder_conf', {}).get('cnn_module_kernel', 31) # Giả định kernel size mặc định
    # Lấy các thông số khác cần cho cache shape
    NUM_BLOCKS = model_config.get('encoder_conf', {}).get('num_blocks', 12) # Giả định số block
    ATTENTION_HEADS = model_config.get('encoder_conf', {}).get('attention_heads', 8) # Giả định số attention heads
    OUTPUT_SIZE = model_config.get('encoder_conf', {}).get('output_size', 256) # Giả định output size

    print(f"Model parameters from config: CHUNK_SIZE={CHUNK_SIZE}, LEFT_CONTEXT_SIZE={LEFT_CONTEXT_SIZE}, RIGHT_CONTEXT_SIZE={RIGHT_CONTEXT_SIZE}, TRUNCATED_CONTEXT_SIZE={TRUNCATED_CONTEXT_SIZE}, CNN_MODULE_KERNEL={CNN_MODULE_KERNEL}, NUM_BLOCKS={NUM_BLOCKS}, ATTENTION_HEADS={ATTENTION_HEADS}, OUTPUT_SIZE={OUTPUT_SIZE}")

except FileNotFoundError:
    print(f"CẢNH BÁO: Không tìm thấy file config tại {CONFIG_PATH}. Sử dụng giá trị mặc định cho Chunkformer parameters.")
    CHUNK_SIZE = 16
    LEFT_CONTEXT_SIZE = 32
    RIGHT_CONTEXT_SIZE = 8
    TRUNCATED_CONTEXT_SIZE = 16
    CNN_MODULE_KERNEL = 31 # Giả định
    NUM_BLOCKS = 12 # Giả định
    ATTENTION_HEADS = 8 # Giả định
    OUTPUT_SIZE = 256 # Giả định
except Exception as e:
     print(f"CẢNH BÁO: Lỗi đọc config file {CONFIG_PATH}: {e}. Sử dụng giá trị mặc định.")
     CHUNK_SIZE = 16
     LEFT_CONTEXT_SIZE = 32
     RIGHT_CONTEXT_SIZE = 8
     TRUNCATED_CONTEXT_SIZE = 16
     CNN_MODULE_KERNEL = 31 # Giả định
     NUM_BLOCKS = 12 # Giả định
     ATTENTION_HEADS = 8 # Giả định
     OUTPUT_SIZE = 256 # Giả định


# Cấu hình cho xử lý theo segment trong chế độ streaming
SEGMENT_DURATION_S = 3 # Độ dài mỗi segment để xử lý (ví dụ 3 giây)
TARGET_SAMPLE_RATE = 16000 # Sample rate mà model mong đợi (đã resample trước khi tính fbank)
SAMPLES_PER_SEGMENT = int(SEGMENT_DURATION_S * TARGET_SAMPLE_RATE) # Số mẫu cần để tạo 1 segment

# Load model (chỉ cần load 1 lần khi script bắt đầu)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

def load_model(device="cpu"):
    """Loads the model, character dictionary."""
    try:
        with open(CONFIG_PATH, 'r') as fin:
            config = yaml.load(fin, Loader=yaml.FullLoader)
        model = init_model(config, CONFIG_PATH)
        model.eval()
        load_checkpoint(model, CHECKPOINT_PATH)
        model.to(device)
        symbol_table = read_symbol_table(VOCAB_PATH)
        char_dict = {v: k for k, v in symbol_table.items()}
        print("Model loaded successfully.")
        return model, char_dict
    except FileNotFoundError:
         print(f"Lỗi: Không tìm thấy file model hoặc config tại {MODEL_DIR}")
         return None, None
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Vui lòng kiểm tra đường dẫn model, file config, checkpoint và vocab.")
        return None, None

# Load model và char_dict toàn cục
model, char_dict = load_model(device=device)

# Khởi tạo Resampler (có thể tạo một lần để tối ưu)
if model is not None:
    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=TARGET_SAMPLE_RATE).to(device)
    # Giả định sample rate mặc định từ Gradio là 48000 Hz
    # Nếu sample rate thực tế khác, resampler sẽ được tạo lại trong preprocess_numpy_audio


# Hàm tiền xử lý audio segment (numpy array) thành features
def preprocess_numpy_audio(audio_np: np.ndarray, sample_rate: int, device: torch.device):
    """Preprocesses a numpy audio array into fbank features."""
    if audio_np.size == 0:
        return None, None # Xử lý trường hợp mảng rỗng

    # Chuyển numpy float sang torch tensor float32
    # Gradio thường cung cấp float64 hoặc float32, cần đảm bảo dtype
    waveform = torch.tensor(audio_np, dtype=torch.float32).unsqueeze(0) # shape (1, num_samples)

    # Resample nếu sample rate không khớp
    if sample_rate != TARGET_SAMPLE_RATE:
        print(f"Warning: Input sample rate {sample_rate} differs from target {TARGET_SAMPLE_RATE}. Resampling.")
        # Tạo resampler mới nếu sample rate đầu vào thay đổi hoặc khác với giả định ban đầu (48k)
        current_resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE).to(device)
        waveform = current_resampler(waveform)
        # sample_rate = TARGET_SAMPLE_RATE # Không cần cập nhật biến sample_rate này nữa, vì waveform đã ở target rate
    else:
         waveform = waveform.to(device) # Đưa waveform lên device ngay cả khi không resample

    # Tính đặc trưng fbank (kaldi cần sample rate là số nguyên, sử dụng TARGET_SAMPLE_RATE)
    # Energy floor nhỏ để tránh lỗi log(0) cho các đoạn im lặng hoàn toàn
    feats = kaldi.fbank(waveform, # Use the (potentially resampled) waveform on the correct device
        num_mel_bins=80,
        frame_length=25, # ms
        frame_shift=10,  # ms
        dither=0.0,
        energy_floor=1e-6, # Sử dụng giá trị nhỏ thay vì 0.0
        sample_frequency=TARGET_SAMPLE_RATE).unsqueeze(0) # Use TARGET_SAMPLE_RATE here

    x_len = torch.tensor([feats.shape[1]], dtype=torch.int)

    # Loại bỏ các đoạn feature quá ngắn nếu có (ví dụ < 3 frames)
    if x_len.item() < 3:
        print(f"Warning: Generated segment feature is too short ({x_len.item()} frames), skipping.")
        return None, None


    return feats, x_len


# Hàm xử lý inference trên feature tensor
def transcribe_features(feats, x_len):
    """Runs inference on precomputed features."""
    if model is None or char_dict is None:
        return "[Lỗi: Model chưa tải]" # Trả về lỗi thay vì raise exception
    if feats is None or x_len is None:
        return "" # Không xử lý nếu không có features hợp lệ

    # Đưa dữ liệu lên device của model
    feats = feats.to(device)
    x_len = x_len.to(device)

    # Khi xử lý độc lập từng segment, ta khởi tạo cache mới và offset=0
    # Đây không phải là streaming context *cho model* mà là streaming context *cho ứng dụng*
    # Mỗi segment được coi là một sequence độc lập để forward qua model
    try:
        att_cache_shape = (NUM_BLOCKS, LEFT_CONTEXT_SIZE, ATTENTION_HEADS, OUTPUT_SIZE * 2 // ATTENTION_HEADS)
        cnn_cache_shape = (NUM_BLOCKS, OUTPUT_SIZE, CNN_MODULE_KERNEL // 2) # Giả định cnn_module_kernel/2 cho right padding

        att_cache = torch.zeros(att_cache_shape, device=device)
        cnn_cache = torch.zeros(cnn_cache_shape, device=device)
        offset = torch.zeros(1, dtype=torch.int, device=device)

        # Sử dụng forward_parallel_chunk. Dù cho input ngắn, nó vẫn hoạt động.
        # Nó coi input là một batch có 1 phần tử (cái segment/file đang xử lý)
        encoder_outs, encoder_lens, n_chunks, _, _, _ = model.encoder.forward_parallel_chunk(
            xs=feats,
            xs_origin_lens=x_len,
            chunk_size=CHUNK_SIZE,
            left_context_size=LEFT_CONTEXT_SIZE,
            right_context_size=RIGHT_CONTEXT_SIZE,
            att_cache=att_cache, # Khởi tạo mới cho mỗi inference segment
            cnn_cache=cnn_cache, # Khởi tạo mới cho mỗi inference segment
            truncated_context_size=TRUNCATED_CONTEXT_SIZE,
            offset=offset # Luôn là 0 khi bắt đầu sequence mới (từng segment)
        )

        # Giải mã CTC
        # ctc_forward cần n_chunks nếu output từ forward_parallel_chunk được cấu trúc theo chunk
        # Ở đây ta đang xử lý độc lập từng segment, n_chunks sẽ tương ứng với số chunk trong segment đó
        hyps = model.encoder.ctc_forward(encoder_outs, encoder_lens, n_chunks)
        text = get_output(hyps, char_dict)[0] # get_output trả về list, lấy phần tử đầu tiên

        return text
    except Exception as e:
        print(f"Error during model inference: {e}")
        # Trả về thông báo lỗi hoặc chuỗi rỗng
        return "[Lỗi xử lý segment]"


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

# Khởi tạo state ban đầu cho Gradio
# Gradio sẽ tự động truyền state này vào hàm lần đầu tiên, sau đó hàm sẽ trả về state đã cập nhật
INITIAL_STREAMING_STATE = {
    'audio_buffer_np': np.array([], dtype=np.float32), # Buffer rỗng ban đầu
    'full_transcript': '',
    'processed_samples': 0, # Số mẫu đã xử lý (đã đưa vào segment)
    'sample_rate': None, # Sample rate của input audio, được cập nhật khi nhận chunk đầu tiên
}

def transcribe_streaming_segments(new_chunk, state):
    """
    Processes incoming audio chunks from streaming recording.
    Args:
        new_chunk: Tuple (sample_rate, audio_array_numpy) or (None, None) on stop.
        state: The current state dictionary maintained by Gradio.
               state = {'audio_buffer_np': np.ndarray, 'full_transcript': str,
                        'processed_samples': int, 'sample_rate': int or None}
    Returns:
        A tuple (updated_transcript_text, updated_state).
        This is a generator function, so it yields results periodically.
    """
    # print(f"Streaming: Received chunk. State before: {len(state['audio_buffer_np'])} samples, {state['processed_samples']} processed, SR: {state['sample_rate']}")

    # Lấy sample rate và audio chunk từ new_chunk
    # Khi stop, audio_chunk_np sẽ là None
    sample_rate, audio_chunk_np = new_chunk

    # --- Xử lý Stop Recording (audio_chunk_np is None) ---
    if audio_chunk_np is None:
        print("Streaming: Stop signal received.")
        # Xử lý bất kỳ audio còn lại trong buffer chưa đủ 1 segment đầy đủ
        remaining_audio_np = state['audio_buffer_np'][state['processed_samples']:]

        if remaining_audio_np.size > 0:
            print(f"Streaming: Processing remaining audio buffer ({remaining_audio_np.size} samples).")

            try:
                 # Tiền xử lý và nhận dạng đoạn cuối
                # Dùng sample_rate từ state vì đó là sample rate của buffer
                feats, x_len = preprocess_numpy_audio(remaining_audio_np, state['sample_rate'], device)

                segment_text = ""
                if feats is not None: # Chỉ xử lý inference nếu preprocess thành công
                    segment_text = transcribe_features(feats, x_len)

                # Thêm kết quả vào bản nháp đầy đủ
                state['full_transcript'] += segment_text.strip() + " "
                state['processed_samples'] = len(state['audio_buffer_np']) # Đánh dấu đã xử lý hết buffer

                print(f"Streaming: Processed final chunk. Text: '{segment_text.strip()}'")

            except Exception as e:
                print(f"Streaming: Error processing final chunk: {e}")
                state['full_transcript'] += "[Lỗi xử lý đoạn cuối] "

        # Trả về kết quả cuối cùng (đã strip khoảng trắng thừa) và reset state
        final_transcript = state['full_transcript'].strip()
        print(f"Streaming: Final transcript: '{final_transcript}'")

        # Reset state cho phiên ghi âm tiếp theo
        reset_state = INITIAL_STREAMING_STATE.copy() # Sử dụng copy để không sửa state gốc
        reset_state['audio_buffer_np'] = np.array([], dtype=np.float32) # Khởi tạo lại buffer rỗng

        # Yield kết quả cuối cùng và state đã reset
        yield final_transcript, reset_state
        return # Kết thúc generator

    # --- Xử lý Incoming Audio Chunk ---

    # Cập nhật sample rate trong state nếu đây là chunk đầu tiên
    if state['sample_rate'] is None:
         state['sample_rate'] = sample_rate
         print(f"Streaming: Initial sample rate detected: {sample_rate} Hz")
    elif state['sample_rate'] != sample_rate:
         # Điều này hiếm khi xảy ra với Gradio nhưng là kiểm tra tốt
         print(f"Streaming Warning: Sample rate changed from {state['sample_rate']} to {sample_rate}. This might cause issues.")
         # Quyết định cách xử lý: có thể bỏ qua chunk này hoặc cố gắng xử lý (phức tạp)
         # Để đơn giản, ta sẽ cố gắng xử lý nhưng cảnh báo. Preprocessor sẽ resample dựa trên state['sample_rate'].
         state['sample_rate'] = sample_rate # Cập nhật sample rate mới trong state


    # Chuyển chunk numpy sang float32 và thêm vào buffer
    # Đảm bảo nó là float32 để nhất quán với torch
    chunk_float32 = audio_chunk_np.astype(np.float32)
    state['audio_buffer_np'] = np.concatenate((state['audio_buffer_np'], chunk_float32))

    total_samples_in_buffer = len(state['audio_buffer_np'])
    # print(f"Streaming: Chunk added. Total samples in buffer: {total_samples_in_buffer}. Processed samples: {state['processed_samples']}.")


    # Xử lý buffer thành các segment đủ dài
    # Lặp trong khi còn đủ mẫu cho ít nhất 1 segment chưa xử lý
    while total_samples_in_buffer - state['processed_samples'] >= SAMPLES_PER_SEGMENT:
        segment_start_idx = state['processed_samples']
        segment_end_idx = state['processed_samples'] + SAMPLES_PER_SEGMENT
        current_segment_np = state['audio_buffer_np'][segment_start_idx : segment_end_idx]

        # print(f"Streaming: Processing segment from index {segment_start_idx} to {segment_end_idx} ({SAMPLES_PER_SEGMENT} samples).")

        try:
            # Tiền xử lý segment và nhận dạng
            # Pass sample rate TỪ STATE đến preprocess_numpy_audio
            feats, x_len = preprocess_numpy_audio(current_segment_np, state['sample_rate'], device)

            segment_text = ""
            if feats is not None: # Chỉ xử lý inference nếu preprocess thành công (không quá ngắn)
                segment_text = transcribe_features(feats, x_len)


            # Thêm kết quả vào bản nháp đầy đủ
            state['full_transcript'] += segment_text.strip() + " " # strip() để loại khoảng trắng thừa

            # Cập nhật chỉ số đã xử lý
            state['processed_samples'] = segment_end_idx

            print(f"Streaming: Processed segment up to {state['processed_samples']} samples. Text: '{segment_text.strip()}'")

            # Sử dụng yield để cập nhật giao diện người dùng với kết quả tích lũy
            # Yield cả state để Gradio duy trì trạng thái
            yield state['full_transcript'].strip(), state

        except Exception as segment_e:
            print(f"Streaming: Error processing segment {segment_start_idx}-{segment_end_idx}: {segment_e}")
            # Thêm thông báo lỗi vào transcript nếu có lỗi xử lý segment
            state['full_transcript'] += "[Lỗi xử lý segment] "
            state['processed_samples'] = segment_end_idx # Vẫn di chuyển con trỏ để không kẹt ở đoạn lỗi
            yield state['full_transcript'].strip(), state # Yield cả lỗi để người dùng biết

        # Cập nhật lại tổng số mẫu trong buffer (có thể buffer đã bị cắt bớt bởi GC)
        total_samples_in_buffer = len(state['audio_buffer_np'])

        # Thêm một chút delay để tránh quá tải CPU/GPU nếu xử lý quá nhanh
        # Tùy chỉnh giá trị này (ví dụ 0.01s - 0.1s)
        time.sleep(0.05)


    # --- Xử lý Buffer Garbage Collection ---
    # Nếu có audio đã xử lý ở đầu buffer (processed_samples > 0) và buffer đủ lớn, cắt bớt
    # Đặt ngưỡng để tránh cắt buffer quá nhỏ liên tục
    BUFFER_CLEANUP_THRESHOLD_SAMPLES = SAMPLES_PER_SEGMENT * 2 # Ví dụ: cắt khi buffer đã xử lý được ít nhất 2 segment
    if state['processed_samples'] > 0 and state['processed_samples'] >= BUFFER_CLEANUP_THRESHOLD_SAMPLES:
        print(f"Streaming: Cleaning buffer. Removing {state['processed_samples']} processed samples.")
        state['audio_buffer_np'] = state['audio_buffer_np'][state['processed_samples']:]
        state['processed_samples'] = 0 # Reset chỉ số đã xử lý về 0 vì buffer đã được cắt
        print(f"Streaming: Buffer cleaned. New buffer length: {len(state['audio_buffer_np'])}. Processed samples index reset to 0.")


    # Nếu không có đủ audio cho một segment mới trong chunk hiện tại,
    # chỉ yield bản nháp hiện tại và state để cập nhật UI và lưu trạng thái.
    # Đây là cần thiết sau khi vòng while kết thúc, hoặc nếu chunk đến quá nhỏ.
    # Yield lần cuối trong hàm này trước khi kết thúc xử lý chunk hiện tại.
    yield state['full_transcript'].strip(), state


# Tạo Gradio Interface
if model is not None: # Chỉ tạo interface nếu model load thành công
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
    # Sử dụng type="numpy" và streaming=True để nhận numpy array chunks
    # Sử dụng gr.State để duy trì trạng thái buffer và transcript giữa các lần gọi hàm
    record_interface = gr.Interface(
        fn=transcribe_streaming_segments,
        # inputs: (sample_rate, audio_numpy_array) + state
        inputs=[
            gr.Audio(
                type="numpy",
                streaming=True,
                # sample_rate=TARGET_SAMPLE_RATE, # <-- Bỏ dòng này như đã thảo luận
                label="Ghi âm trực tiếp"
            ),
            gr.State(value=INITIAL_STREAMING_STATE) # Sử dụng gr.State với giá trị khởi tạo
        ],
        # outputs: updated_transcript + updated_state
        outputs=[
            gr.Text(label=f"Kết quả nhận dạng (hiển thị theo đoạn khoảng {SEGMENT_DURATION_S}s)"),
            gr.State() # Cần trả về State đã cập nhật
        ],
        live=True, # Bật chế độ live update cho streaming
        title="Nhận dạng Giọng nói - Ghi Âm Trực Tiếp (Streaming)",
        description=f"""Nhấn Record để bắt đầu ghi âm. Âm thanh sẽ được xử lý tự động theo các đoạn khoảng {SEGMENT_DURATION_S} giây khi bạn ghi.
        Nhấn Stop khi bạn xong. Kết quả sẽ hiển thị liên tục.""",
        allow_flagging="never",
        # cache_examples=False # Không cache ví dụ cho live streaming
    )

    # Kết hợp hai interface vào TabbedInterface
    full_app = gr.TabbedInterface(
        [upload_interface, record_interface],
        ["Tải File Âm Thanh", "Ghi Âm Trực Tiếp"]
    )

    # Chạy Gradio app
    if __name__ == "__main__":
        print("Starting Gradio app...")
        # Chạy server Gradio. Launch sẽ mở trình duyệt tự động (trừ khi set False)
        # debug=True giúp xem log lỗi chi tiết trong quá trình phát triển
        # share=True để tạo link công khai tạm thời (hữu ích khi chạy trên Colab/server từ xa)
        full_app.launch(debug=True, share=False)
else:
    print("Gradio interface not launched due to model loading error.")
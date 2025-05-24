from optimum.exporters.onnx import main_export

main_export(
    model_name_or_path="khanhld/chunkformer-large-vie",
    task="automatic-speech-recognition",
    output="chunkformer_onnx"
)


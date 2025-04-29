import requests
import torch
import transformers
from PIL import Image
from transformers import AutoModel, AutoProcessor, GenerationConfig
from huggingface_hub import login
import gc

# CUDA関連の設定最適化
torch.backends.cudnn.benchmark = True  # 固定サイズの入力に対して処理を高速化
torch.backends.cuda.matmul.allow_tf32 = True  # TensorFloat32を許可
torch.backends.cudnn.allow_tf32 = True  # cuDNNでTF32を許可

# Log in to Hugging Face
login(token="")

transformers.set_seed(42)
model_path = "MIL-UT/Asagi-8B"
processor = AutoProcessor.from_pretrained(model_path)

# RTX5070ti向けの最適化設定
device_map = "auto"
model_dtype = torch.bfloat16  # RTX5070tiのTensor Coreに最適な形式

# GPUメモリの確保と解放
gc.collect()
torch.cuda.empty_cache()

# モデルロード時に最適化設定を適用
model = AutoModel.from_pretrained(
    model_path, 
    trust_remote_code=True,
    torch_dtype=model_dtype,
    device_map=device_map,
    # Tensor Coreを最大活用するための設定
    use_flash_attention_2=True,
    attn_implementation="flash_attention_2",
)

# より効率的な生成設定
generation_config = GenerationConfig(
    do_sample=True,
    num_beams=5,
    max_new_tokens=256,
    temperature=0.7,
    repetition_penalty=1.5,
    # RTX5070ti向け最適化
    use_cache=True,
)

prompt = ("以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n"
            "### 指示:\n<image>\nこの画像を見て、質問に詳細かつ具体的に答えてください。この写真の問題を理解し、適切な途中式とともに問題に対する正確な回答を答えてください。\n\n### 応答:\n")

# sample image
sample_image_url = "https://cdn.discordapp.com/attachments/1109706306060832780/1366387937947811922/IMG_20250425_1751592.jpg?ex=6810c36e&is=680f71ee&hm=f5330b5f23a0062f2ab85afb164a3b6a17feddca38bb18911818960e4d0b825d&"
image = Image.open(requests.get(sample_image_url, stream=True).raw)

# 入力の最適化処理
with torch.cuda.amp.autocast(dtype=model_dtype):  # 自動混合精度計算を使用
    inputs = processor(
        text=prompt, images=image, return_tensors="pt"
    )
    inputs_text = processor.tokenizer(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs_text['input_ids']
    inputs['attention_mask'] = inputs_text['attention_mask']
    for k, v in inputs.items():
        if v.dtype == torch.float32:
            inputs[k] = v.to(model_dtype)
    inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}

    # バッチサイズ=1でもTensor Coreを活用できるようにする
    generate_ids = model.generate(
        **inputs,
        generation_config=generation_config
    )
    generated_text = processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

# do not print the prompt
if "<image>" in prompt:
    prompt = prompt.replace("<image>", " ")
generated_text = generated_text.replace(prompt, "")

print(f"Generated text: {generated_text}")

# 使用後にキャッシュを解放
torch.cuda.empty_cache()
gc.collect()

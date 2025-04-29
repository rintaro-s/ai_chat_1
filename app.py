from huggingface_hub import login
import streamlit as st
import torch
import gc
from PIL import Image
import os
import time
import uuid
from tempfile import NamedTemporaryFile
from transformers import (
    AutoProcessor, AutoTokenizer, AutoModelForCausalLM, AutoModel,
    LlavaForConditionalGeneration, BitsAndBytesConfig, TextIteratorStreamer
)
import warnings

# deprecation警告を抑制
warnings.filterwarnings("ignore", category=FutureWarning)

# 認証情報 - 本番環境では環境変数から取得するなどセキュアに管理
HF_TOKEN = ""
login(token=HF_TOKEN)

# Streamlit設定
st.set_page_config(page_title="高性能AI画像理解チャット", layout="wide", initial_sidebar_state="expanded")

# モデル情報の定義 - 安定して動作する信頼性の高いモデルのみを含める
VISION_MODELS = {
    "llava": {
        "name": "LLaVA-1.5 (7B)",
        "path": "llava-hf/llava-1.5-7b-hf",
        "vram": 7.0,
        "type": "vision_language",
        "loader_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "description": "英語に強い視覚言語モデル",
        "prompt_template": "USER: <image>\n{prompt}\nASSISTANT:"
    },
    "japanese-llava": {
        "name": "Japanese LLaVA (7B)",
        "path": "llava-jp/llava-jp-13b-instruct-lora-jaster-v1.0-to-llava1.6",
        "vram": 7.0,
        "type": "vision_language",
        "loader_class": LlavaForConditionalGeneration,
        "processor_class": AutoProcessor,
        "description": "日本語に特化した視覚言語モデル",
        "prompt_template": "USER: <image>\n{prompt}\nASSISTANT:"
    }
}

TEXT_MODELS = {
    "standard": {
        "elyza": {
            "name": "ELYZA-japanese-Llama-3-8b",
            "path": "elyza/Llama-3-ELYZA-JP-8B",
            "vram": 8.0,
            "loader_class": AutoModelForCausalLM,
            "processor_class": AutoTokenizer,
            "description": "日本語に最適化されたLlama3モデル"
        },
        "cyberagent": {
            "name": "Calm2-7B",
            "path": "cyberagent/calm2-7b-chat",
            "vram": 7.0,
            "loader_class": AutoModelForCausalLM,
            "processor_class": AutoTokenizer,
            "description": "自然な日本語対話に強いモデル"
        }
    },
    "analytic": {
        "microsoft-phi3": {
            "name": "Phi-3 Mini (7B)",
            "path": "microsoft/phi-3-mini-4k-instruct",
            "vram": 7.0,
            "loader_class": AutoModelForCausalLM,
            "processor_class": AutoTokenizer,
            "description": "推論・分析に特化したコンパクトなモデル"
        },
        "qwen": {
            "name": "Qwen2-7B-Instruct",
            "path": "Qwen/Qwen2-7B-Instruct",
            "vram": 7.0,
            "loader_class": AutoModelForCausalLM,
            "processor_class": AutoTokenizer,
            "description": "詳細な分析に向いた多言語モデル"
        }
    },
    "creative": {
        "stable-lm": {
            "name": "Japanese-StableLM",
            "path": "stabilityai/japanese-stablelm-instruct-gamma-7b",
            "vram": 7.0,
            "loader_class": AutoModelForCausalLM,
            "processor_class": AutoTokenizer,
            "description": "創造的なテキスト生成に優れた日本語モデル"
        }
    }
}

# セッション状態の初期化
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vision_model" not in st.session_state:
    st.session_state.vision_model = None
if "vision_processor" not in st.session_state:
    st.session_state.vision_processor = None
if "text_model" not in st.session_state:
    st.session_state.text_model = None
if "text_processor" not in st.session_state:
    st.session_state.text_processor = None
if "current_mode" not in st.session_state:
    st.session_state.current_mode = "standard"
if "current_vision_model" not in st.session_state:
    st.session_state.current_vision_model = None
if "current_text_model" not in st.session_state:
    st.session_state.current_text_model = None
if "temp_files" not in st.session_state:
    st.session_state.temp_files = []

# 一時ファイル管理
def cleanup_temp_files():
    for file_path in st.session_state.temp_files:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                pass
    st.session_state.temp_files = []

# アプリ終了時のクリーンアップ
if st.session_state.get("cleanup_required", False):
    cleanup_temp_files()
st.session_state["cleanup_required"] = True

# CUDA情報表示
def display_cuda_info():
    if torch.cuda.is_available():
        st.sidebar.success(f"CUDA: {torch.cuda.get_device_name(0)}")
        st.sidebar.success(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        st.sidebar.error("CUDA利用不可")

# メモリ状態と使用状況を表示する関数
def display_memory_status():
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        free_memory = total_memory - reserved_memory
        
        memory_info = (
            f"総VRAM: {total_memory:.2f}GB | "
            f"使用中: {reserved_memory:.2f}GB | "
            f"空き: {free_memory:.2f}GB"
        )
        
        # メモリ使用状況によって色を変える
        memory_percent = (reserved_memory / total_memory) * 100
        if memory_percent > 90:
            st.error(memory_info)
        elif memory_percent > 75:
            st.warning(memory_info)
        else:
            st.success(memory_info)
        
        # プログレスバーでメモリ使用量を表示
        st.progress(reserved_memory / total_memory)
        
        # 現在ロード済みのモデルを表示
        loaded_models = []
        if st.session_state.vision_model is not None:
            loaded_models.append(f"画像: {st.session_state.current_vision_model}")
        if st.session_state.text_model is not None:
            loaded_models.append(f"テキスト: {st.session_state.current_text_model}")
        
        if loaded_models:
            st.info("📌 現在ロード済みモデル: " + " | ".join(loaded_models))
        else:
            st.info("モデルはロードされていません")

# モデル管理関数の最適化
def unload_models(force=False):
    """
    モデルをアンロードする関数
    force=True の場合は強制的にすべてアンロード
    force=False の場合は必要に応じてアンロード
    """
    # モデルロード中は1回だけアンロード処理を実行する
    if not hasattr(st.session_state, "model_loading") or not st.session_state.model_loading:
        # メモリ使用状況を確認
        if torch.cuda.is_available():
            reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            memory_usage_percent = reserved_memory / total_memory * 100
            
            # メモリ使用率が80%を超える場合またはforce=Trueの場合のみアンロード
            if force or memory_usage_percent > 80:
                if st.session_state.vision_model is not None:
                    del st.session_state.vision_model
                    st.session_state.vision_model = None
                if st.session_state.text_model is not None:
                    del st.session_state.text_model
                    st.session_state.text_model = None
                gc.collect()
                torch.cuda.empty_cache()
                return True
        return False
    return False

def load_vision_model(model_key):
    """ビジョンモデルをロードする関数 - 既に同じモデルがロード済みならスキップ"""
    # モデルロード中フラグをセット
    st.session_state.model_loading = True
    
    # 同じモデルが既にロード済みならスキップ
    if (st.session_state.vision_model is not None and 
        st.session_state.current_vision_model == VISION_MODELS[model_key]["name"]):
        st.session_state.model_loading = False
        return st.session_state.vision_model, st.session_state.vision_processor, st.session_state.current_vision_model
    
    # 新しいモデルをロードする前に、必要に応じてメモリを確保
    if st.session_state.vision_model is not None:
        # 異なるモデルがロード済みなら、そのモデルのみ解放
        del st.session_state.vision_model
        st.session_state.vision_model = None
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        model_info = VISION_MODELS[model_key]
        
        # 量子化設定
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        with st.spinner(f"{model_info['name']} をロード中..."):
            # メモリ使用状況をチェック
            if torch.cuda.is_available():
                reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                available_memory = total_memory - reserved_memory
                
                if available_memory < model_info["vram"] + 1.0:  # 1GBのバッファーを追加
                    st.warning("メモリ不足のため既存のモデルを解放します...")
                    unload_models(force=True)  # メモリ不足なら強制解放
            
            # トークナイザー/プロセッサのロード
            processor = model_info["processor_class"].from_pretrained(
                model_info["path"],
                trust_remote_code=True,
                token=HF_TOKEN
            )
            
            # モデルのロード
            model = model_info["loader_class"].from_pretrained(
                model_info["path"],
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
                token=HF_TOKEN
            )
            
            # モデルロード完了後、フラグをリセット
            st.session_state.model_loading = False
            return model, processor, model_info["name"]
    except Exception as e:
        st.session_state.model_loading = False
        st.error(f"ビジョンモデルのロードに失敗しました: {str(e)}")
        return None, None, None

def load_text_model(model_key, mode="standard"):
    """テキストモデルをロードする関数 - 既に同じモデルがロード済みならスキップ"""
    # モデルロード中フラグをセット
    st.session_state.model_loading = True
    
    model_info = TEXT_MODELS[mode][model_key]
    
    # 同じモデルが既にロード済みならスキップ
    if (st.session_state.text_model is not None and 
        st.session_state.current_text_model == model_info["name"]):
        st.session_state.model_loading = False
        return st.session_state.text_model, st.session_state.text_processor, st.session_state.current_text_model
    
    # 新しいモデルをロードする前に、必要に応じてメモリを確保
    if st.session_state.text_model is not None:
        # 異なるモデルがロード済みなら、そのモデルのみ解放
        del st.session_state.text_model
        st.session_state.text_model = None
        gc.collect()
        torch.cuda.empty_cache()
    
    try:
        # 量子化設定 - テキストモデルは常に量子化
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        with st.spinner(f"{model_info['name']} をロード中..."):
            # メモリ使用状況をチェック
            if torch.cuda.is_available():
                reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                available_memory = total_memory - reserved_memory
                
                if available_memory < model_info["vram"] + 1.0:  # 1GBのバッファーを追加
                    st.warning("メモリ不足のため既存のモデルを解放します...")
                    unload_models(force=True)  # メモリ不足なら強制解放
            
            # トークナイザー/プロセッサのロード
            processor = model_info["processor_class"].from_pretrained(
                model_info["path"],
                trust_remote_code=True,
                token=HF_TOKEN
            )
            
            # モデルのロード
            model = model_info["loader_class"].from_pretrained(
                model_info["path"],
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config,
                token=HF_TOKEN
            )
            
            # モデルロード完了後、フラグをリセット
            st.session_state.model_loading = False
            return model, processor, model_info["name"]
    except Exception as e:
        st.session_state.model_loading = False
        st.error(f"テキストモデルのロードに失敗しました: {str(e)}")
        return None, None, None

# 画像処理関数
def optimize_image(image, max_size=1024):
    """画像を最適化"""
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # RGB形式に変換
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def analyze_image(model, processor, image_path, mode="standard"):
    """画像分析を行う関数"""
    try:
        image = Image.open(image_path)
        
        # モードに応じたプロンプト
        if mode == "standard":
            prompt = "この画像を簡潔に説明してください。"
        elif mode == "analytic":
            prompt = "この画像を詳細に分析し、含まれる情報、物体、人物、文字、数字などを詳しく説明してください。"
        else:  # creative
            prompt = "この画像を見て、連想されるストーリーや感情を表現してください。"
        
        # モデルのクラス名を取得
        model_class = model.__class__.__name__.lower()
        
        # LLaVA系のモデルは特殊な処理が必要
        if "llava" in model_class:
            try:
                # LLaVA形式での処理
                messages = [{"role": "user", "content": [{"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}]}]
                
                # LLaVA形式での入力準備
                input_ids = processor.apply_chat_template(
                    messages, 
                    return_tensors="pt"
                ).to(model.device)
                
                # 生成
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    output = model.generate(
                        input_ids=input_ids,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.5,
                        top_p=0.95,
                    )
                
                # 生成されたテキストを取得
                response = processor.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                return response
                
            except Exception as llava_e:
                st.warning(f"LLaVA形式での処理に失敗: {llava_e}")
                # 代替処理方法を試す
                try:
                    # 第2の方法: 単純なテンプレート使用
                    vison_model_info = next((v for k, v in VISION_MODELS.items() if model_class in k.lower()), None)
                    template = vison_model_info['prompt_template'] if vison_model_info else "USER: <image>\n{prompt}\nASSISTANT:"
                    formatted_prompt = template.format(prompt=prompt)
                    
                    # 画像とプロンプトを結合
                    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt").to(model.device)
                    
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = model.generate(
                            **inputs,
                            max_new_tokens=256,
                            do_sample=True,
                            temperature=0.5
                        )
                    
                    response = processor.batch_decode(output, skip_special_tokens=True)[0]
                    if prompt in response:
                        response = response.replace(prompt, "").strip()
                    
                    return response
                    
                except Exception as e2:
                    st.warning(f"代替方法でも失敗: {e2}")
                    # 最終的な代替方法
                    conversation = [{"role": "user", "content": f"{prompt}"}]
                    input_ids = processor.apply_chat_template(conversation, return_tensors="pt").to(model.device)
                    
                    with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                        output = model.generate(input_ids, max_new_tokens=256)
                    
                    response = processor.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
                    return f"画像分析結果: {response}"
        else:
            # 一般的なVLMモデルの処理
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                try:
                    # 標準の方法を試す
                    inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
                    
                    output = model.generate(
                        **inputs,
                        max_new_tokens=256,
                        do_sample=True,
                        temperature=0.5,
                        use_cache=True
                    )
                    
                    response = processor.batch_decode(output, skip_special_tokens=True)[0]
                    if prompt in response:
                        response = response.replace(prompt, "").strip()
                    
                    return response
                    
                except Exception as e:
                    st.warning(f"標準的な解析方法が失敗しました。代替方法を試みます: {e}")
                    
                    # 代替方法を試す
                    try:
                        # 画像のみの入力
                        inputs = processor(images=image, return_tensors="pt").to(model.device)
                        image_embeddings = model.get_image_features(**inputs)
                        
                        # テキスト入力の準備
                        text_inputs = processor(text=prompt, return_tensors="pt").to(model.device)
                        
                        # モデルによる処理
                        outputs = model(
                            input_ids=text_inputs.input_ids,
                            attention_mask=text_inputs.attention_mask,
                            pixel_values=inputs.pixel_values if hasattr(inputs, 'pixel_values') else None,
                            image_embeddings=image_embeddings
                        )
                        
                        return "画像を分析しました。詳細情報は得られませんでしたが、基本的な認識を行いました。"
                    except Exception as e2:
                        return f"画像認識に複数の方法を試みましたが失敗しました。エラー: {str(e2)}"
        
    except Exception as e:
        return f"画像分析中にエラーが発生しました: {str(e)}"

# テキスト生成関数
def generate_text(model, processor, prompt, temperature=0.7, max_tokens=512):
    """テキスト生成を行う関数"""
    try:
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):  # 非推奨警告を修正
            inputs = processor(prompt, return_tensors="pt")
            
            # データをGPUに転送
            for k, v in inputs.items():
                if hasattr(v, "dtype") and v.dtype == torch.float32:
                    inputs[k] = v.to(torch.bfloat16)
            inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}
            
            # 生成
            output = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                repetition_penalty=1.2,
                use_cache=True
            )
            
            # テキスト出力
            generated_text = processor.batch_decode(output, skip_special_tokens=True)[0]
            
            # プロンプト部分の削除
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, "").strip()
            
            return generated_text
    except Exception as e:
        return f"テキスト生成中にエラーが発生しました: {str(e)}"

# ストリーミング生成関数 - TextIteratorStreamerを使用した実装の追加
def generate_text_streaming(model, processor, prompt, temperature=0.7, max_tokens=512):
    try:
        # ストリーミング用の設定
        streamer = TextIteratorStreamer(processor, skip_special_tokens=True)
        
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):  # 修正版のautocast
            inputs = processor(prompt, return_tensors="pt")
            inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}
            
            # 別スレッドで生成
            import threading
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                repetition_penalty=1.1,
            )
            
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            # ストリーマーを返す
            return streamer, thread
            
    except Exception as e:
        st.error(f"ストリーミングセットアップエラー: {str(e)}")
        return None, None

# UIの構築
st.title("ハイパーAI画像理解チャット")
display_cuda_info()

# サイドバー - モデル設定
st.sidebar.title("モデル設定")

# モード選択
mode_options = {
    "standard": "標準モード (一般会話)",
    "analytic": "分析モード (詳細な推論・分析)",
    "creative": "創造モード (物語・創作)"
}
selected_mode = st.sidebar.selectbox(
    "使用モード",
    list(mode_options.keys()),
    format_func=lambda x: mode_options[x],
    index=0
)
st.session_state.current_mode = selected_mode

# ビジョンモデル選択
st.sidebar.subheader("画像理解モデル")
selected_vision = st.sidebar.selectbox(
    "モデルを選択",
    list(VISION_MODELS.keys()),
    format_func=lambda x: f"{VISION_MODELS[x]['name']} ({VISION_MODELS[x]['description']})"
)

# テキストモデル選択
st.sidebar.subheader("テキスト生成モデル")
available_text_models = TEXT_MODELS.get(selected_mode, TEXT_MODELS["standard"])
selected_text = st.sidebar.selectbox(
    "モデルを選択",
    list(available_text_models.keys()),
    format_func=lambda x: f"{available_text_models[x]['name']} ({available_text_models[x]['description']})"
)

# VRAM使用量計算
vision_vram = VISION_MODELS[selected_vision]["vram"]
text_vram = available_text_models[selected_text]["vram"]
total_vram = vision_vram + text_vram

if total_vram > 16:
    st.sidebar.warning(f"⚠️ 警告: 合計VRAM使用量({total_vram:.1f}GB)が16GBを超えています")
else:
    st.sidebar.success(f"VRAM使用量: {total_vram:.1f}GB/16GB")

# モデルロードボタン - 効率化バージョン
if st.sidebar.button("モデルをロード"):
    # モデルロード中フラグをセット (重複アンロード防止)
    st.session_state.model_loading = True
    
    # VRAMをチェック、必要な場合のみ解放
    vision_vram = VISION_MODELS[selected_vision]["vram"]
    text_vram = available_text_models[selected_text]["vram"]
    total_needed_vram = vision_vram + text_vram
    
    # 1回だけメモリチェックと解放を実行
    need_unload = False
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        available_memory = total_memory - reserved_memory
        
        # 必要なメモリが利用可能メモリを超える場合
        if total_needed_vram > available_memory * 0.9:
            need_unload = True
    
    # 必要な場合のみアンロード (1回だけ)
    if need_unload:
        st.warning("必要なVRAM容量が大きいため、既存のモデルを解放します")
        if st.session_state.vision_model is not None:
            del st.session_state.vision_model
            st.session_state.vision_model = None
        if st.session_state.text_model is not None:
            del st.session_state.text_model
            st.session_state.text_model = None
        gc.collect()
        torch.cuda.empty_cache()
    
    # ビジョンモデルのロード (同じモデルなら再利用)
    st.session_state.vision_model, st.session_state.vision_processor, st.session_state.current_vision_model = load_vision_model(selected_vision)
    
    # テキストモデルのロード (同じモデルなら再利用)
    st.session_state.text_model, st.session_state.text_processor, st.session_state.current_text_model = load_text_model(selected_text, selected_mode)
    
    # モデルロード中フラグをリセット
    st.session_state.model_loading = False
    
    # 結果表示
    if st.session_state.vision_model and st.session_state.text_model:
        st.success(f"モデルのロードが完了しました: {st.session_state.current_vision_model} + {st.session_state.current_text_model}")
        
        # メモリ使用状況の表示
        display_memory_status()
    else:
        st.error("一部のモデルをロードできませんでした")

# 生成設定
st.sidebar.subheader("生成設定")
temperature = st.sidebar.slider("Temperature", 0.1, 1.5, 0.7)
max_tokens = st.sidebar.slider("最大トークン数", 100, 2000, 512)

# チャットインターフェース
chat_container = st.container()

with chat_container:
    if not st.session_state.chat_history:
        st.info("👋 モデルをロードして、メッセージを送信してください")
    
    for i, chat in enumerate(st.session_state.chat_history):
        col1, col2 = st.columns([1, 9])
        
        if chat["role"] == "user":
            with col1:
                st.image("https://api.dicebear.com/7.x/identicon/svg?seed=user", width=50)
            with col2:
                st.markdown(f"**ユーザー**")
                st.markdown(chat["content"])
                # 画像表示
                if "images" in chat and chat["images"]:
                    for img in chat["images"]:
                        st.image(img, width=300)
        else:
            with col1:
                st.image("https://api.dicebear.com/7.x/identicon/svg?seed=assistant", width=50)
            with col2:
                if "analysis" in chat and st.session_state.current_mode == "analytic":
                    st.markdown(f"**画像分析 ({st.session_state.current_vision_model})**")
                    st.markdown(chat["analysis"])
                    st.markdown("---")
                st.markdown(f"**AI応答 ({st.session_state.current_text_model})**")
                st.markdown(chat["content"])

# 入力エリア
st.subheader("メッセージ入力")
user_input = st.text_area("テキストを入力", height=100, key="user_input")

# 画像アップロード
uploaded_files = st.file_uploader("画像をアップロード (複数可)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
uploaded_images = []

if uploaded_files:
    cols = st.columns(min(4, len(uploaded_files)))
    for i, uploaded_file in enumerate(uploaded_files):
        # 画像処理
        image = Image.open(uploaded_file)
        processed_image = optimize_image(image)
        
        # 一時保存
        temp_file = NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_file_path = temp_file.name
        processed_image.save(temp_file_path)
        st.session_state.temp_files.append(temp_file_path)
        uploaded_images.append(temp_file_path)
        
        # サムネイル表示
        with cols[i % 4]:
            st.image(processed_image, width=150, caption=f"画像 {i+1}")

# 送信ボタン
if st.button("送信", key="send_button"):
    if st.session_state.vision_model is None or st.session_state.text_model is None:
        st.error("先にモデルをロードしてください")
    elif user_input.strip() or uploaded_images:
        # ユーザーメッセージ
        user_message = {"role": "user", "content": user_input}
        if uploaded_images:
            user_message["images"] = uploaded_images
        st.session_state.chat_history.append(user_message)
        
        # AI応答プレースホルダー
        ai_message = {"role": "assistant", "content": ""}
        st.session_state.chat_history.append(ai_message)
        
        # プレースホルダーと状態表示用
        response_placeholder = st.empty()
        status_placeholder = st.empty()
        
        # 画像分析
        image_descriptions = []
        if uploaded_images:
            with status_placeholder.status("画像を分析中...") as status:
                for i, img_path in enumerate(uploaded_images):
                    status.update(label=f"画像 {i+1}/{len(uploaded_images)} を分析中...")
                    
                    # 画像分析の実行
                    image_desc = analyze_image(
                        st.session_state.vision_model,
                        st.session_state.vision_processor,
                        img_path,
                        mode=st.session_state.current_mode
                    )
                    
                    image_descriptions.append(image_desc)
                    
                    # 分析モードでは分析結果も表示
                    if st.session_state.current_mode == "analytic":
                        if "analysis" not in ai_message:
                            ai_message["analysis"] = "**画像分析結果:**\n\n"
                        ai_message["analysis"] += f"**画像 {i+1}:** {image_desc}\n\n"
                
                status.update(label="画像分析完了", state="complete")
        
        # モードに応じたプロンプト構築
        if st.session_state.current_mode == "standard":
            if image_descriptions:
                prompt = f"{user_input}\n\n画像内容: {' '.join(image_descriptions)}"
            else:
                prompt = user_input
        elif st.session_state.current_mode == "analytic":
            if image_descriptions:
                prompt = f"以下の画像分析に基づいて、ユーザーの質問に詳細かつ論理的に回答してください。\n\n画像分析結果:\n{' '.join(image_descriptions)}\n\nユーザー質問: {user_input}"
            else:
                prompt = f"以下の質問に対して、詳細かつ論理的に回答してください。\n\nユーザー質問: {user_input}"
        else:  # creative
            if image_descriptions:
                prompt = f"以下の画像内容に基づいて、創造的かつ魅力的な回答を作成してください。\n\n画像内容:\n{' '.join(image_descriptions)}\n\nユーザー入力: {user_input}"
            else:
                prompt = f"以下のテーマについて、創造的かつ魅力的な回答を作成してください。\n\nテーマ: {user_input}"
        
        # テキスト生成（ストリーミング表示）
        with status_placeholder.status("回答を生成中...") as status:
            try:
                # ストリーミング生成を設定
                streamer, thread = generate_text_streaming(
                    st.session_state.text_model,
                    st.session_state.text_processor,
                    prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                if streamer and thread:
                    # ストリーミング出力
                    full_response = ""
                    
                    with response_placeholder.container():
                        col1, col2 = st.columns([1, 9])
                        with col1:
                            st.image("https://api.dicebear.com/7.x/identicon/svg?seed=assistant", width=50)
                        with col2:
                            if "analysis" in ai_message and st.session_state.current_mode == "analytic":
                                st.markdown(f"**画像分析 ({st.session_state.current_vision_model})**")
                                st.markdown(ai_message["analysis"])
                                st.markdown("---")
                            
                            # 応答表示用のプレースホルダー
                            response_text = st.empty()
                    
                    # ストリーミングでトークンを受け取る
                    for new_text in streamer:
                        full_response += new_text
                        # UIを更新
                        response_text.markdown(f"**AI応答 ({st.session_state.current_text_model})**\n\n{full_response}▌")
                        status.update(label=f"回答を生成中... ({len(full_response)}文字)")
                    
                    # スレッドが完了するのを待つ
                    thread.join()
                    
                    # 最終的な応答を設定
                    ai_message["content"] = full_response
                    status.update(label="回答生成完了", state="complete")
                else:
                    # フォールバック: 通常の生成
                    final_response = generate_text(
                        st.session_state.text_model,
                        st.session_state.text_processor,
                        prompt,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    
                    # UIを更新
                    with response_placeholder.container():
                        col1, col2 = st.columns([1, 9])
                        with col1:
                            st.image("https://api.dicebear.com/7.x/identicon/svg?seed=assistant", width=50)
                        with col2:
                            if "analysis" in ai_message and st.session_state.current_mode == "analytic":
                                st.markdown(f"**画像分析 ({st.session_state.current_vision_model})**")
                                st.markdown(ai_message["analysis"])
                                st.markdown("---")
                            st.markdown(f"**AI応答 ({st.session_state.current_text_model})**")
                            st.markdown(final_response)
                    
                    ai_message["content"] = final_response
                    status.update(label="回答生成完了", state="complete")
                    
            except Exception as gen_error:
                st.error(f"生成エラー: {gen_error}")
                ai_message["content"] = f"エラーが発生しました: {str(gen_error)}"
        
        # 画面を更新
        st.rerun()

# チャット履歴のクリア
if st.button("会話をクリア"):
    st.session_state.chat_history = []
    st.rerun()  # experimental_rerunから変更

# メモリ情報をサイドバーに表示
with st.sidebar.expander("メモリ使用状況", expanded=False):
    display_memory_status()
    
    # メモリクリアボタン
    if st.button("メモリを解放"):
        # モデルロード中でない場合のみ実行
        if not hasattr(st.session_state, "model_loading") or not st.session_state.model_loading:
            if st.session_state.vision_model is not None:
                del st.session_state.vision_model
                st.session_state.vision_model = None
            if st.session_state.text_model is not None:
                del st.session_state.text_model
                st.session_state.text_model = None
            gc.collect()
            torch.cuda.empty_cache()
            
            # セッション状態をクリア
            st.session_state.current_vision_model = None
            st.session_state.current_text_model = None
            st.success("すべてのモデルをメモリから解放しました")
            st.rerun()

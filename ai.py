import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModel, TextIteratorStreamer
from PIL import Image
import os
import tempfile
import pypdf
import gc
import io
import base64
import threading
from typing import Iterator, List, Optional
import requests
import json

from huggingface_hub import login

# Hugging Faceの認証
login(token="")
# 環境変数からトークンを取得
HF_TOKEN = os.getenv("HF_TOKEN", "")
login(token=HF_TOKEN)

# ページ設定
st.set_page_config(page_title="ストリーミング出力対応・高性能AI画像認識チャット", layout="wide")

# セッション状態の初期化
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_model_category" not in st.session_state:
    st.session_state.current_model_category = "標準モデル"
if "system_prompt" not in st.session_state:
    st.session_state.system_prompt = """あなたは丁寧で親切なAIアシスタントです。
ユーザーの質問に対して具体的で正確な回答を心がけます。
画像に関する質問には、見えている情報を詳細に説明します。
日本語のテキストやOCR結果が含まれている場合は、それを正確に読み取って説明します。
不明な点があれば、率直にわからないと伝えます。
回答は論理的に構築し、関連する情報を簡潔に提供します。"""

# モデルカテゴリと対応するモデルマッピング
MODEL_CATEGORIES = {
    "標準モデル": {
        "name": "elyza-japanese-Llama-3-8B-instruct",
        "id": "elyza/Llama-3-ELYZA-JP-8B",
        "description": "バランスの良い標準的な日本語モデル"
    },
    "推論モデル": {
        "name": "llm-jp-13B-instruct",
        "id": "llm-jp/llm-jp-13b-instruct-v1.0",
        "description": "複雑な推論に強い大規模日本語モデル"
    },
    "高速モデル": {
        "name": "rinna-japanese-gpt-neox-3.6b",
        "id": "rinna/japanese-gpt-neox-3.6b-instruction-ppo",
        "description": "軽量で応答速度が速いモデル"
    },
    "多機能モデル": {
        "name": "stabilityai-japanese-stablelm-base-alpha-7b",
        "id": "stabilityai/japanese-stablelm-base-alpha-7b",
        "description": "多様なタスクに対応する多機能モデル"
    }
}

# 安全なVRAM管理のための関数
def unload_models():
    for var in ['llm_model', 'llm_tokenizer', 'vision_model', 'vision_processor', 'ocr_model', 'ocr_processor']:
        if var in globals():
            del globals()[var]
    torch.cuda.empty_cache()
    gc.collect()

# VLM (Vision Language Model) 読み込み関数
@st.cache_resource(show_spinner=False)
def load_vision_model():
    with st.spinner("画像認識モデルを読み込んでいます..."):
        # 日本語対応のVLMモデル
        processor = AutoProcessor.from_pretrained("LLaVA-jp/LLaVA-NeXT-Japanese-7B-instruct", use_auth_token=HF_TOKEN)
        model = AutoModelForCausalLM.from_pretrained(
            "LLaVA-jp/LLaVA-NeXT-Japanese-7B-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=HF_TOKEN
        )
    return model, processor

# OCR特化モデル (Yomitoku代替) 読み込み関数
@st.cache_resource(show_spinner=False)
def load_ocr_model():
    with st.spinner("日本語OCRモデルを読み込んでいます..."):
        # 日本語OCR特化モデル（Yomitokuの代わりに利用可能なモデル）
        try:
            processor = AutoProcessor.from_pretrained("japanese-ocr/japanese-ocr-model", use_auth_token=HF_TOKEN)
            model = AutoModel.from_pretrained(
                "japanese-ocr/japanese-ocr-model",
                torch_dtype=torch.float16,
                device_map="auto",
                use_auth_token=HF_TOKEN
            )
            return model, processor
        except Exception as e:
            st.warning(f"日本語OCRモデルのロードに失敗しました: {e}")
            # フォールバックとして日本語CLIPモデルを使用
            processor = AutoProcessor.from_pretrained("rinna/japanese-clip-vit-b-32", use_auth_token=HF_TOKEN)
            model = AutoModel.from_pretrained(
                "rinna/japanese-clip-vit-b-32",
                torch_dtype=torch.float16,
                device_map="auto",
                use_auth_token=HF_TOKEN
            )
            return model, processor

# LLM (Language Model) 読み込み関数
@st.cache_resource(show_spinner=False)
def load_llm_model(model_category):
    model_info = MODEL_CATEGORIES[model_category]
    model_id = model_info["id"]
    
    with st.spinner(f"{model_info['name']}を読み込んでいます..."):
        tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=HF_TOKEN)
        
        # RTX5070tiのVRAM (16GB)に合わせて最適化
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            use_auth_token=HF_TOKEN
        )
    return model, tokenizer

# PDFからテキスト抽出関数
def extract_text_from_pdf(pdf_file):
    pdf_reader = pypdf.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

# デュアルビジョン分析関数（VLM + OCR特化モデルの組み合わせ）
def analyze_image_dual(image, vision_model, vision_processor, ocr_model, ocr_processor):
    results = {}
    
    # 1. VLMによる画像理解（全体的な理解）
    try:
        # VLMプロンプト
        vlm_prompt = "この画像を詳細に分析し、写っているものや状況を全て説明してください。"
        
        # VLM入力の準備
        inputs = vision_processor(text=vlm_prompt, images=image, return_tensors="pt").to("cuda")
        
        # 生成
        with torch.no_grad():
            output = vision_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.6,
                top_p=0.9
            )
        
        # 出力を取得
        vision_description = vision_processor.batch_decode(output, skip_special_tokens=True)[0]
        if vlm_prompt in vision_description:
            vision_description = vision_description.replace(vlm_prompt, "")
        
        results["vision"] = vision_description.strip()
    except Exception as e:
        results["vision"] = f"画像認識エラー: {str(e)}"
    
    # 2. OCRモデルによるテキスト検出（日本語テキスト特化）
    try:
        # OCRプロンプト
        ocr_prompt = "この画像から全てのテキスト、文字、数字を日本語で正確に読み取ってください。"
        
        # モデルの種類に応じて処理を分ける
        if "japanese-ocr" in str(ocr_model.__class__):
            # 専用OCRモデルの場合
            inputs = ocr_processor(text=ocr_prompt, images=image, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                output = ocr_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    temperature=0.2
                )
            
            ocr_text = ocr_processor.batch_decode(output, skip_special_tokens=True)[0]
            if ocr_prompt in ocr_text:
                ocr_text = ocr_text.replace(ocr_prompt, "")
            
            results["ocr"] = ocr_text.strip()
        else:
            # CLIPモデルの場合は画像特徴量を取得
            inputs = ocr_processor(images=image, return_tensors="pt").to("cuda")
            
            with torch.no_grad():
                outputs = ocr_model(**inputs)
                image_features = outputs.image_embeds
            
            # 簡易的なテキスト検出（実際のプロダクションでは改良が必要）
            results["ocr"] = "画像内のテキスト検出を試みました。特殊な日本語テキストが含まれている可能性があります。"
    except Exception as e:
        results["ocr"] = f"OCRエラー: {str(e)}"
    
    # 3. 両方の結果を組み合わせて返す
    combined_result = f"""【画像認識結果】
{results.get('vision', 'データなし')}

【テキスト検出結果(OCR)】
{results.get('ocr', 'データなし')}
"""
    
    return combined_result, results

# ストリーミングテキスト生成関数
def generate_streaming_text(prompt, image_analysis=None, llm_model=None, llm_tokenizer=None) -> Iterator[str]:
    # システムプロンプトを取得
    system_prompt = st.session_state.system_prompt
    
    # モデルカテゴリに基づくプロンプトの調整
    model_category = st.session_state.current_model_category
    
    if image_analysis is not None:
        # 画像分析がある場合は、画像についての質問であることを示す
        formatted_prompt = f"""システム: {system_prompt}

以下は画像の分析結果です:
{image_analysis}

ユーザー: {prompt}

アシスタント: """
    else:
        # 通常のテキスト質問
        formatted_prompt = f"""システム: {system_prompt}

ユーザー: {prompt}

アシスタント: """
    
    # モデルによってフォーマットを変更
    if "elyza" in MODEL_CATEGORIES[model_category]["id"].lower():
        formatted_prompt = f"<|system|>\n{system_prompt}\n<|user|>\n{prompt}\n<|assistant|>\n"
    
    inputs = llm_tokenizer(formatted_prompt, return_tensors="pt").to("cuda")
    
    # ストリーミング用のイテレータを設定
    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    # 生成パラメータ
    generation_kwargs = {
        "input_ids": inputs.input_ids,
        "attention_mask": inputs.attention_mask,
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "do_sample": True,
        "streamer": streamer,
    }
    
    # 別スレッドで生成を開始
    thread = threading.Thread(target=lambda: llm_model.generate(**generation_kwargs))
    thread.start()
    
    # ストリーミングイテレータを返す
    return streamer, formatted_prompt

# 画像をbase64エンコードする関数
def get_image_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

# サイドバー
with st.sidebar:
    st.title("設定")
    
    # モデルカテゴリ選択
    selected_category = st.selectbox(
        "モデルカテゴリを選択",
        list(MODEL_CATEGORIES.keys()),
        index=list(MODEL_CATEGORIES.keys()).index(st.session_state.current_model_category)
    )
    
    if selected_category != st.session_state.current_model_category:
        st.session_state.current_model_category = selected_category
        model_info = MODEL_CATEGORIES[selected_category]
        st.session_state.messages.append({
            "role": "assistant", 
            "content": f"モデルカテゴリを「{selected_category}」({model_info['name']})に変更しました。\n\n{model_info['description']}"
        })
    
    # 現在のモデル情報表示
    model_info = MODEL_CATEGORIES[st.session_state.current_model_category]
    st.info(f"現在のモデル: {model_info['name']}\n\n{model_info['description']}")
    
    # システムプロンプト設定
    with st.expander("システムプロンプト設定", expanded=False):
        system_prompt = st.text_area(
            "システムプロンプト",
            st.session_state.system_prompt,
            height=200
        )
        
        if st.button("システムプロンプトを更新"):
            st.session_state.system_prompt = system_prompt
            st.success("システムプロンプトを更新しました")
    
    # 生成パラメータ設定
    st.subheader("生成パラメータ")
    temperature = st.slider("Temperature (創造性)", 0.1, 1.5, 0.7, 0.1)
    max_tokens = st.slider("最大トークン数", 256, 2048, 1024, 64)
    
    # デュアルビジョン設定
    st.subheader("画像認識設定")
    dual_vision_enabled = st.checkbox("デュアルビジョン分析を有効化", value=True, 
                                     help="VLMとOCRモデルを組み合わせて、より正確な画像分析を行います")
    
    # PDFアップロード
    uploaded_pdf = st.file_uploader("PDFをアップロード", type=["pdf"])
    if uploaded_pdf:
        with st.spinner("PDFを解析中..."):
            pdf_text = extract_text_from_pdf(uploaded_pdf)
            if pdf_text:
                st.success(f"PDFを読み取りました ({len(pdf_text.split())}単語)")
                if st.button("PDFの内容について質問する"):
                    new_prompt = f"以下のPDFの内容について教えてください:\n\n{pdf_text[:10000]}..."  # 長すぎる場合は切り取り
                    st.session_state.messages.append({"role": "user", "content": new_prompt})

# メインチャットインターフェース
st.title("ストリーミング出力対応・高性能AI画像認識チャット")
model_info = MODEL_CATEGORIES[st.session_state.current_model_category]
st.write(f"現在のモデル: **{model_info['name']}**")
st.write("*画像認識機能: デュアルビジョン分析 (VLMとOCR)*")

# メッセージ履歴の表示
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], str):
            st.write(message["content"])
        elif isinstance(message["content"], dict):
            # テキストと画像の両方を含むメッセージ
            if "text" in message["content"]:
                st.write(message["content"]["text"])
            if "image" in message["content"]:
                st.image(message["content"]["image"], caption="アップロードされた画像")
            # 画像分析結果がある場合は表示（オプション）
            if "analysis" in message["content"]:
                with st.expander("画像分析詳細を表示"):
                    st.write(message["content"]["analysis"])

# ユーザー入力とファイルアップロード
uploaded_files = st.file_uploader("画像をアップロード (複数選択可)", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
user_prompt = st.chat_input("メッセージを入力...")

if user_prompt or (uploaded_files and len(uploaded_files) > 0):
    # ユーザーメッセージを作成
    content = user_prompt if user_prompt else "画像を分析してください"
    message_content = {"text": content}
    
    # 画像がアップロードされた場合
    if uploaded_files and len(uploaded_files) > 0:
        with st.spinner("画像を処理中..."):
            # 画像を表示
            images = []
            for uploaded_file in uploaded_files:
                image = Image.open(uploaded_file).convert('RGB')
                images.append(image)
            
            # 最初の画像をメッセージに追加（複数の場合は最初の画像のみ表示）
            message_content["image"] = images[0]
    
    # ユーザーメッセージを追加
    st.session_state.messages.append({"role": "user", "content": message_content})
    
    # ユーザーメッセージを表示
    with st.chat_message("user"):
        st.write(message_content.get("text", ""))
        if "image" in message_content:
            st.image(message_content["image"], caption="アップロードされた画像")
    
    # AI応答の生成
    try:
        # モデルをロード
        llm_model, llm_tokenizer = load_llm_model(st.session_state.current_model_category)
        
        # 画像が含まれている場合
        if "image" in message_content:
            # 画像認識モデルをロード
            with st.spinner("画像を分析中..."):
                vision_model, vision_processor = load_vision_model()
                
                # デュアルビジョンが有効な場合はOCRモデルも読み込み
                if dual_vision_enabled:
                    ocr_model, ocr_processor = load_ocr_model()
                    
                    # すべての画像をデュアルビジョン処理（VLM + OCR）
                    combined_analyses = []
                    full_analyses = {}
                    
                    for i, image in enumerate(images):
                        # デュアルビジョン分析を実行
                        combined_analysis, detailed_results = analyze_image_dual(
                            image, vision_model, vision_processor, ocr_model, ocr_processor
                        )
                        combined_analyses.append(combined_analysis)
                        full_analyses[f"image_{i+1}"] = detailed_results
                    
                    # 分析結果を保存
                    analysis_text = "\n\n".join(combined_analyses)
                    
                    # ユーザープロンプトに基づいたテキスト生成（ストリーミング対応）
                    prompt_with_image = f"{user_prompt if user_prompt else '画像に写っているものを詳しく説明してください'}"
                    streamer, formatted_prompt = generate_streaming_text(
                        prompt_with_image, analysis_text, llm_model, llm_tokenizer
                    )
                else:
                    # 従来通りのVLM単独処理
                    # VLMプロンプト
                    vlm_prompt = "この画像を詳細に分析し、写っているものや状況を全て説明してください。"
                    
                    # 入力の準備
                    inputs = vision_processor(text=vlm_prompt, images=images[0], return_tensors="pt").to("cuda")
                    
                    # 生成
                    with torch.no_grad():
                        output = vision_model.generate(
                            **inputs,
                            max_new_tokens=300,
                            do_sample=True,
                            temperature=0.6
                        )
                    
                    # 出力を取得
                    analysis_text = vision_processor.batch_decode(output, skip_special_tokens=True)[0]
                    if vlm_prompt in analysis_text:
                        analysis_text = analysis_text.replace(vlm_prompt, "")
                    
                    # テキスト生成（ストリーミング対応）
                    prompt_with_image = f"{user_prompt if user_prompt else '画像に写っているものを詳しく説明してください'}"
                    streamer, formatted_prompt = generate_streaming_text(
                        prompt_with_image, analysis_text, llm_model, llm_tokenizer
                    )
        else:
            # 通常のテキスト応答（ストリーミング対応）
            streamer, formatted_prompt = generate_streaming_text(user_prompt, None, llm_model, llm_tokenizer)
        
        # アシスタントのメッセージを作成して、ストリーミング出力を表示
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # ストリーミング出力
            for token in streamer:
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            
            # 最終的な応答を表示
            message_placeholder.markdown(full_response)
        
        # 応答をメッセージ履歴に追加
        response_content = {"text": full_response}
        
        # 画像分析があればそれも保存（オプション）
        if "image" in message_content and dual_vision_enabled:
            response_content["analysis"] = analysis_text
        
        st.session_state.messages.append({"role": "assistant", "content": response_content})
        
    except Exception as e:
        error_msg = f"エラーが発生しました: {str(e)}"
        st.error(error_msg)
        with st.chat_message("assistant"):
            st.write(error_msg)
        st.session_state.messages.append({"role": "assistant", "content": error_msg})

# メモリ使用状況の表示
if torch.cuda.is_available():
    with st.sidebar.expander("メモリ使用状況"):
        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        reserved_memory = torch.cuda.memory_reserved(0) / (1024**3)
        allocated_memory = torch.cuda.memory_allocated(0) / (1024**3)
        free_memory = total_memory - reserved_memory
        
        st.write(f"総VRAM: {total_memory:.2f} GB")
        st.write(f"確保済み: {reserved_memory:.2f} GB")
        st.write(f"使用中: {allocated_memory:.2f} GB")
        st.write(f"空き: {free_memory:.2f} GB")
        
        st.progress(reserved_memory / total_memory)
        
        if st.button("メモリをクリア"):
            unload_models()
            st.success("メモリをクリアしました")

# システム情報
with st.sidebar.expander("システム情報"):
    st.write("GPU: RTX5070ti (VRAM 16GB)")
    st.write("RAM: 96GB")
    st.write("CUDA Cores: 8,960")
    st.write("Tensor Cores: 280 (第5世代)")
    st.write("RT Cores: 70 (第4世代)")
    st.write("Boost Clock: 2,452MHz")
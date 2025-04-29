import os
import tempfile
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import uuid
from typing import Optional
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import PyPDF2
from pdf2image import convert_from_path
import pytesseract
import gc

# 画像の最適化
def optimize_image(image, max_size=1024):
    """画像を最適化してメモリ消費を抑える"""
    # アスペクト比を維持しながらリサイズ
    width, height = image.size
    if width > max_size or height > max_size:
        if width > height:
            new_width = max_size
            new_height = int(height * (max_size / width))
        else:
            new_height = max_size
            new_width = int(width * (max_size / height))
        image = image.resize((new_width, new_height), Image.LANCZOS)
    
    # RGB形式に変換（透過画像の場合）
    if image.mode == 'RGBA':
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        image = background
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    
    # JPEGに変換して圧縮
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85, optimize=True)
    buffer.seek(0)
    compressed_image = Image.open(buffer)
    
    return compressed_image

# PDF処理関数
def process_pdf(pdf_path):
    """PDFからテキストを抽出する"""
    text = ""
    
    # PyPDF2でテキスト抽出を試みる
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text += page.extract_text() + "\n\n"
    except Exception as e:
        print(f"PyPDF2でのテキスト抽出に失敗: {e}")
        
    # テキストが抽出できない場合、OCRを試みる
    if not text.strip():
        try:
            print("OCR処理を開始します...")
            # PDFを画像に変換
            images = convert_from_path(pdf_path)
            
            # 各画像にOCRを適用
            for i, image in enumerate(images):
                text += pytesseract.image_to_string(image, lang='jpn+eng') + "\n\n"
                
        except Exception as e:
            print(f"OCR処理に失敗: {e}")
            return f"PDF処理に失敗しました: {e}"
    
    return text.strip()

# 画像生成関数
def generate_image(prompt, width=512, height=512, num_steps=30):
    """テキストプロンプトから画像を生成する"""
    try:
        # クリーンアップ
        gc.collect()
        torch.cuda.empty_cache()
        
        # 日本語に対応した軽量なモデルを選択
        model_id = "stabilityai/stable-diffusion-xl-base-1.0"
        
        # メモリ効率の良い設定
        pipeline = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True
        )
        
        # 効率的なスケジューラ
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
            pipeline.scheduler.config, use_karras_sigmas=True
        )
        
        # xformersの代わりにメモリ効率化のための設定
        # attention_slicingはxformersなしでも使用可能
        pipeline.enable_attention_slicing(slice_size=1)
        
        # VRAM最適化のための追加設定
        pipeline.enable_model_cpu_offload()  # 必要に応じてCPUにオフロード
        
        # 画像生成
        image = pipeline(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=7.5
        ).images[0]
        
        # 一時ファイルに画像を保存
        filename = f"generated_{uuid.uuid4()}.jpg"
        temp_dir = os.path.join(tempfile.gettempdir(), "streamlit_images")
        os.makedirs(temp_dir, exist_ok=True)
        image_path = os.path.join(temp_dir, filename)
        image.save(image_path)
        
        # パイプラインを解放
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        
        return image_path
        
    except Exception as e:
        print(f"画像生成中にエラー発生: {e}")
        raise e

# 画像分析関数（メインから呼び出される用）
def analyze_image(model, processor, image_path, mode="standard"):
    """
    画像を分析してテキスト説明を生成する
    """
    try:
        image = Image.open(image_path)
        
        # モードに応じたプロンプト
        prompts = {
            "standard": "この画像を簡潔に説明してください。",
            "analytic": "この画像を詳細に分析し、含まれている情報を全て説明してください。図表や文字情報があれば、その内容も記述してください。",
            "creative": "この画像から連想されるストーリーや感情を表現してください。"
        }
        
        prompt = prompts.get(mode, prompts["standard"])
        
        # モデル固有のプロンプトフォーマットを適用
        if "yomitoku" in str(model.__class__.__name__).lower() or "yomitoku" in str(type(model)):
            formatted_prompt = f"以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{prompt}\n\n### 応答:"
        elif "qwen" in str(model.__class__.__name__).lower() or "qwen" in str(type(model)):
            formatted_prompt = f"<|im_start|>system\nあなたは優秀なAIアシスタントです。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # 一般的なプロンプト形式
            formatted_prompt = f"ユーザー: {prompt}\nAI: "
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            # プロセッサの形式によって処理を分ける
            try:
                inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
            except:
                # 別の形式を試す
                try:
                    inputs = processor(formatted_prompt, image, return_tensors="pt")
                except:
                    # 最後の手段
                    inputs = processor(formatted_prompt, return_tensors="pt")
                    print("画像入力に失敗しました。テキストのみで処理します。")
            
            # 入力を調整してGPUに転送
            for k, v in inputs.items():
                if v.dtype == torch.float32:
                    inputs[k] = v.to(torch.bfloat16)
            
            inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}
            
            # 生成設定
            generation_params = {
                "max_new_tokens": 150 if mode == "standard" else 300,
                "do_sample": True,
                "temperature": 0.5 if mode == "analytic" else 0.7,
                "top_p": 0.95,
                "use_cache": True,
                "repetition_penalty": 1.1
            }
            
            # 生成
            output = model.generate(**inputs, **generation_params)
            
            # デコード
            try:
                response = processor.batch_decode(output, skip_special_tokens=True)[0]
            except:
                # バッチデコードが使えない場合
                response = processor.decode(output[0], skip_special_tokens=True)
            
            # プロンプト部分を削除
            for prompt_text in [formatted_prompt, prompt]:
                if prompt_text in response:
                    response = response.replace(prompt_text, "").strip()
            
        return response
    except Exception as e:
        print(f"画像分析中にエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return "画像の分析中にエラーが発生しました。"

# Yomitokuを使用した日本語OCR特化の画像分析
def analyze_image_with_ocr(model, processor, image_path, mode="standard"):
    """
    日本語に対応した画像解析モデルを使用して画像内のテキストを特に重視して分析する
    (元々はYomitokuに特化していたが、より広く日本語VLMに対応)
    """
    try:
        image = Image.open(image_path)
        
        # モードに応じたプロンプト選択（日本語OCR向け）
        if mode == "standard":
            prompt = "この画像に写っているテキストや文字を読み取って、内容を説明してください。"
        elif mode == "analytic":
            prompt = "この画像に写っているすべての文字、テキスト、数字を正確に読み取り、さらに図表や構造も詳しく分析してください。"
        else:  # creative
            prompt = "この画像の中の文字情報と視覚的要素を読み取り、その内容から連想されるストーリーを考えてください。"
        
        # モデルのタイプを確認して適切な処理を行う
        model_type = str(type(model)).lower()
        
        if "clip" in model_type:
            # CLIPモデルの場合は特殊処理
            try:
                # CLIPはテキストと画像の類似度を計算するので、テキストエンコーダーを使う
                inputs = processor(text=[prompt], images=image, return_tensors="pt", padding=True)
                for k, v in inputs.items():
                    if hasattr(v, "dtype") and v.dtype == torch.float32:
                        inputs[k] = v.to(torch.bfloat16)
                inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}
                
                # CLIPモデルでの処理
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = torch.nn.functional.softmax(logits_per_image, dim=1)
                
                # シンプルな説明を返す
                response = f"画像にはテキスト情報が含まれています。関連度: {probs[0][0].item():.2f}"
                
            except Exception as e:
                print(f"CLIP処理中にエラー: {e}")
                response = "画像の分析中にエラーが発生しました。一般的な画像認識モデルでは日本語テキストの認識が困難な場合があります。"
        else:
            # 通常の生成モデルの場合
            formatted_prompt = f"以下は、タスクを説明する指示です。要求を適切に満たす応答を書きなさい。\n\n### 指示:\n{prompt}\n\n### 応答:"
            
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                try:
                    inputs = processor(text=formatted_prompt, images=image, return_tensors="pt")
                except:
                    try:
                        inputs = processor(formatted_prompt, image, return_tensors="pt")
                    except:
                        return "このモデルは指定された方法での画像入力に対応していません。"
                
                # 入力を調整してGPUに転送
                for k, v in inputs.items():
                    if hasattr(v, "dtype") and v.dtype == torch.float32:
                        inputs[k] = v.to(torch.bfloat16)
                
                inputs = {k: inputs[k].to(model.device) for k in inputs if k != "token_type_ids"}
                
                # 生成設定
                try:
                    output = model.generate(
                        **inputs,
                        max_new_tokens=300,  # OCRには長めのトークン数
                        do_sample=True,
                        temperature=0.3,  # OCRには低めの温度
                        use_cache=True
                    )
                    
                    response = processor.batch_decode(output, skip_special_tokens=True)[0]
                    
                    # プロンプト部分が含まれる場合は削除
                    if formatted_prompt in response:
                        response = response.replace(formatted_prompt, "").strip()
                except Exception as e:
                    print(f"生成中にエラー: {e}")
                    response = "モデルによる画像テキストの生成処理に失敗しました。"
        
        return response
    except Exception as e:
        print(f"OCR分析中にエラー発生: {e}")
        import traceback
        traceback.print_exc()
        return "画像分析中にエラーが発生しました。別のモデルを試してください。"

# 2つの分析結果を統合する関数
def combine_analyses(ocr_result, visual_result, mode="standard"):
    """
    YomitokuのOCR結果とVLMの視覚的分析結果を統合する
    """
    if mode == "standard":
        return f"【テキスト分析】{ocr_result}\n\n【視覚的分析】{visual_result}"
    elif mode == "analytic":
        # 重複を削除し、より詳細な分析を作成
        combined = "【詳細分析】\n"
        combined += f"テキスト情報: {ocr_result}\n\n"
        combined += f"視覚的情報: {visual_result}"
        return combined
    else:  # creative
        # より創造的な組み合わせ
        return f"この画像には以下の要素が含まれています。\n\n文字情報: {ocr_result}\n\n視覚的要素: {visual_result}"

import torch
import gc
from transformers import AutoModel, AutoProcessor, AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig, TextIteratorStreamer
import warnings
from huggingface_hub import login
from transformers import AutoModelForVision2Seq, LlavaForConditionalGeneration

# VLMモデルをロード
def load_vlm(model_path, quantize=False):
    """
    視覚言語モデル（VLM）をロードする関数
    """
    # 変数を初期化
    model = None
    processor = None
    
    # GPUメモリをクリア
    gc.collect()
    torch.cuda.empty_cache()
    
    # 量子化設定
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        quantization_config = None
    
    # プロセッサ/トークナイザーのロード - trust_remote_code=True を追加
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_auth_token=True)
        print(f"AutoProcessor(trust_remote_code=True)で{model_path}のプロセッサをロードしました")
    except Exception as e:
        warnings.warn(f"プロセッサのロードに失敗しました: {e}")
        try:
            processor = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_auth_token=True)
            print(f"AutoTokenizer(trust_remote_code=True)で{model_path}のトークナイザーをロードしました")
        except Exception as e2:
            warnings.warn(f"トークナイザーのロードにも失敗しました: {e2}")
            processor = None
    
    # モデルパスに基づいて特殊処理
    if "llava" in model_path.lower():
        try:
            # LLaVAは専用クラスを使用
            print(f"LLaVA専用ローダーで{model_path}をロード中...")
            model = LlavaForConditionalGeneration.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,
                quantization_config=quantization_config if quantize else None
            )
            print(f"LlavaForConditionalGeneration で {model_path} をロードしました")
        except Exception as e:
            print(f"LLaVA専用ローダーでのロードに失敗しました: {e}")
            # 以降の通常ローダーでもトライする
    
    elif "qwen" in model_path.lower():
        try:
            # Qwen-VLは完全にtrust_remote_codeに依存
            print(f"Qwen専用モードで{model_path}をロード中...")
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True,  # 必須
                quantization_config=quantization_config if quantize else None
            )
            print(f"QwenのAutoModelForCausalLM (trust_remote_code=True) で {model_path} をロードしました")
        except Exception as e:
            print(f"Qwen専用モードでのロードに失敗しました: {e}")
            # 以降の通常ローダーでもトライする
    
    # モデルがまだロードされていない場合は通常のローダーを試す
    if model is None:
        try:
            # Vision2Seqモデル（マルチモーダルモデル用)
            model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                quantization_config=quantization_config if quantize else None
            )
            print(f"AutoModelForVision2Seqで{model_path}をロードしました")
        except Exception as e:
            print(f"Vision2Seqロードに失敗しました: {e}")
            try:
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    quantization_config=quantization_config if quantize else None
                )
                print(f"AutoModelで{model_path}をロードしました")
            except Exception as e:
                print(f"標準ロードに失敗しました: {e}")
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        quantization_config=quantization_config if quantize else None
                    )
                    print(f"AutoModelForCausalLMで{model_path}をロードしました")
                except Exception as e2:
                    print(f"CausalLMとしてのロードにも失敗しました: {e2}")
                    if quantize:
                        try:
                            print("量子化なしでロードを試みます...")
                            model = AutoModel.from_pretrained(
                                model_path,
                                trust_remote_code=True,
                                torch_dtype=torch.bfloat16,
                                device_map="auto"
                            )
                            print(f"量子化なしで{model_path}をロードしました")
                        except Exception as e3:
                            print(f"すべてのロード方法が失敗しました: {e3}")
    
    # モデルまたはプロセッサがNoneの場合はエラー
    if model is None:
        # トラブルシューティング情報を出力
        print("\n=== トラブルシューティング情報 ===")
        print(f"モデルパス: {model_path}")
        print("依存パッケージの確認:")
        try:
            import transformers
            print(f"- transformers バージョン: {transformers.__version__}")
        except:
            print("- transformers が見つかりません")
        
        try:
            import triton
            print(f"- triton バージョン: {triton.__version__}")
        except:
            print("- triton が見つかりません。インストールを検討してください: pip install triton")
        
        try:
            import transformers_stream_generator
            print(f"- transformers_stream_generator バージョン: {transformers_stream_generator.__version__}")
        except:
            print("- transformers_stream_generator が見つかりません。インストールを検討してください: pip install transformers-stream-generator")
        
        raise RuntimeError(f"モデル'{model_path}'のロードに失敗しました。トラブルシューティング情報を確認してください。")
    
    if processor is None:
        warnings.warn(f"プロセッサが見つかりませんでした。基本的な機能しか使えない可能性があります")
        # 最低限のトークナイザーを作成
        try:
            processor = AutoTokenizer.from_pretrained("gpt2")
        except:
            pass
    
    return model, processor

# LLMモデルをロード
def load_llm(model_path, quantize=True):
    """
    言語モデル（LLM）をロードする関数
    """
    # 変数を初期化
    model = None
    processor = None
    
    # GPUメモリをクリア
    gc.collect()
    torch.cuda.empty_cache()
    
    # 量子化設定
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        quantization_config = None
    
    # トークナイザーのロード
    try:
        processor = AutoTokenizer.from_pretrained(model_path)
        print(f"AutoTokenizerで{model_path}のトークナイザーをロードしました")
    except Exception as e:
        print(f"トークナイザーのロードに失敗しました: {e}")
        try:
            processor = AutoProcessor.from_pretrained(model_path)
            print(f"AutoProcessorで{model_path}のプロセッサをロードしました")
        except Exception as e2:
            warnings.warn(f"プロセッサのロードにも失敗しました: {e2}")
            processor = None
    
    # モデルのロード - 複数の方法を試みる
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=quantization_config if quantize else None
        )
        print(f"AutoModelForCausalLMで{model_path}をロードしました")
    except Exception as e:
        print(f"CausalLMとしてのロードに失敗しました: {e}")
        if quantize:
            try:
                print("量子化なしでロードを試みます...")
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                    device_map="auto"
                )
                print(f"量子化なしで{model_path}をロードしました")
            except Exception as e2:
                try:
                    print("AutoModelとして試してみます...")
                    model = AutoModel.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.bfloat16,
                        device_map="auto"
                    )
                    print(f"AutoModelで{model_path}をロードしました")
                except Exception as e3:
                    print(f"すべてのロード方法が失敗しました: {e3}")
                    raise RuntimeError(f"モデル'{model_path}'をロードできません。詳細: {e3}")
    
    # モデルまたはプロセッサがNoneの場合はエラー
    if model is None:
        raise RuntimeError(f"モデル'{model_path}'のロードに失敗しました")
    if processor is None:
        warnings.warn(f"プロセッサが見つかりませんでした。基本的な機能しか使えない可能性があります")
        # 最低限のトークナイザーを作成
        try:
            processor = AutoTokenizer.from_pretrained("gpt2")
        except:
            pass
    
    return model, processor

# Yomitokuモデルをロード
def load_yomitoku(model_path, quantize=False):
    """
    Yomitokuモデルを特別にロードする関数
    Yomitokuは日本語OCRに特化したVLMです
    """
    # 変数を初期化
    model = None
    processor = None
    
    # GPUメモリをクリア
    gc.collect()
    torch.cuda.empty_cache()
    
    # 量子化設定
    if quantize:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    else:
        quantization_config = None
    
    # プロセッサ/トークナイザーのロード
    try:
        processor = AutoProcessor.from_pretrained(model_path)
        print(f"AutoProcessorで{model_path}のプロセッサをロードしました")
    except Exception as e:
        warnings.warn(f"プロセッサのロードに失敗しました: {e}")
        try:
            processor = AutoTokenizer.from_pretrained(model_path)
            print(f"AutoTokenizerで{model_path}のトークナイザーをロードしました")
        except Exception as e2:
            warnings.warn(f"トークナイザーのロードにも失敗しました: {e2}")
            processor = None
    
    # モデルロードのオプション - Yomitoku専用
    model_options = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    
    # safety checkerは常にFalseに
    model_options["requires_safety_checker"] = False
    
    # 量子化設定を追加
    if quantize and quantization_config:
        model_options["quantization_config"] = quantization_config
    
    # モデルのロード - 複数の方法を試みる
    try:
        model = AutoModel.from_pretrained(model_path, **model_options)
        print(f"AutoModelで{model_path}をロードしました")
    except Exception as e:
        print(f"標準ロードに失敗しました: {e}")
        try:
            # safety_checkerパラメータを削除して試してみる
            model_options.pop("requires_safety_checker", None)
            model = AutoModel.from_pretrained(model_path, **model_options)
            print(f"safety_checker無しで{model_path}をロードしました")
        except Exception as e2:
            try:
                # 最後にAutoModelForCausalLMを試す
                model = AutoModelForCausalLM.from_pretrained(model_path, **model_options)
                print(f"AutoModelForCausalLMで{model_path}をロードしました")
            except Exception as e3:
                print(f"すべてのロード方法が失敗しました: {e3}")
                raise RuntimeError(f"Yomitokuモデル'{model_path}'をロードできません。詳細: {e3}")
    
    # モデルまたはプロセッサがNoneの場合はエラー
    if model is None:
        raise RuntimeError(f"モデル'{model_path}'のロードに失敗しました")
    if processor is None:
        warnings.warn(f"プロセッサが見つかりませんでした。基本的な機能しか使えない可能性があります")
        # 最低限のトークナイザーを作成
        try:
            processor = AutoTokenizer.from_pretrained("gpt2")
        except:
            pass
    
    return model, processor

# モデルをアンロードする関数
def unload_model(model):
    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()

# 互換性のために残しておく関数
def get_available_models():
    return {}

# モデル情報を取得する関数を修正
def get_model_config(model_name):
    # VLMモデル用の設定を返す
    vlm_configs = {
        "yomitoku-v1": {
            "name": "Yomitoku-V1 (8B)",
            "path": "rinna/yomitoku-v1-8b",
            "size": "8B",
            "vram_required": 8.0
        },
        "llava": {
            "name": "LLaVA-1.5-7B",
            "path": "llava-hf/llava-1.5-7b-hf",
            "size": "7B",
            "vram_required": 7.0
        },
        "qwen-vl": {
            "name": "Qwen-VL (7B)",
            "path": "Qwen/Qwen-VL-Chat",
            "size": "7B",
            "vram_required": 7.0
        }
    }
    
    return vlm_configs.get(model_name, {"size": "未知", "vram_required": 8.0})

def get_model_config(model_name):
    return {}

def load_model(model_name):
    raise NotImplementedError("この関数は非推奨です。load_vlm()またはload_llm()を使用してください。")

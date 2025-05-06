import torch
from transformers import RTDetrV2ForObjectDetection
import onnx
import onnxruntime
import os
import torch.nn as nn
import argparse

class RTDetrV2Wrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        outputs = self.model(x)
        
        # クラス予測とバウンディングボックスを取得
        logits = outputs.logits
        pred_boxes = outputs.pred_boxes
        
        # クラス予測を確率に変換
        probs = torch.softmax(logits, dim=-1)
        
        # バウンディングボックスを正規化された形式に変換
        # (center_x, center_y, width, height)形式に変換
        boxes = pred_boxes
        
        return probs, boxes

def download_model(model_type='s'):
    if model_type == 's':
        model_file = 'rtdetrv2_r18vd_120e_coco_rerun_48.1.pth'
        if not os.path.exists(model_file):
            print("RT-DETRv2-Sモデルをダウンロード中...")
            model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
            torch.save(model.state_dict(), model_file)
            print("ダウンロードが完了しました。")
        else:
            print("モデルファイルは既に存在します。")
    elif model_type == 'm':
        model_file = 'rtdetrv2_r50vd_m_7x_coco_ema.pth'
        if not os.path.exists(model_file):
            print("RT-DETRv2-Mモデルをダウンロード中...")
            model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
            torch.save(model.state_dict(), model_file)
            print("ダウンロードが完了しました。")
        else:
            print("モデルファイルは既に存在します。")
    else:  # model_type == 'l'
        model_file = 'rtdetrv2_r50vd_6x_coco_ema.pth'
        if not os.path.exists(model_file):
            print("RT-DETRv2-Lモデルをダウンロード中...")
            model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
            torch.save(model.state_dict(), model_file)
            print("ダウンロードが完了しました。")
        else:
            print("モデルファイルは既に存在します。")
    return model_file

def modify_model_for_input_size(model, input_size):
    # 入力サイズを変更
    model.config.image_size = input_size
    model.config.max_size = input_size
    
    # バックボーンの入力サイズを変更
    if hasattr(model, 'backbone'):
        model.backbone.image_size = input_size
    
    return model

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='RT-DETRv2モデルをONNXに変換')
    parser.add_argument('size', nargs='?', type=int, choices=[160, 224, 256, 320, 640], default=640,
                        help='入力画像サイズ (160, 224, 256, 320, or 640)')
    parser.add_argument('model_type', nargs='?', type=str, choices=['s', 'm', 'l'], default='s',
                        help='モデルタイプ (s: small, m: medium, l: large)')
    args = parser.parse_args()
    
    # デバイスをCPUに設定
    device = torch.device("cpu")
    
    # モデルのダウンロード
    model_file = download_model(args.model_type)
    
    # モデルをロード
    if args.model_type == 's':
        model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r18vd")
    else:
        model = RTDetrV2ForObjectDetection.from_pretrained("PekingU/rtdetr_v2_r50vd")
    
    model.load_state_dict(torch.load(model_file, map_location=device), strict=False)
    
    # モデルの入力サイズを変更
    if args.size != 640:
        model = modify_model_for_input_size(model, args.size)
    
    # モデルをラッパーで包む
    wrapped_model = RTDetrV2Wrapper(model)
    
    # モデルをデバイスに移動
    wrapped_model.to(device)
    
    # モデルを評価モードに設定
    wrapped_model.eval()
    
    # ダミー入力の作成
    dummy_input = torch.randn(1, 3, args.size, args.size).to(device)
    
    # 出力ファイル名の設定
    output_file = f'rtdetr_{args.model_type}_{args.size}.onnx'
    
    # ONNXにエクスポート
    torch.onnx.export(
        wrapped_model,          # 変換するモデル
        dummy_input,           # モデルの入力
        output_file,           # 出力ファイル名
        export_params=True,    # モデルの重みも保存
        opset_version=17,      # ONNXのバージョン
        do_constant_folding=True,  # 定数畳み込みの最適化
        input_names=['input'],     # 入力名
        output_names=['class_probs', 'bbox_coords'],  # 出力名
        dynamic_axes={
            'input': {0: 'batch_size'},    # バッチサイズを動的に
            'class_probs': {0: 'batch_size'},
            'bbox_coords': {0: 'batch_size'}
        }
    )
    
    # ONNXモデルの検証
    onnx_model = onnx.load(output_file)
    onnx.checker.check_model(onnx_model)
    
    print(f"ONNXモデルの変換が完了しました。出力ファイル: {output_file}")

if __name__ == '__main__':
    main() 
import onnxruntime as ort
import cv2
import numpy as np
import argparse

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='RT-DETRv2による物体検出デモ')
    parser.add_argument('size', nargs='?', type=int, choices=[160, 224, 256, 320, 640], default=640,
                      help='入力画像サイズ (160, 224, 256, 320, or 640)')
    parser.add_argument('model_type', nargs='?', type=str, choices=['s', 'm', 'l'], default='s',
                      help='モデルタイプ (s: small, m: medium, l: large)')
    parser.add_argument('--image', type=str, default='street.jpg',
                      help='入力画像のパス')
    parser.add_argument('--threshold', type=float, default=0.5,
                      help='検出閾値 (0.0 - 1.0)')
    args = parser.parse_args()

    # モデルの設定
    dnn_height = args.size
    dnn_width = args.size
    conf_threshold = args.threshold

    # ONNXモデルをロード
    model_path = f'rtdetr_{args.model_type}_{args.size}.onnx'
    try:
        ort_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {model_path}")
        print(f"エラー: {e}")
        print(f"先に python convert_to_onnx.py --size {args.size} --model_type {args.model_type} を実行してモデルを生成してください。")
        return

    # 入力画像の読み込みと前処理
    input_image = cv2.imread(args.image)
    if input_image is None:
        print(f"画像の読み込みに失敗しました: {args.image}")
        return

    # 元の画像サイズを保存
    original_height, original_width = input_image.shape[:2]

    # 画像の前処理
    input_image = cv2.resize(input_image, (dnn_width, dnn_height))  # リサイズ
    input_image = np.transpose(input_image, (2, 0, 1))  # HWC -> CHW
    input_image = np.expand_dims(input_image, axis=0).astype(np.float32) / 255.0  # 正規化

    # 推論
    input_name = ort_session.get_inputs()[0].name
    outputs = ort_session.run(None, {input_name: input_image})

    # 出力の取得
    class_probs = outputs[0]  # クラス予測 (1, 300, 80)
    bbox_coords = outputs[1]  # バウンディングボックス座標 (1, 300, 4)

    # クラスラベル（COCOデータセット）
    class_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

    # 検出結果の描画
    img = cv2.imread(args.image)
    img = cv2.resize(img, (dnn_width, dnn_height))

    # 検出された物体の数を追跡
    detected_objects = 0

    for i in range(class_probs.shape[1]):  # 各検出対象物についてループ
        scores = class_probs[0, i]  # クラススコア
        class_id = np.argmax(scores)  # 最も高いスコアのクラスID
        score = scores[class_id]  # 信頼度スコア

        if score > conf_threshold:  # スコア閾値でフィルタリング
            detected_objects += 1
            label = class_names[class_id]
            # バウンディングボックス座標を取得 (center_x, center_y, width, height)
            center_x, center_y, width, height = bbox_coords[0, i]
            
            # 座標をピクセル座標に変換
            x1 = int((center_x - width/2) * dnn_width)
            y1 = int((center_y - height/2) * dnn_height)
            x2 = int((center_x + width/2) * dnn_width)
            y2 = int((center_y + height/2) * dnn_height)

            # バウンディングボックスとラベルを描画
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{label} {score:.2f}"
            cv2.putText(img, label_text, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 結果画像の保存
    output_file = f'output_{args.model_type}_{args.size}_{args.size}.jpg'
    cv2.imwrite(output_file, img)
    print(f"検出結果を {output_file} に保存しました。")
    print(f"検出された物体の数: {detected_objects}")

if __name__ == '__main__':
    main() 
# pathology_classification.py

import os
import glob
import cv2
import numpy as np
import pandas as pd

# ----------------------------------------------------
# 0. 폴더 기본 경로 설정
# ----------------------------------------------------
BASE_DIR = "dataset"

# ----------------------------------------------------
# 1. Feature Extraction
# ----------------------------------------------------
import cv2
import numpy as np

def extract_features(img_bgr):
    """
    입력: BGR 이미지 (cv2.imread 결과)
    출력: dict 형태의 특징 값들
        {
            "tissue_ratio": float,
            "mean_s": float,
            "mean_v": float,
            "morph_ratio": float,
            "dark_ratio": float,
            "bright_ratio": float,
        }
    """

    h_img, w_img = img_bgr.shape[:2]
    total_pixels = h_img * w_img

    # --- 1) BGR → HSV ---
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # --- 2) Tissue mask (조직 픽셀 추출) ---
    #   - 채도(s)가 너무 낮으면 거의 흰 배경
    #   - 밝기(v)가 너무 높으면 배경/글라스 쪽일 확률↑
    #   → 적당한 기준으로 tissue 영역 정의
    tissue_mask = (s > 30) & (v < 240)    # 필요하면 threshold 조정

    tissue_pixels = int(tissue_mask.sum())
    tissue_ratio = tissue_pixels / float(total_pixels)

    # --- 3) mean S, mean V (조직 기준) ---
    if tissue_pixels > 0:
        mean_s = float(s[tissue_mask].mean())  # 0~255
        mean_v = float(v[tissue_mask].mean())  # 0~255
    else:
        # 조직이 거의 없으면 전체 평균으로 fallback
        mean_s = float(s.mean())
        mean_v = float(v.mean())

    # --- 4) Dark / Bright ratio (조직 내 밝기 분포) ---
    # threshold는 대충 예시이므로 example 보고 튜닝하면 됨
    if tissue_pixels > 0:
        v_tissue = v[tissue_mask]

        dark_threshold = 50      # V < 50 이면 매우 어두운 픽셀
        bright_threshold = 210   # V > 210 이면 매우 밝은 픽셀

        dark_ratio = float((v_tissue < dark_threshold).mean())
        bright_ratio = float((v_tissue > bright_threshold).mean())
    else:
        dark_ratio = 0.0
        bright_ratio = 0.0

    # --- 5) Edge 기반 morphology 비율 (morph_ratio) ---
    #   - Gray로 변환 후 Canny edge
    #   - 조직 영역 내에서 edge가 차지하는 비율
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    if tissue_pixels > 0:
        edge_in_tissue = np.logical_and(edges > 0, tissue_mask)
        morph_ratio = float(edge_in_tissue.sum() / tissue_pixels)
    else:
        morph_ratio = 0.0

    features = {
        "tissue_ratio": tissue_ratio,
        "mean_s": mean_s,
        "mean_v": mean_v,
        "morph_ratio": morph_ratio,
        "dark_ratio": dark_ratio,
        "bright_ratio": bright_ratio,
    }

    return features

# ----------------------------------------------------
# 2. Rule-based Classifier
# ----------------------------------------------------
def classify_patch_by_rules(img):
    """
    입력: BGR 이미지 (cv2.imread 결과)
    출력: "Good" 또는 "Ungood"
    """

    f = extract_features(img_rgb)

    tissue_ratio = f["tissue_ratio"]
    mean_s = f["mean_s"]
    mean_v = f["mean_v"]
    morph_ratio = f["morph_ratio"]
    dark_ratio = f["dark_ratio"]
    bright_ratio = f["bright_ratio"]

    # 1) 조직이 너무 적으면 Ungood
    if tissue_ratio < 0.2:
        return "Ungood"
    
    # 2) 조직이 너무 어둡거나 (괴사/덩어리) 너무 밝은 부분이 많으면 Ungood
    if dark_ratio > 0.3 or bright_ratio > 0.3:
        return "Ungood"

    # 3) 색이 날아간 슬라이드라면 Ungood
    if mean_s < 30 or mean_v > 200:
        return "Ungood"

    # 4) 모양이 이상한 조직의 비율이 높으면 Ungodd
    if morph_ratio > 0.4:
        return "Ungood"

    # 위 조건에 해당하지 않으면 Good
    return "Good"

# ===============================================


# ----------------------------------------------------
# 3. example/ 에서 example_label.csv랑 비교 → accuracy 계산
# ----------------------------------------------------
def evaluate_on_example():
    example_dir = os.path.join(BASE_DIR, "example")
    label_csv_path = os.path.join(example_dir, "example_label.csv")

    if not os.path.exists(label_csv_path):
        print(f"[ERROR] example_label.csv 못 찾음: {label_csv_path}")
        return

    df = pd.read_csv(label_csv_path)

    gt_labels = []
    pred_labels = []

    print("=== [Example Set Evaluation] ===")

    for _, row in df.iterrows():
        filename = row["filename"]
        gt = row["label"]  # "Good" / "Ungood"

        img_path = os.path.join(example_dir, filename)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] 이미지 로드 실패: {img_path}")
            pred = "Unknown"
        else:
            pred = classify_patch_by_rules(img)

        gt_labels.append(gt)
        pred_labels.append(pred)

        correct = (gt == pred)
        print(f"{filename:>6} | GT: {gt:7} | Pred: {pred:7} | {'O' if correct else 'X'}")

    # 정확도 계산
    correct_list = [int(g == p) for g, p in zip(gt_labels, pred_labels)]
    acc = np.mean(correct_list) * 100.0

    print("\n=== [Summary] ===")
    print(f"Accuracy on example/: {acc:.2f}%")

# ----------------------------------------------------
# 4. test/ 에 대해 predict.csv 생성
# ----------------------------------------------------
def generate_predict_for_test():
    test_dir = os.path.join(BASE_DIR, "test")
    output_csv_path = os.path.join(test_dir, "predict.csv")

    # t*.png 파일 리스트 정렬해서 가져오기
    img_paths = sorted(glob.glob(os.path.join(test_dir, "t*.png")))

    if len(img_paths) == 0:
        print(f"[WARNING] test/ 폴더에서 t*.png를 찾지 못함: {test_dir}")
        return

    filenames = []
    preds = []

    print("\n=== [Test Set Prediction → predict.csv 생성] ===")

    for img_path in img_paths:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)

        if img is None:
            print(f"[WARNING] 이미지 로드 실패: {img_path}")
            pred = "Good"    # 혹시라도 실패하면 기본값
        else:
            pred = classify_patch_by_rules(img)

        filenames.append(filename)
        preds.append(pred)

        print(f"{filename:>6} | Pred: {pred}")

    # predict.csv 저장
    pred_df = pd.DataFrame({
        "filename": filenames,
        "pred": preds,
    })
    pred_df.to_csv(output_csv_path, index=False)

    print(f"\npredict.csv 저장 완료: {output_csv_path}")


# ----------------------------------------------------
# 5. main
# ----------------------------------------------------
if __name__ == "__main__":
    # 1) example/로 accuracy 확인
    evaluate_on_example()

    # 2) test/에 대한 predict.csv 생성
    generate_predict_for_test()
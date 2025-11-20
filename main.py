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
# 1. Rule-based Classifier
# ----------------------------------------------------
def classify_image(img: np.ndarray) -> str:
    """
    입력: BGR 이미지 (cv2.imread 결과)
    출력: "Good" 또는 "Ungood"
    """

    #TODO: Rule-based 분류기 작성

    return "Good"

# ===============================================


# ----------------------------------------------------
# 2. example/ 에서 example_label.csv랑 비교 → accuracy 계산
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
            pred = classify_image(img)

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
# 3. test/ 에 대해 predict.csv 생성
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
            pred = classify_image(img)

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
# 4. main
# ----------------------------------------------------
if __name__ == "__main__":
    # 1) example/로 accuracy 확인
    evaluate_on_example()

    # 2) test/에 대한 predict.csv 생성
    generate_predict_for_test()
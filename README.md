# DIP_Task2_PSC

Task 2 – Pathology Slide Classification 과제 설명

과제 제출 기한 : 2025년 12월 12일 (금) – 15주차 금요일

**[과제 목표]**

본 과제의 목표는 병리 슬라이드 이미지 패치에서 

"병리학적 이상소견이 관찰되지 않는 조직 슬라이드 (Good/Normal)"와

"병리학적으로 암이 의심되는 조직 슬라이드 (Ungood/Abnormal)"를 
기본 영상처리 기법만 사용하여 구분하는 것입니다.

**[기본 규칙]**

※ 수업에서 배운 기본 영상처리 기법만을 사용해야 합니다.

※ 딥러닝 / 머신러닝 기반 feature extraction 또는 classifier 사용은 금지됩니다.

**[Dataset 제공 방식]**
```
dataset/
 ├─ example/
 │    ├─ e01.png
 │    ├─ e02.png
 │    ├─ e03.png
 │    ├─ ...
 │    ├─ e10.png
 │    └─ example_label.csv
 └─ test/
      ├─ t01.png
      ├─ t02.png
      ├─ t03.png
      ├─ ...
      └─ t10.png
```
- example 폴더: 학습 및 검증에 사용할 수 있는 예시 데이터 (라벨 포함)
- test 폴더: 최종 평가용 데이터 (라벨 제공되지 않음)
- example_label.csv 형식:
    filename,label
    e01.png,good
    e02.png,ungood
    ...

**[제출 파일]**
1. Source Code
   - 구현 언어: Python (기본)

2. Report
   - 사용한 영상처리 기반 feature와 규칙(rule) 설명
   - Good / Ungood 패치 각각 최소 1개 이상을 선택하여 비교 분석
   - 시각화 이미지(전처리 단계, 마스크, edge 등) 포함을 권장

3. Results
   - Test set(t01~t10)에 대한 predict.csv 제출
   - predict.csv 형식은 example_label.csv와 동일하며,
     label 대신 pred 컬럼을 사용합니다.

제출 파일명:
   ‘이름_학번_PSC.zip’

※ 제공된 디렉토리의 파일들은 템플릿이므로, 이름/학번을 채우고 본인 결과물을 채워 넣으면 됩니다.

※ 파일 용량이 너무 큰 경우, Word/한글 문서 대신 PDF로 변환해 제출해 주세요.
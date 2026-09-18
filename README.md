# GNN 기반 조직적 사기 리뷰 탐지

YelpZip 레스토랑 리뷰 데이터에서 **조직적으로 동원된 사기 리뷰**를 그래프 신경망(GNN)으로 탐지하는 프로젝트입니다.
텍스트만으로는 드러나지 않는 구조적 패턴(같은 유저의 동시다발 리뷰, 특정 식당에 단기간 몰리는 리뷰어 집단 등)을 그래프의 연결 구조로 포착합니다.

- **팀**: 경희대학교 CODE 3조 빅크크 (박서현 · 이승아 · 조서영)
- **데이터셋**: YelpZip (608,458건 → 밀도 중심 샘플링 50,950건, 사기 비율 24.8%)
- **평가지표**: PR-AUC, Macro F1

## 최종 성능

| 단계 | 모델 | 그래프 | PR-AUC | Macro F1 |
|---|---|---|---|---|
| 04 | ReviewRGCN (Baseline) | 동질 그래프, 2관계 | 0.6041 | 0.5896 |
| 05_1 | PC-GNN (예선) | 동질 그래프, 4관계 | 0.8634 | 0.7701 |
| 06 | CARE-HAN (GNN 단독) | 이종 그래프(HIN) | 0.8221 | 0.7417 |
| **06** | **CARE-HAN + CatBoost (최종)** | **이종 그래프(HIN)** | **0.9279** | **0.9136** |

수치는 `results_04.json`, `results_05.json`, `results_06.json` 기준입니다.

## 파이프라인

```
원본 데이터 (608,458건)
   │  [01] 라벨 변환(-1/1 → 1/0) + 밀도 중심 샘플링
   ▼
샘플 데이터 (50,950건)
   ├─ [02]   버스트 EDA          → burst_features.csv, burst_months.csv
   ├─ [03_1] 텍스트 네트워크 분석 → custom_edges_textSim.csv, features_text_vocab.csv
   └─ [03_2] 별점 기반 파생 피처  → features_rating.csv
   ▼
[04]   ReviewRGCN (Baseline)
[05_1] PC-GNN (예선)
[06]   CARE-HAN + CatBoost 하이브리드 (최종)
[07]   Gradio 대시보드
```

## 폴더 구조

```
GNN/
├── 01 ~ 07 *.ipynb        # 최종 파이프라인 노트북 (순서대로 실행)
├── common_utils.py        # 공통 함수 (데이터 로드, 버스트 월 계산, 시각화 설정)
├── *.pt, *.cbm            # 학습된 모델
├── *.csv, results_*.json  # 중간 산출물(파생 피처, 커스텀 엣지) 및 성능 결과
├── requirements.txt
├── figures/               # 분석·결과 시각화
├── docs/
│   ├── report/            # 최종 분석 보고서, EDA 보고서, 파생변수, 데이터 누수 처리
│   ├── process/           # 피처 로그, 모델 전환 기록 등 작업 과정 문서
│   ├── presentation/      # 발표 자료 · 보고서 PDF
│   └── team/              # 팀원 정리 문서
└── archive/               # 최종에 쓰이지 않은 이전 버전 (노트북, 모델, 스크립트, 문서)
```

노트북은 같은 폴더의 산출물을 파일 이름으로 읽고 쓰기 때문에, 노트북·모델·중간 산출물은 루트에 둡니다.

## 파일 설명

### 노트북

| 파일 | 내용 | 구분 |
|---|---|---|
| `01_Data_Preprocessing.ipynb` | 라벨 변환, 밀도 중심 샘플링 | 최종 |
| `02_EDA_Burst.ipynb` | 매크로·마이크로·협력 버스트 분석 | 최종 |
| `03_1_Text_Network_Analysis.ipynb` | PMI 단어 네트워크, 텍스트 유사도 커스텀 엣지 | 최종 |
| `03_2_Rating_Fraud_Analysis.ipynb` | 별점 기반 파생 피처 | 최종 |
| `04_Baseline.ipynb` | ReviewRGCN 베이스라인 | 최종 |
| `05_1_PCGNN.ipynb` | PC-GNN | 최종 |
| `06_HIN_CARE_Model.ipynb` | CARE-HAN + CatBoost 하이브리드 | 최종 |
| `07_Dashboard.ipynb` | Gradio 대시보드 | 최종 |

### 모델

| 파일 | 생성 노트북 | 구분 |
|---|---|---|
| `best_hin_care.pt` | 06 | **최종** |
| `best_catboost.cbm` | 06 | **최종** |
| `best_catboost_82.cbm` | 07 대시보드에서 사용 | **최종** |
| `best_pcgnn.pt` | 05_1 | |
| `best_review_rgcn.pt` | 04 | |
| `archive/models/*.pt` | 이전 실험 (DualRelGNN, FiveRelCAREGNN, SixRelCAREGNN 등) | |

## 실행 방법

1. **환경 설치**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cpu
   pip install -r requirements.txt
   ```
2. **데이터 준비**: 원본 데이터는 용량(약 400MB)과 라이선스 문제로 저장소에 포함하지 않았습니다.
   YelpZip 데이터(`text, user_id, prod_id, date, rating, tag, label` 컬럼)를 `yelpzip.csv`로 이 폴더에 넣어 주세요.
3. **노트북 실행**: 이 폴더를 작업 디렉토리로 두고 `01 → 02 → 03_1 → 03_2 → 04 → 05_1 → 06 → 07` 순서로 실행합니다.
   - 04 이후만 다시 돌릴 경우, 01에서 생성되는 `yelpzip_sampled.parquet`이 필요합니다.
   - 02~03의 산출물(csv)은 저장소에 포함되어 있습니다.

## 문서

**보고서**
- [최종 분석 보고서](docs/report/분석보고서_최종.md)
- [모델 성능 정리](docs/report/모델_성능_정리.md)
- [EDA 분석 과정 상세 기록](docs/report/EDA_Report.md)
- [파생변수 정리](docs/report/파생변수.md)
- [데이터 누수 처리 보고서](docs/report/데이터_누수_처리_보고서.md)
- [추가예정 고도화 전략](docs/report/추가예정_고도화_전략.md)

**작업 과정**
- [시행착오 과정 정리](docs/process/시행착오_과정_정리.md)
- [DualRelGNN → FiveRelCAREGNN 전환 기록](docs/process/model_transition_log.md)
- [피처 엔지니어링 로그](docs/process/feature_log.md)

**발표**
- [발표 대본 (최종)](docs/presentation/발표_대본_최종.md)
- [예상 QnA](docs/presentation/예상_QnA.md)
- [발표 자료 PDF](docs/presentation/경희대학교_빅크크_발표_PDF.pdf)

**팀원 정리**
- [전체 정리](docs/team/전체_정리.md) · [RGCN & PC-GNN 모델 정리](docs/team/RGCN_PC-GNN_모델_정리.md)

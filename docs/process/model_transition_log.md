# DualRelGNN에서 FiveRelCAREGNN으로 전환한 시행착오 정리

## 1. 전환 배경

초기 모델은 `DualRelGNN`이었다. 구조는 두 개의 관계 브랜치를 병렬로 학습한 뒤 게이트로 합치는 방식이었다.

| 브랜치 | 사용 모델 | 사용 관계 |
|---|---|---|
| 기본 관계 브랜치 | GraphSAGE 2-layer | `R-U-R` + `R-T-R` 합산 엣지 |
| 커스텀 관계 브랜치 | GAT 2-layer | 텍스트 유사도 KNN 엣지 |
| 병합부 | Gated Fusion | `h_basic`과 `h_custom` 가중합 |

이 방식은 구현이 단순하고 기준 모델로 쓰기 좋았지만, 사기 리뷰 탐지 문제에서는 한계가 있었다.

1. 서로 다른 관계가 하나의 엣지 묶음으로 섞였다.
   - `R-U-R`와 `R-T-R`가 모두 기본 관계로 합쳐져 있어, 모델이 "같은 사용자 패턴"과 "같은 시기 캠페인 패턴"을 분리해서 해석하기 어려웠다.

2. 커스텀 관계가 텍스트 유사도에 과하게 의존했다.
   - 텍스트가 비슷한 리뷰끼리 연결하는 KNN 방식은 유용하지만, TF-IDF/SVD 기반 유사도는 봇성 문장 재사용을 충분히 잡지 못하고 노이즈를 만들 수 있었다.

3. camouflage 문제에 대한 직접 방어가 약했다.
   - 사기 노드는 정상처럼 보이는 리뷰와 일부러 연결될 수 있다.
   - DualRelGNN은 엣지가 들어오면 대체로 메시지를 받아들이는 구조라, 관계별로 "이 이웃을 믿을지 말지"를 조절하는 장치가 부족했다.

4. 관계별 해석력이 부족했다.
   - 최종 결과에서 어떤 관계가 사기 판별에 기여했는지, 어떤 관계에서 노이즈가 많았는지 설명하기 어려웠다.

## 2. 중간 시행착오: 피처 개선만으로는 부족

모델을 바꾸기 전에 피처 쪽에서 여러 개선을 먼저 시도했다. 상세 기록은 `feature_log.md`에 남겨져 있다.

| 단계 | 핵심 변경 | 관찰 |
|---|---|---|
| Baseline | `rating` + 텍스트 임베딩 | 그래프/행동 정보가 부족해 사기 탐지력이 낮음 |
| v1 | 사용자 행동, 버스트, 텍스트 통계 추가 | PR-AUC가 크게 개선됨 |
| v2 | 행동 분산, `rpr_suspicion` 추가 | Macro F1은 개선됐지만 PR-AUC는 일부 하락 |
| v3 초기 | `avg_text_sim`, Focal Loss 추가 | PR-AUC가 하락해 일부 롤백 |
| v3 Fix | 노이즈 피처 제거, CrossEntropyLoss 복귀, trust score 유지 | 최종 입력은 수치 23차원 + 텍스트 64차원 = 87차원 |

이 과정에서 얻은 결론은 다음과 같다.

- `user_burst_max7d`, `user_burst_ratio` 같은 행동 피처는 강한 신호다.
- `upper_ratio`, `excl_count`, `avg_word_len`, `avg_text_sim`은 현재 서브그래프에서는 노이즈 가능성이 컸다.
- 단순히 피처를 늘리는 것보다, 관계 구조 자체를 더 정교하게 분리해야 했다.
- PR-AUC를 우선 지표로 보면, 정상 클래스 F1만 올리는 개선은 충분하지 않았다.

## 3. FiveRelCAREGNN으로 바꾼 핵심 아이디어

전환 후 모델은 관계를 5개로 명시적으로 분리했다.

| 관계 | 의미 | 기대 역할 |
|---|---|---|
| `R-U-R` | 같은 사용자가 작성한 리뷰 연결 | 동일 계정의 반복 사기 패턴 전파 |
| `R-P-R` | 같은 레스토랑 리뷰 연결 | 특정 업장 대상 조직적 홍보/공격 탐지 |
| `R-T-R` | 같은 월 또는 시기 리뷰 연결 | 캠페인성 시간 집중 패턴 탐지 |
| `R-S-R` | 같은 rating 리뷰 연결 | 극단 별점 도배 패턴 탐지 |
| `R-B-R` | burst 상위 노드 연결 | 단기 폭발 작성 계정의 집단 행동 탐지 |

DualRelGNN이 "기본 관계 vs 커스텀 관계"의 2분법이었다면, FiveRelCAREGNN은 사기 리뷰의 행동 가설을 5개 관계로 쪼개서 각각 학습한다.

## 4. CARE 필터를 넣은 이유

FiveRelCAREGNN의 핵심은 `CARERelationLayer`다. 각 관계마다 이웃을 그대로 받지 않고, 유사도와 threshold를 통해 메시지를 부드럽게 걸러낸다.

```text
sim_ij = sigmoid(MLP(|x_i - x_j|))
gate_ij = sigmoid((sim_ij - threshold_relation) / temperature)
message_ij = gate_ij * sqrt(trust_i * trust_j) * h_j
```

이 구조의 의도는 다음과 같다.

1. 관계별 필터링
   - `R-U-R`, `R-P-R`, `R-T-R`, `R-S-R`, `R-B-R`마다 별도 `rel_score` MLP와 threshold를 둔다.
   - 같은 이웃이라도 어떤 관계에서 연결됐는지에 따라 신뢰도가 달라질 수 있다.

2. adaptive threshold
   - 관계별 threshold는 학습 가능한 파라미터다.
   - 모델이 "이 관계는 노이즈가 많다"라고 판단하면 keep ratio를 낮추는 방향으로 조정할 수 있다.

3. trust-aware message
   - `trust_score`가 낮은 노드의 메시지는 `sqrt(trust_i * trust_j)`로 감쇄된다.
   - 저신뢰 노드가 정상 노드처럼 위장해서 주변 표현을 오염시키는 효과를 줄인다.

4. relation attention
   - 5개 관계에서 나온 임베딩을 단순 평균하지 않고 attention으로 병합한다.
   - 최종적으로 어떤 관계가 더 중요했는지 해석할 수 있다.

## 5. 구조 비교

| 항목 | DualRelGNN | FiveRelCAREGNN |
|---|---|---|
| 관계 수 | 2개 묶음 | 5개 명시 관계 |
| 관계 처리 | GraphSAGE/GAT 브랜치 분리 | 관계별 CARE 필터 |
| 이웃 필터링 | GAT attention 중심 | 유사도 gate + adaptive threshold |
| trust 반영 | 제한적 또는 edge weight 중심 | 메시지 가중치에 직접 반영 |
| 병합 방식 | gated fusion | relation attention |
| 해석성 | 두 브랜치 중요도 정도 | 관계별 attention, threshold, keep ratio 확인 가능 |
| 저장 모델 | `best_dualrelgnn.pt` | `best_fiverelcaregnn.pt` |

## 6. 현재 FiveRelCAREGNN 학습 설정

노트북 기준 현재 설정은 다음과 같다.

| 항목 | 값 |
|---|---|
| 입력 차원 | 87차원 |
| 수치 피처 | 23차원 |
| 텍스트 임베딩 | TF-IDF + SVD 64차원 |
| hidden dimension | 256 |
| 관계 수 | 5개 |
| 클래스 수 | 2개, 정상/사기 |
| 클래스 가중치 | 정상 1.0, 사기 약 8.55 |
| optimizer | AdamW |
| learning rate | `8e-4` |
| scheduler | CosineAnnealingLR, `T_max=180` |
| loss | CrossEntropyLoss + `0.08 * CARE threshold aux loss` |
| 파라미터 수 | 1,171,223개 |
| best checkpoint | `best_fiverelcaregnn.pt` |

파라미터는 대부분 관계별 메시지 변환과 2층 CARE 레이어에 사용된다.

| 구성 요소 | 파라미터 수 | 역할 |
|---|---:|---|
| `care1` | 476,810 | 87차원 입력을 256차원으로 변환하고 5개 관계별 1차 집계 |
| `care2` | 628,234 | 1층 임베딩을 다시 5개 관계별로 필터링/집계 |
| `rel_attention` | 33,025 | 5개 관계 임베딩 병합 |
| `classifier` | 33,154 | 정상/사기 분류 |
| 합계 | 1,171,223 | 전체 학습 파라미터 |

## 7. 전환 과정에서의 판단

이번 전환은 단순히 모델을 더 크게 만든 것이 아니라, 사기 탐지 문제의 구조를 모델에 직접 반영한 변경이다.

DualRelGNN에서 확인한 문제는 다음과 같았다.

- 관계가 너무 거칠게 합쳐져 있었다.
- 텍스트 유사도 엣지가 노이즈를 만들 수 있었다.
- 위장 사기 노드에 대한 방어가 약했다.
- 결과 해석이 브랜치 단위에 머물렀다.

FiveRelCAREGNN에서는 이를 다음 방식으로 대응했다.

- 관계를 5개로 분리해 행동 가설을 명시화했다.
- 관계별 유사도 필터와 threshold를 넣어 노이즈 이웃을 줄였다.
- `trust_score`를 메시지 가중치에 직접 넣었다.
- relation attention과 CARE threshold/keep ratio로 해석 가능성을 높였다.

## 8. 남은 확인 포인트

최종 보고서나 발표에 넣기 전에는 다음을 추가 확인하는 것이 좋다.

1. DualRelGNN과 FiveRelCAREGNN의 동일 split 기준 성능 비교
   - Test PR-AUC
   - Test Macro F1
   - 사기 클래스 precision/recall

2. 관계별 해석 결과
   - 평균 relation attention
   - 관계별 threshold
   - 관계별 keep ratio

3. ablation
   - `R-B-R` 제거 시 성능 변화
   - CARE threshold aux loss 제거 시 성능 변화
   - trust weighting 제거 시 성능 변화

4. 과적합 여부
   - 파라미터 수가 약 117만 개로 DualRelGNN보다 커졌으므로 validation PR-AUC와 test PR-AUC 차이를 확인해야 한다.

## 9. 한 줄 요약

DualRelGNN은 두 관계 묶음을 학습하는 기준 모델이었고, FiveRelCAREGNN은 사기 리뷰의 사용자/업장/시간/별점/버스트 관계를 분리한 뒤 CARE 필터와 trust-aware message passing으로 위장 노이즈를 줄이도록 바꾼 모델이다.

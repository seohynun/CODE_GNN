# Image #1 네트워크 추출 방식 설명

이 문서는 `network_analysis.png`에 표시된 리뷰 네트워크가 어떻게 만들어졌는지 정리한 것이다. 해당 이미지는 전체 학습용 GNN 그래프가 아니라, 네트워크 구조를 사람이 보기 쉽게 설명하기 위해 만든 **시각화용 부분 그래프**다.

## 1. 기본 단위

- 노드: 리뷰 1개
- 엣지: 리뷰와 리뷰 사이의 공유 관계
- 라벨: `label=1`은 사기 리뷰, `label=0`은 정상 리뷰
- 시각화 도구: `networkx.Graph()`

즉, 사용자나 식당을 노드로 만든 것이 아니라 **리뷰 자체를 노드로 놓고**, 리뷰끼리 같은 사용자, 같은 시간/식당, 같은 의심 패턴을 공유하면 엣지로 연결했다.

## 2. 시각화용 리뷰 샘플링

먼저 전체 YelpZip 데이터에서 리뷰 수가 많은 상위 5개 식당을 고른다.

```python
top5_prods = (
    df.groupby("prod_id").size()
      .sort_values(ascending=False)
      .head(5)
      .index.tolist()
)

df_top5 = df[df["prod_id"].isin(top5_prods)].copy()
```

그 다음 상위 5개 식당의 리뷰 중에서 사기 리뷰 80개, 정상 리뷰 220개를 샘플링해 총 300개 노드를 만든다.

```python
N_FRAUD_VIS = 80
N_NORMAL_VIS = 220

fraud_pool = df_top5[df_top5["label"] == 1]
normal_pool = df_top5[df_top5["label"] == 0]

sampled_fraud = fraud_pool.sample(min(N_FRAUD_VIS, len(fraud_pool)), random_state=42)
sampled_normal = normal_pool.sample(min(N_NORMAL_VIS, len(normal_pool)), random_state=42)

df_vis = pd.concat([sampled_fraud, sampled_normal]).reset_index(drop=True)
df_vis["node_id"] = range(len(df_vis))
```

이미지 제목의 `nodes=300`은 이 샘플링 결과다. 사기 리뷰 80개와 정상 리뷰 220개를 합쳐 만든 설명용 네트워크다.

## 3. 노드 속성 구성

각 리뷰 노드에는 다음 속성을 저장했다.

| 속성 | 의미 |
|---|---|
| `label` | 실제 라벨. 사기 리뷰면 1, 정상 리뷰면 0 |
| `suspicious` | 텍스트 기반 의심 패턴 여부 |
| `prod_id` | 리뷰가 달린 식당 ID |
| `user_id` | 리뷰 작성자 ID |
| `week_bin` | 리뷰 작성 주 단위 시간 구간 |

```python
G = nx.Graph()

for _, row in df_vis.iterrows():
    G.add_node(
        int(row["node_id"]),
        label=int(row["label"]),
        suspicious=int(row["suspicious"]),
        prod_id=str(row["prod_id"]),
        user_id=str(row["user_id"]),
        week_bin=str(row["date"].to_period("W")),
    )
```

## 4. R-P-R 의심 패턴 정의

이미지에서 노란색 또는 진한 빨간색 노드는 `suspicious=1`인 노드다. 이 값은 텍스트에서 다음 조건을 동시에 만족하면 1로 표시했다.

1. 리뷰 길이가 짧음: 전체 시각화 샘플의 35% 분위수 이하
2. 메뉴 관련 단어가 없음: `menu_cnt == 0`
3. 긍정 단어 비율이 높음: 전체 시각화 샘플의 65% 분위수 이상

```python
p35_tlen = df_vis["tlen"].quantile(0.35)
p65_pos = df_vis["pos_r"].quantile(0.65)

df_vis["suspicious"] = (
    (df_vis["tlen"] <= p35_tlen) &
    (df_vis["menu_cnt"] == 0) &
    (df_vis["pos_r"] >= p65_pos)
).astype(int)
```

이 조건은 "짧고, 구체적인 메뉴 언급은 없는데, 긍정 표현은 강한 리뷰"를 의심 패턴으로 본 것이다. 이미지에서는 이 의심 패턴 노드가 19개였고, 그중 실제 사기 라벨 비율이 47.4%였다.

## 5. 엣지 3종 구성

이미지의 네트워크는 세 종류의 엣지를 합쳐 만든다.

| 엣지 | 의미 | 이미지 색상 |
|---|---|---|
| R-U-R | 같은 사용자가 작성한 리뷰끼리 연결 | 회색 |
| R-T-R | 같은 식당에 같은 주에 작성된 리뷰끼리 연결 | 하늘색 |
| R-P-R | 같은 의심 텍스트 패턴을 가진 리뷰끼리 연결 | 주황색 |

### 5.1 R-U-R: 같은 사용자 리뷰 연결

같은 `user_id`를 가진 리뷰 노드들을 서로 연결한다.

```python
for uid, grp in df_vis.groupby("user_id"):
    ids = grp["node_id"].tolist()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if not G.has_edge(ids[i], ids[j]):
                G.add_edge(ids[i], ids[j], etype="RUR")
```

Image #1에서는 이 관계로 만들어진 엣지가 1개다.

### 5.2 R-T-R: 같은 식당, 같은 주 리뷰 연결

같은 `prod_id`이고 같은 `week_bin`에 속한 리뷰를 연결한다. 단, 한 그룹에 리뷰가 너무 많으면 최대 4개만 뽑아 과도한 연결을 막았다.

```python
MAX_RTR = 4

for (pid, wk), grp in df_vis.groupby(["prod_id", "week_bin"]):
    ids = grp["node_id"].tolist()
    if len(ids) > MAX_RTR:
        ids = list(np.random.default_rng(42).choice(ids, MAX_RTR, replace=False))

    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            if not G.has_edge(ids[i], ids[j]):
                G.add_edge(ids[i], ids[j], etype="RTR")
```

Image #1에서는 이 관계로 만들어진 엣지가 24개다.

### 5.3 R-P-R: 의심 패턴 리뷰 연결

`suspicious=1`인 노드들끼리 연결한다. 단, 한 노드에서 최대 5개까지만 연결하도록 제한했다.

```python
sus_ids = df_vis[df_vis["suspicious"] == 1]["node_id"].tolist()
MAX_RPR = 5

for i in range(len(sus_ids)):
    connected = 0
    for j in range(i + 1, len(sus_ids)):
        if connected >= MAX_RPR:
            break
        if not G.has_edge(sus_ids[i], sus_ids[j]):
            G.add_edge(sus_ids[i], sus_ids[j], etype="RPR")
            connected += 1
```

Image #1에서는 이 관계로 만들어진 엣지가 80개다. 왼쪽 네트워크에서 주황색으로 조밀하게 보이는 부분이 이 R-P-R 관계다.

## 6. 최종 네트워크 규모

Image #1의 최종 그래프는 다음과 같다.

| 항목 | 값 |
|---|---:|
| 노드 수 | 300 |
| 전체 엣지 수 | 105 |
| R-U-R 엣지 | 1 |
| R-T-R 엣지 | 24 |
| R-P-R 엣지 | 80 |
| 평균 degree | 0.70 |

엣지 수 105개는 세 관계를 합친 값이다.

```text
전체 엣지 = R-U-R 1 + R-T-R 24 + R-P-R 80 = 105
```

## 7. 이미지에서 색상과 크기의 의미

노드 색상은 실제 라벨과 의심 패턴 여부를 조합해 정했다.

| 색상 | 의미 |
|---|---|
| 진한 빨강 | 사기 리뷰이면서 의심 패턴 |
| 빨강 | 사기 리뷰 |
| 노랑 | 정상 리뷰이지만 의심 패턴 |
| 파랑 | 정상 리뷰 |

노드 크기는 degree에 비례한다.

```python
node_sizes = [max(degrees[n] * 18 + 30, 40) for n in G.nodes()]
```

연결이 많은 리뷰일수록 더 크게 보인다.

## 8. 그래프 배치

노드 위치는 실제 좌표나 시간순 배치가 아니라 `spring_layout`으로 계산한 힘 기반 배치다.

```python
pos = nx.spring_layout(G, k=1.5, iterations=80, seed=42)
```

따라서 가까이 보이는 노드는 실제 물리적 위치가 가까운 것이 아니라, 그래프 연결 구조상 서로 강하게 묶인 노드라는 뜻이다.

## 9. 오른쪽 분석 차트 계산 방식

### Degree Distribution

각 노드의 연결 수를 구한 뒤, 사기 노드와 정상 노드의 degree 분포를 비교했다.

```python
degrees = dict(G.degree())
deg_fraud = [degrees[n] for n in fraud_nodes]
deg_normal = [degrees[n] for n in normal_nodes]
```

### Homophily

사기-사기 엣지가 전체 엣지 중 얼마나 많은지 계산하고, 무작위 기대값과 비교했다.

```python
ff_edges = sum(
    1 for u, v in G.edges()
    if G.nodes[u]["label"] == 1 and G.nodes[v]["label"] == 1
)

p_fraud = len(fraud_nodes) / G.number_of_nodes()
ff_ratio = ff_edges / total_edges
expected_ff = p_fraud ** 2
homophily = ff_ratio / expected_ff
```

Image #1에서는 실제 사기-사기 엣지 비율이 0.3143이고, 무작위 기대값은 0.0711이다. 따라서 사기 노드끼리 무작위보다 약 4.42배 더 많이 연결된 것으로 해석했다.

### Suspicious Pattern vs True Label

`suspicious` 여부와 실제 `label`을 교차표로 만든 것이다.

```python
pd.crosstab(df_vis["suspicious"], df_vis["label"])
```

Image #1의 값은 다음과 같다.

| 구분 | 정상 | 사기 |
|---|---:|---:|
| Not Suspicious | 210 | 71 |
| Suspicious | 10 | 9 |

따라서 의심 패턴으로 잡힌 리뷰 중 실제 사기 비율은 `9 / (10 + 9) = 47.4%`다.

## 10. 학습용 GNN 그래프와의 차이

Image #1은 설명용 시각화 그래프다. 실제 GNN 학습용 그래프는 더 큰 `df_sub`를 기반으로 만들며, 관계도 더 확장되어 있다.

- 기본 관계: R-U-R, R-T-R
- 커스텀 관계: 텍스트 KNN, 정제된 R-P-R
- 학습 입력: `edge_index_basic`, `edge_index_custom`

즉, Image #1은 전체 모델 구조를 직관적으로 보여주기 위한 축소판이고, 실제 학습 그래프는 더 많은 노드와 엣지를 사용한다.

## 요약

Image #1의 네트워크는 다음 순서로 뽑았다.

1. 전체 리뷰 중 리뷰 수가 많은 상위 5개 식당 선택
2. 그 안에서 사기 리뷰 80개, 정상 리뷰 220개 샘플링
3. 리뷰 1개를 노드 1개로 등록
4. 짧은 글, 메뉴 미언급, 고긍정 표현 조건으로 의심 패턴 노드 표시
5. 같은 사용자 리뷰는 R-U-R로 연결
6. 같은 식당, 같은 주 리뷰는 R-T-R로 연결
7. 의심 패턴 리뷰끼리는 R-P-R로 연결
8. `spring_layout`으로 배치해 `network_analysis.png`로 저장

결과적으로 `nodes=300`, `edges=105`, `R-U-R=1`, `R-T-R=24`, `R-P-R=80`인 리뷰 네트워크가 만들어졌다.

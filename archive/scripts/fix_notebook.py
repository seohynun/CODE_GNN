# -*- coding: utf-8 -*-
import json

nb_path = r"C:\Users\bella\Desktop\대학\CODE\GNN 학술제\05_Model_Training.ipynb"

with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 수정할 패턴: (찾을 문자열, 교체할 문자열)
fixes = [
    # Cell [10]: make_pairs 독스트링
    (
        '"""그룹 내 노드 쌍 생성. max_per_node 지정 시 랜덤 샘플링."""    src_list',
        '"""그룹 내 노드 쌍 생성. max_per_node 지정 시 랜덤 샘플링."""\n    src_list'
    ),
    # Cell [10]: to_edge 독스트링
    (
        '"""numpy 배열 → undirected PyG edge_index."""    if len',
        '"""numpy 배열 → undirected PyG edge_index."""\n    if len'
    ),
    # Cell [11]: CAREConv 클래스 독스트링
    (
        '"""Context-Aware Relation Encoder: 유사도 게이트 + Trust-Aware 메시지."""    def __init__',
        '"""Context-Aware Relation Encoder: 유사도 게이트 + Trust-Aware 메시지."""\n    def __init__'
    ),
    # Cell [11]: SixRelCAREGNN 클래스 독스트링
    (
        '"""6-Relation CARE GNN: 관계별 분리 학습 + 관계 Attention 병합."""    def __init__',
        '"""6-Relation CARE GNN: 관계별 분리 학습 + 관계 Attention 병합."""\n    def __init__'
    ),
    # Cell [11]: BaselineGCN 클래스 독스트링
    (
        '"""베이스라인: 2-layer GCN, R-U-R 엣지만, 65차원 피처."""    def __init__',
        '"""베이스라인: 2-layer GCN, R-U-R 엣지만, 65차원 피처."""\n    def __init__'
    ),
]

total_fixed = 0
for cell in nb["cells"]:
    if cell["cell_type"] != "code":
        continue
    src = "".join(cell["source"])
    original = src
    for old, new in fixes:
        if old in src:
            src = src.replace(old, new)
            print(f"수정됨: {old[:50]}...")
            total_fixed += 1
    if src != original:
        # source를 다시 줄 단위 리스트로 변환
        lines = src.splitlines(keepends=True)
        cell["source"] = lines

print(f"\n총 {total_fixed}개 패턴 수정 완료.")

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("노트북 저장 완료.")

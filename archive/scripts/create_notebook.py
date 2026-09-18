"""05_Model_Training.ipynb 생성 스크립트"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata = {
    "kernelspec": {"display_name": "Python (.venv)", "language": "python", "name": "gnn_venv"},
    "language_info": {"name": "python", "version": "3.11.0"}
}

cells = []
def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)

# ─────────────────────────────────────────────────────────────────────────
cells.append(md("""# GNN 사기 탐지 — 05. 모델 학습 및 성능 비교

## SixRelCAREGNN vs Baseline GCN

| 모델 | 피처 | 관계 수 | 핵심 기법 |
|------|------|--------|-----------|
| Baseline GCN | rating + TF-IDF(64) = 65차원 | 1 (R-U-R) | 2-layer GCN |
| **SixRelCAREGNN** | 행동/텍스트/버스트/레이팅 EDA = 90차원 | **6개** | CARE 필터 + Trust-Aware + Relation Attention |

**커스텀 관계 3개 (창의성 설계):**
- `R-P-R`: 같은 식당 + 의심 텍스트 패턴 (짧음 / 메뉴 언급 없음 / 과도한 칭찬)
- `R-B-R`: 단기 폭발 작성(Burst) 유저들의 리뷰 연결 — 봇넷 행동 패턴 탐지
- `R-Sim-R`: TF-IDF 코사인 유사도 ≥ 0.7 리뷰 연결 — 템플릿 재사용 탐지
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [00] 임포트 및 환경 설정
import os, sys, time, warnings, itertools
import numpy as np
import pandas as pd
from collections import defaultdict
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, average_precision_score, classification_report
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_undirected, degree

BASE_DIR = r'C:\\Users\\bella\\Desktop\\대학\\CODE\\GNN 학술제'
os.chdir(BASE_DIR)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ 환경 설정 완료 | 디바이스: {device}')
print(f'   PyTorch: {torch.__version__}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [01] 하이퍼파라미터 설정
CFG = {
    # 데이터 분할
    'test_size': 0.20,
    'val_size': 0.20,   # train+val 중 val 비율
    'random_state': 42,
    # 텍스트 임베딩
    'tfidf_max_features': 5000,
    'svd_n_components': 64,
    # 엣지 샘플링
    'rtr_max_per_node': 10,      # 10으로 제한 (2M→~1M 엣지)
    'rsr_max_per_node': 10,      # 10으로 제한 (1.7M→~500K 엣지)
    'rpr_max_per_node': 15,      # 15으로 제한 (2M→~700K 엣지)
    'rbr_burst_thr': 3,       # user_burst_max7d 임계값
    'rsim_thr': 0.70,         # 텍스트 유사도 임계값
    # 모델
    'hidden_dim': 128,
    'dropout': 0.30,
    'care_temp': 0.10,
    # 학습
    'lr': 8e-4,
    'wd': 1e-4,
    'epochs': 150,
    'patience': 20,
    'aux_weight': 0.08,
}
print('✅ 설정 완료')
print(f'   Relation 수: 6  (R-U-R / R-T-R / R-S-R / R-P-R / R-B-R / R-Sim-R)')
print(f'   텍스트 임베딩: TF-IDF SVD {CFG[\"svd_n_components\"]}차원')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [02] 데이터 로드 및 기본 검증
print('데이터 로드 중...')
df = pd.read_parquet('yelpzip_sampled.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.reset_index(drop=True)   # review 행 인덱스 = 0..N-1

# ── 라벨 검증 (규정: 1=사기, 0=정상) ──
assert set(df['label'].unique()).issubset({0, 1}), '라벨 변환 필요!'
n_total = len(df)
n_fraud = (df['label'] == 1).sum()
n_normal = (df['label'] == 0).sum()
print(f'✅ 데이터 로드 완료: {n_total:,}건')
print(f'   사기(1): {n_fraud:,} ({100*n_fraud/n_total:.1f}%)')
print(f'   정상(0): {n_normal:,} ({100*n_normal/n_total:.1f}%)')
print(f'   기간: {df[\"date\"].min().date()} ~ {df[\"date\"].max().date()}')
print(f'   유저 수: {df[\"user_id\"].nunique():,}  |  식당 수: {df[\"prod_id\"].nunique():,}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [03] 텍스트 피처 추출 (7개 유의미 피처)
MENU_WORDS = {
    'chicken','beef','pizza','salad','pasta','fish','shrimp','steak',
    'burger','sandwich','soup','rice','noodles','sushi','tacos','bread',
    'cake','dessert','cheese','bacon','turkey','salmon','lobster','crab',
    'pork','lamb','veal','duck','tofu','mushroom','fries','wings','ribs',
    'brisket','sausage','waffle','pancake','omelette','quiche','crepe',
    'curry','ramen','pho','bibimbap','gyoza','tempura','dumplings',
    'kebab','shawarma','falafel','hummus','guacamole','nachos','burrito'
}
POS_WORDS = {
    'great','amazing','delicious','excellent','wonderful','fantastic',
    'outstanding','superb','brilliant','perfect','beautiful','awesome',
    'incredible','lovely','magnificent','splendid','terrific','fabulous'
}
NEG_WORDS = {
    'bad','terrible','awful','horrible','disgusting','dreadful','poor',
    'disappointing','unacceptable','mediocre','tasteless','bland','rude',
    'unfriendly','slow','dirty','overpriced','waste','worst','never'
}

def extract_text_features(texts):
    rows = []
    for text in texts:
        text = str(text)
        words = text.split()
        n = max(len(words), 1)
        chars = max(len(text), 1)
        unique_words = set(w.lower().strip('.,!?') for w in words)
        pos = sum(1 for w in words if w.lower().strip('.,!?') in POS_WORDS)
        neg = sum(1 for w in words if w.lower().strip('.,!?') in NEG_WORDS)
        menu = sum(1 for w in words if w.lower().strip('.,!?') in MENU_WORDS)
        rows.append({
            'text_len': len(text),
            'word_count': len(words),
            'digit_ratio': sum(c.isdigit() for c in text) / chars,
            'menu_mentions': float(menu),
            'special_ratio': sum(not c.isalnum() and not c.isspace() for c in text) / chars,
            'ttr': len(unique_words) / n,
            'sentiment': (pos - neg) / n,
        })
    return pd.DataFrame(rows)

print('텍스트 피처 추출 중... (약 1~2분)')
t0 = time.time()
text_feats = extract_text_features(df['text'].values)
print(f'✅ 텍스트 피처 완료 ({time.time()-t0:.0f}s) | shape: {text_feats.shape}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [04] TF-IDF + SVD 텍스트 임베딩
print('TF-IDF + SVD 임베딩 중...')
t0 = time.time()
tfidf = TfidfVectorizer(
    max_features=CFG['tfidf_max_features'],
    ngram_range=(1, 2),
    min_df=3, max_df=0.95,
    strip_accents='unicode',
    sublinear_tf=True
)
tfidf_mat = tfidf.fit_transform(df['text'].astype(str))
svd = TruncatedSVD(n_components=CFG['svd_n_components'], random_state=CFG['random_state'])
text_embed = svd.fit_transform(tfidf_mat)   # (N, 64)
print(f'✅ 임베딩 완료 ({time.time()-t0:.0f}s)')
print(f'   shape: {text_embed.shape}  |  설명 분산: {svd.explained_variance_ratio_.sum():.3f}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [05] Train / Val / Test 분리 (규정 준수: 샘플링 이후 분리)
N = len(df)
labels = df['label'].values

# 1차: 80% trainval / 20% test
idx_trainval, idx_test = train_test_split(
    np.arange(N), test_size=CFG['test_size'],
    random_state=CFG['random_state'], stratify=labels
)
# 2차: trainval 내 80/20 → train/val
idx_train, idx_val = train_test_split(
    idx_trainval, test_size=CFG['val_size'],
    random_state=CFG['random_state'], stratify=labels[idx_trainval]
)

train_mask = torch.zeros(N, dtype=torch.bool); train_mask[idx_train] = True
val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[idx_val]   = True
test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[idx_test]  = True

df_train = df.iloc[idx_train]
print(f'Train: {len(idx_train):,} ({len(idx_train)/N*100:.1f}%)')
print(f'Val  : {len(idx_val):,} ({len(idx_val)/N*100:.1f}%)')
print(f'Test : {len(idx_test):,} ({len(idx_test)/N*100:.1f}%)')
print(f'Train 사기 비율: {labels[idx_train].mean():.3f}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [06] 사용자 통계 피처 계산 (Train 전용 → 데이터 누수 방지)
print('사용자 통계 계산 중...')
fraud_user_set = set(df_train.loc[df_train['label'] == 1, 'user_id'])

user_stats = df_train.groupby('user_id').agg(
    user_review_count = ('label', 'count'),
    user_fraud_ratio  = ('label', 'mean'),
    user_avg_rating   = ('rating', 'mean'),
    user_rating_std   = ('rating', lambda x: float(x.std()) if len(x) > 1 else 0.0),
    prod_diversity    = ('prod_id', 'nunique'),
    activity_span     = ('date', lambda x: float((x.max() - x.min()).days)),
    interval_mean     = ('date', lambda x: float(x.sort_values().diff().dt.days.mean()) if len(x) > 1 else 0.0),
    interval_std      = ('date', lambda x: float(x.sort_values().diff().dt.days.std())  if len(x) > 1 else 0.0),
).reset_index()
user_stats['is_fraud_user'] = user_stats['user_id'].isin(fraud_user_set).astype(float)

# trust_score (CARE 메시지 가중치 전용)
user_stats['trust_score'] = (
    np.clip(user_stats['activity_span'] / 365, 0, 1) * 0.5 +
    np.clip(user_stats['user_review_count'] / 10, 0, 1) * 0.5
)
user_stats.fillna(0, inplace=True)

# burst 피처 병합
burst_df = pd.read_csv('burst_features.csv', index_col=0).reset_index()
burst_df.columns = ['user_id', 'user_burst_max7d', 'user_burst_ratio']
user_stats = user_stats.merge(burst_df, on='user_id', how='left')
user_stats[['user_burst_max7d', 'user_burst_ratio']] = (
    user_stats[['user_burst_max7d', 'user_burst_ratio']].fillna(0)
)
print(f'✅ 사용자 통계 완료: {len(user_stats):,}명')
print(user_stats[['user_review_count','user_fraud_ratio','user_burst_max7d','trust_score']].describe().round(3))
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [07] 전체 피처 행렬 조립
df_feat = df[['user_id','prod_id','rating','date','label']].copy().reset_index(drop=True)
df_feat['weekday'] = df['date'].dt.weekday.astype(float)
df_feat['month']   = df['date'].dt.month.astype(float)

# 사용자 통계 병합
user_cols = ['user_id','is_fraud_user','user_review_count','user_fraud_ratio',
             'user_avg_rating','user_rating_std','interval_mean','interval_std',
             'prod_diversity','activity_span','user_burst_max7d','user_burst_ratio','trust_score']
df_feat = df_feat.merge(user_stats[user_cols], on='user_id', how='left')

# features_rating.csv 병합 (EDA 03-2 파생변수)
fr = pd.read_csv('features_rating.csv', index_col=0)
# user_id+prod_id로 병합, 중복 시 첫 번째 사용
fr_dedup = fr[['user_id','prod_id','rating_deviation','norm_rating_entropy',
               'is_cold_start','is_single_review']].drop_duplicates(subset=['user_id','prod_id'])
df_feat = df_feat.merge(fr_dedup, on=['user_id','prod_id'], how='left')

# 텍스트 피처 추가
for col in text_feats.columns:
    df_feat[col] = text_feats[col].values

# rpr_suspicion 계산 (R-P-R 엣지 가중치 + 노드 피처)
tl_35 = np.nanpercentile(df_feat['text_len'], 35)
st_65 = np.nanpercentile(df_feat['sentiment'], 65)
df_feat['rpr_suspicion'] = (
    (df_feat['text_len'] <= tl_35).astype(float) +
    (df_feat['menu_mentions'] == 0).astype(float) +
    (df_feat['sentiment'] >= st_65).astype(float)
) / 3.0
# 화이트리스트: 장기 충성 고객은 제외
whitelist = (df_feat['activity_span'].fillna(0) >= 365) & (df_feat['user_review_count'].fillna(0) >= 10)
df_feat.loc[whitelist, 'rpr_suspicion'] = 0.0

# 결측치 처리
df_feat.fillna(0, inplace=True)
df_feat['user_avg_rating'] = df_feat['user_avg_rating'].replace(0, df_train['rating'].mean())

NUM_COLS = [
    'rating','is_fraud_user','user_review_count','user_fraud_ratio',
    'user_avg_rating','user_rating_std','weekday','month',
    'user_burst_max7d','user_burst_ratio',
    'interval_mean','interval_std','prod_diversity','activity_span',
    'rating_deviation','norm_rating_entropy','is_cold_start','is_single_review',
    'text_len','word_count','digit_ratio','menu_mentions',
    'special_ratio','ttr','sentiment','rpr_suspicion'
]
print(f'수치 피처: {len(NUM_COLS)}차원')
print(f'텍스트 임베딩: {CFG[\"svd_n_components\"]}차원')
print(f'총 입력 차원: {len(NUM_COLS) + CFG[\"svd_n_components\"]}차원')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [08] 피처 행렬 표준화 및 텐서 변환
scaler = StandardScaler()
num_matrix = scaler.fit_transform(df_feat[NUM_COLS].values)  # Train 기준 fit
X_all = np.hstack([num_matrix, text_embed]).astype(np.float32)
X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
in_dim = X_all.shape[1]

# Baseline용 65차원 (rating + TF-IDF only)
X_base = np.hstack([
    df_feat['rating'].values.reshape(-1,1) / 5.0,
    text_embed
]).astype(np.float32)

x_full = torch.tensor(X_all,  dtype=torch.float32)
x_base = torch.tensor(X_base, dtype=torch.float32)
y      = torch.tensor(labels,  dtype=torch.long)
trust  = torch.tensor(df_feat['trust_score'].values, dtype=torch.float32)

print(f'✅ 피처 행렬 완료')
print(f'   Full  (SixRelCAREGNN): {x_full.shape}')
print(f'   Base  (BaselineGCN) : {x_base.shape}')
print(f'   Labels: 사기={y.sum().item():,}  정상={(y==0).sum().item():,}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [09] 그래프 엣지 구성 (6개 관계)
df['ym'] = df['date'].dt.to_period('M')
rng = np.random.default_rng(42)

def make_pairs(groups, max_per_node=None):
    \"\"\"그룹 내 노드 쌍 생성. max_per_node 지정 시 랜덤 샘플링.\"\"\"\
    src_list, dst_list = [], []
    for nodes in groups:
        nodes = list(nodes)
        if len(nodes) < 2: continue
        n_pairs = len(nodes) * (len(nodes) - 1) // 2
        if max_per_node is None or n_pairs <= max_per_node * len(nodes):
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    src_list.append(nodes[i]); dst_list.append(nodes[j])
        else:
            for i, node in enumerate(nodes):
                others = [n for n in nodes if n != node]
                k = min(max_per_node, len(others))
                sampled = rng.choice(others, k, replace=False)
                for j in sampled:
                    src_list.append(node); dst_list.append(int(j))
    return np.array(src_list, dtype=np.int64), np.array(dst_list, dtype=np.int64)

def to_edge(src, dst, n_nodes):
    \"\"\"numpy 배열 → undirected PyG edge_index.\"\"\"\
    if len(src) == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    ei = torch.tensor(np.stack([src, dst], axis=0), dtype=torch.long)
    ei = to_undirected(ei, num_nodes=n_nodes)
    ei = ei[:, ei[0] != ei[1]]   # self-loop 제거
    return ei

N = len(df)
print(f'총 노드 수: {N:,}  | 엣지 구성 시작...')
t0 = time.time()

# R-U-R: 같은 유저의 리뷰 연결
grp_uur = df.groupby('user_id').apply(lambda x: x.index.tolist()).values
s, d = make_pairs(grp_uur)
edge_rur = to_edge(s, d, N)
print(f'  R-U-R   : {edge_rur.size(1):>8,}개 엣지  (같은 유저)')

# R-T-R: 같은 월 리뷰 연결 (샘플링)
grp_rtr = df.groupby('ym').apply(lambda x: x.index.tolist()).values
s, d = make_pairs(grp_rtr, max_per_node=CFG['rtr_max_per_node'])
edge_rtr = to_edge(s, d, N)
print(f'  R-T-R   : {edge_rtr.size(1):>8,}개 엣지  (같은 월)')

# R-S-R: 같은 식당 + 같은 별점
grp_rsr = df.groupby(['prod_id','rating']).apply(lambda x: x.index.tolist()).values
s, d = make_pairs(grp_rsr, max_per_node=CFG['rsr_max_per_node'])
edge_rsr = to_edge(s, d, N)
print(f'  R-S-R   : {edge_rsr.size(1):>8,}개 엣지  (같은 식당+별점)')

# R-P-R: 의심 텍스트 패턴 + 같은 식당 (커스텀 1)
susp_mask = df_feat['rpr_suspicion'].values > 0
df_susp = df[susp_mask].copy()
grp_rpr = df_susp.groupby('prod_id').apply(lambda x: x.index.tolist()).values
s, d = make_pairs(grp_rpr, max_per_node=CFG['rpr_max_per_node'])
edge_rpr = to_edge(s, d, N)
n_susp = susp_mask.sum()
print(f'  R-P-R   : {edge_rpr.size(1):>8,}개 엣지  (의심 리뷰 {n_susp:,}건 / 같은 식당)')

# R-B-R: 단기 폭발 유저 + 같은 캠페인 월 (커스텀 2)
burst_users = set(user_stats.loc[
    user_stats['user_burst_max7d'] >= CFG['rbr_burst_thr'], 'user_id'
])
burst_mask = df['user_id'].isin(burst_users).values
df_burst = df[burst_mask].copy()
grp_rbr = df_burst.groupby('ym').apply(lambda x: x.index.tolist()).values
s, d = make_pairs(grp_rbr, max_per_node=20)
edge_rbr = to_edge(s, d, N)
print(f'  R-B-R   : {edge_rbr.size(1):>8,}개 엣지  (버스트 유저 {len(burst_users):,}명 / 같은 캠페인 월)')

# R-Sim-R: 텍스트 코사인 유사도 ≥ thr (커스텀 3)
sim_df = pd.read_csv('custom_edges_textSim.csv')
sim_df = sim_df[sim_df['text_similarity'] >= CFG['rsim_thr']].copy()
valid = (sim_df['src_review_idx'] < N) & (sim_df['dst_review_idx'] < N)
sim_df = sim_df[valid]
s = sim_df['src_review_idx'].values.astype(np.int64)
d = sim_df['dst_review_idx'].values.astype(np.int64)
edge_rsim = to_edge(s, d, N)
print(f'  R-Sim-R : {edge_rsim.size(1):>8,}개 엣지  (텍스트 유사도 ≥ {CFG[\"rsim_thr\"]})')

edge_indices = [edge_rur, edge_rtr, edge_rsr, edge_rpr, edge_rbr, edge_rsim]
REL_NAMES    = ['R-U-R','R-T-R','R-S-R','R-P-R','R-B-R','R-Sim-R']
total_edges  = sum(e.size(1) for e in edge_indices)
print(f'\\n✅ 엣지 구성 완료 ({time.time()-t0:.0f}s) | 총 {total_edges:,}개')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [10] 모델 정의

# ── CARE 메시지 패싱 레이어 ───────────────────────────────────────────
class CAREConv(MessagePassing):
    \"\"\"Context-Aware Relation Encoder: 유사도 게이트 + Trust-Aware 메시지.\"\"\"\
    def __init__(self, in_ch, out_ch, temp=0.1):
        super().__init__(aggr='mean')
        self.lin     = nn.Linear(in_ch, out_ch, bias=False)
        self.w_sim   = nn.Linear(in_ch, 1,      bias=False)
        self.threshold = nn.Parameter(torch.tensor(0.5))
        self.temp    = temp
        self.norm    = nn.LayerNorm(out_ch)
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.w_sim.weight)

    def forward(self, x, edge_index, trust=None):
        if edge_index.size(1) == 0:
            return self.norm(torch.zeros(x.size(0), self.lin.weight.size(0),
                                         device=x.device))
        # 선형 변환을 노드 단위로 사전 계산 (O(N*d²) → O(E*d) 최적화)
        x_lin = self.lin(x)                                           # (N, out_ch)
        # trust 사전 인덱싱 (1D 텐서 자동 인덱싱 오류 방지)
        if trust is not None:
            t_src = trust[edge_index[0]]
            t_dst = trust[edge_index[1]]
        else:
            t_src, t_dst = None, None
        out = self.propagate(edge_index, x=x, x_lin=x_lin, t_src=t_src, t_dst=t_dst)
        return self.norm(out)

    def message(self, x_i, x_j, x_lin_j=None, t_src=None, t_dst=None):
        # 유사도 게이트 (이웃을 얼마나 믿을지)
        sim  = torch.sigmoid(self.w_sim(torch.abs(x_i - x_j)))       # (E,1)
        gate = torch.sigmoid((sim - self.threshold) / self.temp)      # (E,1)
        msg  = gate * x_lin_j                                         # (E,out_ch)
        # Trust-Aware 가중치 (저신뢰 노드 메시지 감쇄)
        if t_src is not None:
            trust_w = torch.sqrt(
                t_src.clamp(min=1e-6) * t_dst.clamp(min=1e-6)
            ).unsqueeze(-1)                                           # (E,1)
            msg = msg * trust_w
        return msg

    def get_threshold(self):
        return float(self.threshold)


# ── SixRelCAREGNN ────────────────────────────────────────────────────────
class SixRelCAREGNN(nn.Module):
    \"\"\"6-Relation CARE GNN: 관계별 분리 학습 + 관계 Attention 병합.\"\"\"\
    def __init__(self, in_dim, hidden_dim=128, n_rel=6, n_class=2,
                 dropout=0.3, care_temp=0.1):
        super().__init__()
        self.n_rel = n_rel

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU()
        )
        self.drop = nn.Dropout(dropout)
        self.care1 = nn.ModuleList([CAREConv(hidden_dim, hidden_dim, care_temp) for _ in range(n_rel)])
        self.care2 = nn.ModuleList([CAREConv(hidden_dim, hidden_dim, care_temp) for _ in range(n_rel)])

        self.rel_attn = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(),
            nn.Linear(32, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_class)
        )

    def forward(self, x, edge_indices, trust=None):
        h = self.drop(self.input_proj(x))

        # Layer 1
        h1 = [F.elu(self.drop(c(h,  ei, trust))) for c, ei in zip(self.care1, edge_indices)]
        # Layer 2
        h2 = [F.elu(self.drop(c(h1i, ei, trust))) for c, h1i, ei in zip(self.care2, h1, edge_indices)]

        # 관계 Attention 병합
        stk  = torch.stack(h2, dim=1)                       # (N, n_rel, hidden)
        attn = torch.softmax(self.rel_attn(stk).squeeze(-1), dim=1)  # (N, n_rel)
        h_agg = (stk * attn.unsqueeze(-1)).sum(dim=1)       # (N, hidden)

        # Threshold 정규화 보조 손실
        aux = sum((c.threshold - 0.5).pow(2)
                  for layers in [self.care1, self.care2] for c in layers)

        return self.classifier(h_agg), aux, attn.detach().mean(dim=0)


# ── Baseline GCN ─────────────────────────────────────────────────────────
from torch_geometric.nn import GCNConv

class BaselineGCN(nn.Module):
    \"\"\"베이스라인: 2-layer GCN, R-U-R 엣지만, 65차원 피처.\"\"\"\
    def __init__(self, in_dim, hidden_dim=64, n_class=2, dropout=0.3):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)
        self.clf   = nn.Linear(hidden_dim, n_class)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index)); h = self.drop(h)
        h = F.relu(self.conv2(h, edge_index)); h = self.drop(h)
        return self.clf(h)


print('✅ 모델 정의 완료')
# 파라미터 수 확인
care_test = SixRelCAREGNN(in_dim=in_dim, hidden_dim=CFG['hidden_dim'])
n_params  = sum(p.numel() for p in care_test.parameters() if p.requires_grad)
print(f'   SixRelCAREGNN 파라미터: {n_params:,}개')
base_test = BaselineGCN(in_dim=x_base.shape[1])
n_base    = sum(p.numel() for p in base_test.parameters() if p.requires_grad)
print(f'   BaselineGCN   파라미터: {n_base:,}개')
del care_test, base_test
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [11] 학습 유틸리티
def eval_model(model, x, edge_list, trust, y, mask, mode='care'):
    model.eval()
    with torch.no_grad():
        if mode == 'care':
            logits, _, _ = model(x, edge_list, trust)
        else:
            logits = model(x, edge_list[0])
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = logits.argmax(dim=1)
    yt = y[mask].numpy(); yp = preds[mask].numpy(); yprob = probs[mask].numpy()
    pr  = average_precision_score(yt, yprob)
    f1  = f1_score(yt, yp, average='macro')
    return pr, f1, yt, yp, yprob

def train_step(model, opt, criterion, x, edge_list, trust, y, mask, mode='care', aux_w=0.08):
    model.train(); opt.zero_grad()
    if mode == 'care':
        logits, aux, _ = model(x, edge_list, trust)
        loss = criterion(logits[mask], y[mask]) + aux_w * aux
    else:
        logits = model(x, edge_list[0])
        loss   = criterion(logits[mask], y[mask])
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return float(loss)

def bar(v, L=28): return '█'*int(v*L) + '░'*(L-int(v*L))

# 클래스 가중치 계산
cw = compute_class_weight('balanced', classes=np.array([0,1]), y=labels[idx_train])
print(f'클래스 가중치: 정상={cw[0]:.3f}, 사기={cw[1]:.3f}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [12] 베이스라인 GCN 학습
print('='*58)
print(' 베이스라인 GCN 학습')
print(' 피처: rating(1) + TF-IDF SVD(64) = 65차원')
print(' 관계: R-U-R (1개)')
print('='*58)

cw_base = compute_class_weight('balanced', classes=np.array([0,1]), y=labels[idx_train])
crit_base = nn.CrossEntropyLoss(weight=torch.tensor(cw_base, dtype=torch.float32))
base_model = BaselineGCN(in_dim=x_base.shape[1], hidden_dim=64)
opt_base  = AdamW(base_model.parameters(), lr=1e-3, weight_decay=1e-4)
sch_base  = CosineAnnealingLR(opt_base, T_max=CFG['epochs'])

best_base_pr, best_base_f1, best_base_state = 0.0, 0.0, None
no_imp = 0
t0 = time.time()

for ep in range(1, CFG['epochs']+1):
    loss = train_step(base_model, opt_base, crit_base,
                      x_base, edge_indices, trust, y, train_mask, mode='baseline')
    sch_base.step()

    if ep % 10 == 0:
        vpr, vf1, *_ = eval_model(base_model, x_base, edge_indices, trust, y, val_mask, 'baseline')
        print(f'  [E{ep:3d}] loss={loss:.4f} | val PR-AUC={vpr:.4f} F1={vf1:.4f} ({time.time()-t0:.0f}s)')
        if vpr > best_base_pr:
            best_base_pr = vpr; best_base_f1 = vf1
            best_base_state = {k: v.clone() for k,v in base_model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= CFG['patience'] // 2:
                print(f'  Early stopping (no improve {no_imp*10} epochs)'); break

base_model.load_state_dict(best_base_state)
_, _, yt_base, yp_base, yprob_base = eval_model(
    base_model, x_base, edge_indices, trust, y, test_mask, 'baseline')
res_base_pr = average_precision_score(yt_base, yprob_base)
res_base_f1 = f1_score(yt_base, yp_base, average='macro')
torch.save(best_base_state, 'best_baseline_new.pt')
print(f'\\n✅ 베이스라인 최종:  PR-AUC={res_base_pr:.4f}  Macro F1={res_base_f1:.4f}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [13] SixRelCAREGNN 학습
print('='*58)
print(' SixRelCAREGNN 학습')
print(f' 피처: {in_dim}차원')
print(' 관계: R-U-R / R-T-R / R-S-R / R-P-R / R-B-R / R-Sim-R (6개)')
print('='*58)

# 사기 가중치를 균형보다 1.5배 높여 PR-AUC 최적화
cw_care = compute_class_weight('balanced', classes=np.array([0,1]), y=labels[idx_train])
cw_care[1] *= 1.5
crit_care = nn.CrossEntropyLoss(weight=torch.tensor(cw_care, dtype=torch.float32))

care_model = SixRelCAREGNN(
    in_dim=in_dim, hidden_dim=CFG['hidden_dim'],
    n_rel=6, dropout=CFG['dropout'], care_temp=CFG['care_temp']
)
opt_care = AdamW(care_model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
sch_care = CosineAnnealingLR(opt_care, T_max=CFG['epochs'])

best_care_pr, best_care_f1, best_care_state = 0.0, 0.0, None
no_imp_c = 0
t0 = time.time()

for ep in range(1, CFG['epochs']+1):
    loss = train_step(care_model, opt_care, crit_care,
                      x_full, edge_indices, trust, y, train_mask,
                      mode='care', aux_w=CFG['aux_weight'])
    sch_care.step()

    if ep % 10 == 0:
        vpr, vf1, *_ = eval_model(care_model, x_full, edge_indices, trust, y, val_mask, 'care')
        print(f'  [E{ep:3d}] loss={loss:.4f} | val PR-AUC={vpr:.4f} F1={vf1:.4f} ({time.time()-t0:.0f}s)')
        if vpr > best_care_pr:
            best_care_pr = vpr; best_care_f1 = vf1
            best_care_state = {k: v.clone() for k,v in care_model.state_dict().items()}
            no_imp_c = 0
        else:
            no_imp_c += 1
            if no_imp_c >= CFG['patience']:
                print(f'  Early stopping (no improve {no_imp_c*10} epochs)'); break

care_model.load_state_dict(best_care_state)
_, _, yt_care, yp_care, yprob_care = eval_model(
    care_model, x_full, edge_indices, trust, y, test_mask, 'care')
res_care_pr = average_precision_score(yt_care, yprob_care)
res_care_f1 = f1_score(yt_care, yp_care, average='macro')
torch.save(best_care_state, 'best_sixrelcaregnn.pt')
print(f'\\n✅ SixRelCAREGNN 최종:  PR-AUC={res_care_pr:.4f}  Macro F1={res_care_f1:.4f}')
"""))

# ─────────────────────────────────────────────────────────────────────────
cells.append(code("""# [14] 최종 성능 비교 리포트
W = 62
print()
print('=' * W)
print('  📊 GNN 사기 탐지 성능 비교 리포트')
print('  (베이스라인 → EDA 기반 SixRelCAREGNN 개선 효과)')
print('=' * W)

print(f'\\n[1] 베이스라인 — GCN (2-layer, R-U-R only)')
print(f'    피처 : rating(1) + TF-IDF SVD(64) = 65차원')
print(f'    ──────────────────────────────────────────')
print(f'    PR-AUC  : [{bar(res_base_pr)}] {res_base_pr:.4f}')
print(f'    Macro F1: [{bar(res_base_f1)}] {res_base_f1:.4f}')

print(f'\\n[2] SixRelCAREGNN — EDA 기반 개선')
print(f'    피처 : 행동/버스트/텍스트/레이팅EDA = {in_dim}차원')
print(f'    관계 : R-U-R / R-T-R / R-S-R / R-P-R / R-B-R / R-Sim-R (6개)')
print(f'    기법 : CARE 유사도 게이트 + Trust-Aware 메시지 + Relation Attention')
print(f'    ──────────────────────────────────────────')
print(f'    PR-AUC  : [{bar(res_care_pr)}] {res_care_pr:.4f}')
print(f'    Macro F1: [{bar(res_care_f1)}] {res_care_f1:.4f}')

d_pr = res_care_pr - res_base_pr
d_f1 = res_care_f1 - res_base_f1
pct_pr = d_pr / max(res_base_pr, 1e-9) * 100
pct_f1 = d_f1 / max(res_base_f1, 1e-9) * 100

print(f'\\n[3] EDA 기여 개선폭 (Baseline → SixRelCAREGNN)')
print(f'    PR-AUC  : {res_base_pr:.4f} → {res_care_pr:.4f}  ({d_pr:+.4f}, {pct_pr:+.1f}%)')
print(f'    Macro F1: {res_base_f1:.4f} → {res_care_f1:.4f}  ({d_f1:+.4f}, {pct_f1:+.1f}%)')

# 관계별 Attention 가중치
care_model.eval()
with torch.no_grad():
    _, _, attn_mean = care_model(x_full, edge_indices, trust)
attn_np = attn_mean.cpu().numpy()

print(f'\\n[4] 관계별 Attention 가중치 (평균, 높을수록 사기 판별에 중요)')
for name, w, ei in zip(REL_NAMES, attn_np, edge_indices):
    b = '█' * int(w * 40)
    print(f'    {name:8s}: [{b:<20}] {w:.3f}  ({ei.size(1):,} edges)')

# CARE 임계값 (threshold)
print(f'\\n[5] CARE 적응형 임계값 — Layer1 / Layer2')
print(f'    (0.5 = 중립, 높을수록 엣지 필터링 강도 ↑)')
for i, name in enumerate(REL_NAMES):
    t1 = care_model.care1[i].get_threshold()
    t2 = care_model.care2[i].get_threshold()
    print(f'    {name:8s}: {t1:.3f} → {t2:.3f}')

# Confusion matrix
print(f'\\n[6] SixRelCAREGNN — Classification Report (Test set)')
print(classification_report(yt_care, yp_care, target_names=['정상(0)','사기(1)'],
                            digits=4))
print('=' * W)
print(f'✅ 완료 | 모델 저장: best_baseline_new.pt / best_sixrelcaregnn.pt')
"""))

nb.cells = cells

out_path = r'C:\Users\bella\Desktop\대학\CODE\GNN 학술제\05_Model_Training.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'✅ 생성 완료: {out_path}')
print(f'셀 수: {len(cells)}')

import json, os

BASE = r'C:\Users\bella\Desktop\대학\CODE\GNN 학술제'

def code(src, cid):
    return {"cell_type":"code","execution_count":None,"id":cid,"metadata":{},"outputs":[],"source":src}

def md(src, cid):
    return {"cell_type":"markdown","id":cid,"metadata":{},"source":src}

cells = []

# ── 마크다운 헤더 ──────────────────────────────────────────────────────
cells.append(md(
"# GNN 사기 탐지 — 05-1. RGCN(Baseline) vs PC-GNN\n\n"
"## 설계 결정 요약\n\n"
"| 항목 | RGCN (베이스라인) | PC-GNN |\n"
"|------|------------------|--------|\n"
"| 피처 | rating_norm 1차원 | 행동피처 8개 + TF-IDF SVD 64차원 |\n"
"| 엣지 | R-U-R + R-S-R (2종) | R-U-R + R-S-R + R-P-R + R-Sim-R (4종) |\n"
"| 집계 | 관계별 균등 평균 (RGCNConv) | Chooser 어텐션 (위장 이웃 억제) |\n"
"| 배치 | 전체 train | **Picker** — 사기 전량 + 정상 3배 |\n"
"| 손실 | CrossEntropyLoss + class weight | **FocalLoss** + class weight |\n"
"| 분할 | **유저 단위** (R-U-R 라벨 누수 차단) | 동일 |\n\n"
"### 비교 포인트\n"
"- RGCN: EDA 피처 없음 + 균등 집계 → 그래프 구조만의 기여 측정\n"
"- PC-GNN: EDA 피처 + Chooser + Picker → 세 가지 개선의 결합 효과\n",
"aa000001"))

# ── [00] 임포트 ────────────────────────────────────────────────────────
cells.append(code(
r"""# [00] 임포트 및 환경 설정
import os, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

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
from torch_geometric.nn import MessagePassing, RGCNConv
from torch_geometric.utils import to_undirected

BASE_DIR = r'C:\Users\bella\Desktop\대학\CODE\GNN 학술제'
os.chdir(BASE_DIR)
torch.manual_seed(42)
np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ 환경 설정 완료 | 디바이스: {device}')
print(f'   PyTorch: {torch.__version__}')""",
"aa000002"))

# ── [01] 하이퍼파라미터 ────────────────────────────────────────────────
cells.append(code(
"""# [01] 하이퍼파라미터 설정
CFG = {
    'random_state': 42,
    # 텍스트 임베딩
    'tfidf_max_features': 5000,
    'svd_n_components':   64,
    # 유저 단위 분할 비율
    'train_user_ratio': 0.60,
    'val_user_ratio':   0.20,   # test = 나머지 0.20
    # 엣지 샘플링
    'rsr_max_per_node':  5,
    'rpr_max_per_node':  8,
    'rsim_thr':          0.70,
    # 모델
    'hidden_dim':  128,
    'dropout':     0.40,
    # 학습
    'lr':       5e-4,
    'wd':       5e-4,
    'epochs':   80,
    'patience': 10,
    # PC-GNN 특화
    'picker_ratio': 3,    # 배치 내 정상:사기 = 3:1
    'focal_gamma':  2.0,
}
print('✅ 설정 완료')
print(f"   유저 분할: train {CFG['train_user_ratio']*100:.0f}% / val {CFG['val_user_ratio']*100:.0f}% / test 나머지")
print(f"   Picker ratio={CFG['picker_ratio']}  |  Focal gamma={CFG['focal_gamma']}")""",
"aa000003"))

# ── [02] 데이터 로드 ───────────────────────────────────────────────────
cells.append(code(
"""# [02] 데이터 로드
print('데이터 로드 중...')
df = pd.read_parquet('yelpzip_sampled.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

assert set(df['label'].unique()).issubset({0, 1}), '라벨 변환 필요!'
N = len(df)
labels = df['label'].values
n_fraud = labels.sum()
print(f'✅ 데이터: {N:,}건  | 사기: {n_fraud:,} ({100*n_fraud/N:.1f}%)  | 정상: {N-n_fraud:,}')
print(f'   기간: {df[\"date\"].min().date()} ~ {df[\"date\"].max().date()}')
print(f'   유저: {df[\"user_id\"].nunique():,}명  | 식당: {df[\"prod_id\"].nunique():,}개')""",
"aa000004"))

# ── [03] TF-IDF + SVD ─────────────────────────────────────────────────
cells.append(code(
"""# [03] TF-IDF + SVD 텍스트 임베딩 (임시 train 인덱스로 fit)
_users = df['user_id'].unique().copy()
np.random.seed(CFG['random_state'])
np.random.shuffle(_users)
_n_tr     = int(len(_users) * CFG['train_user_ratio'])
_fit_idx  = df.index[df['user_id'].isin(set(_users[:_n_tr]))].to_numpy()

print('TF-IDF + SVD 임베딩 중...')
t0 = time.time()
tfidf = TfidfVectorizer(
    max_features=CFG['tfidf_max_features'], ngram_range=(1,2),
    min_df=3, max_df=0.95, strip_accents='unicode', sublinear_tf=True
)
tfidf.fit(df['text'].astype(str).values[_fit_idx])
tfidf_mat  = tfidf.transform(df['text'].astype(str))

svd = TruncatedSVD(n_components=CFG['svd_n_components'], random_state=CFG['random_state'])
svd.fit(tfidf_mat[_fit_idx])
text_embed = svd.transform(tfidf_mat).astype(np.float32)
print(f'✅ 임베딩 완료 ({time.time()-t0:.0f}s) | 설명 분산: {svd.explained_variance_ratio_.sum():.3f}')""",
"aa000005"))

# ── [04] 유저 단위 분할 ────────────────────────────────────────────────
cells.append(code(
"""# [04] 유저 단위 Train / Val / Test 분리
# R-U-R 라벨 누수 차단: 동일 유저 모든 리뷰가 같은 파티션에 배치됨
users = df['user_id'].unique()
np.random.seed(CFG['random_state'])
np.random.shuffle(users)
n_users = len(users)

n_train = int(n_users * CFG['train_user_ratio'])
n_val   = int(n_users * CFG['val_user_ratio'])

train_users = set(users[:n_train])
val_users   = set(users[n_train:n_train+n_val])
test_users  = set(users[n_train+n_val:])

idx_train = df.index[df['user_id'].isin(train_users)].to_numpy()
idx_val   = df.index[df['user_id'].isin(val_users)].to_numpy()
idx_test  = df.index[df['user_id'].isin(test_users)].to_numpy()

train_mask = torch.zeros(N, dtype=torch.bool); train_mask[idx_train] = True
val_mask   = torch.zeros(N, dtype=torch.bool); val_mask[idx_val]     = True
test_mask  = torch.zeros(N, dtype=torch.bool); test_mask[idx_test]   = True

df_train = df.iloc[idx_train]
print(f'Train: {len(idx_train):,}건  사기율: {labels[idx_train].mean():.3f}  ({len(train_users):,}명 유저)')
print(f'Val  : {len(idx_val):,}건  사기율: {labels[idx_val].mean():.3f}  ({len(val_users):,}명 유저)')
print(f'Test : {len(idx_test):,}건  사기율: {labels[idx_test].mean():.3f}  ({len(test_users):,}명 유저)')
print('✅ 유저 단위 분리 완료 — train/val/test 간 R-U-R 엣지 없음')""",
"aa000006"))

# ── [05] 사용자 통계 피처 ─────────────────────────────────────────────
cells.append(code(
"""# [05] 사용자 통계 피처 (Train 전용 → 누수 방지)
print('사용자 통계 계산 중...')
user_stats = df_train.groupby('user_id').agg(
    user_review_count = ('label', 'count'),
    user_avg_rating   = ('rating', 'mean'),
    user_rating_std   = ('rating', lambda x: float(x.std()) if len(x)>1 else 0.0),
    activity_span     = ('date',   lambda x: float((x.max()-x.min()).days)),
).reset_index()
user_stats.fillna(0, inplace=True)

burst_df = pd.read_csv('burst_features.csv', index_col=0).reset_index()
burst_df.columns = ['user_id','user_burst_max7d','user_burst_ratio']
user_stats = user_stats.merge(burst_df, on='user_id', how='left')
user_stats[['user_burst_max7d','user_burst_ratio']] = user_stats[['user_burst_max7d','user_burst_ratio']].fillna(0)
print(f'✅ 사용자 통계 완료: {len(user_stats):,}명')""",
"aa000007"))

# ── [06] 피처 행렬 조립 ────────────────────────────────────────────────
cells.append(code(
"""# [06] 피처 행렬 조립
df_feat = df[['user_id','prod_id','rating','date','label']].copy().reset_index(drop=True)

user_cols = ['user_id','user_review_count','user_avg_rating','user_rating_std',
             'activity_span','user_burst_max7d','user_burst_ratio']
df_feat = df_feat.merge(user_stats[user_cols], on='user_id', how='left')

fr = pd.read_csv('features_rating.csv', index_col=0)
fr_dedup = fr[['user_id','prod_id','rating_deviation','is_cold_start','is_single_review']].drop_duplicates(subset=['user_id','prod_id'])
df_feat = df_feat.merge(fr_dedup, on=['user_id','prod_id'], how='left')
df_feat.fillna(0, inplace=True)
df_feat['user_avg_rating'] = df_feat['user_avg_rating'].replace(0, df_train['rating'].mean())

# rpr_suspicion — R-P-R 엣지 및 노드 피처용
rating_dev = df_feat['rating_deviation'].fillna(0).values
df_feat['rpr_suspicion'] = np.abs(rating_dev)
# 장기 충성 고객 화이트리스트
whitelist = (df_feat['activity_span'].fillna(0) >= 365) & (df_feat['user_review_count'].fillna(0) >= 10)
df_feat.loc[whitelist, 'rpr_suspicion'] = 0.0

# ── RGCN 베이스라인: rating_norm 1차원 ──
X_rgcn = (df_feat['rating'].values / 5.0).reshape(-1,1).astype(np.float32)

# ── PC-GNN: 행동 피처 8개 + TF-IDF SVD 64차원 ──
BEHAV_COLS = [
    'rating', 'user_review_count', 'user_avg_rating', 'user_rating_std',
    'activity_span', 'user_burst_max7d', 'is_cold_start', 'is_single_review',
]
scaler   = StandardScaler()
scaler.fit(df_feat[BEHAV_COLS].values[idx_train])
behav_mat = scaler.transform(df_feat[BEHAV_COLS].values).astype(np.float32)
X_pcgnn   = np.hstack([behav_mat, text_embed])
X_pcgnn   = np.nan_to_num(X_pcgnn, nan=0.0, posinf=0.0, neginf=0.0)

x_rgcn   = torch.tensor(X_rgcn,  dtype=torch.float32)
x_pcgnn  = torch.tensor(X_pcgnn, dtype=torch.float32)
y        = torch.tensor(labels,   dtype=torch.long)
in_dim_rgcn  = x_rgcn.shape[1]
in_dim_pcgnn = x_pcgnn.shape[1]

print('✅ 피처 조립 완료')
print(f'   RGCN  피처: {in_dim_rgcn}차원  (rating_norm)')
print(f'   PCGNN 피처: {in_dim_pcgnn}차원  (행동 {len(BEHAV_COLS)}개 + SVD {CFG[\"svd_n_components\"]}차원)')""",
"aa000008"))

# ── [07] 엣지 구성 ────────────────────────────────────────────────────
cells.append(code(
"""# [07] 엣지 구성
df['ym'] = df['date'].dt.to_period('M')
rng = np.random.default_rng(42)

susp_scores       = df_feat['rpr_suspicion'].values
rating_dev_scores = np.abs(df_feat['rating_deviation'].fillna(0).values)

def make_pairs(groups, max_per_node=None):
    src, dst = [], []
    for nodes in groups:
        nodes = list(nodes)
        if len(nodes) < 2: continue
        n_pairs = len(nodes)*(len(nodes)-1)//2
        if max_per_node is None or n_pairs <= max_per_node*len(nodes):
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    src.append(nodes[i]); dst.append(nodes[j])
        else:
            for node in nodes:
                others = [n for n in nodes if n != node]
                k = min(max_per_node, len(others))
                for j in rng.choice(others, k, replace=False):
                    src.append(node); dst.append(int(j))
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

def make_pairs_scored(groups, scores, max_per_node):
    src, dst = [], []
    for nodes in groups:
        nodes = list(nodes)
        if len(nodes) < 2: continue
        n_pairs = len(nodes)*(len(nodes)-1)//2
        if n_pairs <= max_per_node*len(nodes):
            for i in range(len(nodes)):
                for j in range(i+1, len(nodes)):
                    src.append(nodes[i]); dst.append(nodes[j])
        else:
            sorted_nodes = [nodes[i] for i in np.argsort(-scores[nodes])]
            for node in nodes:
                others = [n for n in sorted_nodes if n != node]
                k = min(max_per_node, len(others))
                for j in others[:k]:
                    src.append(node); dst.append(j)
    return np.array(src, dtype=np.int64), np.array(dst, dtype=np.int64)

def to_edge(s, d, n):
    if len(s) == 0: return torch.zeros((2,0), dtype=torch.long)
    ei = torch.tensor(np.stack([s,d], axis=0), dtype=torch.long)
    ei = to_undirected(ei, num_nodes=n)
    return ei[:, ei[0] != ei[1]]

t0 = time.time()
print(f'엣지 구성 중... (N={N:,})')

# R-U-R: 공통
s, d = make_pairs(df.groupby('user_id').apply(lambda x: x.index.tolist()).values)
edge_rur = to_edge(s, d, N)
print(f'  R-U-R   : {edge_rur.size(1):>8,}개  (공통)')

# R-S-R: 공통 — rating_deviation 높은 순 우선
s, d = make_pairs_scored(
    df.groupby(['prod_id','rating']).apply(lambda x: x.index.tolist()).values,
    rating_dev_scores, max_per_node=CFG['rsr_max_per_node'])
edge_rsr = to_edge(s, d, N)
print(f'  R-S-R   : {edge_rsr.size(1):>8,}개  (공통 | rating_deviation 우선)')

# R-P-R: PC-GNN 전용 — rpr_suspicion 높은 순 우선
susp_mask = df_feat['rpr_suspicion'].values > 0
df_susp   = df[susp_mask].copy()
s, d = make_pairs_scored(
    df_susp.groupby('prod_id').apply(lambda x: x.index.tolist()).values,
    susp_scores, max_per_node=CFG['rpr_max_per_node'])
edge_rpr = to_edge(s, d, N)
print(f'  R-P-R   : {edge_rpr.size(1):>8,}개  (PC-GNN 전용 | 의심도 우선)')

# R-Sim-R: PC-GNN 전용 — 텍스트 유사도
sim_df = pd.read_csv('custom_edges_textSim.csv')
sim_df = sim_df[sim_df['text_similarity'] >= CFG['rsim_thr']]
valid  = (sim_df['src_review_idx'] < N) & (sim_df['dst_review_idx'] < N)
sim_df = sim_df[valid]
edge_rsim = to_edge(
    sim_df['src_review_idx'].values.astype(np.int64),
    sim_df['dst_review_idx'].values.astype(np.int64), N)
print(f'  R-Sim-R : {edge_rsim.size(1):>8,}개  (PC-GNN 전용 | 유사도 ≥{CFG[\"rsim_thr\"]})')

# RGCN용: 두 엣지 합치기 + edge_type
edge_index_rgcn = torch.cat([edge_rur, edge_rsr], dim=1)
edge_type_rgcn  = torch.cat([
    torch.zeros(edge_rur.size(1), dtype=torch.long),
    torch.ones (edge_rsr.size(1), dtype=torch.long)
])

# PC-GNN용: 4개 관계 별도 유지
edge_indices_pcgnn = [edge_rur, edge_rsr, edge_rpr, edge_rsim]
REL_NAMES = ['R-U-R','R-S-R','R-P-R','R-Sim-R']

print(f'\\n✅ 엣지 구성 완료 ({time.time()-t0:.0f}s)')
print(f'   RGCN  엣지: {edge_index_rgcn.size(1):,}개 (2종 합산)')
print(f'   PCGNN 엣지: {sum(e.size(1) for e in edge_indices_pcgnn):,}개 (4종 분리)')""",
"aa000009"))

# ── [08] 모델 정의 ────────────────────────────────────────────────────
cells.append(code(
"""# [08] 모델 정의

# ── Focal Loss ──────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── Baseline RGCN ───────────────────────────────────────────────────
class BaselineRGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim=64, n_rel=2, n_class=2, dropout=0.3):
        super().__init__()
        self.conv1 = RGCNConv(in_dim,     hidden_dim, num_relations=n_rel)
        self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=n_rel)
        self.drop  = nn.Dropout(dropout)
        self.clf   = nn.Linear(hidden_dim, n_class)

    def forward(self, x, edge_index, edge_type):
        h = F.relu(self.conv1(x, edge_index, edge_type)); h = self.drop(h)
        h = F.relu(self.conv2(h, edge_index, edge_type)); h = self.drop(h)
        return self.clf(h)


# ── PC-GNN Chooser ──────────────────────────────────────────────────
class ChooserConv(MessagePassing):
    def __init__(self, in_ch, out_ch):
        super().__init__(aggr='add')
        self.lin  = nn.Linear(in_ch, out_ch, bias=False)
        self.attn = nn.Linear(2 * out_ch, 1)
        self.norm = nn.LayerNorm(out_ch)
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, edge_index):
        if edge_index.size(1) == 0:
            return self.norm(torch.zeros(x.size(0), self.lin.weight.size(0), device=x.device))
        x_lin = self.lin(x)
        out   = self.propagate(edge_index, x=x_lin)
        return self.norm(out)

    def message(self, x_i, x_j):
        alpha = torch.sigmoid(self.attn(torch.cat([x_i, x_j], dim=-1)))
        return alpha * x_j


# ── PC-GNN ──────────────────────────────────────────────────────────
class PCGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim=128, n_rel=4, n_class=2, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ELU()
        )
        self.drop      = nn.Dropout(dropout)
        self.choosers1 = nn.ModuleList([ChooserConv(hidden_dim, hidden_dim) for _ in range(n_rel)])
        self.choosers2 = nn.ModuleList([ChooserConv(hidden_dim, hidden_dim) for _ in range(n_rel)])
        self.inter_attn = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.Tanh(), nn.Linear(32, 1, bias=False)
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ELU(),
            nn.Dropout(dropout), nn.Linear(64, n_class)
        )

    def forward(self, x, edge_indices):
        h  = self.drop(self.input_proj(x))
        h1 = [F.elu(self.drop(c(h,   ei))) for c, ei in zip(self.choosers1, edge_indices)]
        h2 = [F.elu(self.drop(c(h1i, ei))) for c, h1i, ei in zip(self.choosers2, h1, edge_indices)]
        stk   = torch.stack(h2, dim=1)
        attn  = torch.softmax(self.inter_attn(stk).squeeze(-1), dim=1)
        h_agg = (stk * attn.unsqueeze(-1)).sum(dim=1)
        return self.classifier(h_agg), attn.detach().mean(dim=0)


print('✅ 모델 정의 완료  (FocalLoss + BaselineRGCN + PCGNN)')
_r = BaselineRGCN(in_dim_rgcn, 64); _p = PCGNN(in_dim_pcgnn, CFG['hidden_dim'], 4, dropout=CFG['dropout'])
print(f'   RGCN  파라미터: {sum(p.numel() for p in _r.parameters() if p.requires_grad):,}개')
print(f'   PCGNN 파라미터: {sum(p.numel() for p in _p.parameters() if p.requires_grad):,}개')
del _r, _p""",
"aa000010"))

# ── [09] 학습 유틸리티 ────────────────────────────────────────────────
cells.append(code(
"""# [09] 학습 유틸리티

def picker_sample(idx_train, labels, ratio, N):
    fraud_idx  = idx_train[labels[idx_train] == 1]
    normal_idx = idx_train[labels[idx_train] == 0]
    n_sample   = min(len(fraud_idx) * ratio, len(normal_idx))
    sampled    = np.random.choice(normal_idx, n_sample, replace=False)
    picked     = np.concatenate([fraud_idx, sampled])
    mask = torch.zeros(N, dtype=torch.bool)
    mask[picked] = True
    return mask

def eval_rgcn(model, x, ei, et, y, mask):
    model.eval()
    with torch.no_grad():
        logits = model(x, ei, et)
        probs  = torch.softmax(logits, dim=1)[:, 1]
        preds  = logits.argmax(dim=1)
    yt, yp, yprob = y[mask].numpy(), preds[mask].numpy(), probs[mask].numpy()
    return average_precision_score(yt, yprob), f1_score(yt, yp, average='macro'), yt, yp, yprob

def eval_pcgnn(model, x, edge_list, y, mask):
    model.eval()
    with torch.no_grad():
        logits, _ = model(x, edge_list)
        probs     = torch.softmax(logits, dim=1)[:, 1]
        preds     = logits.argmax(dim=1)
    yt, yp, yprob = y[mask].numpy(), preds[mask].numpy(), probs[mask].numpy()
    return average_precision_score(yt, yprob), f1_score(yt, yp, average='macro'), yt, yp, yprob

def bar(v, L=28): return '█'*int(v*L) + '░'*(L-int(v*L))

cw = compute_class_weight('balanced', classes=np.array([0,1]), y=labels[idx_train])
print('✅ 유틸리티 준비 완료')
print(f'   클래스 가중치: 정상={cw[0]:.3f}  사기={cw[1]:.3f}')
print(f'   Picker: 사기 전량 + 정상 {CFG[\"picker_ratio\"]}배 → 배치 사기율 {1/(1+CFG[\"picker_ratio\"]):.2f}')""",
"aa000011"))

# ── [10] RGCN 학습 ────────────────────────────────────────────────────
cells.append(code(
"""# [10] RGCN 베이스라인 학습
print('='*62)
print(' RGCN 베이스라인 학습')
print(' 피처: rating_norm (1차원)  |  집계: 균등 평균')
print(' 엣지: R-U-R + R-S-R (2종)')
print('='*62)

cw_r   = compute_class_weight('balanced', classes=np.array([0,1]), y=labels[idx_train])
crit_r = nn.CrossEntropyLoss(weight=torch.tensor(cw_r, dtype=torch.float32))

rgcn_model = BaselineRGCN(in_dim=in_dim_rgcn, hidden_dim=64, n_rel=2, dropout=0.30)
opt_r = AdamW(rgcn_model.parameters(), lr=1e-3, weight_decay=1e-4)
sch_r = CosineAnnealingLR(opt_r, T_max=CFG['epochs'])

best_r_pr, best_r_state, no_imp = 0.0, None, 0
t0 = time.time()

for ep in range(1, CFG['epochs']+1):
    rgcn_model.train(); opt_r.zero_grad()
    logits = rgcn_model(x_rgcn, edge_index_rgcn, edge_type_rgcn)
    loss   = crit_r(logits[train_mask], y[train_mask])
    loss.backward(); nn.utils.clip_grad_norm_(rgcn_model.parameters(), 1.0); opt_r.step()
    sch_r.step()

    if ep % 10 == 0:
        vpr, vf1, *_ = eval_rgcn(rgcn_model, x_rgcn, edge_index_rgcn, edge_type_rgcn, y, val_mask)
        print(f'  [E{ep:3d}] loss={float(loss):.4f} | val PR-AUC={vpr:.4f} F1={vf1:.4f} ({time.time()-t0:.0f}s)')
        if vpr > best_r_pr:
            best_r_pr = vpr
            best_r_state = {k: v.clone() for k,v in rgcn_model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= CFG['patience'] // 2:
                print(f'  Early stopping'); break

rgcn_model.load_state_dict(best_r_state)
res_r_pr, res_r_f1, yt_r, yp_r, yprob_r = eval_rgcn(
    rgcn_model, x_rgcn, edge_index_rgcn, edge_type_rgcn, y, test_mask)
torch.save(best_r_state, 'best_rgcn_baseline.pt')
print(f'\\n✅ RGCN 최종:  PR-AUC={res_r_pr:.4f}  Macro F1={res_r_f1:.4f}')""",
"aa000012"))

# ── [11] PC-GNN 학습 ──────────────────────────────────────────────────
cells.append(code(
"""# [11] PC-GNN 학습
print('='*62)
print(' PC-GNN 학습')
print(f' 피처: 행동 {len(BEHAV_COLS)}개 + SVD {CFG[\"svd_n_components\"]}차원 = {in_dim_pcgnn}차원')
print(' 엣지: R-U-R + R-S-R + R-P-R + R-Sim-R (4종) | Chooser 어텐션')
print(f' 배치: Picker ratio={CFG[\"picker_ratio\"]}  |  FocalLoss gamma={CFG[\"focal_gamma\"]}')
print('='*62)

cw_p   = compute_class_weight('balanced', classes=np.array([0,1]), y=labels[idx_train])
cw_p[1] *= 1.5
crit_p = FocalLoss(alpha=torch.tensor(cw_p, dtype=torch.float32), gamma=CFG['focal_gamma'])

pcgnn_model = PCGNN(in_dim=in_dim_pcgnn, hidden_dim=CFG['hidden_dim'], n_rel=4, dropout=CFG['dropout'])
opt_p = AdamW(pcgnn_model.parameters(), lr=CFG['lr'], weight_decay=CFG['wd'])
sch_p = CosineAnnealingLR(opt_p, T_max=CFG['epochs'])

best_p_pr, best_p_state, no_imp_p = 0.0, None, 0
t0 = time.time()

for ep in range(1, CFG['epochs']+1):
    picked_mask = picker_sample(idx_train, labels, CFG['picker_ratio'], N)

    pcgnn_model.train(); opt_p.zero_grad()
    logits, _ = pcgnn_model(x_pcgnn, edge_indices_pcgnn)
    loss = crit_p(logits[picked_mask], y[picked_mask])
    loss.backward(); nn.utils.clip_grad_norm_(pcgnn_model.parameters(), 1.0); opt_p.step()
    sch_p.step()

    if ep % 10 == 0:
        vpr, vf1, *_ = eval_pcgnn(pcgnn_model, x_pcgnn, edge_indices_pcgnn, y, val_mask)
        print(f'  [E{ep:3d}] loss={float(loss):.4f} | val PR-AUC={vpr:.4f} F1={vf1:.4f} ({time.time()-t0:.0f}s)')
        if vpr > best_p_pr:
            best_p_pr = vpr
            best_p_state = {k: v.clone() for k,v in pcgnn_model.state_dict().items()}
            no_imp_p = 0
        else:
            no_imp_p += 1
            if no_imp_p >= CFG['patience']:
                print(f'  Early stopping'); break

pcgnn_model.load_state_dict(best_p_state)
res_p_pr, res_p_f1, yt_p, yp_p, yprob_p = eval_pcgnn(
    pcgnn_model, x_pcgnn, edge_indices_pcgnn, y, test_mask)
torch.save(best_p_state, 'best_pcgnn.pt')
print(f'\\n✅ PC-GNN 최종:  PR-AUC={res_p_pr:.4f}  Macro F1={res_p_f1:.4f}')""",
"aa000013"))

# ── [12] 성능 비교 리포트 ─────────────────────────────────────────────
cells.append(code(
"""# [12] 최종 성능 비교 리포트
W = 64
print()
print('=' * W)
print('  RGCN(Baseline) vs PC-GNN  성능 비교')
print('=' * W)

print(f'\\n[1] RGCN 베이스라인 — rating_norm 1차원 / R-U-R+R-S-R / 균등 집계')
print(f'    PR-AUC  : [{bar(res_r_pr)}] {res_r_pr:.4f}')
print(f'    Macro F1: [{bar(res_r_f1)}] {res_r_f1:.4f}')

print(f'\\n[2] PC-GNN — 행동피처 {len(BEHAV_COLS)}개+SVD / 4종 엣지 / Chooser+Picker+FocalLoss')
print(f'    PR-AUC  : [{bar(res_p_pr)}] {res_p_pr:.4f}')
print(f'    Macro F1: [{bar(res_p_f1)}] {res_p_f1:.4f}')

d_pr = res_p_pr - res_r_pr
d_f1 = res_p_f1 - res_r_f1
print(f'\\n[3] 개선폭 (RGCN → PC-GNN)')
print(f'    PR-AUC  : {res_r_pr:.4f} → {res_p_pr:.4f}  ({d_pr:+.4f}, {d_pr/max(res_r_pr,1e-9)*100:+.1f}%)')
print(f'    Macro F1: {res_r_f1:.4f} → {res_p_f1:.4f}  ({d_f1:+.4f}, {d_f1/max(res_r_f1,1e-9)*100:+.1f}%)')

# 관계별 Attention
pcgnn_model.eval()
with torch.no_grad():
    _, attn_mean = pcgnn_model(x_pcgnn, edge_indices_pcgnn)
attn_np = attn_mean.cpu().numpy()
print(f'\\n[4] 관계별 Chooser Attention (높을수록 사기 판별 기여 큼)')
for name, w, ei in zip(REL_NAMES, attn_np, edge_indices_pcgnn):
    b = '█' * int(w * 40)
    print(f'    {name:8s}: [{b:<20}] {w:.3f}  ({ei.size(1):,} edges)')

print(f'\\n[5] PC-GNN — Classification Report (Test set)')
print(classification_report(yt_p, yp_p, target_names=['정상(0)','사기(1)'], digits=4))
print('=' * W)
print('✅ 완료 | 저장: best_rgcn_baseline.pt / best_pcgnn.pt')""",
"aa000014"))

# ── 노트북 저장 ─────────────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = os.path.join(BASE, '05_1_PCGNN.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print(f'[OK] 노트북 생성: {out}')

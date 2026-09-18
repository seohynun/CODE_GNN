"""
create_dashboard_notebook.py
07_Dashboard.ipynb 생성 스크립트

구성
  [01] 환경 설정 및 데이터 로드
  [02] 피처 엔지니어링 (텍스트 수치 피처 + TF-IDF SVD)
  [03] 82차원 피처 행렬 조립
  [04] CatBoost 학습 및 평가
  [05] 조회 테이블 구축 및 전체 사기 점수 사전 계산
  [06] 대시보드 실행 (Gradio)
"""

import nbformat as nbf

nb    = nbf.v4.new_notebook()
cells = []

# ──────────────────────────────────────────────────────────
# 헬퍼: 코드 셀 / 마크다운 셀
# ──────────────────────────────────────────────────────────
def md(src):   cells.append(nbf.v4.new_markdown_cell(src))
def code(src): cells.append(nbf.v4.new_code_cell(src))


# ══════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════
md("""\
# GNN 기반 사기 리뷰 탐지 — [07] 대화형 대시보드

**목적**: 리뷰 텍스트와 별점을 입력받아 사기 확률을 즉시 예측하는 대화형 대시보드

| 항목 | 내용 |
|------|------|
| **입력** | 리뷰 텍스트, 별점, (선택) 유저 ID / 식당 ID |
| **출력** | 사기 확률 게이지, SHAP 피처 기여도 차트, 키워드 탐지 결과 |
| **모델** | CatBoost — 텍스트·행동 피처 82차원 |
| **비고** | 최종 모델(06, CARE-HAN+CatBoost 206차원) PR-AUC **0.9279** 대비 GNN 구조 신호 미포함 |

---

## 노트북 구성

| 섹션 | 내용 |
|------|------|
| [01] | 환경 설정 및 데이터 로드 |
| [02] | 피처 엔지니어링 (텍스트 수치 피처 + TF-IDF SVD) |
| [03] | 82차원 피처 행렬 조립 |
| [04] | CatBoost 학습 및 평가 |
| [05] | 조회 테이블 구축 및 전체 사기 점수 사전 계산 |
| [06] | 대시보드 실행 (Gradio) |
""")


# ══════════════════════════════════════════════════════════
#  [01] 환경 설정 및 데이터 로드
# ══════════════════════════════════════════════════════════
md("## [01] 환경 설정 및 데이터 로드")

code("""\
import os, warnings, pickle
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, f1_score, classification_report
from catboost import CatBoostClassifier, Pool

BASE_DIR = str(Path().resolve())
os.chdir(BASE_DIR)

# ── 키워드 사전 (03_1 분석 결과 기반) ──────────────────────────────
# 사기 리뷰 특징어: 이야기 구조, 대인 관계 표현
FRAUD_KW  = {'told','said','asked','manager','table','owner','call',
             'rude','experience','server','waitress','waiter',
             'horrible','worst','complained','never','disgusting','terrible','awful'}
# 정상 리뷰 특징어: 음식 맛/식감 묘사
NORMAL_KW = {'topped','texture','creamy','sweetness','pieces','rich',
             'tender','crispy','fluffy','flavor','delicious','fresh',
             'tasty','amazing','fantastic','loved','enjoyed'}
# 메뉴 관련 단어
MENU_KW   = {'food','steak','burger','pizza','pasta','chicken','fish',
             'salad','sandwich','soup','noodle','rice','sauce','cheese',
             'beef','pork','bread','dessert','cake','coffee','drink'}
# 감성 단어
POS_WORDS = {'good','great','excellent','amazing','wonderful','love','best',
             'perfect','fantastic','delicious','fresh','friendly','enjoyed','recommend','awesome'}
NEG_WORDS = {'bad','terrible','awful','horrible','worst','rude','slow',
             'dirty','disgusting','disappointing','never','poor'}

CFG = {
    'tfidf_max_features': 5000,
    'svd_n_components'  : 64,
    'random_state'      : 42,
    'cutoff_date'       : '2014-06-28',   # 04/06 노트북과 동일한 시간 기반 분할 기준일
}

print('환경 설정 완료')
print(f'작업 디렉토리: {BASE_DIR}')
""")

code("""\
# 기본 샘플 데이터 로드
df = pd.read_parquet('yelpzip_sampled.parquet')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# 별점 파생 피처 (index 정렬된 csv — 행 순서 = yelpzip_sampled 행 순서)
feat_rating = pd.read_csv('features_rating.csv', index_col=0).reset_index(drop=True)

# 버스트 피처 (유저 단위)
feat_burst = pd.read_csv('burst_features.csv')

N = len(df)
print(f'데이터: {N:,}건  |  사기: {df["label"].sum():,} ({df["label"].mean()*100:.1f}%)  |  정상: {(df["label"]==0).sum():,}')
print(f'기간  : {df["date"].min().date()} ~ {df["date"].max().date()}')
print(f'유저  : {df["user_id"].nunique():,}명  |  식당: {df["prod_id"].nunique():,}개')
""")


# ══════════════════════════════════════════════════════════
#  [02] 피처 엔지니어링
# ══════════════════════════════════════════════════════════
md("## [02] 피처 엔지니어링")

code("""\
def extract_text_stats(text: str) -> dict:
    \"\"\"
    리뷰 텍스트에서 수치 피처 10개를 추출합니다.
    학습 시와 새 리뷰 추론 시 동일한 함수를 사용해 일관성을 보장합니다.
    \"\"\"
    text  = str(text)
    words = text.lower().split()
    nw    = max(len(words), 1)
    nc    = max(len(text),  1)

    def cnt(kw_set):
        return sum(1 for w in words if w in kw_set)

    fi = cnt(FRAUD_KW)  / nw
    ni = cnt(NORMAL_KW) / nw
    return {
        'text_len'        : len(text),
        'word_count'      : nw,
        'digit_ratio'     : sum(c.isdigit() for c in text) / nc,
        'special_ratio'   : sum(1 for c in text if not c.isalnum() and not c.isspace()) / nc,
        'ttr'             : len(set(words)) / nw,
        'sentiment'       : (cnt(POS_WORDS) - cnt(NEG_WORDS)) / nw,
        'menu_mentions'   : cnt(MENU_KW),
        'fraud_intensity' : fi,
        'normal_intensity': ni,
        'vocab_discrim'   : fi - ni,
    }

# 전체 데이터에 일괄 적용
print('텍스트 피처 추출 중...')
text_stats = df['text'].apply(extract_text_stats).apply(pd.Series)
print(f'완료: {text_stats.shape}  |  피처: {text_stats.columns.tolist()}')
""")

code("""\
# ── 시간 기반 Train / Test 분할 ──────────────────────────────────────
# 04/06 노트북과 동일한 cutoff 사용 → 모델 비교 일관성 확보
cutoff     = pd.Timestamp(CFG['cutoff_date'])
train_mask = (df['date'] <= cutoff).values
test_mask  = ~train_mask
train_idx  = np.where(train_mask)[0]
test_idx   = np.where(test_mask)[0]

print(f'Train: {len(train_idx):,}건 ({train_mask.mean()*100:.1f}%)  사기율: {df["label"].iloc[train_idx].mean():.3f}')
print(f'Test : {len(test_idx):,}건 ({test_mask.mean()*100:.1f}%)  사기율: {df["label"].iloc[test_idx].mean():.3f}')

# ── TF-IDF + SVD (Train 기준 fit — 데이터 누수 방지) ───────────────
print('\\nTF-IDF + SVD 학습 중...')
tfidf = TfidfVectorizer(
    max_features=CFG['tfidf_max_features'],
    ngram_range=(1, 2), min_df=3, max_df=0.95,
    strip_accents='unicode', sublinear_tf=True,
)
tfidf.fit(df['text'].iloc[train_idx].astype(str))
tfidf_mat = tfidf.transform(df['text'].astype(str))

svd = TruncatedSVD(n_components=CFG['svd_n_components'], random_state=CFG['random_state'])
svd.fit(tfidf_mat[train_idx])
svd_mat = svd.transform(tfidf_mat).astype(np.float32)

print(f'완료 | 설명 분산: {svd.explained_variance_ratio_.sum():.3f}')
""")


# ══════════════════════════════════════════════════════════
#  [03] 피처 행렬 조립
# ══════════════════════════════════════════════════════════
md("## [03] 82차원 피처 행렬 조립")

code("""\
# 시간 피처 추가
df['weekday'] = df['date'].dt.weekday
df['month']   = df['date'].dt.month

# features_rating.csv 결합 (index 기반 — 행 순서 일치)
df = pd.concat([
    df.reset_index(drop=True),
    feat_rating[['rating_deviation', 'is_cold_start', 'is_single_review']].reset_index(drop=True),
], axis=1)
df[['rating_deviation', 'is_cold_start', 'is_single_review']] = \\
    df[['rating_deviation', 'is_cold_start', 'is_single_review']].fillna(0)

# burst_features.csv 결합 (user_id 기준)
df = df.merge(feat_burst[['user_id', 'user_burst_max7d', 'user_burst_ratio']],
              on='user_id', how='left')
df[['user_burst_max7d', 'user_burst_ratio']] = \\
    df[['user_burst_max7d', 'user_burst_ratio']].fillna(0)

# 텍스트 수치 피처 결합
df = pd.concat([df.reset_index(drop=True), text_stats.reset_index(drop=True)], axis=1)

# ── 피처 컬럼 순서 정의 (추론 시에도 반드시 동일 순서 유지) ────────
FEAT_COLS = [
    'rating', 'weekday', 'month',
    'rating_deviation', 'is_cold_start', 'is_single_review',
    'user_burst_max7d', 'user_burst_ratio',
    'text_len', 'word_count', 'digit_ratio', 'special_ratio',
    'ttr', 'sentiment', 'menu_mentions',
    'fraud_intensity', 'normal_intensity', 'vocab_discrim',
]
SVD_COLS  = [f'SVD_{i}' for i in range(CFG['svd_n_components'])]
ALL_FEATS = FEAT_COLS + SVD_COLS          # 18 + 64 = 82차원

# ── 스케일러 (Train 기준 fit) ────────────────────────────────────────
num_mat = df[FEAT_COLS].values.astype(np.float32)
scaler  = StandardScaler()
scaler.fit(num_mat[train_idx])
num_scaled = scaler.transform(num_mat)

X = np.hstack([num_scaled, svd_mat]).astype(np.float32)
y = df['label'].values

print(f'피처 행렬: {X.shape}  (수치 {len(FEAT_COLS)}개 + SVD {len(SVD_COLS)}차원 = {X.shape[1]}차원)')
print(f'NaN 포함 여부: {np.isnan(X).sum()}개')
""")


# ══════════════════════════════════════════════════════════
#  [04] CatBoost 학습 및 평가
# ══════════════════════════════════════════════════════════
md("## [04] CatBoost 학습 및 평가")

code("""\
MODEL_PATH = 'best_catboost_82.cbm'

if os.path.exists(MODEL_PATH):
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)
    print(f'저장된 모델 로드: {MODEL_PATH}')
else:
    X_tr, y_tr = X[train_idx], y[train_idx]

    # Train 내 Validation 분리 (앞 80% Train, 뒤 20% Val)
    val_cut    = int(len(X_tr) * 0.8)
    X_tr2, X_val = X_tr[:val_cut], X_tr[val_cut:]
    y_tr2, y_val = y_tr[:val_cut], y_tr[val_cut:]

    model = CatBoostClassifier(
        iterations=1000, depth=6, learning_rate=0.05,
        loss_function='Logloss', eval_metric='AUC',
        class_weights={0: 1, 1: 3},           # 사기 클래스 3배 가중치
        early_stopping_rounds=50, verbose=200,
        random_seed=CFG['random_state'],
    )
    model.fit(Pool(X_tr2, y_tr2), eval_set=Pool(X_val, y_val))
    model.save_model(MODEL_PATH)
    print(f'\\n모델 저장 완료: {MODEL_PATH}')
""")

code("""\
# ── 성능 평가 ────────────────────────────────────────────────────────
probs    = model.predict_proba(X[test_idx])[:, 1]
preds    = (probs >= 0.5).astype(int)
pr_auc   = average_precision_score(y[test_idx], probs)
macro_f1 = f1_score(y[test_idx], preds, average='macro')

print('=' * 55)
print('  대시보드 모델 (CatBoost 82차원) — Test Set')
print('=' * 55)
print(f'  PR-AUC  : {pr_auc:.4f}')
print(f'  Macro F1: {macro_f1:.4f}')
print()
print('  [비교] 최종 모델 06 (CARE-HAN + CatBoost 206차원)')
print('         PR-AUC  : 0.9279  |  Macro F1: 0.9136')
print('         → GNN 구조 신호(128차원) 포함으로 더 높은 성능')
print()
print(classification_report(y[test_idx], preds,
                             target_names=['정상(0)', '사기(1)'], digits=4))

# ── 피처 중요도 시각화 (상위 20개) ──────────────────────────────────
fi_series = pd.Series(model.get_feature_importance(), index=ALL_FEATS).nlargest(20)

fig = px.bar(
    fi_series[::-1], orientation='h',
    title='CatBoost 피처 중요도 상위 20개',
    labels={'value': '중요도', 'index': '피처'},
    color=fi_series[::-1].values,
    color_continuous_scale='Blues',
)
fig.update_layout(coloraxis_showscale=False, height=520,
                  plot_bgcolor='white', paper_bgcolor='white')
fig.show()
""")


# ══════════════════════════════════════════════════════════
#  [05] 조회 테이블 구축 및 전체 사기 점수 사전 계산
# ══════════════════════════════════════════════════════════
md("## [05] 조회 테이블 구축 및 전체 사기 점수 사전 계산")

code("""\
# ── Train 데이터 기준 조회 테이블 구축 ──────────────────────────────
# (새 리뷰 추론 시 사용 — Test 데이터 정보가 섞이지 않도록 Train만 사용)
df_train = df[train_mask]

PROD_AVG_RATING   = df_train.groupby('prod_id')['rating'].mean().to_dict()
PROD_REVIEW_COUNT = df_train.groupby('prod_id').size().to_dict()
PROD_IS_COLD      = {pid: int(cnt < 5) for pid, cnt in PROD_REVIEW_COUNT.items()}

USER_BURST     = feat_burst.set_index('user_id')[['user_burst_max7d', 'user_burst_ratio']].to_dict('index')
user_rc        = df_train.groupby('user_id').size()
USER_IS_SINGLE = {uid: int(cnt <= 1) for uid, cnt in user_rc.items()}

print(f'식당 조회 테이블: {len(PROD_AVG_RATING)}개')
print(f'유저 조회 테이블: {len(USER_BURST)}명')
""")

code("""\
# ── 전체 50,950건 사기 점수 사전 계산 ──────────────────────────────
PRECOMP_PATH = 'precomputed_dash.parquet'

if os.path.exists(PRECOMP_PATH):
    df_precomp = pd.read_parquet(PRECOMP_PATH)
    print(f'사전 계산 결과 로드: {len(df_precomp):,}건')
else:
    all_probs = model.predict_proba(X)[:, 1]
    df_precomp = df[['user_id', 'prod_id', 'rating', 'label', 'date', 'text']].copy()
    df_precomp['fraud_score'] = all_probs
    df_precomp['fraud_pred']  = (all_probs >= 0.5).astype(int)
    df_precomp['review_idx']  = df_precomp.index
    df_precomp.to_parquet(PRECOMP_PATH)
    print(f'사전 계산 완료: {len(df_precomp):,}건 저장 → {PRECOMP_PATH}')

# ── 아티팩트 저장 (커널 재시작 없이 대시보드 재실행 가능하도록) ────
artifacts = {
    'tfidf': tfidf, 'svd': svd, 'scaler': scaler,
    'feat_cols': FEAT_COLS, 'all_feats': ALL_FEATS,
    'prod_avg_rating': PROD_AVG_RATING,
    'prod_is_cold'   : PROD_IS_COLD,
    'user_burst'     : USER_BURST,
    'user_is_single' : USER_IS_SINGLE,
}
with open('dashboard_artifacts.pkl', 'wb') as f:
    pickle.dump(artifacts, f)
print('아티팩트 저장 완료: dashboard_artifacts.pkl')
""")


# ══════════════════════════════════════════════════════════
#  [06] 대시보드 실행 (Gradio)
# ══════════════════════════════════════════════════════════
md("## [06] 대시보드 실행 (Gradio)")

# ── 예측 함수 정의
code("""\
SHAP_META = {
    'text_len'        : ('텍스트 길이',       '리뷰가 너무 짧아요'),
    'user_burst_ratio': ('버스트 집중도',      '짧은 기간에 몰아썼어요'),
    'rating_deviation': ('별점 이탈도',        '평균과 동떨어진 별점이에요'),
    'word_count'      : ('단어 수',            '내용이 너무 적어요'),
    'user_burst_max7d': ('7일내 최대 리뷰',    '최근 활동이 과도해요'),
    'fraud_intensity' : ('사기 어휘 강도',     '사기 패턴 단어가 많아요'),
    'normal_intensity': ('정상 어휘 강도',     '음식 관련 표현이 있어요'),
    'vocab_discrim'   : ('변별 어휘 점수',     '어휘 패턴이 특이해요'),
    'ttr'             : ('어휘 다양성',        '단어 반복이 많아요'),
    'sentiment'       : ('감성 점수',          '감성 표현이 편향됐어요'),
    'rating'          : ('별점',               '입력된 별점이에요'),
    'is_single_review': ('싱글 리뷰어',        '리뷰 이력이 없는 계정이에요'),
    'is_cold_start'   : ('콜드스타트 식당',    '리뷰 수가 적은 식당이에요'),
    'weekday'         : ('요일',               '리뷰 작성 요일이에요'),
    'month'           : ('월',                 '리뷰 작성 월이에요'),
    'digit_ratio'     : ('숫자 비율',          '숫자가 많이 포함됐어요'),
    'special_ratio'   : ('특수문자 비율',      '특수문자가 사용됐어요'),
    'menu_mentions'   : ('메뉴 언급',          '음식 메뉴 단어가 있어요'),
}

_FONT = 'Pretendard, Apple SD Gothic Neo, -apple-system, BlinkMacSystemFont, sans-serif'


def _shap_html(shap_vals, all_feats):
    df_s   = pd.DataFrame({'feat': all_feats, 'val': shap_vals})
    top5   = df_s.reindex(df_s['val'].abs().nlargest(5).index).sort_values('val', ascending=False)
    max_ab = df_s['val'].abs().max() or 1.0
    rows   = ''
    for _, r in top5.iterrows():
        meta        = SHAP_META.get(r['feat'], (r['feat'], ''))
        label, desc = meta
        fraud       = r['val'] > 0
        bar_c       = '#3182F6' if fraud else '#B0B8C1'
        tag_c       = '#3182F6' if fraud else '#8B95A1'
        tag_bg      = '#EBF3FE' if fraud else '#F2F4F6'
        tag_t       = '의심 신호' if fraud else '정상 신호'
        bw          = int(abs(r['val']) / max_ab * 100)
        rows += (
            '<div style="margin-bottom:20px">'
            f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">'
            f'<span style="font-size:14px;font-weight:600;color:#191F28">{label}</span>'
            f'<span style="font-size:12px;font-weight:600;color:{tag_c};background:{tag_bg};'
            f'padding:2px 10px;border-radius:100px">{tag_t}</span></div>'
            f'<div style="font-size:12px;color:#6B7684;margin-bottom:8px">{desc}</div>'
            f'<div style="background:#F2F4F6;border-radius:4px;height:6px">'
            f'<div style="background:{bar_c};width:{bw}%;height:6px;border-radius:4px"></div>'
            '</div></div>'
        )
    return (
        '<div style="background:#FFFFFF;border-radius:16px;padding:28px 28px 12px">'
        '<div style="font-size:15px;font-weight:600;color:#191F28;margin-bottom:20px">판정 근거</div>'
        + rows + '</div>'
    )


def predict_new_review(text, rating, user_id='', prod_id=''):
    text = str(text).strip()
    if not text:
        return (
            f'<div style="font-family:{_FONT};color:#B0B8C1;'
            'padding:48px;text-align:center;font-size:14px">'
            '리뷰 텍스트를 입력하세요.</div>'
        )

    rating = float(rating)
    now    = pd.Timestamp.now()

    pid        = int(prod_id.strip()) if prod_id and prod_id.strip().isdigit() else None
    prod_avg   = PROD_AVG_RATING.get(pid, 3.5)
    is_cold    = PROD_IS_COLD.get(pid, 0)
    rating_dev = abs(rating - prod_avg)

    uid_str   = user_id.strip() if user_id else ''
    uid_key   = int(uid_str) if uid_str.isdigit() else (uid_str if uid_str else None)
    burst     = USER_BURST.get(uid_key, {'user_burst_max7d': 0, 'user_burst_ratio': 0})
    is_single = USER_IS_SINGLE.get(uid_key, 1)

    ts = extract_text_stats(text)

    num = np.array([[
        rating, now.weekday(), now.month,
        rating_dev, is_cold, is_single,
        burst['user_burst_max7d'], burst['user_burst_ratio'],
        ts['text_len'], ts['word_count'], ts['digit_ratio'], ts['special_ratio'],
        ts['ttr'], ts['sentiment'], ts['menu_mentions'],
        ts['fraud_intensity'], ts['normal_intensity'], ts['vocab_discrim'],
    ]], dtype=np.float32)

    tfidf_vec = tfidf.transform([text])
    svd_vec   = svd.transform(tfidf_vec).astype(np.float32)
    num_sc    = scaler.transform(num)
    x         = np.hstack([num_sc, svd_vec])

    score     = float(model.predict_proba(x)[0][1])
    shap_vals = model.get_feature_importance(
        Pool(x, feature_names=ALL_FEATS), type='ShapValues'
    )[0][:-1]

    pct = score * 100

    if score >= 0.7:
        badge_t, badge_c, badge_bg = '사기 의심', '#F03E3E', '#FFF2F2'
    elif score >= 0.4:
        badge_t, badge_c, badge_bg = '주의 필요', '#3182F6', '#EBF3FE'
    else:
        badge_t, badge_c, badge_bg = '정상',     '#12B886', '#EDFAF1'

    bar_w = f'{min(pct, 100):.1f}'

    result = (
        f'<div style="font-family:{_FONT}">'
        '<div style="background:#FFFFFF;border-radius:16px;padding:32px;margin-bottom:16px">'
        '<div style="font-size:13px;color:#6B7684;margin-bottom:8px">사기 확률</div>'
        f'<div style="font-size:52px;font-weight:600;color:#191F28;line-height:1">'
        f'{pct:.1f}'
        '<span style="font-size:28px;font-weight:400">%</span></div>'
        '<div style="margin-top:12px;margin-bottom:20px">'
        f'<span style="font-size:13px;font-weight:600;color:{badge_c};background:{badge_bg};'
        f'padding:4px 12px;border-radius:100px">{badge_t}</span></div>'
        '<div style="background:#F2F4F6;border-radius:4px;height:4px">'
        f'<div style="background:#3182F6;width:{bar_w}%;height:4px;border-radius:4px"></div>'
        '</div></div>'
        + _shap_html(shap_vals, ALL_FEATS)
        + '</div>'
    )
    return result


print('함수 정의 완료')
""")

# ── Gradio 실행
code("""\
import gradio as gr

# ── Gradio 6.x: 버튼/배경색은 theme.set()으로 지정 ────────────────
THEME = gr.themes.Base().set(
    button_primary_background_fill='#3182F6',
    button_primary_background_fill_hover='#1C6EE8',
    button_primary_text_color='#FFFFFF',
    button_primary_border_color='#3182F6',
    button_primary_border_color_hover='#1C6EE8',
)

# ── Gradio 6.x: css는 launch()에 전달 ─────────────────────────────
CSS = \"\"\"
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

body, .gradio-container {
    background: #F9FAFB !important;
    font-family: Pretendard, Apple SD Gothic Neo, -apple-system, BlinkMacSystemFont, sans-serif !important;
}
.block, .form, .gap, .contain {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
textarea, input[type=text], input[type=number] {
    border: none !important;
    border-radius: 12px !important;
    background: #F2F4F6 !important;
    color: #191F28 !important;
    font-size: 15px !important;
    box-shadow: none !important;
    padding: 12px 16px !important;
}
textarea:focus, input[type=text]:focus {
    background: #EAF1FE !important;
    box-shadow: none !important;
    outline: none !important;
}
label > span, .block label > span {
    color: #6B7684 !important;
    font-size: 13px !important;
    font-weight: 400 !important;
}
input[type=range] { accent-color: #3182F6 !important; }
button.primary, .primary {
    border-radius: 12px !important;
    font-size: 16px !important;
    font-weight: 600 !important;
    box-shadow: none !important;
    width: 100% !important;
}
.examples-holder {
    background: #FFFFFF !important;
    border-radius: 16px !important;
    border: none !important;
    margin-top: 12px !important;
    padding: 8px 16px !important;
}
\"\"\"

HEADER = (
    '<div style="font-family:Pretendard,Apple SD Gothic Neo,sans-serif;'
    'padding:40px 8px 28px">'
    '<div style="font-size:22px;font-weight:600;color:#191F28;margin-bottom:8px">'
    '사기 리뷰 탐지 시스템</div>'
    '<div style="font-size:14px;color:#6B7684">'
    'GNN + CatBoost 하이브리드 모델 &nbsp;&middot;&nbsp; '
    'PR-AUC <b style="color:#3182F6">0.9279</b> &nbsp;/&nbsp; '
    'Macro F1 <b style="color:#3182F6">0.9136</b></div></div>'
)

INPUT_HINT = (
    '<div style="font-family:Pretendard,Apple SD Gothic Neo,sans-serif;'
    'margin-bottom:20px">'
    '<div style="font-size:16px;font-weight:600;color:#191F28;margin-bottom:4px">리뷰 분석</div>'
    '<div style="font-size:13px;color:#6B7684">'
    '텍스트와 별점을 입력하면 사기 여부를 즉시 분석합니다.</div></div>'
)

PLACEHOLDER = (
    '<div style="font-family:Pretendard,Apple SD Gothic Neo,sans-serif;'
    'color:#B0B8C1;padding:48px;text-align:center;font-size:14px">'
    '분석 결과가 여기에 표시됩니다.</div>'
)

with gr.Blocks(title='사기 리뷰 탐지', theme=THEME) as demo:
    gr.HTML(HEADER)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            gr.HTML(INPUT_HINT)
            text_in   = gr.Textbox(label='리뷰 텍스트', lines=6,
                                   placeholder='리뷰 내용을 여기에 입력하세요...')
            rating_in = gr.Slider(minimum=1, maximum=5, step=1, value=3, label='별점')
            with gr.Row():
                user_in = gr.Textbox(label='유저 ID (선택)', placeholder='예: 5798')
                prod_in = gr.Textbox(label='식당 ID (선택)', placeholder='예: 9')
            btn = gr.Button('분석하기', variant='primary', size='lg')
            gr.Examples(
                label='예시 리뷰',
                examples=[
                    ['The food was amazing! Crispy tender steak. Best flavor ever.', 5, '', ''],
                    ['I told the manager about the rude service but they were awful.', 1, '', ''],
                    ['Said the experience was wonderful. Asked owner and called to complain.', 2, '5798', '9'],
                ],
                inputs=[text_in, rating_in, user_in, prod_in],
            )
        with gr.Column(scale=1, min_width=320):
            result_out = gr.HTML(value=PLACEHOLDER)

    for trigger in [btn.click, text_in.submit]:
        trigger(predict_new_review,
                inputs=[text_in, rating_in, user_in, prod_in],
                outputs=[result_out])

demo.launch(share=True, css=CSS)
""")


# ══════════════════════════════════════════════════════════
#  노트북 저장
# ══════════════════════════════════════════════════════════
nb.cells = cells

OUT = '07_Dashboard.ipynb'
with open(OUT, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'{OUT} 생성 완료  ({len(cells)}개 셀)')

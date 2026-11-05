# src/modeling.py
# خطوط المعالجة (Pipelines)، النموذج المرجعي، النموذجان، وشبكات الإعدادات.
# run_experiments تستقبل التدريب والتحقق فقط: الاختبار النهائي لا يلمس اليوم.

import itertools
import time

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src.splitting import get_xy
from src.target import TARGET

BASELINE = "baseline_prior"
LOGISTIC = "logistic_regression"
HGB = "hist_gradient_boosting"

# شبكات الإعدادات (صغيرة عمداً: الاختيار يتم على التحقق فقط)
LR_GRID = [{"C": c, "class_weight": cw}
           for c, cw in itertools.product([0.01, 0.1, 1.0, 10.0], [None, "balanced"])]
HGB_GRID = [{"learning_rate": lr, "max_leaf_nodes": leaves, "class_weight": cw}
            for lr, leaves, cw in itertools.product([0.05, 0.1], [15, 31], [None, "balanced"])]


def build_preprocessor(scale):
    # كل الإحصاءات (الوسيط، المتوسط، الفئات) تتعلمها fit على ما يمر بها،
    # وداخل Pipeline لا يمر بها إلا التدريب.
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist",
                                 min_frequency=100, sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def make_baseline():
    # يتنبأ دائماً بالفئة الأكثر تكراراً، واحتماله = نسبة التأخر في التدريب
    return Pipeline([
        ("prep", build_preprocessor(scale=False)),
        ("model", DummyClassifier(strategy="prior")),
    ])


def make_logistic_pipeline(C=1.0, class_weight=None):
    return Pipeline([
        ("prep", build_preprocessor(scale=True)),     # الخطي يحتاج تحجيماً
        ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=2000,
                                     random_state=config.RANDOM_SEED)),
    ])


def make_hgb_pipeline(learning_rate=0.1, max_leaf_nodes=31, class_weight=None):
    return Pipeline([
        ("prep", build_preprocessor(scale=False)),    # الأشجار لا تتأثر بالتحجيم
        ("model", HistGradientBoostingClassifier(
            learning_rate=learning_rate, max_leaf_nodes=max_leaf_nodes,
            max_iter=200, l2_regularization=1.0, class_weight=class_weight,
            early_stopping=False, random_state=config.RANDOM_SEED)),
    ])


def describe(params):
    return ", ".join(f"{k}={v}" for k, v in params.items())


def score_probabilities(y_true, proba):
    return {"average_precision": average_precision_score(y_true, proba),
            "roc_auc": roc_auc_score(y_true, proba)}


def run_experiments(train, val):
    # يدرب كل مرشح على train ويقيسه على val. لا توجد وسيطة للاختبار عمداً.
    X_tr, y_tr = get_xy(train)
    X_va, y_va = get_xy(val)
    val_prevalence = float(y_va.mean())

    candidates = [(BASELINE, "strategy=prior", make_baseline())]
    candidates += [(LOGISTIC, describe(p), make_logistic_pipeline(**p)) for p in LR_GRID]
    candidates += [(HGB, describe(p), make_hgb_pipeline(**p)) for p in HGB_GRID]

    rows, fitted, probas = [], [], []
    for name, settings, pipe in candidates:
        start = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        seconds = time.perf_counter() - start

        p_val = pipe.predict_proba(X_va)[:, 1]
        p_train = pipe.predict_proba(X_tr)[:, 1]
        val_scores = score_probabilities(y_va, p_val)
        train_scores = score_probabilities(y_tr, p_train)

        rows.append({
            "model": name,
            "settings": settings,
            "val_average_precision": round(val_scores["average_precision"], 4),
            "val_roc_auc": round(val_scores["roc_auc"], 4),
            "ap_lift": round(val_scores["average_precision"] / val_prevalence, 3),
            "train_average_precision": round(train_scores["average_precision"], 4),
            "fit_seconds": round(seconds, 1),
        })
        fitted.append(pipe)
        probas.append(p_val)

    log = pd.DataFrame(rows)

    # أفضل إعداد لكل نموذج = الأعلى AP على التحقق
    best_index = {m: g["val_average_precision"].idxmax() for m, g in log.groupby("model")}
    log["selected"] = log.index.isin(list(best_index.values()))
    selected = {m: (fitted[i], log.loc[i, "settings"], probas[i])
                for m, i in best_index.items()}
    return log, selected


def save_models(selected):
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, (pipe, _settings, _proba) in selected.items():
        joblib.dump(pipe, config.MODELS_DIR / f"{name}.joblib")


def load_model(name):
    path = config.MODELS_DIR / f"{name}.joblib"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python run_day7.py")
    return joblib.load(path)


def save_val_predictions(val, selected):
    out = val[["order_id", TARGET]].copy()
    for name, (_pipe, _settings, proba) in selected.items():
        out[f"proba_{name}"] = proba
    out.to_csv(config.VAL_PREDICTIONS_FILE, index=False)
    return out


def learned_medians(pipe):
    # الأوساط التي تعلمها خط المعالجة من التدريب (لإثبات ضبطه على التدريب فقط)
    imputer = pipe.named_steps["prep"].named_transformers_["num"].named_steps["impute"]
    return dict(zip(NUMERIC_FEATURES, imputer.statistics_))
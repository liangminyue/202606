# -*- coding: utf-8 -*-
"""
Streamlit Web应用 - 机器学习预测工具（SVR优化版）

针对SVR模型的特殊优化：
1. 从模型训练数据中提取特征范围
2. 使用训练数据作为SHAP背景数据
3. 添加特征值验证和提示
4. 使用TreeExplainer替代KernelExplainer（如果可能）
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import re
import warnings
warnings.filterwarnings('ignore')



def _patch_gradient_boosting_loss(model):
    """
    修复 scikit-learn 版本不一致导致的预测失败。

    旧版 scikit-learn（本项目序列化环境为 1.3.2）保存的 GradientBoosting 模型，
    在新版 sklearn 中反序列化后，其损失对象（如 LeastSquaresError）会丢失
    `get_init_raw_predictions` 等方法，predict 时抛出
    AttributeError: 'LeastSquaresError' object has no attribute 'get_init_raw_predictions'。

    该函数为缺失该方法的损失对象所属类补齐一个等价于官方实现的
    get_init_raw_predictions，使预测可正常进行（无需重训或降级 sklearn）。
    """
    try:
        from sklearn.ensemble import (
            GradientBoostingRegressor, GradientBoostingClassifier,
        )
    except Exception:
        return

    estimators = []
    if hasattr(model, 'named_steps'):
        estimators.extend(model.named_steps.values())
    if isinstance(model, (GradientBoostingRegressor, GradientBoostingClassifier)):
        estimators.append(model)

    for est in estimators:
        if not isinstance(est, (GradientBoostingRegressor, GradientBoostingClassifier)):
            continue
        loss = getattr(est, 'loss_', None)
        if loss is None or hasattr(loss, 'get_init_raw_predictions'):
            continue

        def get_init_raw_predictions(self, X, estimator):
            raw = np.asarray(estimator.predict(X), dtype=np.float64)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            if raw.shape[1] == 1:
                return raw
            if raw.shape[1] == 2:
                return raw[:, 1:2]
            return raw

        type(loss).get_init_raw_predictions = get_init_raw_predictions


# 设置页面配置
st.set_page_config(
    page_title="基于机器学习的地贫输血疗效血红蛋白智能测算工具",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
def load_custom_css():
    """加载自定义CSS样式"""
    st.markdown("""
    <style>
    .stApp { background: #f5f5f5; min-height: 100vh; }
    .main-header {
        font-size: 1.8rem;
        font-weight: 600;
        text-align: center;
        padding: 0.8rem;
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
        border-radius: 8px;
        color: white;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(67, 56, 202, 0.25);
    }
    .card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
        margin: 0.5rem 0;
        border: 1px solid #e5e5e5;
    }
    .prediction-card {
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(67, 56, 202, 0.25);
    }
    .prediction-value { font-size: 2.5rem; font-weight: 600; margin: 0.3rem 0; }
    .stButton > button {
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.15s ease;
        border: none;
        padding: 0.45rem 1rem;
        font-size: 0.875rem;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 1px 4px rgba(67, 56, 202, 0.2);
    }
    .stNumberInput > div > div > input {
        border-radius: 6px;
        border: 1px solid #e5e5e5;
        transition: all 0.15s ease;
        padding: 0.35rem;
        font-size: 0.875rem;
    }
    .stNumberInput > div > div > input:focus {
        border-color: #4338ca;
        box-shadow: 0 0 0 2px rgba(67, 56, 202, 0.08);
        outline: none;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e5e5;
    }
    .stDataFrame { border-radius: 6px; overflow: hidden; box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02); }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        padding: 0.2rem;
        background: #f0f0f0;
        border-radius: 6px;
        margin-bottom: 0.8rem;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 4px;
        padding: 6px 12px;
        background-color: #e5e5e5;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
        color: white;
    }
    [data-testid="stMetric"] {
        background: white;
        border-radius: 6px;
        padding: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        border: 1px solid #e5e5e5;
    }
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 6px;
        padding: 0.6rem;
        border: 1px solid transparent;
        font-size: 0.875rem;
    }
    .stSuccess { background: #f0fdf4; border-color: #86efac; }
    .stWarning { background: #fffbeb; border-color: #fde047; }
    .stError { background: #fef2f2; border-color: #fca5a5; }
    .stInfo { background: #eff6ff; border-color: #93c5fd; }
    .stProgress > div > div {
        border-radius: 4px;
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
    }
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: #f0f0f0; border-radius: 2px; }
    ::-webkit-scrollbar-thumb { background: #d4d4d4; border-radius: 2px; }
    h2 { font-size: 1.1rem; color: #333; font-weight: 600; margin-bottom: 0.6rem; }
    hr { border: none; height: 1px; background: #e5e5e5; margin: 0.6rem 0; }
    .stMarkdown, .stText { font-size: 0.875rem; }
    .stLabel { font-size: 0.825rem; margin-bottom: 0.25rem; }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# Web应用配置路径（部署优化：相对路径 + 环境变量覆盖，便于部署到 Streamlit Cloud）
import os
_MODEL_DIR_ENV = os.environ.get("STREAMLIT_MODEL_DIR")

def _resolve_web_dir():
    """
    按优先级定位模型目录（需包含 *.pkl 与特征 *.csv）。
    设计目标：streamlit/ 文件夹可整体直接部署到 Streamlit Cloud，
    因此首选「应用同目录下的 models/」（即 streamlit/models/，已随仓库提交）。

    优先级：
      1. 环境变量 STREAMLIT_MODEL_DIR（最高优先，便于云端覆盖路径）
      2. 应用同目录 models/      -> <本文件所在目录>/models   （部署首选）
      3. 仓库根 models/          -> <本文件上级目录>/models
      4. 本地调试：仓库根 多次实验结果/多次结果_<时间戳>/streamlit/models/
         （取名字最大者 = 最近一次实验运行，仅本地存在）
    """
    if _MODEL_DIR_ENV:
        return Path(_MODEL_DIR_ENV)
    base = Path(__file__).resolve().parent      # streamlit/
    repo_root = base.parent                      # 仓库根
    candidates = []
    # 2. 自包含部署目录（随仓库提交，Streamlit Cloud 必用此项）
    candidates.append(base / "models")
    # 3. 仓库根 models/
    candidates.append(repo_root / "models")
    # 4. 本地调试：实验产出目录
    results_root = repo_root / "多次实验结果"
    if results_root.exists():
        for rd in sorted(results_root.glob("多次结果_*"), reverse=True):
            d = rd / "streamlit" / "models"
            if d.exists():
                candidates.append(d)
    for d in candidates:
        if d.exists() and (list(d.glob("*.pkl")) or list(d.glob("*.csv"))):
            return d
    # 兜底：返回部署首选目录（即使为空），保持行为一致
    return base / "models"

WEB_DIR = _resolve_web_dir()

# 按文件名中的 R2 值排序，取效果最好的模型（如 高斯过程_R2_0.8935 > 梯度提升_R2_0.8775）
_R2_RE = re.compile(r"R2_(\d+(?:\.\d+)?)")
def _r2_of(path):
    m = _R2_RE.search(path.name)
    return float(m.group(1)) if m else -1.0

def find_model_file():
    """自动查找最佳模型文件（按文件名中的 R2 取最高）"""
    if not WEB_DIR.exists():
        return None
    pkl_files = list(WEB_DIR.glob("*.pkl"))
    if pkl_files:
        return max(pkl_files, key=_r2_of)
    return None

def find_features_file():
    """自动查找特征配置文件"""
    if not WEB_DIR.exists():
        return None
    csv_files = list(WEB_DIR.glob("*.csv"))
    if not csv_files:
        return None
    
    def priority_score(filename):
        score = 0
        if "特征筛选" in filename:
            score += 10
        if "A表7" in filename:
            score += 10
        if "特征" in filename:
            score += 5
        if "feature" in filename.lower():
            score += 5
        return score
    
    csv_files.sort(key=lambda p: priority_score(p.name), reverse=True)
    return csv_files[0]

MODEL_FILE = find_model_file()
FEATURES_FILE = find_features_file()


class ModelPredictor:
    """模型预测器类 - 针对SVR优化"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.feature_names_out = []
        self.model_info = {}
        self.model_loaded = False
        self._shap_explainer = None
        self._shap_background_df = None
        self._force_recompute_shap = True
        self._training_data = None  # 存储训练数据用于SHAP背景
        self._feature_ranges = {}   # 存储特征范围
        
    def load_model(self, model_path=None):
        """加载模型（自动跳过无法反序列化的旧版本模型，使用首个可成功加载的模型）"""
        if model_path is not None:
            candidates = [Path(model_path)]
        else:
            # 按 R2 降序排列，优先尝试效果最好的模型
            candidates = sorted(WEB_DIR.glob("*.pkl"), key=_r2_of, reverse=True)
            if not candidates:
                candidates = [None]

        last_err = None
        for mp in candidates:
            if mp is None or not mp.exists():
                continue
            try:
                self.model = joblib.load(mp)
                # 修复可能存在的 sklearn 版本不一致导致的损失对象方法缺失
                _patch_gradient_boosting_loss(self.model)
                self.model_loaded = True

                # 提取模型信息
                if hasattr(self.model, 'named_steps'):
                    if 'preprocessor' in self.model.named_steps:
                        self.preprocessor = self.model.named_steps['preprocessor']

                        # 提取特征范围（从预处理器）
                        self._extract_feature_ranges()

                    if 'model' in self.model.named_steps:
                        model_obj = self.model.named_steps['model']
                        self.model_info['model_type'] = type(model_obj).__name__

                        if hasattr(model_obj, 'feature_importances_'):
                            self.model_info['feature_importances'] = model_obj.feature_importances_
                        elif hasattr(model_obj, 'coef_'):
                            self.model_info['feature_importances'] = np.abs(model_obj.coef_)

                # 尝试获取特征名称
                self._extract_feature_names()

                # 如果没有从CSV获取特征名，尝试从模型获取
                if len(self.feature_names) == 0:
                    self._extract_feature_names_from_model()

                # 提取特征范围
                self._extract_feature_ranges_from_model()

                # 从文件名中提取R2值
                import re
                match = re.search(r'R2_(\d+\.\d+)', mp.name)
                if match:
                    self.model_info['train_r2'] = float(match.group(1))
                self.model_info['model_path'] = str(mp)

                return True
            except Exception as e:
                # 单个模型（如旧版本 sklearn 序列化产物）加载失败时跳过，尝试下一个
                last_err = e
                continue

        if last_err is not None:
            st.error(f"加载模型失败: {str(last_err)}")
        else:
            st.error("未找到可用的模型文件(.pkl)，请确保web文件夹中存在模型文件")
        return False
    
    def _extract_feature_ranges(self):
        """从预处理器提取特征范围"""
        if self.preprocessor is None:
            return
        
        # 尝试从MinMaxScaler提取范围
        if hasattr(self.preprocessor, 'named_transformers_'):
            for name, transformer in self.preprocessor.named_transformers_.items():
                if hasattr(transformer, 'data_min_') and hasattr(transformer, 'data_max_'):
                    # 获取对应特征名
                    if hasattr(self.preprocessor, 'feature_names_in_'):
                        feature_names = self.preprocessor.feature_names_in_
                        if hasattr(feature_names, 'tolist'):
                            feature_names = feature_names.tolist()
                        else:
                            feature_names = list(feature_names)
                        
                        for i, feat in enumerate(feature_names):
                            if i < len(transformer.data_min_):
                                self._feature_ranges[feat] = {
                                    'min': transformer.data_min_[i],
                                    'max': transformer.data_max_[i]
                                }
    
    def _extract_feature_ranges_from_model(self):
        """从模型提取特征范围（备用方法）"""
        if self.model is None:
            return
        
        # 如果是Pipeline，尝试从预处理器获取
        if hasattr(self.model, 'named_steps') and 'preprocessor' in self.model.named_steps:
            preprocessor = self.model.named_steps['preprocessor']
            if hasattr(preprocessor, 'data_min_') and hasattr(preprocessor, 'data_max_'):
                feature_names = self.feature_names
                data_min = preprocessor.data_min_
                data_max = preprocessor.data_max_
                
                for i, feat in enumerate(feature_names):
                    if i < len(data_min):
                        self._feature_ranges[feat] = {
                            'min': data_min[i],
                            'max': data_max[i]
                        }
    
    def get_feature_range_info(self, feature_name):
        """获取特征范围信息"""
        if feature_name in self._feature_ranges:
            return self._feature_ranges[feature_name]
        return None
    
    def validate_feature_value(self, feature_name, value):
        """验证特征值是否在合理范围内"""
        range_info = self.get_feature_range_info(feature_name)
        if range_info:
            min_val = range_info['min']
            max_val = range_info['max']
            if value < min_val or value > max_val:
                return {
                    'valid': False,
                    'min': min_val,
                    'max': max_val,
                    'value': value,
                    'message': f"值 {value} 超出训练数据范围 [{min_val:.2f}, {max_val:.2f}]"
                }
        return {'valid': True}
    
    def _extract_feature_names(self):
        """提取特征名称"""
        # 首先尝试从CSV文件加载特征名
        csv_feature_names, csv_defaults = load_features_from_csv()
        if csv_feature_names and len(csv_feature_names) > 0:
            self.feature_names = csv_feature_names
            self.model_info['feature_names'] = csv_feature_names
            self.model_info['feature_defaults'] = csv_defaults
            return
        
        # 尝试从预处理器获取
        if self.preprocessor is not None:
            if hasattr(self.preprocessor, 'feature_names_in_'):
                try:
                    feature_names_in = self.preprocessor.feature_names_in_
                    if hasattr(feature_names_in, 'tolist'):
                        self.feature_names = feature_names_in.tolist()
                    else:
                        self.feature_names = list(feature_names_in)
                    self.model_info['feature_names'] = self.feature_names
                    return
                except:
                    pass
            
            if hasattr(self.preprocessor, 'get_feature_names_out'):
                try:
                    feature_names_out = self.preprocessor.get_feature_names_out()
                    if hasattr(feature_names_out, 'tolist'):
                        self.feature_names_out = feature_names_out.tolist()
                    else:
                        self.feature_names_out = list(feature_names_out)
                    if not self.feature_names and self.feature_names_out:
                        self.feature_names = self.feature_names_out
                        self.model_info['feature_names'] = self.feature_names
                    return
                except:
                    pass
        
        self._extract_feature_names_from_model()
    
    def _extract_feature_names_from_model(self):
        """从模型提取特征名"""
        if self.model is None:
            return
        
        if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
            model_obj = self.model.named_steps['model']
            if hasattr(model_obj, 'feature_names_in_'):
                try:
                    model_feature_names = model_obj.feature_names_in_
                    if hasattr(model_feature_names, 'tolist'):
                        self.feature_names = model_feature_names.tolist()
                    else:
                        self.feature_names = list(model_feature_names)
                    self.model_info['feature_names'] = self.feature_names
                    return
                except:
                    pass
        
        if hasattr(self.model, 'feature_names_in_'):
            try:
                model_feature_names = self.model.feature_names_in_
                if hasattr(model_feature_names, 'tolist'):
                    self.feature_names = model_feature_names.tolist()
                else:
                    self.feature_names = list(model_feature_names)
                self.model_info['feature_names'] = self.feature_names
                return
            except:
                pass
        
        # 从coef_或feature_importances_推断
        if hasattr(self.model, 'coef_'):
            n_features = self.model.coef_.shape[-1] if hasattr(self.model.coef_, 'shape') else len(self.model.coef_)
            self.feature_names = [f"特征_{i}" for i in range(n_features)]
            self.model_info['feature_names'] = self.feature_names
        elif hasattr(self.model, 'feature_importances_'):
            n_features = len(self.model.feature_importances_)
            self.feature_names = [f"特征_{i}" for i in range(n_features)]
            self.model_info['feature_names'] = self.feature_names
    
    def _prepare_input_dataframe(self, input_data):
        """准备输入DataFrame"""
        expected_features = self.feature_names
        
        if not expected_features:
            raise ValueError("未定义特征名称，请检查模型加载")
        
        if isinstance(input_data, dict):
            input_dict = {}
            for feat in expected_features:
                input_dict[feat] = input_data.get(feat, 0.0)
            return pd.DataFrame([input_dict])
        
        elif isinstance(input_data, list):
            if all(isinstance(item, dict) for item in input_data):
                return self._prepare_input_dataframe(input_data[0])
            
            if len(input_data) == len(expected_features):
                return pd.DataFrame([input_data], columns=expected_features)
            else:
                if len(input_data) < len(expected_features):
                    padded = list(input_data) + [0.0] * (len(expected_features) - len(input_data))
                    return pd.DataFrame([padded], columns=expected_features)
                else:
                    return pd.DataFrame([input_data[:len(expected_features)]], columns=expected_features)
        
        elif isinstance(input_data, pd.DataFrame):
            result_df = input_data.copy()
            matched_cols = [col for col in expected_features if col in result_df.columns]
            
            if len(matched_cols) == len(expected_features):
                return result_df[expected_features]
            else:
                for col in expected_features:
                    if col not in result_df.columns:
                        result_df[col] = 0.0
                result_df = result_df[expected_features]
                return result_df
        
        else:
            raise ValueError(f"不支持的输入数据类型: {type(input_data)}")
    
    def predict_with_validation(self, input_data):
        """带验证的预测"""
        # 验证输入值
        warnings_list = []
        for feat in self.feature_names:
            value = input_data.get(feat, 0.0) if isinstance(input_data, dict) else 0.0
            validation = self.validate_feature_value(feat, value)
            if not validation['valid']:
                warnings_list.append(validation['message'])
        
        # 执行预测
        prediction = self.predict(input_data)
        
        return prediction, warnings_list
    
    def predict(self, input_data):
        """预测单个样本"""
        if not self.model_loaded:
            raise ValueError("模型未加载")
        
        try:
            input_df = self._prepare_input_dataframe(input_data)
            # 兜底：确保反序列化的损失对象具备预测所需方法
            _patch_gradient_boosting_loss(self.model)
            prediction = self.model.predict(input_df)
            return prediction[0]
        except Exception as e:
            raise ValueError(f"预测失败: {str(e)}")
    
    def predict_batch(self, input_df):
        """批量预测"""
        if not self.model_loaded:
            raise ValueError("模型未加载")
        
        try:
            prepared_df = self._prepare_input_dataframe(input_df)
            _patch_gradient_boosting_loss(self.model)
            predictions = self.model.predict(prepared_df)
            return predictions
        except Exception as e:
            raise ValueError(f"批量预测失败: {str(e)}")
    
    def _build_shap_background(self, n=50):
        """从模型内部保存的训练点构建 SHAP 背景数据

        本模型管道无缩放器（num=KNNImputer、bin=SimpleImputer），GPR 的 X_train_
        即插补后的原始值空间。直接用真实训练点做背景最贴近训练分布；
        用随机范围采样会严重偏离分布，GPR 输出近似常数（先验均值），SHAP 全为 0。
        注意：X_train_ 是 ColumnTransformer 输出列顺序，需用 get_feature_names_out()
        去前缀标注后重排为 self.feature_names（模型输入顺序），否则贡献张冠李戴。
        """
        try:
            import numpy as np
            if self.model is None:
                return None
            steps = getattr(self.model, 'named_steps', None)
            est = steps['model'] if steps and 'model' in steps else self.model
            if not hasattr(est, 'X_train_'):
                return None
            Xt = np.asarray(est.X_train_)
            if Xt.ndim != 2 or Xt.shape[1] != len(self.feature_names):
                return None
            pre = steps['preprocessor'] if steps and 'preprocessor' in steps else None
            cols = list(self.feature_names)
            if pre is not None and hasattr(pre, 'get_feature_names_out'):
                try:
                    out_cols = [str(cn).split('__', 1)[-1] for cn in pre.get_feature_names_out()]
                    if len(out_cols) == Xt.shape[1]:
                        cols = out_cols
                except Exception:
                    pass
            df = pd.DataFrame(Xt, columns=cols)
            missing = [c for c in self.feature_names if c not in df.columns]
            if missing:
                return None
            df = df[self.feature_names]
            if df.shape[0] > n:
                np.random.seed(42)
                idx = np.random.choice(df.shape[0], n, replace=False)
                np.random.seed(None)
                df = df.iloc[idx].reset_index(drop=True)
            return df
        except Exception:
            return None

    def get_shap_values(self, input_data):

        """获取SHAP解释值 - SVR优化版"""
        try:
            import shap
            
            # 准备输入数据
            input_df = self._prepare_input_dataframe(input_data)

            # 兜底：确保反序列化的损失对象具备预测所需方法（SHAP 也会调用 model.predict）
            _patch_gradient_boosting_loss(self.model)

            # 定义完整的预测函数
            def full_model_predict(X):
                """完整的预测函数"""
                if isinstance(X, np.ndarray):
                    if X.shape[1] == len(self.feature_names):
                        X = pd.DataFrame(X, columns=self.feature_names)
                    else:
                        try:
                            X = pd.DataFrame(X, columns=self.feature_names[:X.shape[1]])
                        except:
                            X = pd.DataFrame(X)
                elif isinstance(X, list):
                    if len(X) == len(self.feature_names):
                        X = pd.DataFrame([X], columns=self.feature_names)
                    else:
                        X = pd.DataFrame(X)
                
                try:
                    X_prepared = self._prepare_input_dataframe(X)
                    return self.model.predict(X_prepared)
                except:
                    return self.model.predict(X)
            
            # 创建背景数据：优先使用模型保存的真实训练点（分布内样本，SHAP 才有意义）
            # 用随机范围采样会严重偏离训练分布，GPR 输出近似常数，导致 SHAP 全为 0。
            if self._shap_explainer is None or self._force_recompute_shap:
                bg_df = self._build_shap_background(n=50)
                if bg_df is None:
                    st.warning("SHAP 背景数据构建失败，跳过 SHAP 分析")
                    return None, None, None
                self._shap_background_df = bg_df
                self._shap_explainer = shap.KernelExplainer(full_model_predict, bg_df)
                self._force_recompute_shap = False
            
            # 计算SHAP值
            np.random.seed(42)
            shap_values = self._shap_explainer.shap_values(input_df, nsamples=100)  # 增加采样数
            np.random.seed(None)
            
            # 处理SHAP值格式
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                if shap_values.shape[0] == 1:
                    shap_values = shap_values[0]
            
            # 调整SHAP值长度
            if len(shap_values) != len(self.feature_names):
                if len(shap_values) > len(self.feature_names):
                    shap_values = shap_values[:len(self.feature_names)]
                elif len(shap_values) < len(self.feature_names):
                    shap_values = list(shap_values) + [0] * (len(self.feature_names) - len(shap_values))
            
            return self._shap_explainer, shap_values, self.feature_names
        except Exception as e:
            st.warning(f"SHAP计算失败: {str(e)}")
            return None, None, None


def load_features_from_csv():
    """从特征CSV文件动态加载特征列表

    特征表同时含「特征名称」（显示名，如 血型（A型））与「原始特征名称」
    （模型输入列名，如 血型_0）。模型期望的输入列是「原始特征名称」，因此：
      - feature_names 使用原始特征名称（模型输入列）
      - feature_defaults 附带 display（显示名），并识别独热编码列（xx_数字）以便分组控件
    """
    if FEATURES_FILE is None or not FEATURES_FILE.exists():
        return None, None

    try:
        features_df = pd.read_csv(FEATURES_FILE, encoding='utf-8-sig')

        header = [str(h).strip() for h in features_df.columns]
        raw_col_idx = None
        disp_col_idx = None
        for i, h in enumerate(header):
            if '原始特征名称' in h:
                raw_col_idx = i
            elif '特征名称' in h:
                disp_col_idx = i
        if raw_col_idx is None:
            raw_col_idx = disp_col_idx if disp_col_idx is not None else (1 if len(features_df.columns) >= 2 else 0)

        def _clean(series):
            return [str(f).strip() for f in series if pd.notna(f) and str(f).strip()]

        raw_names = _clean(features_df.iloc[:, raw_col_idx].tolist())
        disp_names = _clean(features_df.iloc[:, disp_col_idx].tolist()) if disp_col_idx is not None else list(raw_names)

        # 过滤标题行
        title_keywords = ['特征', '名称', '变量', 'feature', 'name', 'variable']
        if raw_names:
            first_few = raw_names[:3]
            if any(str(f).lower() in title_keywords for f in first_few if f):
                raw_names = raw_names[1:]
                if len(disp_names) > len(raw_names):
                    disp_names = disp_names[1:]

        if not raw_names:
            return None, None

        if len(disp_names) < len(raw_names):
            disp_names = disp_names + raw_names[len(disp_names):]

        feature_names = raw_names
        display_map = {raw: disp for raw, disp in zip(raw_names, disp_names) if disp}

        feature_defaults = {}
        for feat in feature_names:
            feat_lower = str(feat).lower()

            default_config = {
                'default': 0.0,
                'min': 0.0,
                'max': 100.0,
                'step': 0.1,
                'type': 'numerical',
                'display': display_map.get(feat, feat)
            }

            # 独热编码列（如 血型_0 / 脾_2 / 基因_2）：0/1 开关 + 分组（用于下拉合并）
            if re.fullmatch(r'.+_[0-9]+', feat) and feat.rsplit('_', 1)[0] not in ('0', '1', '2', '3'):
                base = feat.rsplit('_', 1)[0]
                default_config.update({
                    'default': 0,
                    'min': 0,
                    'max': 1,
                    'step': 1,
                    'type': 'categorical',
                    'options': [0, 1],
                    'group': base,
                    'level': int(feat.rsplit('_', 1)[1]),
                })
            elif any(keyword in feat_lower for keyword in ['hb', 'hgb', '血红蛋白']):
                default_config.update({
                    'default': 80.0,
                    'min': 5.0,
                    'max': 200.0,
                    'step': 1.0
                })
            elif any(keyword in feat_lower for keyword in ['年龄', 'age']):
                default_config.update({
                    'default': 20.0,
                    'min': 0.0,
                    'max': 120.0,
                    'step': 1.0
                })
            elif any(keyword in feat_lower for keyword in ['身高', 'height']):
                default_config.update({
                    'default': 150.0,
                    'min': 50.0,
                    'max': 250.0,
                    'step': 1.0
                })
            elif any(keyword in feat_lower for keyword in ['体重', 'weight']):
                default_config.update({
                    'default': 45.0,
                    'min': 2.0,
                    'max': 200.0,
                    'step': 0.5
                })
            elif any(keyword in feat_lower for keyword in ['输血量', 'volume']):
                default_config.update({
                    'default': 2.0,
                    'min': 0.0,
                    'max': 20.0,
                    'step': 0.5
                })
            elif any(keyword in feat_lower for keyword in ['性别', 'sex', 'gender']):
                default_config.update({
                    'default': 1,
                    'min': 0,
                    'max': 2,
                    'step': 1,
                    'type': 'categorical',
                    'options': [0, 1, 2]
                })
            elif any(keyword in feat_lower for keyword in ['肝']):
                default_config.update({
                    'default': 0,
                    'min': 0,
                    'max': 1,
                    'step': 1,
                    'type': 'categorical',
                    'options': [0, 1]
                })

            feature_defaults[feat] = default_config

        return feature_names, feature_defaults
    except Exception as e:
        st.warning(f"读取特征文件失败: {str(e)}")
        return None, None
def create_input_widgets(feature_names, feature_defaults=None, predictor=None):
    """创建输入控件 - 带范围提示

    独热编码列（如 血型_0/血型_2、脾_0/脾_2）自动合并为下拉分组：
      - 同一基础名 >=2 个成员 -> 一个下拉框（如 血型（A型）/血型（O型）/未选择）
      - 单个成员（如 基因_2）-> 是否开关（否/是）
    其余特征保持原有数值/类别控件，控件标签使用 CSV 中的「特征名称」（显示名）。
    """
    inputs = {}
    cols = st.columns(3)

    groups = {}
    plain = []
    if feature_defaults:
        for feat in feature_names:
            cfg = feature_defaults.get(feat)
            if cfg and cfg.get('group'):
                groups.setdefault(cfg['group'], []).append((feat, cfg))
            else:
                plain.append((feat, cfg))
    else:
        plain = [(f, None) for f in feature_names]

    units = [('plain', feat, cfg) for feat, cfg in plain]
    for base in sorted(groups.keys()):
        units.append(('group', base, groups[base]))

    for i, unit in enumerate(units):
        col_idx = i % 3
        with cols[col_idx]:
            if unit[0] == 'group':
                base, members = unit[1], unit[2]
                if len(members) >= 2:
                    disp_options = [cfg.get('display', feat) for feat, cfg in members]
                    selected = st.selectbox(
                        base,
                        options=["未选择（其他）"] + disp_options,
                        key=f"input_group_{base}"
                    )
                    for feat, cfg in members:
                        inputs[feat] = 0
                    if selected != "未选择（其他）":
                        idx = disp_options.index(selected)
                        inputs[members[idx][0]] = 1
                else:
                    feat, cfg = members[0]
                    label = cfg.get('display', feat)
                    selected = st.selectbox(
                        label,
                        options=["否 (0)", "是 (1)"],
                        index=int(cfg.get('default', 0)),
                        key=f"input_solo_{feat}"
                    )
                    inputs[feat] = 1 if selected.startswith("是") else 0
                continue

            feature = unit[1]
            if feature_defaults and feature in feature_defaults:
                config = feature_defaults[feature]
                default_val = config.get('default', 0.0)
                min_val = config.get('min', 0.0)
                max_val = config.get('max', 100.0)
                step = config.get('step', 0.1)
                feature_type = config.get('type', 'numerical')
                options = config.get('options', None)
                display = config.get('display', feature)

                training_range = None
                if predictor:
                    training_range = predictor.get_feature_range_info(feature)

                if feature_type == 'categorical' and options:
                    option_labels = [str(opt) for opt in options]
                    selected_idx = options.index(default_val) if default_val in options else 0
                    selected_label = st.selectbox(
                        display,
                        options=option_labels,
                        index=selected_idx,
                        key=f"input_{feature}"
                    )
                    inputs[feature] = options[option_labels.index(selected_label)]
                else:
                    format_str = "%.1f" if step < 1 else "%.0f"
                    if training_range:
                        st.caption(f"训练范围: [{training_range['min']:.2f}, {training_range['max']:.2f}]")
                    col_label, col_input = st.columns([1, 2])
                    with col_label:
                        st.markdown(f"**{display[:12]}**")
                    with col_input:
                        inputs[feature] = st.number_input(
                            label=display,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default_val),
                            step=step,
                            format=format_str,
                            key=f"input_{feature}",
                            label_visibility="collapsed"
                        )
            else:
                feat_lower = str(feature).lower()
                if any(keyword in feat_lower for keyword in ['hb', 'hgb', '血红蛋白']):
                    min_val, max_val, step, default_val = 5.0, 200.0, 1.0, 80.0
                    format_str = "%.0f"
                elif any(keyword in feat_lower for keyword in ['年龄', 'age']):
                    min_val, max_val, step, default_val = 0.0, 120.0, 1.0, 20.0
                    format_str = "%.0f"
                elif any(keyword in feat_lower for keyword in ['身高', 'height']):
                    min_val, max_val, step, default_val = 50.0, 250.0, 1.0, 150.0
                    format_str = "%.0f"
                elif any(keyword in feat_lower for keyword in ['体重', 'weight']):
                    min_val, max_val, step, default_val = 2.0, 200.0, 0.5, 45.0
                    format_str = "%.1f"
                elif any(keyword in feat_lower for keyword in ['输血量', 'volume']):
                    min_val, max_val, step, default_val = 0.0, 20.0, 0.5, 2.0
                    format_str = "%.1f"
                else:
                    min_val, max_val, step, default_val = 0.0, 1000.0, 0.1, 0.0
                    format_str = "%.1f"

                training_range = None
                if predictor:
                    training_range = predictor.get_feature_range_info(feature)
                if training_range:
                    st.caption(f"训练范围: [{training_range['min']:.2f}, {training_range['max']:.2f}]")

                col_label, col_input = st.columns([1, 2])
                with col_label:
                    st.markdown(f"**{feature}**")
                with col_input:
                    inputs[feature] = st.number_input(
                        label=feature,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        step=step,
                        format=format_str,
                        key=f"input_{feature}",
                        label_visibility="collapsed"
                    )

    return inputs
def display_prediction_results(prediction, model_info, shap_explainer=None, shap_values=None, 
                               shap_feature_names=None, predictor=None, confidence_interval=None, 
                               hb_before=None, warnings_list=None):
    """显示预测结果"""
    st.markdown("---")
    
    # 显示警告
    if warnings_list:
        for warning in warnings_list:
            st.warning(f"⚠️ {warning}")
    
    # ==================== 预测结果 ====================
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
            <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">🔮 预测结果</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937;">{prediction:.2f} g/L</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        if hb_before is not None:
            delta_hb = prediction - hb_before
            if delta_hb >= 0:
                arrow = "↑"
                color = "#ef4444"
            else:
                arrow = "↓"
                color = "#3b82f6"
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
                <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">📊 HGB变化量</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: {color};">{abs(delta_hb):.2f} g/L {arrow}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
                <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">📊 HGB变化量</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937;">-</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        if confidence_interval:
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
                <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">📈 95%置信区间</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937;">[{confidence_interval[0]:.2f}, {confidence_interval[1]:.2f}]</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
                <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">📈 95%置信区间</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937;">-</div>
            </div>
            """, unsafe_allow_html=True)
    
    with col4:
        if 'train_r2' in model_info:
            st.markdown(f"""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
                <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">🎯 模型 R²</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937;">{model_info['train_r2']:.3f}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
                <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">🎯 模型 R²</div>
                <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937;">-</div>
            </div>
            """, unsafe_allow_html=True)
    
    # ==================== SHAP分析 ====================
    if shap_values is not None and shap_explainer is not None:
        shap_array = np.array(shap_values) if not isinstance(shap_values, list) else np.array(shap_values)
        if np.all(shap_array == 0):
            st.warning("⚠️ SHAP值全为0")
            st.info("""
            **可能的原因和解决方案：**
            
            1. **特征值超出训练范围**：请调整输入值到训练数据范围内（查看输入框下方的范围提示）
            
            2. **SVR模型特性**：SVR模型对输入变化不敏感，建议：
               - 尝试使用RandomForest或XGBoost模型
               - 使用不同的SHAP解释器
            
            3. **背景数据问题**：已使用训练数据范围生成背景样本
            
            **建议**：如果需要更敏感的特征贡献分析，请考虑使用树模型（RandomForest/XGBoost）重新训练。
            """)
        
        st.subheader("🔍 特征贡献分析")
        
        try:
            # 使用 Plotly 横向条形图：在浏览器端渲染，中文正常显示，
            # 无需依赖服务器上的中文字体（原 matplotlib + SimHei 在 Streamlit Cloud 上会显示方块）。
            # 获取特征名称
            if shap_feature_names:
                feature_names_list = shap_feature_names
            elif predictor and hasattr(predictor, 'feature_names'):
                feature_names_list = predictor.feature_names
            elif model_info and 'feature_names' in model_info:
                feature_names_list = model_info['feature_names']
            else:
                if hasattr(shap_values, '__len__'):
                    feature_names_list = [f"特征_{i}" for i in range(len(shap_values))]
                else:
                    feature_names_list = []

            # 处理SHAP值
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values

            if hasattr(shap_vals, 'tolist'):
                shap_vals = shap_vals.tolist()
            elif not isinstance(shap_vals, list):
                shap_vals = list(shap_vals) if hasattr(shap_vals, '__iter__') else [shap_vals]

            # 匹配特征名称和SHAP值
            if feature_names_list and len(feature_names_list) != len(shap_vals):
                if len(feature_names_list) > len(shap_vals):
                    feature_names_list = feature_names_list[:len(shap_vals)]
                else:
                    extra_names = [f"特征_{i}" for i in range(len(feature_names_list), len(shap_vals))]
                    feature_names_list = list(feature_names_list) + extra_names

            if feature_names_list and len(feature_names_list) == len(shap_vals):
                contrib_df = pd.DataFrame({
                    '特征': feature_names_list,
                    'SHAP值': shap_vals
                })
                contrib_df = contrib_df.sort_values('SHAP值', key=abs, ascending=False)
                contrib_df['贡献方向'] = contrib_df['SHAP值'].apply(
                    lambda x: '⬆️ 正向' if x > 0 else ('⬇️ 负向' if x < 0 else '➡️ 无影响')
                )
                contrib_df['SHAP值'] = contrib_df['SHAP值'].round(4)

                colors = ['#2ecc71' if v > 0 else '#e74c3c' if v < 0 else '#95a5a6' for v in contrib_df['SHAP值']]
                fig = go.Figure(go.Bar(
                    x=contrib_df['SHAP值'],
                    y=contrib_df['特征'],
                    orientation='h',
                    marker_color=colors,
                    text=contrib_df['SHAP值'].apply(lambda v: f'{v:.4f}'),
                    textposition='outside',
                ))
                fig.update_layout(
                    title='特征贡献分析',
                    xaxis_title='SHAP值',
                    yaxis=dict(autorange='reversed'),
                    height=max(300, len(contrib_df) * 60),
                    margin=dict(l=120, r=40, t=50, b=40),
                    font=dict(family='Microsoft YaHei, SimHei, PingFang SC, sans-serif', size=13),
                )
                fig.add_vline(x=0, line_color='black', line_width=0.5)
                st.plotly_chart(fig, use_container_width=True)
                
                st.subheader("📋 特征贡献明细")
                st.dataframe(contrib_df, use_container_width=True)
            else:
                st.info(f"特征名称数量与SHAP值数量不匹配")
                
        except Exception as e:
            st.info(f"SHAP可视化失败: {str(e)}")
    
    # ==================== 预测值可视化 ====================
    st.subheader("📊 Hb 输血前后可视化")
    
    fig_gauge = go.Figure(go.Indicator(
        mode="number+gauge",
        value=prediction,
        domain={'x': [0, 1], 'y': [0.2, 0.8]},
        title={'text': "预测值", 'font': {'size': 16}, 'align': 'center'},
        number={'valueformat': '.2f', 'suffix': ' g/L', 'font': {'size': 28, 'color': "#4f46e5"}},
        gauge={
            'shape': "bullet",
            'axis': {'range': [50, 200], 'tickwidth': 1, 'tickcolor': "#6b7280", 'tickfont': {'size': 12}},
            'bar': {'color': "#4f46e5", 'thickness': 0.7},
            'steps': [
                {'range': [50, 90], 'color': '#fee2e2'},
                {'range': [90, 160], 'color': '#d1fae5'},
                {'range': [160, 200], 'color': '#fef3c7'}
            ],
            'threshold': {
                'line': {'color': "#4f46e5", 'width': 3},
                'thickness': 0.75,
                'value': prediction
            }
        }
    ))
    
    if hb_before is not None:
        try:
            hb_before_ratio = (hb_before - 50) / (200 - 50)
            fig_gauge.add_shape(
                type="line",
                x0=hb_before_ratio, y0=0.2,
                x1=hb_before_ratio, y1=0.8,
                line=dict(color="#ef4444", width=3, dash="dash"),
            )
            fig_gauge.add_annotation(
                x=hb_before_ratio, y=0.85,
                text=f"输血前: {hb_before:.2f}",
                showarrow=False,
                font={'size': 11, 'color': "#ef4444"},
                xanchor='center', yanchor='bottom'
            )
            delta_hb = prediction - hb_before
            delta_color = "#ef4444" if delta_hb >= 0 else "#3b82f6"
            delta_arrow = "↑" if delta_hb >= 0 else "↓"
            fig_gauge.add_annotation(
                x=1.0, y=0.5,
                text=f"{delta_hb:+.2f} g/L {delta_arrow}",
                showarrow=False,
                font={'size': 16, 'color': delta_color},
                xanchor='left', yanchor='middle'
            )
        except:
            pass
    
    fig_gauge.update_layout(height=120, margin=dict(l=20, r=120, t=40, b=30), showlegend=False)
    st.plotly_chart(fig_gauge)


def display_model_info(model_info, predictor=None):
    """显示模型信息"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 模型信息")
    
    if model_info:
        if 'model_type' in model_info:
            st.sidebar.write(f"**模型类型:** {model_info['model_type']}")
            
            # SVR特殊提示
            if 'SVR' in model_info['model_type']:
                st.sidebar.info("💡 SVR模型对输入变化不敏感，SHAP值可能趋近于0")
        
        if 'feature_importances' in model_info and model_info['feature_importances'] is not None:
            st.sidebar.write("**特征重要性**")
            
            importances = model_info['feature_importances']
            
            if 'feature_names' in model_info:
                names = model_info['feature_names']
            else:
                names = [f"特征_{i}" for i in range(len(importances))]
            
            imp_df = pd.DataFrame({
                '特征': names[:len(importances)],
                '重要性': importances
            })
            imp_df = imp_df.sort_values('重要性', ascending=False).head(10)
            
            for _, row in imp_df.iterrows():
                st.sidebar.progress(
                    float(row['重要性'] / imp_df['重要性'].max()),
                    text=f"{row['特征']}: {row['重要性']:.3f}"
                )
        
        if 'model_path' in model_info:
            st.sidebar.write(f"**模型文件:** {Path(model_info['model_path']).name}")
        
        if 'feature_names' in model_info and model_info['feature_names']:
            st.sidebar.write(f"**特征数量:** {len(model_info['feature_names'])}")
            with st.sidebar.expander("查看特征列表"):
                for feat in model_info['feature_names']:
                    # 显示特征范围
                    range_info = None
                    if predictor:
                        range_info = predictor.get_feature_range_info(feat)
                    if range_info:
                        st.write(f"- {feat} [{range_info['min']:.2f}, {range_info['max']:.2f}]")
                    else:
                        st.write(f"- {feat}")


def load_model_for_app():
    """加载模型"""
    predictor = ModelPredictor()
    
    model_path = find_model_file()
    
    if model_path is None:
        return None, "未找到模型文件，请确保web文件夹中存在.pkl模型文件"
    
    success = predictor.load_model(model_path)
    
    if success:
        predictor.model_info['model_path'] = str(model_path)
        
        feature_names, feature_defaults = load_features_from_csv()
        
        if feature_names:
            predictor.feature_names = feature_names
            predictor.model_info['feature_defaults'] = feature_defaults
            predictor.model_info['feature_names'] = feature_names
        
        return predictor, f"成功加载模型: {model_path.name}"
    else:
        return None, "模型加载失败"


def main():
    """主函数"""
    
    # ==================== 初始化Session State ====================
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    
    # ==================== 标题区域 ====================
    st.markdown("""
    <div class="main-header">
        🏥 基于机器学习的地贫输血疗效血红蛋白智能测算工具
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== 侧边栏 ====================
    st.sidebar.title("⚙️ 设置")
    
    with st.sidebar.expander("📖 使用帮助"):
        st.markdown("""
        **单样本预测：**
        1. 输入患者特征值
        2. 点击"预测"按钮
        3. 查看预测结果和SHAP分析
        
        **批量预测：**
        1. 准备CSV文件（包含所有特征列）
        2. 上传文件
        3. 点击"执行批量预测"
        4. 下载结果
        
        **📌 注意事项：**
        - SVR模型对输入变化不敏感
        - 建议输入值在训练数据范围内
        - 每个输入框下方显示了训练数据范围
        """)
    
    # ==================== 加载模型 ====================
    with st.spinner("正在加载模型..."):
        predictor, load_msg = load_model_for_app()
    
    if predictor is None or not predictor.model_loaded:
        st.error(f"❌ {load_msg}")
        st.info("请确保web文件夹中存在模型文件(.pkl)和特征配置文件(.csv)")
        return
    
    st.sidebar.success(f"✅ {load_msg}")
    
    # 显示模型信息
    display_model_info(predictor.model_info, predictor)
    
    # 获取特征名称
    feature_names = predictor.feature_names
    
    if len(feature_names) == 0:
        st.warning("未能获取特征名称，请检查模型")
        return
    
    # ==================== 主界面 ====================
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 单样本预测", "📤 批量预测", "📊 模型信息", "📜 预测历史"])
    
    # Tab 1: 单样本预测
    with tab1:
        st.subheader("输入患者信息")
        
        feature_defaults = predictor.model_info.get('feature_defaults', {})
        
        inputs = create_input_widgets(feature_names, feature_defaults, predictor)
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            predict_btn = st.button("🔮 预测", type="primary", use_container_width=True)
        
        with col2:
            reset_btn = st.button("🔄 重置", use_container_width=True)
            if reset_btn:
                predictor._force_recompute_shap = True
                st.rerun()
        
        if predict_btn:
            try:
                # 执行带验证的预测
                prediction, warnings_list = predictor.predict_with_validation(inputs)
                
                # 计算置信区间
                rmse = predictor.model_info.get('train_rmse', 5.0)
                confidence_interval = 1.96 * rmse
                lower_bound = prediction - confidence_interval
                upper_bound = prediction + confidence_interval
                
                # 获取SHAP值
                predictor._force_recompute_shap = True
                shap_explainer, shap_values, shap_feature_names = predictor.get_shap_values(inputs)
                
                # 保存预测历史
                history_record = {
                    '时间': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    '预测值': round(prediction, 2),
                    '置信区间下限': round(lower_bound, 2),
                    '置信区间上限': round(upper_bound, 2),
                    **inputs
                }
                st.session_state['prediction_history'].append(history_record)
                
                # 获取输血前Hb
                hb_before = None
                for feat in feature_names:
                    if any(keyword in str(feat).lower() for keyword in ['输血前', 'before', '前hb']):
                        hb_before = inputs.get(feat)
                        break
                
                display_prediction_results(
                    prediction, 
                    predictor.model_info,
                    shap_explainer,
                    shap_values,
                    shap_feature_names,
                    predictor,
                    confidence_interval=(lower_bound, upper_bound),
                    hb_before=hb_before,
                    warnings_list=warnings_list
                )
                
                with st.expander("📋 输入数据摘要"):
                    input_df = pd.DataFrame([inputs])
                    st.dataframe(input_df, use_container_width=True)
                    
            except Exception as e:
                st.error(f"预测失败: {str(e)}")
                import traceback
                st.error(f"详细错误: {traceback.format_exc()}")
    
    # Tab 2: 批量预测
    with tab2:
        st.subheader("📤 批量预测")
        st.markdown("上传CSV文件进行批量预测")
        
        template_df = pd.DataFrame(columns=feature_names)
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 下载CSV模板",
            data=template_csv,
            file_name="prediction_template.csv",
            mime="text/csv",
            help="下载模板文件，填写数据后上传进行批量预测"
        )
        
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="文件应包含所有特征列，列名与模型特征名称一致"
        )
        
        if uploaded_file is not None:
            try:
                input_df = None
                for encoding in ['utf-8', 'gbk', 'gb2312', 'utf-8-sig', 'latin1']:
                    try:
                        uploaded_file.seek(0)
                        input_df = pd.read_csv(uploaded_file, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                
                if input_df is None:
                    st.error("无法识别文件编码，请确保文件为UTF-8或GBK编码")
                    st.stop()
                
                missing_cols = set(feature_names) - set(input_df.columns)
                if missing_cols:
                    st.warning(f"缺少以下列: {missing_cols}")
                    st.info("将使用默认值填充")
                    for col in missing_cols:
                        input_df[col] = 0.0
                
                st.subheader("📊 数据预览")
                st.dataframe(input_df.head(10), use_container_width=True)
                st.caption(f"共 {len(input_df)} 行")
                
                if st.button("🚀 执行批量预测", type="primary"):
                    with st.spinner("正在预测..."):
                        predictions = predictor.predict_batch(input_df)
                        
                        result_df = input_df.copy()
                        result_df['预测值'] = predictions
                        
                        st.subheader("📊 预测结果")
                        st.dataframe(result_df, use_container_width=True)
                        
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 下载预测结果",
                            data=csv,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                        
                        st.subheader("📈 预测值分布")
                        fig = px.histogram(
                            result_df, 
                            x='预测值',
                            nbins=30,
                            title="预测值分布"
                        )
                        fig.update_layout(
                            xaxis_title="预测值",
                            yaxis_title="频数"
                        )
                        st.plotly_chart(fig)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("平均值", f"{predictions.mean():.2f}")
                        with col2:
                            st.metric("标准差", f"{predictions.std():.2f}")
                        with col3:
                            st.metric("范围", f"[{predictions.min():.2f}, {predictions.max():.2f}]")
                        
            except Exception as e:
                st.error(f"批量预测失败: {str(e)}")
    
    # Tab 3: 模型信息
    with tab3:
        st.subheader("📊 模型详细信息")
        
        if 'model_type' in predictor.model_info:
            st.write(f"**模型类型:** {predictor.model_info['model_type']}")
            if 'SVR' in predictor.model_info['model_type']:
                st.info("💡 **SVR模型说明**：支持向量回归模型对输入变化不敏感，SHAP值可能趋近于0。建议使用RandomForest或XGBoost模型获取更有意义的特征贡献分析。")
        
        st.subheader("📋 特征列表")
        st.write(f"共 {len(feature_names)} 个特征")
        
        # 显示特征范围
        st.subheader("📊 特征训练范围")
        range_data = []
        for feat in feature_names:
            range_info = predictor.get_feature_range_info(feat)
            if range_info:
                range_data.append({
                    '特征': feat,
                    '最小值': f"{range_info['min']:.2f}",
                    '最大值': f"{range_info['max']:.2f}"
                })
        if range_data:
            st.dataframe(pd.DataFrame(range_data), use_container_width=True)
        
        if 'feature_importances' in predictor.model_info:
            importances = predictor.model_info['feature_importances']
            if importances is not None:
                imp_df = pd.DataFrame({
                    '特征名称': feature_names[:len(importances)],
                    '重要性': importances
                })
                imp_df = imp_df.sort_values('重要性', ascending=False)
                
                fig = px.bar(
                    imp_df.head(20),
                    x='重要性',
                    y='特征名称',
                    orientation='h',
                    title="特征重要性排序",
                    color='重要性',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(
                    height=600,
                    yaxis={'categoryorder': 'total ascending'}
                )
                st.plotly_chart(fig)
        
        if 'model_path' in predictor.model_info:
            model_path = Path(predictor.model_info['model_path'])
            st.subheader("📁 模型文件")
            st.write(f"**路径:** {model_path}")
            st.write(f"**大小:** {model_path.stat().st_size / 1024:.2f} KB")
            st.write(f"**修改时间:** {pd.Timestamp.fromtimestamp(model_path.stat().st_mtime)}")
    
    # Tab 4: 预测历史
    with tab4:
        st.subheader("📜 预测历史记录")
        
        history_count = len(st.session_state['prediction_history'])
        st.info(f"📊 共保存了 {history_count} 条预测记录")
        
        if history_count > 0:
            history_df = pd.DataFrame(st.session_state['prediction_history'])
            
            st.dataframe(history_df, use_container_width=True)
            
            st.subheader("📈 预测值趋势")
            fig_trend = px.line(
                history_df,
                x='时间',
                y='预测值',
                title='预测值历史趋势',
                markers=True
            )
            fig_trend.update_traces(line_color='#667eea')
            st.plotly_chart(fig_trend)
            
            st.subheader("📊 预测值统计")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("平均值", f"{history_df['预测值'].mean():.2f}")
            with col2:
                st.metric("最大值", f"{history_df['预测值'].max():.2f}")
            with col3:
                st.metric("最小值", f"{history_df['预测值'].min():.2f}")
            with col4:
                st.metric("标准差", f"{history_df['预测值'].std():.2f}")
            
            st.subheader("📥 导出历史记录")
            csv_history = history_df.to_csv(index=False)
            st.download_button(
                label="下载预测历史记录 (CSV)",
                data=csv_history,
                file_name=f"prediction_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
            
            if st.button("🗑️ 清除历史记录", type="secondary"):
                st.session_state['prediction_history'] = []
                st.success("历史记录已清除")
                st.rerun()
        else:
            st.info("暂无预测历史记录，请先进行预测")


if __name__ == "__main__":
    main()
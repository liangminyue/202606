# -*- coding: utf-8 -*-
"""
Streamlit Web应用 - 机器学习预测工具

功能：
1. 加载训练好的最佳模型
2. 交互式输入特征值
3. 实时预测并显示结果
4. 显示模型性能指标
5. 支持批量预测（CSV上传）
6. 显示SHAP解释（如果可用）

配置：
- 模型文件：web文件/多层感知机_best_model_R2_0.8965.pkl
- 特征配置：web文件/A表7_被选中特征筛选的变量表.csv
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import sys
import os
import json
import warnings
warnings.filterwarnings('ignore')

# 导入配置模块
from config import Config

# 设置页面配置
st.set_page_config(
    page_title="基于机器学习的地贫输血疗效血红蛋白智能测算工具",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== 自定义CSS样式 ====================
def load_custom_css():
    """加载自定义CSS样式，简约紧凑"""
    st.markdown("""
    <style>
    /* 主背景 - 简约纯色 */
    .stApp {
        background: #f5f5f5;
        min-height: 100vh;
    }
    
    /* 标题样式 - 紧凑大气 */
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
    
    /* 卡片样式 - 紧凑干净 */
    .card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 4px rgba(0, 0, 0, 0.03);
        margin: 0.5rem 0;
        border: 1px solid #e5e5e5;
    }
    
    /* 预测结果卡片 - 紧凑优雅 */
    .prediction-card {
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(67, 56, 202, 0.25);
    }
    
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 600;
        margin: 0.3rem 0;
    }
    
    /* 按钮样式 - 紧凑现代 */
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
    
    /* 输入框样式 - 紧凑精致 */
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
    
    /* 侧边栏样式 - 紧凑 */
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e5e5;
    }
    
    /* 侧边栏进度条文字样式 - 确保在蓝色背景上可见 */
    [data-testid="stSidebar"] .stProgress > div > div {
        color: white !important;
        background: #4f46e5 !important;
    }
    
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] div[data-testid="stProgress"] {
        color: white;
    }
    
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #333333;
    }
    
    /* 进度条容器内的文字颜色 */
    [data-testid="stSidebar"] div[role="progressbar"] + div,
    [data-testid="stSidebar"] div[data-testid="stProgress"] > div:last-child {
        color: white !important;
        font-weight: bold;
    }
    
    /* 表格样式 */
    .stDataFrame {
        border-radius: 6px;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
    }
    
    /* Tab样式 - 紧凑 */
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
    
    .stTabs [data-baseweb="tab"]:hover {
        background-color: #d4d4d4;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
        color: white;
    }
    
    /* Metric样式 - 紧凑 */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 6px;
        padding: 0.75rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.02);
        border: 1px solid #e5e5e5;
    }
    
    /* 消息提示样式 - 紧凑 */
    .stSuccess, .stWarning, .stError, .stInfo {
        border-radius: 6px;
        padding: 0.6rem;
        border: 1px solid transparent;
        font-size: 0.875rem;
    }
    
    .stSuccess {
        background: #f0fdf4;
        border-color: #86efac;
    }
    
    .stWarning {
        background: #fffbeb;
        border-color: #fde047;
    }
    
    .stError {
        background: #fef2f2;
        border-color: #fca5a5;
    }
    
    .stInfo {
        background: #eff6ff;
        border-color: #93c5fd;
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        border-radius: 4px;
        background: linear-gradient(90deg, #4338ca 0%, #6366f1 100%);
    }
    
    /* 文件上传区域样式 */
    .stFileUploader {
        border: 1px dashed #d4d4d4;
        border-radius: 6px;
        padding: 1rem;
        background: #fafafa;
        transition: all 0.15s ease;
    }
    
    .stFileUploader:hover {
        border-color: #4338ca;
        background: #f5f3ff;
    }
    
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 4px;
        height: 4px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f0f0f0;
        border-radius: 2px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d4d4d4;
        border-radius: 2px;
    }
    
    /* 标题样式 */
    h2 {
        font-size: 1.1rem;
        color: #333;
        font-weight: 600;
        margin-bottom: 0.6rem;
    }
    
    /* 分割线样式 */
    hr {
        border: none;
        height: 1px;
        background: #e5e5e5;
        margin: 0.6rem 0;
    }
    
    /* 整体紧凑化 */
    .stMarkdown, .stText {
        font-size: 0.875rem;
    }
    
    /* 减小标签间距 */
    .stLabel {
        font-size: 0.825rem;
        margin-bottom: 0.25rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_custom_css()

# ==================== 缓存机制 ====================
@st.cache_resource
def load_model_cached():
    """缓存模型加载，提高性能"""
    predictor = ModelPredictor()
    success = predictor.load_model()
    return predictor if success else None

@st.cache_data
def load_features_cached():
    """缓存特征加载"""
    return load_features_from_csv()

# Web应用配置路径
WEB_DIR = Path("web文件")

# 查找模型文件（支持多种命名方式）
def find_model_file():
    """查找模型文件"""
    possible_names = [
        "多层感知机_best_model_R2_0.8965.pkl",
        "多层感知机_best_model_R2_0.9049.pkl",
        "best_model.pkl"
    ]
    for name in possible_names:
        model_path = WEB_DIR / name
        if model_path.exists():
            return model_path
    
    # 查找任何.pkl文件
    pkl_files = list(WEB_DIR.glob("*.pkl"))
    if pkl_files:
        return pkl_files[0]
    return None

MODEL_FILE = find_model_file()
FEATURES_FILE = WEB_DIR / "A表7_被选中特征筛选的变量表.csv"


class ModelPredictor:
    """模型预测器类"""
    
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = []
        self.model_info = {}
        self.model_loaded = False
        
    def load_model(self, model_path):
        """加载模型"""
        try:
            self.model = joblib.load(model_path)
            self.model_loaded = True
            
            # 提取模型信息
            if hasattr(self.model, 'named_steps'):
                if 'preprocessor' in self.model.named_steps:
                    self.preprocessor = self.model.named_steps['preprocessor']
                
                if 'model' in self.model.named_steps:
                    model_obj = self.model.named_steps['model']
                    self.model_info['model_type'] = type(model_obj).__name__
                    
                    # 获取特征重要性
                    if hasattr(model_obj, 'feature_importances_'):
                        self.model_info['feature_importances'] = model_obj.feature_importances_
                    elif hasattr(model_obj, 'coef_'):
                        self.model_info['feature_importances'] = np.abs(model_obj.coef_)
            
            # 尝试获取特征名称
            self._extract_feature_names()
            
            return True
        except Exception as e:
            st.error(f"加载模型失败: {str(e)}")
            return False
    
    def _extract_feature_names(self):
        """提取特征名称"""
        # 尝试从预处理器获取
        if self.preprocessor is not None and hasattr(self.preprocessor, 'get_feature_names_out'):
            try:
                feature_names_out = self.preprocessor.get_feature_names_out()
                # 转换为列表以避免numpy数组布尔判断问题
                if hasattr(feature_names_out, 'tolist'):
                    self.feature_names = feature_names_out.tolist()
                else:
                    self.feature_names = list(feature_names_out)
            except:
                pass
        
        # 如果还没有特征名称，尝试获取feature_names_in_
        if not self.feature_names and self.preprocessor is not None:
            if hasattr(self.preprocessor, 'feature_names_in_'):
                try:
                    feature_names_in = self.preprocessor.feature_names_in_
                    if hasattr(feature_names_in, 'tolist'):
                        self.feature_names = feature_names_in.tolist()
                    else:
                        self.feature_names = list(feature_names_in)
                except:
                    pass
        
        # 如果还是没有特征名称，尝试从config获取
        if len(self.feature_names) == 0:
            try:
                from config import Config
                # 使用配置中的特征
                pass
            except:
                pass
        
        # 如果还是没有，使用模型的feature_names_in_
        if len(self.feature_names) == 0 and self.model is not None:
            if hasattr(self.model, 'feature_names_in_'):
                try:
                    model_feature_names = self.model.feature_names_in_
                    if hasattr(model_feature_names, 'tolist'):
                        self.feature_names = model_feature_names.tolist()
                    else:
                        self.feature_names = list(model_feature_names)
                except:
                    pass
        
        # 将特征名称保存到model_info中供后续使用
        if self.feature_names:
            self.model_info['feature_names'] = self.feature_names
    
    def predict(self, input_data):
        """预测单个样本"""
        if not self.model_loaded:
            raise ValueError("模型未加载")
        
        # 转换为DataFrame并确保列名正确
        if isinstance(input_data, dict):
            # 确保字典的键与特征名称匹配
            input_dict = {}
            for feat in self.feature_names:
                input_dict[feat] = input_data.get(feat, 0.0)
            input_df = pd.DataFrame([input_dict])
        elif isinstance(input_data, list):
            input_df = pd.DataFrame([input_data], columns=self.feature_names)
        else:
            # 确保DataFrame有正确的列名
            if isinstance(input_data, pd.DataFrame):
                # 重新排列列顺序以匹配特征名称
                available_cols = [col for col in self.feature_names if col in input_data.columns]
                missing_cols = [col for col in self.feature_names if col not in input_data.columns]
                
                # 为缺失的列添加默认值
                for col in missing_cols:
                    input_data[col] = 0.0
                
                input_df = input_data[self.feature_names]
            else:
                input_df = input_data
        
        # 预测
        try:
            prediction = self.model.predict(input_df)
            return prediction[0]
        except Exception as e:
            raise ValueError(f"预测失败: {str(e)}")
    
    def predict_batch(self, input_df):
        """批量预测"""
        if not self.model_loaded:
            raise ValueError("模型未加载")
        
        try:
            # 确保DataFrame有正确的列名
            if isinstance(input_df, pd.DataFrame):
                # 检查缺失的列
                missing_cols = [col for col in self.feature_names if col not in input_df.columns]
                
                # 为缺失的列添加默认值
                for col in missing_cols:
                    input_df[col] = 0.0
                
                # 重新排列列顺序以匹配特征名称
                input_df = input_df[self.feature_names]
            
            predictions = self.model.predict(input_df)
            return predictions
        except Exception as e:
            raise ValueError(f"批量预测失败: {str(e)}")
    
    def get_shap_values(self, input_data):
        """获取SHAP解释值
        
        对于不同类型的模型使用不同的SHAP解释方法：
        - 高斯过程回归(GaussianProcessRegressor)：使用数值梯度方法（leave-one-out边际贡献）
        - 其他模型：使用KernelExplainer在原始特征空间计算SHAP值
        """
        try:
            import shap
            
            # 确保输入是DataFrame格式
            if isinstance(input_data, dict):
                input_dict = {}
                for feat in self.feature_names:
                    input_dict[feat] = input_data.get(feat, 0.0)
                input_df = pd.DataFrame([input_dict])
            elif isinstance(input_data, pd.DataFrame):
                input_df = input_data.copy()
            else:
                input_df = pd.DataFrame([input_data], columns=self.feature_names)
            
            # 获取底层模型对象（处理Pipeline情况）
            model_obj = self.model
            if hasattr(self.model, 'named_steps') and 'model' in self.model.named_steps:
                model_obj = self.model.named_steps['model']
            
            # 判断是否为高斯过程回归模型
            is_gp_model = False
            try:
                from sklearn.gaussian_process import GaussianProcessRegressor
                is_gp_model = isinstance(model_obj, GaussianProcessRegressor)
            except:
                pass
            
            # 定义完整的预测函数（包括预处理）
            def full_model_predict(X):
                """完整的预测函数，接受原始特征DataFrame或numpy数组"""
                if isinstance(X, np.ndarray):
                    # 确保二维数组
                    if X.ndim == 1:
                        X = X.reshape(1, -1)
                    X = pd.DataFrame(X, columns=self.feature_names)
                elif isinstance(X, list):
                    X = pd.DataFrame([X], columns=self.feature_names)
                
                # 确保列顺序正确
                if isinstance(X, pd.DataFrame):
                    X = X[self.feature_names]
                
                # 预测
                result = self.model.predict(X)
                
                # 确保返回的是1D数组
                if hasattr(result, 'ndim') and result.ndim > 1:
                    result = result.flatten()
                
                # 确保返回numpy数组（KernelExplainer要求）
                if not isinstance(result, np.ndarray):
                    result = np.array([result])
                
                return result
            
            # 创建背景数据集 - 使用基于特征名称的合理临床范围
            background_samples = []
            for i in range(50):  # 创建50个背景样本
                sample_dict = {}
                for feat in self.feature_names:
                    feat_lower = str(feat).lower()
                    if '输血' in str(feat):
                        sample_dict[feat] = np.random.uniform(0, 15)
                    elif 'hb' in feat_lower or 'hgb' in feat_lower:
                        sample_dict[feat] = np.random.uniform(40, 180)
                    elif '年龄' in str(feat):
                        sample_dict[feat] = np.random.randint(0, 100)
                    elif '身高' in str(feat):
                        sample_dict[feat] = np.random.randint(80, 220)
                    elif '体重' in str(feat):
                        sample_dict[feat] = np.random.uniform(10, 200)
                    elif 'plt' in feat_lower:
                        sample_dict[feat] = np.random.uniform(50, 500)
                    else:
                        sample_dict[feat] = np.random.uniform(0, 100)
                background_samples.append(sample_dict)
            
            background_df = pd.DataFrame(background_samples)
            
            # 对于高斯过程回归模型，直接使用数值梯度方法
            # GradientExplainer不支持sklearn模型，KernelExplainer对GP不稳定
            if is_gp_model:
                shap_values = self._compute_gp_shap_numerical(input_df, background_df)
                explainer = None
            else:
                # 使用KernelExplainer在原始特征空间计算SHAP值
                explainer = shap.KernelExplainer(full_model_predict, background_df)
                shap_values = explainer.shap_values(input_df)
                
                # 处理SHAP值格式
                if isinstance(shap_values, list):
                    shap_values = shap_values[0]
                
                # 确保shap_values是正确的形状（单个样本）
                if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                    if shap_values.shape[0] == 1:
                        shap_values = shap_values[0]
            
            return explainer, shap_values, self.feature_names
        except Exception as e:
            st.warning(f"SHAP计算失败: {str(e)}")
            import traceback
            st.warning(f"详细错误: {traceback.format_exc()}")
            return None, None, None
    
    def _compute_gp_shap_numerical(self, input_df, background_df):
        """使用数值梯度方法计算高斯过程回归的SHAP值
        
        对于高斯过程回归等难以使用标准SHAP解释器的模型，
        使用leave-one-out边际贡献方法近似计算每个特征的贡献。
        
        参数:
            input_df (pd.DataFrame): 待解释的输入数据
            background_df (pd.DataFrame): 背景数据集
        
        返回:
            np.ndarray: SHAP值数组
        """
        try:
            # 获取输入数据的预测值（Pipeline.predict返回ndarray）
            input_pred = self.model.predict(input_df[self.feature_names])
            # 确保获取标量值
            if hasattr(input_pred, 'ndim') and input_pred.ndim > 0:
                input_pred = input_pred[0]
            
            # 使用数值方法计算每个特征的SHAP值
            shap_values = []
            input_array = input_df.values[0] if len(input_df) == 1 else input_df.values
            
            for i, feat in enumerate(self.feature_names):
                # 获取该特征在背景数据中的分布
                feat_distribution = background_df[feat].values
                
                # 计算该特征被移除时的预测值
                # 使用背景数据中该特征的均值替代输入值
                masked_input = input_array.copy()
                masked_input[i] = np.mean(feat_distribution)
                
                masked_df = pd.DataFrame([masked_input], columns=self.feature_names)
                masked_pred = self.model.predict(masked_df)
                
                # 确保获取标量值
                if hasattr(masked_pred, 'ndim') and masked_pred.ndim > 0:
                    masked_pred = masked_pred[0]
                
                # SHAP值 = 原始预测 - 特征被移除后的预测
                shap_value = input_pred - masked_pred
                shap_values.append(shap_value)
            
            return np.array(shap_values)
        
        except Exception as e:
            st.warning(f"数值梯度SHAP计算失败: {str(e)}")
            return np.zeros(len(self.feature_names))


def create_input_widgets(feature_names, feature_defaults=None, categorical_options_desc=None):
    """创建输入控件
    
    根据特征配置创建合适的输入控件，包括数值输入和分类选择
    
    参数:
        feature_names (list): 特征名称列表
        feature_defaults (dict): 特征默认值和范围配置
        categorical_options_desc (dict): 分类特征选项说明
    """
    inputs = {}
    cols = st.columns(3)
    
    for i, feature in enumerate(feature_names):
        col_idx = i % 3
        with cols[col_idx]:
            # 从配置获取默认值和范围
            if feature_defaults and feature in feature_defaults:
                config = feature_defaults[feature]
                default_val = config.get('default', 0.0)
                min_val = config.get('min', 0.0)
                max_val = config.get('max', 100.0)
                step = config.get('step', 0.1)
                feature_type = config.get('type', 'numerical')
                options = config.get('options', None)
                
                # 分类特征使用选择框
                if feature_type == 'categorical' and options:
                    # 获取选项说明
                    if categorical_options_desc and feature in categorical_options_desc:
                        option_desc = categorical_options_desc[feature]
                        option_labels = [option_desc.get(str(opt), str(opt)) for opt in options]
                    else:
                        option_labels = [str(opt) for opt in options]
                    
                    # 创建选择框
                    selected_idx = options.index(default_val) if default_val in options else 0
                    selected_label = st.selectbox(
                        f"{feature}",
                        options=option_labels,
                        index=selected_idx,
                        key=f"input_{feature}"
                    )
                    # 将标签转换为数值
                    inputs[feature] = options[option_labels.index(selected_label)]
                else:
                    # 数值特征使用数字输入
                    format_str = "%.1f" if step < 1 else "%.0f"
                    # 使用水平容器布局，特征名称和输入框在同一行完美对齐
                    col_label, col_input = st.columns([1, 2])
                    with col_label:
                        st.markdown(f"**{feature[:10]}**")
                    with col_input:
                        inputs[feature] = st.number_input(
                            label=feature,
                            min_value=float(min_val),
                            max_value=float(max_val),
                            value=float(default_val),
                            step=step,
                            format=format_str,
                            key=f"input_{feature}",
                            label_visibility="collapsed"
                        )
            else:
                # 没有配置时使用默认设置
                # 根据特征名称建议合理的范围和默认值
                if 'Hb' in feature or 'HGB' in feature or 'Hb后' in feature:
                    min_val, max_val, step, default_val = 5.0, 200.0, 1.0, 100.0
                    format_str = "%.0f"
                elif '年龄' in feature or 'age' in feature.lower():
                    min_val, max_val, step, default_val = 0.0, 120.0, 1.0, 50.0
                    format_str = "%.0f"
                elif '身高' in feature or 'height' in feature.lower():
                    min_val, max_val, step, default_val = 50.0, 250.0, 1.0, 170.0
                    format_str = "%.0f"
                elif '体重' in feature or 'weight' in feature.lower():
                    min_val, max_val, step, default_val = 2.0, 200.0, 0.5, 65.0
                    format_str = "%.1f"
                elif '输血量' in feature or 'volume' in feature.lower():
                    min_val, max_val, step, default_val = 0.0, 20.0, 0.5, 2.0
                    format_str = "%.1f"
                else:
                    min_val, max_val, step, default_val = 0.0, 1000.0, 0.1, 0.0
                    format_str = "%.1f"

                # 使用容器布局确保水平对齐
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


def display_prediction_results(prediction, model_info, shap_explainer=None, shap_values=None, shap_feature_names=None, predictor=None, confidence_interval=None, hb_before=None):
    """显示预测结果（包含置信区间和改进的可视化）"""
    st.markdown("---")
    
    # ==================== 预测结果（与指标同行） ====================
    col1, col2, col3, col4 = st.columns(4)
    
    # 预测结果
    with col1:
        st.markdown(f"""
        <div style="background: #f8fafc; padding: 1rem; border-radius: 6px; border: 1px solid #e5e5e5;">
            <div style="font-size: 0.825rem; color: #6b7280; margin-bottom: 0.25rem;">🔮 预测结果</div>
            <div style="font-size: 1.5rem; font-weight: 600; color: #1f2937;">{prediction:.2f} g/L</div>
        </div>
        """, unsafe_allow_html=True)
    
    # 显示预测结果与输血前Hb的差值
    with col2:
        if hb_before is not None:
            delta_hb = prediction - hb_before
            if delta_hb >= 0:
                arrow = "↑"
                color = "#ef4444"  # 红色
            else:
                arrow = "↓"
                color = "#3b82f6"  # 蓝色
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
    
    # 显示95%置信区间
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
    
    # 如果模型有训练集统计信息
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
    # 显示SHAP解释
    if shap_values is not None and shap_explainer is not None:
        st.subheader("🔍 特征贡献分析")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            
            # 设置中文字体支持
            # 尝试使用系统中文字体
            try:
                # Windows系统常用中文字体
                plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
                plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
            except:
                pass
            
            # 使用传入的特征名称或predictor的特征名称
            feature_names_list = shap_feature_names if shap_feature_names else (predictor.feature_names if predictor else None)
            
            # 处理SHAP值
            if hasattr(shap_values, 'shape') and len(shap_values.shape) > 1:
                shap_vals = shap_values[0]
            else:
                shap_vals = shap_values
            
            # 确保是数组格式
            if hasattr(shap_vals, 'tolist'):
                shap_vals = shap_vals.tolist()
            elif not isinstance(shap_vals, list):
                shap_vals = list(shap_vals) if hasattr(shap_vals, '__iter__') else [shap_vals]
            
            # 创建贡献DataFrame
            if feature_names_list and len(feature_names_list) == len(shap_vals):
                contrib_df = pd.DataFrame({
                    '特征': feature_names_list,
                    'SHAP值': shap_vals
                })
                contrib_df = contrib_df.sort_values('SHAP值', key=abs, ascending=False)
                contrib_df['贡献方向'] = contrib_df['SHAP值'].apply(
                    lambda x: '⬆️ 正向' if x > 0 else '⬇️ 负向'
                )
                contrib_df['SHAP值'] = contrib_df['SHAP值'].round(4)
                
                # 先创建条形图
                fig, ax = plt.subplots(figsize=(10, 3))
                colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in contrib_df['SHAP值']]
                bars = ax.barh(range(len(contrib_df)), contrib_df['SHAP值'], color=colors)
                
                # 设置y轴标签为特征名称（中文）
                ax.set_yticks(range(len(contrib_df)))
                ax.set_yticklabels(contrib_df['特征'].tolist(), fontsize=12)
                ax.tick_params(axis='x', labelsize=11)
                
                ax.set_xlabel('SHAP值', fontsize=12)
                ax.set_title('特征贡献分析', fontsize=14)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                
                # 添加数值标签
                for i, (bar, val) in enumerate(zip(bars, contrib_df['SHAP值'])):
                    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{val:.4f}', va='center', fontsize=10)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # 再显示特征贡献明细表格
                st.subheader("📋 特征贡献明细")
                st.dataframe(contrib_df, use_container_width=True)
            else:
                st.info(f"特征名称数量({len(feature_names_list) if feature_names_list else 0})与SHAP值数量({len(shap_vals)})不匹配")
                
        except Exception as e:
            st.info(f"SHAP可视化失败: {str(e)}")
    
    # ==================== 预测值可视化 ====================
    st.subheader("📊 Hb 输血前后可视化")
    
    # 创建Hb输血前后可视化（整合到一条横条）
    fig_gauge = go.Figure(go.Indicator(
        mode = "number+gauge",
        value = prediction,
        domain = {'x': [0, 1], 'y': [0.2, 0.8]},
        title = {'text': "Hb 输血前后可视化", 'font': {'size': 18}, 'align': 'center'},
        number = {'valueformat': '.2f', 'suffix': ' g/L', 'font': {'size': 28, 'weight': 'bold', 'color': "#4f46e5"}},
        gauge = {
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
    
    # 如果有输血前数据，添加输血前标记
    if hb_before is not None:
        # 计算输血前值在横条上的位置比例
        hb_before_ratio = (hb_before - 50) / (200 - 50)
        
        # 添加输血前标记线
        fig_gauge.add_shape(
            type="line",
            x0=hb_before_ratio, y0=0.2,
            x1=hb_before_ratio, y1=0.8,
            line=dict(color="#ef4444", width=3, dash="dash"),
        )
        
        # 添加输血前标签（在横条上方）
        fig_gauge.add_annotation(
            x=hb_before_ratio, y=0.85,
            text=f"输血前: {hb_before:.2f}",
            showarrow=False,
            font={'size': 11, 'color': "#ef4444", 'weight': 'bold'},
            xanchor='center', yanchor='bottom'
        )
        
        # 计算变化量（升高为红色，减低为蓝色）
        delta_hb = prediction - hb_before
        delta_color = "#ef4444" if delta_hb >= 0 else "#3b82f6"
        delta_arrow = "↑" if delta_hb >= 0 else "↓"
        
        # 在右侧显示变化量
        fig_gauge.add_annotation(
            x=1.0, y=0.5,
            text=f"{delta_hb:+.2f} g/L {delta_arrow}",
            showarrow=False,
            font={'size': 16, 'color': delta_color, 'weight': 'bold'},
            xanchor='left', yanchor='middle'
        )
    
    fig_gauge.update_layout(
        height=120,
        margin=dict(l=20, r=120, t=40, b=30),
        showlegend=False
    )
    
    st.plotly_chart(fig_gauge, use_container_width=True)


def display_model_info(model_info):
    """显示模型信息"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 模型信息")
    
    if model_info:
        if 'model_type' in model_info:
            st.sidebar.write(f"**模型类型:** {model_info['model_type']}")
        
        if 'feature_importances' in model_info and model_info['feature_importances'] is not None:
            st.sidebar.write("**特征重要性**")
            
            # 获取特征名称和重要性
            importances = model_info['feature_importances']
            
            if 'feature_names' in model_info:
                names = model_info['feature_names']
            else:
                names = [f"特征_{i}" for i in range(len(importances))]
            
            # 创建重要性DataFrame
            imp_df = pd.DataFrame({
                '特征': names[:len(importances)],
                '重要性': importances
            })
            imp_df = imp_df.sort_values('重要性', ascending=False).head(10)
            
            # 在侧边栏显示
            for _, row in imp_df.iterrows():
                st.sidebar.progress(
                    float(row['重要性'] / imp_df['重要性'].max()),
                    text=f"{row['特征']}: {row['重要性']:.3f}"
                )
        
        # 显示模型文件信息
        if 'model_path' in model_info:
            st.sidebar.write(f"**模型文件:** {Path(model_info['model_path']).name}")


def find_best_model():
    """查找最佳模型
    
    从web文件夹中加载预置的最佳模型文件
    """
    # 检查web文件夹是否存在
    if not WEB_DIR.exists():
        return None
    
    # 检查模型文件是否存在
    if not MODEL_FILE.exists():
        return None
    
    return MODEL_FILE


def load_features_from_csv():
    """从A表7 CSV文件加载特征列表"""
    if not FEATURES_FILE.exists():
        return None, None
    
    try:
        # 读取特征CSV文件
        features_df = pd.read_csv(FEATURES_FILE, encoding='utf-8-sig')
        
        # 获取特征名称列表（第二列是特征名称）
        feature_names = features_df.iloc[:, 1].tolist()
        
        # 准备默认特征配置（用于输入控件）
        feature_defaults = {}
        for feat in feature_names:
            # 根据特征名称设置合理的默认值和范围
            if 'Hb' in feat or 'HGB' in feat or 'Hb后' in feat:
                feature_defaults[feat] = {
                    'default': 80.0,
                    'min': 5.0,
                    'max': 200.0,
                    'step': 1.0,
                    'type': 'numerical'
                }
            elif '年龄' in feat or 'age' in feat.lower():
                feature_defaults[feat] = {
                    'default': 20.0,
                    'min': 0.0,
                    'max': 120.0,
                    'step': 1.0,
                    'type': 'numerical'
                }
            elif '身高' in feat or 'height' in feat.lower():
                feature_defaults[feat] = {
                    'default': 150.0,
                    'min': 50.0,
                    'max': 250.0,
                    'step': 1.0,
                    'type': 'numerical'
                }
            elif '体重' in feat or 'weight' in feat.lower():
                feature_defaults[feat] = {
                    'default': 45.0,
                    'min': 2.0,
                    'max': 200.0,
                    'step': 0.5,
                    'type': 'numerical'
                }
            elif '输血量' in feat or 'volume' in feat.lower():
                feature_defaults[feat] = {
                    'default': 2.0,
                    'min': 0.0,
                    'max': 20.0,
                    'step': 0.5,
                    'type': 'numerical'
                }
            elif 'PLT' in feat or 'plt' in feat.lower():
                feature_defaults[feat] = {
                    'default': 300.0,
                    'min': 30.0,
                    'max': 1200.0,
                    'step': 1.0,
                    'type': 'numerical'
                }
            else:
                feature_defaults[feat] = {
                    'default': 0.0,
                    'min': 0.0,
                    'max': 1000.0,
                    'step': 0.1,
                    'type': 'numerical'
                }
        
        return feature_names, feature_defaults
    except Exception as e:
        return None, None


def load_model_for_app():
    """加载模型"""
    predictor = ModelPredictor()
    
    # 查找最佳模型
    model_path = find_best_model()
    
    if model_path is None:
        return None, "未找到模型文件，请确保web文件夹中存在.pkl模型文件"
    
    # 加载模型
    success = predictor.load_model(model_path)
    
    if success:
        predictor.model_info['model_path'] = str(model_path)
        
        # 从文件名中提取R2值
        import re
        match = re.search(r'R2_(\d+\.\d+)', model_path.name)
        if match:
            predictor.model_info['train_r2'] = float(match.group(1))
        
        # 从CSV文件加载特征列表
        feature_names, feature_defaults = load_features_from_csv()
        if feature_names:
            predictor.feature_names = feature_names
            predictor.model_info['feature_defaults'] = feature_defaults
            predictor.model_info['feature_names'] = feature_names  # 确保特征名称也保存到model_info中
        else:
            # 如果CSV加载失败，使用模型自身的特征名
            if not predictor.feature_names:
                predictor.feature_names = [f"特征_{i}" for i in range(10)]
                predictor.model_info['feature_defaults'] = {}
        
        return predictor, f"成功加载模型: {model_path.name}"
    else:
        return None, "模型加载失败"


def main():
    """主函数"""
    
    # 验证配置参数（必须在任何操作之前执行）
    Config.validate_config()
    
    # ==================== 初始化Session State ====================
    if 'prediction_history' not in st.session_state:
        st.session_state['prediction_history'] = []
    if 'template_selected' not in st.session_state:
        st.session_state['template_selected'] = None
    
    # ==================== 标题区域 ====================
    st.markdown("""
    <div class="main-header">
        🏥 基于机器学习的地贫输血疗效血红蛋白智能测算工具
    </div>
    """, unsafe_allow_html=True)
    
    # ==================== 侧边栏 ====================
    st.sidebar.title("⚙️ 设置")
    
    # 显示帮助信息
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
        
        **快捷模板：**
        - 选择预设模板快速填充
        - 或自定义输入值
        """)
    
    # ==================== 加载模型 ====================
    with st.spinner("正在加载模型..."):
        predictor, load_msg = load_model_for_app()
    
    if predictor is None or not predictor.model_loaded:
        st.error(f"❌ {load_msg}")
        st.info("请先运行主程序训练模型，确保模型文件存在于模型目录中")
        return
    
    st.sidebar.success(f"✅ {load_msg}")
    
    # 显示模型信息
    display_model_info(predictor.model_info)
    
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
        
        # 创建输入控件（使用特征配置）
        feature_defaults = predictor.model_info.get('feature_defaults', {})
        categorical_options_desc = predictor.model_info.get('categorical_options_desc', {})
        
        inputs = create_input_widgets(feature_names, feature_defaults, categorical_options_desc)
        
        # 预测按钮
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            predict_btn = st.button("🔮 预测", type="primary", use_container_width=True)
        
        with col2:
            reset_btn = st.button("🔄 重置", use_container_width=True)
            if reset_btn:
                st.experimental_rerun()
        
        # 执行预测
        if predict_btn:
            try:
                # 准备输入数据
                input_data = pd.DataFrame([inputs])
                
                # 预测
                prediction = predictor.predict(input_data)
                
                # 计算置信区间（基于模型RMSE）
                rmse = predictor.model_info.get('train_rmse', 5.0)
                confidence_interval = 1.96 * rmse  # 95%置信区间
                lower_bound = prediction - confidence_interval
                upper_bound = prediction + confidence_interval
                
                # 获取SHAP值
                shap_explainer, shap_values, shap_feature_names = predictor.get_shap_values(input_data)
                
                # 保存预测历史
                history_record = {
                    '时间': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
                    '预测值': round(prediction, 2),
                    '置信区间下限': round(lower_bound, 2),
                    '置信区间上限': round(upper_bound, 2),
                    **inputs
                }
                st.session_state['prediction_history'].append(history_record)
                
                # 显示结果（包含置信区间）
                display_prediction_results(
                    prediction, 
                    predictor.model_info,
                    shap_explainer,
                    shap_values,
                    shap_feature_names,
                    predictor,
                    confidence_interval=(lower_bound, upper_bound),
                    hb_before=inputs.get('输血前Hb')
                )
                
                # 显示输入摘要
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
        
        # 下载模板按钮
        template_df = pd.DataFrame(columns=feature_names)
        template_csv = template_df.to_csv(index=False)
        st.download_button(
            label="📥 下载CSV模板",
            data=template_csv,
            file_name="prediction_template.csv",
            mime="text/csv",
            help="下载模板文件，填写数据后上传进行批量预测"
        )
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择CSV文件",
            type=['csv'],
            help="文件应包含所有特征列，列名与模型特征名称一致"
        )
        
        if uploaded_file is not None:
            try:
                # 读取文件，尝试多种编码
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
                
                # 检查列
                missing_cols = set(feature_names) - set(input_df.columns)
                if missing_cols:
                    st.warning(f"缺少以下列: {missing_cols}")
                    st.info("将使用默认值填充")
                    for col in missing_cols:
                        input_df[col] = 0.0
                
                # 显示数据预览
                st.subheader("📊 数据预览")
                st.dataframe(input_df.head(10), use_container_width=True)
                st.caption(f"共 {len(input_df)} 行")
                
                # 预测按钮
                if st.button("🚀 执行批量预测", type="primary"):
                    with st.spinner("正在预测..."):
                        # 预测
                        predictions = predictor.predict_batch(input_df)
                        
                        # 添加预测结果
                        result_df = input_df.copy()
                        result_df['HGB后_预测值'] = predictions
                        
                        # 显示结果
                        st.subheader("📊 预测结果")
                        st.dataframe(result_df, use_container_width=True)
                        
                        # 下载按钮
                        csv = result_df.to_csv(index=False)
                        st.download_button(
                            label="📥 下载预测结果",
                            data=csv,
                            file_name="prediction_results.csv",
                            mime="text/csv"
                        )
                        
                        # 可视化预测分布
                        st.subheader("📈 预测值分布")
                        fig = px.histogram(
                            result_df, 
                            x='HGB后_预测值',
                            nbins=30,
                            title="预测值分布"
                        )
                        fig.update_layout(
                            xaxis_title="预测值",
                            yaxis_title="频数"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 显示统计信息
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
        
        # 模型类型
        if 'model_type' in predictor.model_info:
            st.write(f"**模型类型:** {predictor.model_info['model_type']}")
        
        # 特征信息
        st.subheader("📋 特征列表")
        st.write(f"共 {len(feature_names)} 个特征")
        
        # 创建特征DataFrame
        if 'feature_importances' in predictor.model_info:
            importances = predictor.model_info['feature_importances']
            if importances is not None:
                imp_df = pd.DataFrame({
                    '特征名称': feature_names[:len(importances)],
                    '重要性': importances
                })
                imp_df = imp_df.sort_values('重要性', ascending=False)
                
                # 显示重要性图表
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
                st.plotly_chart(fig, use_container_width=True)
        
        # 模型文件信息
        if 'model_path' in predictor.model_info:
            model_path = Path(predictor.model_info['model_path'])
            st.subheader("📁 模型文件")
            st.write(f"**路径:** {model_path}")
            st.write(f"**大小:** {model_path.stat().st_size / 1024:.2f} KB")
            st.write(f"**修改时间:** {pd.Timestamp.fromtimestamp(model_path.stat().st_mtime)}")
    
    # Tab 4: 预测历史
    with tab4:
        st.subheader("📜 预测历史记录")
        
        # 显示历史记录数量
        history_count = len(st.session_state['prediction_history'])
        st.info(f"📊 共保存了 {history_count} 条预测记录")
        
        if history_count > 0:
            # 创建历史记录DataFrame
            history_df = pd.DataFrame(st.session_state['prediction_history'])
            
            # 显示历史记录表格
            st.dataframe(history_df, use_container_width=True)
            
            # 预测值趋势图
            st.subheader("📈 预测值趋势")
            fig_trend = px.line(
                history_df,
                x='时间',
                y='预测值',
                title='预测值历史趋势',
                markers=True
            )
            fig_trend.update_traces(line_color='#667eea')
            st.plotly_chart(fig_trend, use_container_width=True)
            
            # 预测值分布统计
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
            
            # 下载历史记录
            st.subheader("📥 导出历史记录")
            csv_history = history_df.to_csv(index=False)
            st.download_button(
                label="下载预测历史记录 (CSV)",
                data=csv_history,
                file_name=f"prediction_history_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                type="primary"
            )
            
            # 清除历史记录按钮
            if st.button("🗑️ 清除历史记录", type="secondary"):
                st.session_state['prediction_history'] = []
                st.success("历史记录已清除")
                st.experimental_rerun()
        else:
            st.info("暂无预测历史记录，请先进行预测")


if __name__ == "__main__":
    main()

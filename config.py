# -*- coding: utf-8 -*-  # 指定文件编码为UTF-8，支持中文字符
"""
配置参数模块，集中管理所有机器学习流水线的配置参数

该模块定义config类，包含数据文件路径、目标变量、模型超参数、
特征选择设置、输出目录等所有配置信息。
"""

import os  # 导入操作系统接口模块，用于读取环境变量
from pathlib import Path  # 从pathlib导入Path类，用于路径处理

try:
    from skopt.space import Real, Integer, Categorical
except ImportError:
    Real = Integer = Categorical = None
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet, BayesianRidge  # 从sklearn导入线性模型，包括线性回归、Lasso回归、岭回归、弹性网络和贝叶斯岭回归
from sklearn.neural_network import MLPRegressor  # 从sklearn导入多层感知机回归器，用于神经网络回归
from sklearn.neighbors import KNeighborsRegressor  # 从sklearn导入K近邻回归器
from sklearn.tree import DecisionTreeRegressor  # 从sklearn导入决策树回归器
from sklearn.ensemble import (GradientBoostingRegressor, AdaBoostRegressor,
                              RandomForestRegressor, ExtraTreesRegressor)  # 从sklearn导入集成学习模型，包括梯度提升、AdaBoost、随机森林和极端随机树
from sklearn.gaussian_process import GaussianProcessRegressor  # 从sklearn导入高斯过程回归模型
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel  # 从sklearn导入高斯过程核函数
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
from sklearn.svm import SVR  # 从sklearn导入支持向量回归器，用于支持向量机回归
from sklearn.preprocessing import PolynomialFeatures  # 从sklearn导入多项式特征生成器，用于多项式回归
from sklearn.pipeline import Pipeline  # 从sklearn导入管道工具，用于组合多个处理步骤
from sklearn.ensemble import HistGradientBoostingRegressor  # 从sklearn导入直方图梯度提升树回归器
from sklearn.linear_model import SGDRegressor  # 从sklearn导入随机梯度下降回归器

# CatBoost导入（需要单独安装catboost包）
try:
    from catboost import CatBoostRegressor
except ImportError:
    CatBoostRegressor = None


class Config:
    """配置参数类，集中管理所有参数

    该类用于定义项目中使用的各种常量和路径配置。包括数据文件、目标列名、
    模型超参数、特征选择设置以及输出目录等信息。
    """  # 定义config类的文档字符串，描述该类的作用

    # ========== 路径配置 ==========
    BASE_RESULTS_DIR = Path(os.environ.get("RESULTS_DIR", "多次实验结果")).absolute()  # 基础结果目录（绝对路径），存储所有分析结果的根目录
    RESULTS_DIR = BASE_RESULTS_DIR  # 结果主目录（兼容旧代码）
    TABLES_DIR_1 = RESULTS_DIR / "表格1"  # 表格输出目录1，存储第一类表格结果
    TABLES_DIR_2 = RESULTS_DIR / "表格2"  # 表格输出目录2，存储第二类表格结果
    FEATURE_SELECTION_DIR = RESULTS_DIR / "特征选择"  # 特征选择结果目录，存储特征选择相关结果
    FIGURES_DIR = RESULTS_DIR / "图表"  # 图表保存目录，存储生成的图表文件
    MODEL_DIR = RESULTS_DIR / "模型"  # 模型保存目录，存储训练好的模型文件
    SHAP_DIR = RESULTS_DIR / "SHAP分析"  # SHAP分析结果目录，存储SHAP解释分析结果
    INSPECTION_TABLES_DIR = RESULTS_DIR / "检验表"  # 检验表目录，存储基线特征检验表和公式误差检验表
    LOGS_DIR = RESULTS_DIR / "logs"  # 日志目录，存储日志文件
    
    @classmethod
    def ensure_directories(cls):
        """
        确保所有必要的目录存在（统一目录创建入口）
        
        该方法会创建所有配置中定义的目录，避免在多个地方重复创建目录
        """
        directories = [
            cls.BASE_RESULTS_DIR,
            cls.TABLES_DIR_1,
            cls.TABLES_DIR_2,
            cls.FEATURE_SELECTION_DIR,
            cls.FIGURES_DIR,
            cls.MODEL_DIR,
            cls.SHAP_DIR,
            cls.INSPECTION_TABLES_DIR,
            cls.LOGS_DIR
        ]
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def with_paths(cls, **path_overrides):
        """
        创建带有临时路径覆盖的上下文管理器
        
        使用示例:
            with Config.with_paths(
                TABLES_DIR_1=temp_tables_dir,
                TABLES_DIR_2=temp_tables_dir
            ):
                # 在这个上下文中，config.TABLES_DIR_1/2 被临时覆盖
                pass  # 退出上下文后自动恢复
        
        参数:
            **path_overrides: 要临时覆盖的路径配置键值对
            
        返回:
            PathContext: 上下文管理器对象
        """
        class PathContext:
            def __enter__(self):
                self.original = {}
                for key, value in path_overrides.items():
                    if hasattr(Config, key):
                        self.original[key] = getattr(Config, key)
                        setattr(Config, key, value)
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                for key, value in self.original.items():
                    setattr(Config, key, value)
        return PathContext()

    # ========== 数据配置 ==========
    DATA_FILE = "data/6.24-3.csv"  # 输入数据文件名，指定要读取的CSV数据文件（位于data文件夹中）
    TARGET_COLUMN = "HGB后"  # 目标变量列名，指定回归分析的目标变量
    TEST_SIZE = 0.3  # 测试集所占比例，将数据集的30%用作测试集
    KNN_NEIGHBORS = 5  # K近邻算法邻居数，用于KNN缺失值填充和KNN回归
    BINS = 10  # 分箱数量，用于连续变量分箱处理
    NUM_CATEGORICAL_FEATURES = 5  # 前N列作为分类特征列名列表的数量
    FORMULA_COL_VOLUME = '输血量'  # 公式法输血量列名
    FORMULA_COL_WEIGHT = '体重'  # 公式法体重列名
    FORMULA_COL_AGE = '年龄'  # 公式法年龄列名
    FORMULA_COL_HGB = '输血前Hb'  # 公式法输血前Hb列名
    FORMULA_REQUIRED_COLS = [FORMULA_COL_VOLUME, FORMULA_COL_WEIGHT, FORMULA_COL_AGE, FORMULA_COL_HGB]  # 公式法所需的输入字段列表，动态构建以避免重复

    # ========== 训练配置 ==========
    try:
        RANDOM_STATE = int(os.environ.get('RANDOM_SEED', 42))  # 随机种子，从环境变量获取或使用默认值71
    except ValueError:
        raise ValueError("环境变量 RANDOM_SEED 必须是整数类型")
    MAX_ITER = 5000  # 最大迭代次数（通用），控制模型训练的最大迭代次数
    ALPHA_LEVEL = 0.05  # 显著性水平，用于统计检验的显著性判断
    N_JOBS = 1  # 并行计算线程数，设置为1以确保结果完全可重现
    REPRODUCIBLE = True  # 是否启用结果可重现机制，控制是否设置随机种子
    LASSO_CV_FOLDS = 10  # 特征筛选Lasso交叉验证折数，用于Lasso回归的交叉验证
    LASSO_MAX_ITER = MAX_ITER  # Lasso回归最大迭代次数，设置Lasso模型的最大迭代次数
    RF_MAX_DEPTH = 5  # 随机森林最大深度，限制随机森林中每棵树的最大深度
    RF_N_ESTIMATORS = 100  # 随机森林基学习器数量，指定随机森林中决策树的数量
    GB_N_ESTIMATORS = 100  # 梯度提升弱学习器数量
    ADA_N_ESTIMATORS = 50  # AdaBoost弱学习器数量
    XGB_N_ESTIMATORS = 100  # XGBoost弱学习器数量
    LGB_N_ESTIMATORS = 100  # LightGBM弱学习器数量
    ET_N_ESTIMATORS = 100  # 极端随机树基学习器数量
    CV_FOLDS = 5  # 通用交叉验证折数，用于模型评估的交叉验证折数
    BAYES_N_ITER = 50  # 贝叶斯优化迭代次数，控制贝叶斯优化的迭代次数
    BAYES_CV_FOLDS = 5  # 贝叶斯优化交叉验证折数，用于贝叶斯优化中的交叉验证
    KFOLD_SPLITS = 5  # K折交叉验证划分次数，指定K折交叉验证的折数

    # ========== 开关配置 ==========
    ENABLE_BAYESIAN_TUNING = False  # 是否启用贝叶斯调参，设置为False时仅使用默认参数训练模型False
    ENABLE_CROSS_VALIDATION = False  # 是否启用交叉验证，设置为False时跳过交叉验证步骤
    ENABLE_EXTERNAL_VALIDATION = True  # 是否启用外部验证，设置为False时跳过外部验证步骤

    # ========== 外部验证配置 ==========
    EXTERNAL_VALIDATION_SEED = 42  # 外部验证专用随机种子，用于Bootstrap置信区间计算
    EXTERNAL_VALIDATION_FILES = [
        "data/11.04少少少.csv",  # 第一个外部验证数据集文件路径
        "data/11.04少少少2.csv",  # 第二个外部验证数据集文件路径
        "data/11.04少少少3.csv",  # 第三个外部验证数据集文件路径
    ]

    # ========== Bootstrap配置 ==========
    BOOTSTRAP_N_ITER = 1000  # Bootstrap重采样次数
    BOOTSTRAP_CONFIDENCE_LEVEL = 0.95  # Bootstrap置信水平
    BOOTSTRAP_SEED = 42  # Bootstrap专用随机种子，确保所有Bootstrap计算结果可重现且一致

    # ========== SHAP配置 ==========
    SHAP_SAMPLE_SIZE = 100  # SHAP分析使用的样本数量（最多100个样本）
    SHAP_MAX_DISPLAY = 20  # SHAP图中显示的最大特征数量

    # ========== 特征选择配置 ==========
    FEATURE_SELECTION_RF_N_ESTIMATORS = 100  # 特征选择随机森林决策树数量
    FEATURE_SELECTION_TOP_N = 20  # 特征选择显示的特征数量
    
    # ========== SVM-RFE特征选择配置 ==========
    SVM_RFE_STEP = 1  # SVM-RFE每次迭代移除的特征数量
    SVM_RFE_KERNEL = "linear"  # SVM-RFE使用的核函数类型
    
    # ========== LASSO特征选择配置 ==========
    LASSO_ALPHA_MAX_MULTIPLIER = 2  # LASSO最大alpha的乘数系数
    LASSO_ALPHA_MIN_RATIO = 0.001  # LASSO最小alpha与最大alpha的比例
    
    # ========== 特征选择交叉验证配置 ==========
    FEATURE_SELECTION_CV_FOLDS = 10  # 特征选择交叉验证折数，统一用于随机森林、LASSO和SVM-RFE特征选择的交叉验证
    FEATURE_SELECTION_CV_TOP_N = 20  # 交叉验证时评估的最大特征数量
    
    # ========== 特征筛选选中次数阈值配置 ==========
    FS_RF_SELECTION_THRESHOLD = 8  # 随机森林特征选中次数阈值，超过此值认为该特征稳定选中（例如10折中选中8次）
    FS_LASSO_SELECTION_THRESHOLD = 8  # LASSO特征选中次数阈值，超过此值认为该特征稳定选中
    FS_SVM_RFE_SELECTION_THRESHOLD = 8  # SVM-RFE特征选中次数阈值，超过此值认为该特征稳定选中
    
    # ========== LASSO配置 ==========
    LASSO_ALPHA_SEARCH_COUNT = 100  # LASSO alpha搜索数量
    
    # ========== 统计检验配置 ==========
    STATS_SHAPIRO_MIN_SAMPLES = 3  # Shapiro-Wilk检验最小样本量
    STATS_SHAPIRO_MAX_SAMPLES = 5000  # Shapiro-Wilk检验最大样本量
    
    # ========== 数据处理配置 ==========
    DATA_HISTOGRAM_BINS = 10  # 直方图分箱数量
    LASSO_PATH_N_ALPHAS = 200  # LASSO路径的alpha数量
    LASSO_PATH_EPS = 1e-8  # LASSO路径的epsilon值
    MAX_SEED_ATTEMPTS = 100  # 最大尝试种子数
    N_BOOTSTRAPS = 1000  # 自助采样数量
    
    # ========== 数据缩放配置 ==========
    ENABLE_SCALER = True  # 是否启用数据缩放（标准化/归一化），设置为False时不进行任何缩放处理
    SCALER_TYPE = 'normalization'  # 缩放类型，可选值：'standardization'（标准化，均值为0，标准差为1）或 'normalization'（归一化，缩放到[0,1]区间）
    
    # ========== 执行流程控制配置 ==========
    ENABLE_MODEL_TRAINING = True  # 是否启用模型训练（包括贝叶斯调参、模型评估等），设置为False时跳过模型训练阶段，显著提升运行速度
    ENABLE_PLOTS = True  # 是否启用图形生成（包括特征选择图、模型结果可视化、SHAP分析图等），设置为False时跳过所有图形生成，提升运行速度
    ENABLE_MODEL_OUTPUT = True  # 是否启用模型文件输出（保存.pkl模型文件），设置为False时跳过模型文件保存，提升运行速度并减少磁盘占用（依赖ENABLE_MODEL_TRAINING，仅当模型训练启用时生效）
    ENABLE_BOOTSTRAP = True  # 是否启用Bootstrap分析（包括B表10、B表12等），设置为False时跳过Bootstrap计算，显著提升运行速度（依赖ENABLE_MODEL_TRAINING，仅当模型训练启用时生效）
    
    # ========== 统计检验配置 ==========
    STATISTICAL_ALPHA = 0.05  # 显著性水平（95%置信区间）

    # ========== 文件输出配置 ==========
    OUTPUT_FORMAT = 'png'  # 默认输出格式
    OUTPUT_BACKGROUND_COLOR = 'white'  # 输出背景色
    OUTPUT_BBOX_INCHES = 'tight'  # 边框处理方式

    # ========== 算法名称常量 ==========
    ALGO_LINEAR_REGRESSION = '线性回归'
    ALGO_RIDGE_REGRESSION = '岭回归'
    ALGO_LASSO_REGRESSION = 'Lasso回归'
    ALGO_KNN_REGRESSION = 'K近邻回归'
    ALGO_DECISION_TREE = '决策树'
    ALGO_RANDOM_FOREST = '随机森林'
    ALGO_GRADIENT_BOOSTING = '梯度提升'
    ALGO_ADABOOST = 'AdaBoost'
    ALGO_MLP = '多层感知机'
    ALGO_XGBOOST = 'XGBoost'
    ALGO_LIGHTGBM = 'LightGBM'
    ALGO_EXTRA_TREES = '极端随机树'
    ALGO_ELASTIC_NET = '弹性网络'
    ALGO_GAUSSIAN_PROCESS = '高斯过程回归'
    ALGO_BAYESIAN_RIDGE = '贝叶斯岭回归'
    ALGO_POLYNOMIAL_REGRESSION = '多项式回归'
    ALGO_SUPPORT_VECTOR_MACHINE = '支持向量机'
    ALGO_CATBOOST = 'CatBoost回归'
    ALGO_HIST_GRADIENT_BOOSTING = '直方图梯度提升树回归'
    ALGO_SGD_REGRESSION = '随机梯度下降回归'
    
    # ========== 特征选择方法名称常量 ==========
    FS_RANDOM_FOREST = '随机森林'
    FS_LASSO = 'LASSO'
    FS_SVM_RFE = 'SVM-RFE'

    # ========== 模型开关配置 ==========
    ENABLE_LINEAR_REGRESSION = True  # 是否启用线性回归
    ENABLE_RIDGE_REGRESSION = True  # 是否启用岭回归
    ENABLE_LASSO_REGRESSION = False  # 是否启用Lasso回归
    ENABLE_KNN_REGRESSION = True  # 是否启用K近邻回归
    ENABLE_DECISION_TREE = True  # 是否启用决策树
    ENABLE_RANDOM_FOREST = True  # 是否启用随机森林
    ENABLE_GRADIENT_BOOSTING = True  # 是否启用梯度提升
    ENABLE_ADABOOST = True  # 是否启用AdaBoost
    ENABLE_MLP = True  # 是否启用多层感知机
    ENABLE_XGBOOST = True  # 是否启用XGBoost
    ENABLE_LIGHTGBM = True  # 是否启用LightGBM
    ENABLE_EXTRA_TREES = True  # 是否启用极端随机树
    ENABLE_ELASTIC_NET = False  # 是否启用弹性网络
    ENABLE_GAUSSIAN_PROCESS = True  # 是否启用高斯过程回归
    ENABLE_BAYESIAN_RIDGE = False  # 是否启用贝叶斯岭回归
    ENABLE_POLYNOMIAL_REGRESSION = True  # 是否启用多项式回归
    ENABLE_SUPPORT_VECTOR_MACHINE = True  # 是否启用支持向量机
    ENABLE_CATBOOST = False  # 是否启用CatBoost回归
    ENABLE_HIST_GRADIENT_BOOSTING = True  # 是否启用直方图梯度提升树回归
    ENABLE_SGD_REGRESSION = True  # 是否启用随机梯度下降回归

    # ========== 可视化配置 ==========
    PLOT_FONT_SIZE = 14  # 图表字体大小
    PLOT_LABEL_SIZE = 14  # 轴标签字体大小
    PLOT_TITLE_SIZE = 14  # 标题字体大小
    PLOT_TICK_SIZE = 14  # 刻度标签字体大小
    PLOT_LINE_WIDTH = 2  # 线条宽度
    PLOT_MARKER_SIZE = 60  # 标记点大小
    PLOT_ALPHA = 0.7  # 透明度
    PLOT_CONFIDENCE_ALPHA = 0.12  # 置信区间透明度
    PLOT_COLOR_TRAIN = '#1f77b4'  # 训练集颜色（蓝色）
    PLOT_COLOR_TEST = '#ff7f0e'  # 测试集颜色（橙色）
    PLOT_COLOR_11_LINE = 'k--'  # 1:1参考线颜色和样式
    PLOT_DPI = 600  # 图像分辨率
    PLOT_FIGSIZE_LARGE = (12, 10)  # 大图尺寸 (宽, 高)
    PLOT_FIGSIZE_MEDIUM = (12, 8)  # 中图尺寸
    PLOT_FIGSIZE_SMALL = (15, 6)  # 小图尺寸
    PLOT_FIGSIZE_SHAP = (12, 8)  # SHAP图尺寸
    PLOT_FIGSIZE_COMPARISON = (14, 4)  # 比较图尺寸
    PLOT_TITLE_FONT_SIZE = 18  # 标题字体大小（大）
    PLOT_LABEL_FONT_SIZE = 14  # 标签字体大小
    PLOT_SPINE_WIDTH = 1.5  # 边框线宽
    PLOT_GRID_LINEWIDTH = 0.8  # 网格线宽
    PLOT_GRID_ALPHA = 0.7  # 网格透明度
    PLOT_HIST_BINS = 10  # 直方图bins数量
    PLOT_GRID_COLOR = '#cccccc'  # 网格颜色
    PLOT_REF_LINE_COLOR = 'red'  # 参考线颜色
    PLOT_ERROR_COLOR = 'red'  # 误差分布颜色
    PLOT_HIST_COLOR = 'skyblue'  # 直方图颜色
    PLOT_EDGE_COLOR = 'black'  # 图形边缘颜色

    # ========== 可视化字体配置 ==========
    PLOT_LEGEND_FONT_SIZE = 11  # 图例字体大小
    PLOT_LEGEND_FONT_SIZE_SMALL = 10  # 图例字体大小（小）
    PLOT_LEGEND_FONT_SIZE_LARGE = 14  # 图例字体大小（大）
    PLOT_SUBTITLE_FONT_SIZE = 18  # 子标题字体大小
    PLOT_TITLE_FONT_SIZE_LARGE = 22  # 标题字体大小（大）
    PLOT_FONT_SIZE_VERY_SMALL = 10  # 极小字体大小
    PLOT_FONT_SIZE_SMALL = 11  # 小字体大小
    PLOT_FONT_SIZE_NORMAL = 12  # 正常字体大小
    PLOT_FONT_SIZE_MEDIUM = 14  # 中等字体大小
    PLOT_FONT_SIZE_LARGE = 16  # 大字体大小
    PLOT_FONT_SIZE_VERY_LARGE = 18  # 极大字体大小
    PLOT_FONT_SIZE_TITLE = 22  # 标题字体大小

    # ========== 可视化图形大小配置 ==========
    PLOT_FIGSIZE_WIDE = (14, 8)  # 宽图尺寸

    # ========== 可视化线宽配置 ==========
    PLOT_LINE_WIDTH_VERY_THIN = 0.5  # 极细线宽
    PLOT_LINE_WIDTH_THIN = 0.8  # 细线宽
    PLOT_LINE_WIDTH_NORMAL = 1  # 正常线宽
    PLOT_LINE_WIDTH_MEDIUM = 1.5  # 中等线宽
    PLOT_LINE_WIDTH_THICK = 2  # 粗线宽

    # ========== 可视化标记大小配置 ==========
    PLOT_MARKER_SIZE_SMALL = 4  # 小标记大小
    PLOT_MARKER_SIZE_MEDIUM = 6  # 中等标记大小
    PLOT_MARKER_SIZE_LARGE = 8  # 大标记大小

    # ========== 可视化透明度配置 ==========
    PLOT_ALPHA_HIGH = 0.8  # 高透明度
    PLOT_ALPHA_MEDIUM = 0.7  # 中等透明度
    PLOT_ALPHA_MEDIUM_LOW = 0.6  # 中低透明度
    PLOT_ALPHA_LOW = 0.15  # 低透明度
    PLOT_ALPHA_VERY_LOW = 0.1  # 极低透明度
    PLOT_ALPHA_GRID = 0.3  # 网格透明度
    PLOT_ALPHA_CONFIDENCE = 0.12  # 置信区间透明度
    PLOT_ALPHA_FILL = 0.2  # 填充区域透明度
    PLOT_ALPHA_LINE = 0.9  # 线条高透明度
    PLOT_ALPHA_LINE5 = 0.95  # 高透明度（图例/文本框边框）
    PLOT_ALPHA_TEXT_BOX = 0.95  # 文本框透明度
    PLOT_ALPHA_REF_LINE = 0.5  # 参考线透明度

    # ========== 可视化散点大小配置 ==========
    PLOT_SCATTER_SIZE_SMALL = 60  # 小散点大小
    PLOT_SCATTER_SIZE_LARGE = 200  # 大散点大小

    # ========== 可视化标签间距配置 ==========
    PLOT_LABELPAD_SMALL = 6  # 小标签间距
    PLOT_LABELPAD_NORMAL = 10  # 正常标签间距

    # ========== 可视化标题间距配置 ==========
    PLOT_TITLEPAD_MEDIUM = 15  # 子图标题间距
    PLOT_TITLEPAD_LARGE = 20  # 主图标题间距
    PLOT_TEXTBOX_PAD = 0.4  # 文本框内边距

    # ========== 可视化采样点配置 ==========
    PLOT_LINE_SAMPLES = 100  # 绘制拟合线时的采样点数量

    # ========== 可视化文本框位置配置 ==========
    PLOT_TEXTBOX_X_POS = 0.98  # 文本框x位置（右下角）
    PLOT_TEXTBOX_Y_POS = 0.02  # 文本框y位置（右下角）

    # ========== 可视化颜色配置 ==========
    PLOT_COLOR_BLUE = '#1f77b4'  # 蓝色
    PLOT_COLOR_ORANGE = '#ff7f0e'  # 橙色
    PLOT_COLOR_GRAY = '#cccccc'  # 灰色
    PLOT_COLOR_BLACK = '#000000'  # 黑色
    PLOT_COLOR_SKY_BLUE = 'skyblue'  # 天蓝色

    # ========== 可视化总标题位置配置 ==========
    PLOT_SUPTITLE_Y_POS = 0.95  # 总标题y位置

    @classmethod
    def validate_config(cls):
        """验证配置参数的有效性
        
        检查所有配置参数是否在合理范围内，确保实验可以正常运行。
        将所有验证错误收集到列表中，最后统一抛出ValueError异常，
        使用户能够一次性看到所有问题。
        """
        errors = []
        
        # ========== 数据配置验证 ==========
        # 验证测试集比例
        if cls.TEST_SIZE <= 0 or cls.TEST_SIZE >= 1:
            errors.append(f"测试集比例必须在(0, 1)之间，当前值: {cls.TEST_SIZE}")
        
        # 验证K近邻邻居数
        if cls.KNN_NEIGHBORS < 1:
            errors.append(f"K近邻邻居数必须大于等于1，当前值: {cls.KNN_NEIGHBORS}")
        
        # 验证分箱数量
        if cls.BINS < 1:
            errors.append(f"分箱数量必须大于等于1，当前值: {cls.BINS}")
        
        # 验证分类特征数量
        if cls.NUM_CATEGORICAL_FEATURES < 0:
            errors.append(f"分类特征数量必须大于等于0，当前值: {cls.NUM_CATEGORICAL_FEATURES}")
        
        # ========== 训练配置验证 ==========
        # 验证随机种子
        if cls.RANDOM_STATE < 0:
            errors.append(f"随机种子必须大于等于0，当前值: {cls.RANDOM_STATE}")
        
        # 验证最大迭代次数
        if cls.MAX_ITER < 1:
            errors.append(f"最大迭代次数必须大于等于1，当前值: {cls.MAX_ITER}")
        
        # 验证显著性水平
        if cls.ALPHA_LEVEL <= 0 or cls.ALPHA_LEVEL >= 1:
            errors.append(f"显著性水平必须在(0, 1)之间，当前值: {cls.ALPHA_LEVEL}")
        
        # 验证并行计算线程数
        if cls.N_JOBS < 1:
            errors.append(f"并行计算线程数必须大于等于1，当前值: {cls.N_JOBS}")
        
        # 验证Lasso交叉验证折数
        if cls.LASSO_CV_FOLDS < 2:
            errors.append(f"Lasso交叉验证折数必须大于等于2，当前值: {cls.LASSO_CV_FOLDS}")
        
        # 验证随机森林最大深度
        if cls.RF_MAX_DEPTH < 1:
            errors.append(f"随机森林最大深度必须大于等于1，当前值: {cls.RF_MAX_DEPTH}")
        
        # 验证随机森林基学习器数量
        if cls.RF_N_ESTIMATORS < 1:
            errors.append(f"随机森林基学习器数量必须大于等于1，当前值: {cls.RF_N_ESTIMATORS}")
        
        # 验证梯度提升弱学习器数量
        if cls.GB_N_ESTIMATORS < 1:
            errors.append(f"梯度提升弱学习器数量必须大于等于1，当前值: {cls.GB_N_ESTIMATORS}")
        
        # 验证AdaBoost弱学习器数量
        if cls.ADA_N_ESTIMATORS < 1:
            errors.append(f"AdaBoost弱学习器数量必须大于等于1，当前值: {cls.ADA_N_ESTIMATORS}")
        
        # 验证XGBoost弱学习器数量
        if cls.XGB_N_ESTIMATORS < 1:
            errors.append(f"XGBoost弱学习器数量必须大于等于1，当前值: {cls.XGB_N_ESTIMATORS}")
        
        # 验证LightGBM弱学习器数量
        if cls.LGB_N_ESTIMATORS < 1:
            errors.append(f"LightGBM弱学习器数量必须大于等于1，当前值: {cls.LGB_N_ESTIMATORS}")
        
        # 验证极端随机树基学习器数量
        if cls.ET_N_ESTIMATORS < 1:
            errors.append(f"极端随机树基学习器数量必须大于等于1，当前值: {cls.ET_N_ESTIMATORS}")
        
        # 验证通用交叉验证折数
        if cls.CV_FOLDS < 2:
            errors.append(f"通用交叉验证折数必须大于等于2，当前值: {cls.CV_FOLDS}")
        
        # 验证贝叶斯优化迭代次数
        if cls.BAYES_N_ITER < 1:
            errors.append(f"贝叶斯优化迭代次数必须大于等于1，当前值: {cls.BAYES_N_ITER}")
        
        # 验证贝叶斯优化交叉验证折数
        if cls.BAYES_CV_FOLDS < 2:
            errors.append(f"贝叶斯优化交叉验证折数必须大于等于2，当前值: {cls.BAYES_CV_FOLDS}")
        
        # 验证K折交叉验证划分次数
        if cls.KFOLD_SPLITS < 2:
            errors.append(f"K折交叉验证划分次数必须大于等于2，当前值: {cls.KFOLD_SPLITS}")
        
        # ========== Bootstrap配置验证 ==========
        # 验证Bootstrap重采样次数
        if cls.BOOTSTRAP_N_ITER < 1:
            errors.append(f"Bootstrap重采样次数必须大于等于1，当前值: {cls.BOOTSTRAP_N_ITER}")
        
        # 验证Bootstrap置信水平
        if cls.BOOTSTRAP_CONFIDENCE_LEVEL <= 0 or cls.BOOTSTRAP_CONFIDENCE_LEVEL >= 1:
            errors.append(f"Bootstrap置信水平必须在(0, 1)之间，当前值: {cls.BOOTSTRAP_CONFIDENCE_LEVEL}")
        
        # ========== SHAP配置验证 ==========
        # 验证SHAP分析样本数量
        if cls.SHAP_SAMPLE_SIZE < 1:
            errors.append(f"SHAP分析样本数量必须大于等于1，当前值: {cls.SHAP_SAMPLE_SIZE}")
        
        # 验证SHAP显示特征数量
        if cls.SHAP_MAX_DISPLAY < 1:
            errors.append(f"SHAP图显示特征数量必须大于等于1，当前值: {cls.SHAP_MAX_DISPLAY}")
        
        # ========== 特征选择配置验证 ==========
        # 验证特征选择交叉验证折数
        if cls.FEATURE_SELECTION_CV_FOLDS < 2:
            errors.append(f"特征选择交叉验证折数必须大于等于2，当前值: {cls.FEATURE_SELECTION_CV_FOLDS}")
        
        # 验证特征选择显示数量
        if cls.FEATURE_SELECTION_TOP_N < 1:
            errors.append(f"特征选择显示数量必须大于等于1，当前值: {cls.FEATURE_SELECTION_TOP_N}")
        
        # 验证特征选中次数阈值
        if cls.FS_RF_SELECTION_THRESHOLD < 1 or cls.FS_RF_SELECTION_THRESHOLD > cls.FEATURE_SELECTION_CV_FOLDS:
            errors.append(f"随机森林特征选中次数阈值必须在1到{cls.FEATURE_SELECTION_CV_FOLDS}之间，当前值: {cls.FS_RF_SELECTION_THRESHOLD}")
        
        if cls.FS_LASSO_SELECTION_THRESHOLD < 1 or cls.FS_LASSO_SELECTION_THRESHOLD > cls.FEATURE_SELECTION_CV_FOLDS:
            errors.append(f"LASSO特征选中次数阈值必须在1到{cls.FEATURE_SELECTION_CV_FOLDS}之间，当前值: {cls.FS_LASSO_SELECTION_THRESHOLD}")
        
        if cls.FS_SVM_RFE_SELECTION_THRESHOLD < 1 or cls.FS_SVM_RFE_SELECTION_THRESHOLD > cls.FEATURE_SELECTION_CV_FOLDS:
            errors.append(f"SVM-RFE特征选中次数阈值必须在1到{cls.FEATURE_SELECTION_CV_FOLDS}之间，当前值: {cls.FS_SVM_RFE_SELECTION_THRESHOLD}")
        
        # ========== LASSO配置验证 ==========
        # 验证LASSO alpha搜索数量
        if cls.LASSO_ALPHA_SEARCH_COUNT < 1:
            errors.append(f"LASSO alpha搜索数量必须大于等于1，当前值: {cls.LASSO_ALPHA_SEARCH_COUNT}")
        
        # ========== 统计检验配置验证 ==========
        # 验证Shapiro-Wilk检验最小样本量
        if cls.STATS_SHAPIRO_MIN_SAMPLES < 1:
            errors.append(f"Shapiro-Wilk检验最小样本量必须大于等于1，当前值: {cls.STATS_SHAPIRO_MIN_SAMPLES}")
        
        # 验证Shapiro-Wilk检验最大样本量
        if cls.STATS_SHAPIRO_MAX_SAMPLES < cls.STATS_SHAPIRO_MIN_SAMPLES:
            errors.append(f"Shapiro-Wilk检验最大样本量必须大于等于最小样本量，当前值: {cls.STATS_SHAPIRO_MAX_SAMPLES}")
        
        # 验证显著性水平（统计检验）
        if cls.STATISTICAL_ALPHA <= 0 or cls.STATISTICAL_ALPHA >= 1:
            errors.append(f"统计检验显著性水平必须在(0, 1)之间，当前值: {cls.STATISTICAL_ALPHA}")
        
        # ========== 数据处理配置验证 ==========
        # 验证直方图分箱数量
        if cls.DATA_HISTOGRAM_BINS < 1:
            errors.append(f"直方图分箱数量必须大于等于1，当前值: {cls.DATA_HISTOGRAM_BINS}")
        
        # 验证LASSO路径alpha数量
        if cls.LASSO_PATH_N_ALPHAS < 1:
            errors.append(f"LASSO路径alpha数量必须大于等于1，当前值: {cls.LASSO_PATH_N_ALPHAS}")
        
        # 验证最大尝试种子数
        if cls.MAX_SEED_ATTEMPTS < 1:
            errors.append(f"最大尝试种子数必须大于等于1，当前值: {cls.MAX_SEED_ATTEMPTS}")
        
        # 验证自助采样数量
        if cls.N_BOOTSTRAPS < 1:
            errors.append(f"自助采样数量必须大于等于1，当前值: {cls.N_BOOTSTRAPS}")
        
        # ========== 数据缩放配置验证 ==========
        # 验证缩放类型
        if cls.SCALER_TYPE not in ['standardization', 'normalization']:
            errors.append(f"无效的缩放类型: {cls.SCALER_TYPE}，请使用 'standardization' 或 'normalization'")
        
        # ========== 文件输出配置验证 ==========
        # 验证输出格式
        if cls.OUTPUT_FORMAT not in ['png', 'pdf', 'svg']:
            errors.append(f"无效的输出格式: {cls.OUTPUT_FORMAT}，请使用 'png'、'pdf' 或 'svg'")
        
        # ========== 开关配置验证（布尔类型检查） ==========
        boolean_configs = [
            'ENABLE_BAYESIAN_TUNING', 'ENABLE_CROSS_VALIDATION', 'ENABLE_EXTERNAL_VALIDATION',
            'ENABLE_SCALER', 'ENABLE_MODEL_TRAINING', 'ENABLE_PLOTS', 'ENABLE_MODEL_OUTPUT',
            'ENABLE_BOOTSTRAP', 'REPRODUCIBLE',
            'ENABLE_LINEAR_REGRESSION', 'ENABLE_RIDGE_REGRESSION', 'ENABLE_LASSO_REGRESSION',
            'ENABLE_KNN_REGRESSION', 'ENABLE_DECISION_TREE', 'ENABLE_RANDOM_FOREST',
            'ENABLE_GRADIENT_BOOSTING', 'ENABLE_ADABOOST', 'ENABLE_MLP', 'ENABLE_XGBOOST',
            'ENABLE_LIGHTGBM', 'ENABLE_EXTRA_TREES', 'ENABLE_ELASTIC_NET',
            'ENABLE_GAUSSIAN_PROCESS', 'ENABLE_BAYESIAN_RIDGE', 'ENABLE_POLYNOMIAL_REGRESSION',
            'ENABLE_SUPPORT_VECTOR_MACHINE', 'ENABLE_CATBOOST', 'ENABLE_HIST_GRADIENT_BOOSTING',
            'ENABLE_SGD_REGRESSION'
        ]
        
        for config_name in boolean_configs:
            value = getattr(cls, config_name)
            if not isinstance(value, bool):
                errors.append(f"{config_name} 必须是布尔类型，当前类型: {type(value).__name__}")
        
        # ========== 文件路径验证 ==========
        # 验证主数据文件存在
        data_file_path = Path(cls.DATA_FILE)
        if not data_file_path.exists():
            errors.append(f"主数据文件不存在: {data_file_path.resolve()}")
        
        # 验证外部验证数据文件存在（仅当启用外部验证时）
        if cls.ENABLE_EXTERNAL_VALIDATION:
            for ext_file in cls.EXTERNAL_VALIDATION_FILES:
                ext_file_path = Path(ext_file)
                if not ext_file_path.exists():
                    errors.append(f"外部验证数据文件不存在: {ext_file_path.resolve()}")
        
        # ========== 依赖关系验证 ==========
        # 验证至少启用一个模型
        model_enable_flags = [
            cls.ENABLE_LINEAR_REGRESSION, cls.ENABLE_RIDGE_REGRESSION, cls.ENABLE_LASSO_REGRESSION,
            cls.ENABLE_KNN_REGRESSION, cls.ENABLE_DECISION_TREE, cls.ENABLE_RANDOM_FOREST,
            cls.ENABLE_GRADIENT_BOOSTING, cls.ENABLE_ADABOOST, cls.ENABLE_MLP, cls.ENABLE_XGBOOST,
            cls.ENABLE_LIGHTGBM, cls.ENABLE_EXTRA_TREES, cls.ENABLE_ELASTIC_NET,
            cls.ENABLE_GAUSSIAN_PROCESS, cls.ENABLE_BAYESIAN_RIDGE, cls.ENABLE_POLYNOMIAL_REGRESSION,
            cls.ENABLE_SUPPORT_VECTOR_MACHINE, cls.ENABLE_CATBOOST, cls.ENABLE_HIST_GRADIENT_BOOSTING,
            cls.ENABLE_SGD_REGRESSION
        ]
        
        if not any(model_enable_flags):
            errors.append("必须至少启用一个模型（ENABLE_*），当前所有模型均被禁用")
        
        # 验证ENABLE_MODEL_OUTPUT依赖ENABLE_MODEL_TRAINING
        if cls.ENABLE_MODEL_OUTPUT and not cls.ENABLE_MODEL_TRAINING:
            errors.append("ENABLE_MODEL_OUTPUT 为 True 时，ENABLE_MODEL_TRAINING 必须也为 True")
        
        # 验证ENABLE_BOOTSTRAP依赖ENABLE_MODEL_TRAINING
        if cls.ENABLE_BOOTSTRAP and not cls.ENABLE_MODEL_TRAINING:
            errors.append("ENABLE_BOOTSTRAP 为 True 时，ENABLE_MODEL_TRAINING 必须也为 True")
        
        # 验证贝叶斯调参依赖模型训练
        if cls.ENABLE_BAYESIAN_TUNING and not cls.ENABLE_MODEL_TRAINING:
            errors.append("ENABLE_BAYESIAN_TUNING 为 True 时，ENABLE_MODEL_TRAINING 必须也为 True")
        
        # ========== 如果有错误，统一抛出异常 ==========
        if errors:
            error_message = "配置验证失败，发现以下问题:\n"
            for i, error in enumerate(errors, 1):
                error_message += f"  {i}. {error}\n"
            raise ValueError(error_message)
        
        print("配置参数验证通过")


MODEL_CONFIG = {
    '线性回归': {
        'model': LinearRegression(),
        'param_space': {}
    }
}

if Real is not None:
    MODEL_CONFIG.update({
        '岭回归': {
            'model': Ridge(random_state=Config.RANDOM_STATE),
            'param_space': {'model__alpha': Real(0.001, 100, prior='log-uniform')}
        },
        'Lasso回归': {
            'model': Lasso(max_iter=Config.LASSO_MAX_ITER, random_state=Config.RANDOM_STATE),
            'param_space': {'model__alpha': Real(0.001, 100, prior='log-uniform')}
        },
        'K近邻回归': {
            'model': KNeighborsRegressor(n_neighbors=5),
            'param_space': {
                'model__n_neighbors': Integer(3, 15),
                'model__weights': Categorical(['uniform', 'distance']),
                'model__metric': Categorical(['euclidean', 'manhattan'])
            }
        },
        '决策树': {
            'model': DecisionTreeRegressor(random_state=Config.RANDOM_STATE),
            'param_space': {
                'model__max_depth': Integer(2, 10),
                'model__min_samples_split': Integer(2, 20),
                'model__min_samples_leaf': Integer(1, 10),
                'model__max_features': Categorical(['sqrt', 'log2', None])
            }
        },
        '随机森林': {
            'model': RandomForestRegressor(n_estimators=Config.RF_N_ESTIMATORS, random_state=Config.RANDOM_STATE),
            'param_space': {
                'model__n_estimators': Integer(50, 200),
                'model__max_depth': Integer(3, 12),
                'model__min_samples_split': Integer(2, 20),
                'model__min_samples_leaf': Integer(1, 10),
                'model__max_features': Categorical(['sqrt', 'log2', None])
            }
        },
        '梯度提升': {
            'model': GradientBoostingRegressor(n_estimators=Config.GB_N_ESTIMATORS, random_state=Config.RANDOM_STATE),
            'param_space': {
                'model__n_estimators': Integer(50, 200),
                'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'model__max_depth': Integer(2, 8),
                'model__min_samples_split': Integer(2, 20),
                'model__min_samples_leaf': Integer(1, 10)
            }
        },
        'AdaBoost': {
            'model': AdaBoostRegressor(n_estimators=Config.ADA_N_ESTIMATORS, random_state=Config.RANDOM_STATE),
            'param_space': {
                'model__n_estimators': Integer(50, 200),
                'model__learning_rate': Real(0.01, 1.0, prior='log-uniform')
            }
        },
        '多层感知机': {
            'model': MLPRegressor(hidden_layer_sizes=(100,), max_iter=Config.MAX_ITER,
                                  random_state=Config.RANDOM_STATE, early_stopping=True),
            'param_space': {
                'model__hidden_layer_sizes': Categorical([(50,), (100,), (50, 50), (100, 50), (100, 100)]),
                'model__activation': Categorical(['relu', 'tanh', 'logistic']),
                'model__solver': Categorical(['adam', 'sgd']),
                'model__learning_rate_init': Real(0.0001, 0.1, prior='log-uniform'),
                'model__alpha': Real(0.0001, 0.1, prior='log-uniform')
            }
        },
        '极端随机树': {
            'model': ExtraTreesRegressor(n_estimators=Config.ET_N_ESTIMATORS, random_state=Config.RANDOM_STATE),
            'param_space': {
                'model__n_estimators': Integer(50, 200),
                'model__max_depth': Integer(3, 12),
                'model__min_samples_split': Integer(2, 20),
                'model__min_samples_leaf': Integer(1, 10),
                'model__max_features': Categorical(['sqrt', 'log2', None])
            }
        },
        '弹性网络': {
            'model': ElasticNet(max_iter=Config.MAX_ITER, random_state=Config.RANDOM_STATE),
            'param_space': {
                'model__alpha': Real(0.0001, 100, prior='log-uniform'),
                'model__l1_ratio': Real(0.0, 1.0)
            }
        },
        '高斯过程回归': {
            'model': GaussianProcessRegressor(random_state=Config.RANDOM_STATE,
                                               kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0),
                                               normalize_y=True, n_restarts_optimizer=3),
            'param_space': {'model__alpha': Real(1e-10, 1.0, prior='log-uniform')}
        },
        '贝叶斯岭回归': {
            'model': BayesianRidge(compute_score=True, max_iter=300),
            'param_space': {
                'model__alpha_1': Real(1e-10, 1e-5, prior='log-uniform'),
                'model__alpha_2': Real(1e-10, 1e-5, prior='log-uniform'),
                'model__lambda_1': Real(1e-10, 1e-5, prior='log-uniform'),
                'model__lambda_2': Real(1e-10, 1e-5, prior='log-uniform'),
                'model__alpha_init': Real(0.001, 1, prior='log-uniform'),
                'model__lambda_init': Real(0.001, 1, prior='log-uniform')
            }
        },
        '多项式回归': {
            'model': Pipeline(steps=[
                ('poly', PolynomialFeatures(degree=2, include_bias=False)),
                ('linear', LinearRegression())
            ]),
            'param_space': {
                'model__poly__degree': Integer(2, 5),
                'model__poly__include_bias': Categorical([False])
            }
        },
        '支持向量机': {
            'model': SVR(kernel='rbf', gamma='scale', max_iter=Config.MAX_ITER),
            'param_space': {
                'model__kernel': Categorical(['linear', 'rbf', 'poly']),
                'model__C': Real(0.001, 1000, prior='log-uniform'),
                'model__gamma': Real(0.0001, 100, prior='log-uniform'),
                'model__degree': Integer(2, 5)
            }
        },
        '直方图梯度提升树回归': {
            'model': HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.05, max_depth=6,
                                                    random_state=Config.RANDOM_STATE, early_stopping=True,
                                                    validation_fraction=0.1),
            'param_space': {
                'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
                'model__max_depth': Integer(4, 10),
                'model__max_leaf_nodes': Integer(31, 255),
                'model__l2_regularization': Real(0.0, 1.0),
                'model__min_samples_leaf': Integer(1, 20)
            }
        },
        '随机梯度下降回归': {
            'model': SGDRegressor(loss='squared_error', penalty='l2', alpha=0.0001,
                                   max_iter=Config.MAX_ITER, random_state=Config.RANDOM_STATE,
                                   early_stopping=True, validation_fraction=0.1),
            'param_space': {
                'model__loss': Categorical(['squared_error', 'huber', 'epsilon_insensitive']),
                'model__penalty': Categorical(['l2', 'l1', 'elasticnet']),
                'model__alpha': Real(1e-6, 1, prior='log-uniform'),
                'model__learning_rate': Categorical(['constant', 'optimal', 'invscaling', 'adaptive']),
                'model__eta0': Real(0.001, 1, prior='log-uniform')
            }
        }
    })

if xgb is not None and Real is not None:
    MODEL_CONFIG['XGBoost'] = {
        'model': xgb.XGBRegressor(n_estimators=Config.XGB_N_ESTIMATORS, objective='reg:squarederror',
                                    random_state=Config.RANDOM_STATE, verbosity=0),
        'param_space': {
            'model__n_estimators': Integer(50, 200),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__max_depth': Integer(2, 8),
            'model__subsample': Real(0.5, 1.0),
            'model__colsample_bytree': Real(0.5, 1.0),
            'model__reg_alpha': Real(0.0, 10.0),
            'model__reg_lambda': Real(0.0, 10.0)
        }
    }

if lgb is not None and Real is not None:
    MODEL_CONFIG['LightGBM'] = {
        'model': lgb.LGBMRegressor(n_estimators=Config.LGB_N_ESTIMATORS, random_state=Config.RANDOM_STATE, verbose=-1),
        'param_space': {
            'model__n_estimators': Integer(50, 200),
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__max_depth': Integer(2, 8),
            'model__num_leaves': Integer(15, 127),
            'model__subsample': Real(0.5, 1.0),
            'model__colsample_bytree': Real(0.5, 1.0),
            'model__reg_alpha': Real(0.0, 10.0),
            'model__reg_lambda': Real(0.0, 10.0)
        }
    }

if CatBoostRegressor is not None and Real is not None:
    MODEL_CONFIG['CatBoost回归'] = {
        'model': CatBoostRegressor(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            random_seed=Config.RANDOM_STATE,
            verbose=0,
            early_stopping_rounds=100
        ),
        'param_space': {
            'model__learning_rate': Real(0.01, 0.3, prior='log-uniform'),
            'model__depth': Integer(4, 10),
            'model__l2_leaf_reg': Real(0.01, 10, prior='log-uniform'),
            'model__bagging_temperature': Real(0.0, 1.0)
        }
    }

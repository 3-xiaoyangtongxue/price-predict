import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor  # 导入 RandomForestRegressor

# 加载训练集和测试集
Train_data = pd.read_csv('./used_car_train_20200313.csv', sep=' ')
TestA_data = pd.read_csv('./used_car_testA_20200313.csv/used_car_testA_20200313.csv', sep=' ')

# 数据预处理
# 处理 Train_data
Train_data.dropna(inplace=True)
Train_data['regDate'] = pd.to_datetime(Train_data['regDate'], format='%Y%m%d', errors='coerce')
Train_data['creatDate'] = pd.to_datetime(Train_data['creatDate'], format='%Y%m%d', errors='coerce')

# 提取日期字段中的年、月、日
Train_data['regYear'] = Train_data['regDate'].dt.year
Train_data['regMonth'] = Train_data['regDate'].dt.month
Train_data['regDay'] = Train_data['regDate'].dt.day
Train_data['creatYear'] = Train_data['creatDate'].dt.year
Train_data['creatMonth'] = Train_data['creatDate'].dt.month
Train_data['creatDay'] = Train_data['creatDate'].dt.day

# 删除日期字段
Train_data = Train_data.drop(columns=['regDate', 'creatDate'], errors='ignore')

# 对 TestA_data 进行同样的处理
TestA_data['regDate'] = pd.to_datetime(TestA_data['regDate'], format='%Y%m%d', errors='coerce')
TestA_data['creatDate'] = pd.to_datetime(TestA_data['creatDate'], format='%Y%m%d', errors='coerce')

TestA_data['regYear'] = TestA_data['regDate'].dt.year
TestA_data['regMonth'] = TestA_data['regDate'].dt.month
TestA_data['regDay'] = TestA_data['regDate'].dt.day
TestA_data['creatYear'] = TestA_data['creatDate'].dt.year
TestA_data['creatMonth'] = TestA_data['creatDate'].dt.month
TestA_data['creatDay'] = TestA_data['creatDate'].dt.day

# 删除日期字段
TestA_data = TestA_data.drop(columns=['regDate', 'creatDate'], errors='ignore')

# 对类别字段进行 One-Hot Encoding
columns_to_encode = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'seller', 'offerType']
Train_data = pd.get_dummies(Train_data, columns=columns_to_encode, drop_first=True, dummy_na=True)
TestA_data = pd.get_dummies(TestA_data, columns=columns_to_encode, drop_first=True, dummy_na=True)

# 对齐 Train_data 和 TestA_data 的特征
TestA_data = TestA_data.reindex(columns=Train_data.columns, fill_value=0)

# 特征和目标变量
X = Train_data.drop(columns=['price', 'SaleID'])
y = Train_data['price']
X_test = TestA_data.drop(columns=['price', 'SaleID'], errors='ignore')

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 划分训练集和验证集
X_train, X_valid, y_train, y_valid = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 定义 RandomForest 回归模型
model = RandomForestRegressor(
    n_estimators=200,  # 树的数量
    max_depth=10,  # 树的最大深度
    random_state=42,
    n_jobs=-1  # 使用所有可用的CPU核心
)

# 使用训练集进行训练
model.fit(X_train, y_train)

# 在验证集上进行预测并计算 MAE
y_valid_pred = model.predict(X_valid)
mae = mean_absolute_error(y_valid, y_valid_pred)
print(f"Mean Absolute Error (MAE) on validation set: {mae:.2f}")

# 在测试集上进行预测
y_pred = model.predict(X_test_scaled)

# 对预测结果进行四舍五入，保留整数
y_pred_rounded = np.round(y_pred)  # 四舍五入到最接近的整数

# 创建提交的 DataFrame
submission = pd.DataFrame({
    'SaleID': TestA_data['SaleID'],
    'price': y_pred_rounded.astype(int)  # 将结果转换为整数类型
})

# 保存预测结果为 CSV 文件
submission.to_csv('submission.csv', index=False)

print("Submission file has been saved as 'submission_with_mae.csv'")

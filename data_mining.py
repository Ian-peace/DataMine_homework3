import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate, cross_val_score

df = pd.read_csv("data/hotel_bookings.csv")
print("df.shape:\n", df.shape)  #  (119390, 32)
print("df.head:\n", df.head())
#            hotel  is_canceled  lead_time  arrival_date_year arrival_date_month  ...   adr  required_car_parking_spaces  total_of_special_requests  reservation_status  reservation_status_date
# 0  Resort Hotel            0        342               2015               July  ...   0.0                            0                          0           Check-Out               2015-07-01
# 1  Resort Hotel            0        737               2015               July  ...   0.0                            0                          0           Check-Out               2015-07-01
# 2  Resort Hotel            0          7               2015               July  ...  75.0                            0                          0           Check-Out               2015-07-02
# 3  Resort Hotel            0         13               2015               July  ...  75.0                            0                          0           Check-Out               2015-07-02
# 4  Resort Hotel            0         14               2015               July  ...  98.0                            0                          1           Check-Out               2015-07-03


# 缺失值处理
print("缺失值处理前：")
print(df.isnull().any()) # 每列是否有缺失值
print(df.isnull().sum()) # 每列的缺失值总行数
nan_replace = {"children": 0,"country": "Unknown", "agent": 0, "company": 0}
df_cln = df.fillna(nan_replace)
df_cln["meal"].replace("Undefined", "SC", inplace=True)
zero_guests = list(df_cln.loc[df_cln["adults"]
                   + df_cln["children"]
                   + df_cln["babies"]==0].index)
df_cln.drop(df_cln.index[zero_guests], inplace=True)
df = df_cln
print("缺失值处理后：")
print("df.shape:\n", df.shape)  #  (119390, 32)
print(df.isnull().any()) # 每列是否有缺失值
print(df.isnull().sum()) # 每列的缺失值总行数


## 1. 基本情况：城市酒店和假日酒店预订需求和入住率比较
print("df.columns:\n", df.columns)
print("df.hotel.value_counts():\n", df.hotel.value_counts())

sns.countplot(df.hotel)
plt.show()
city_count_book = df.hotel.value_counts()[0] # 城市酒店的预定情况
resort_count_book = df.hotel.value_counts()[1] # 假日酒店的预定情况

city_check_in = df[df['hotel'] == 'City Hotel'].is_canceled.value_counts()[0] # 城市酒店取消预订情况
resort_check_in = df[df['hotel'] == 'Resort Hotel'].is_canceled.value_counts() # 假日酒店取消预定情况

# 入住率=入住总数/预定总数
city_rate = city_check_in/city_count_book
resort_rate = resort_check_in/resort_count_book

print('城市酒店入住率：{}, 假日酒店入住率：{}'.format(city_rate, resort_rate))


## 2. 用户行为：提前预订时间、入住时长、预订间隔、餐食预订情况
# 2.1 提前预订时间
time_list = list(df['lead_time'])
print('均值:', np.mean(time_list))
print("中位数：",np.median(time_list))
print("最小值：",min(time_list))
print("最大值：",max(time_list))
print("四分位数:",np.percentile(time_list, (25, 50, 75), interpolation='midpoint'))
counts = np.bincount(time_list)
print("众数：",np.argmax(counts))

# 2.2 入住时长
df['stay_time'] = df['stays_in_weekend_nights'] + df['stays_in_week_nights']
stay_list = list(df['stay_time'])
print('均值:', np.mean(stay_list))
print("中位数：",np.median(stay_list))
print("最小值：",min(stay_list))
print("最大值：",max(stay_list))
print("四分位数:",np.percentile(stay_list, (25, 50, 75), interpolation='midpoint'))
counts = np.bincount(stay_list)
print("众数：",np.argmax(counts))

# 2.3 餐食预订情况
sns.countplot(df.meal)
plt.show()


## 3. 一年中最佳预订酒店时间
# 酒店入住情况柱状图
df.groupby(['arrival_date_month','arrival_date_year'])['children'].sum().plot.bar(figsize=(15,5))
# 酒店平均价格-时间段折线图
name_list = []
price_list = []
for s in df.groupby(['arrival_date_month','arrival_date_year'])['adr']:
    name_list.append(list(s)[0])
    price_list.append(np.mean(list(s)[1].values))

plt.figure(figsize=(15,10))
font = FontProperties(fname=r"C:\windows\\fonts\simsun.ttc", size=13)
plt.xlabel(u'时间段（年/月）', fontproperties=font, fontdict={'family': 'Times New Roman',
                                                 'color': 'black',
                                                 'weight': 'normal',
                                                 'size': 13})
plt.ylabel(u'价格', fontproperties=font, fontdict={'family': 'Times New Roman',
                                                  'fontstyle': 'italic',
                                                  'color': 'black',
                                                  'weight': 'normal',
                                                  'size': 13})
x = range(len(price_list))
plt.xticks(range(len(price_list)), name_list, fontsize=15, rotation=75)

value = np.array(price_list)
l1, = plt.plot(x, value, '--', color='b', linewidth=1, marker='^')

plt.show()


## 4. 利用Logistic预测酒店预订
cancel_corr = df.corr()["is_canceled"]
cancel_corr.abs().sort_values(ascending=False)[1:]

df.groupby("is_canceled")["reservation_status"].value_counts()
num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled","agent","company",
                "required_car_parking_spaces", "total_of_special_requests", "adr"]

cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]

features = num_features + cat_features
X = df.drop(["is_canceled"], axis=1)[features]
y = df["is_canceled"]


num_transformer = SimpleImputer(strategy="constant")

cat_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
    ("onehot", OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                               ("cat", cat_transformer, cat_features)])
base_models = [("LR_model", LogisticRegression(random_state=42,n_jobs=-1))]
kfolds = 4
split = KFold(n_splits=kfolds, shuffle=True, random_state=42)

for name, model in base_models:
    
    model_steps = Pipeline(steps=[('preprocessor', preprocessor),
                              ('model', model)])
    
    
    cv_results = cross_val_score(model_steps, 
                                 X, y, 
                                 cv=split,
                                 scoring="accuracy",
                                 n_jobs=-1)
   
    min_score = round(min(cv_results), 4)
    max_score = round(max(cv_results), 4)
    mean_score = round(np.mean(cv_results), 4)
    std_dev = round(np.std(cv_results), 4)
    print(f"{name} cross validation accuarcy score: {mean_score} +/- {std_dev} (std) min: {min_score}, max: {max_score}")



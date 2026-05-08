#KNN手写数字识别


#导入需要用到的包

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import joblib

#导入数据
#将第0行数据看作column名，并且跳过第1行
data_df = pd.read_csv('mnist_test.csv', header = 0, skiprows = 1)

#将数据转化为28x28的图片
def show_num(ide : int = 0,img_path : str = 'img.png', save : bool = False):
    #拿到数据部分
    x = data_df.iloc[ide, 1:]

    #将Series对象转化为Numpy数组
    data = x.values

    #重塑数组
    data = data.reshape(28, 28)

    #类型转化
    data = data.astype(float)

    #绘图
    plt.imshow(data, cmap = 'gray')
    if save:
        plt.imsave(img_path, data)
    plt.show()

#训练模型
#由gridsearch_cv可知n = 1时准确率最高
def train_model(n : int = 1, path : str = 'model.bin', save : bool = False):
    #整理数据，并且归一化
    x = data_df.iloc[:, 1:] / 255.
    y = data_df.iloc[:, 0]

    #分割数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, stratify = y, random_state = 0)

    #模型的训练
    estimator = KNeighborsClassifier(n_neighbors = n)
    estimator.fit(x_train, y_train)

    #模型评估
    acc = estimator.score(x_test, y_test)
    print(f"测试集准确率：{acc}")

    #模型保存
    if save:
        joblib.dump(estimator, path)

#交叉验证网格搜索
def gridsearch_cv(grid : list):
    #数据整理
    x = data_df.iloc[:, 1:]
    y = data_df.iloc[:, 0]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
    #预处理
    pre = StandardScaler()
    x_train = pre.fit_transform(x_train)
    x_test = pre.transform(x_test)

    #超参数
    param_grid = {'n_neighbors' : grid}
    #实例化
    estimator = KNeighborsClassifier()
    #实例化交叉验证网格搜索对象
    estimator = GridSearchCV(estimator, param_grid = param_grid, cv = 5)
    estimator.fit(x_train, y_train)
    #输出结果
    print('estimator.best_score_-->', estimator.best_score_)
    print('estimator.best_params_-->', estimator.best_params_)
    print('estimator.cv_results_-->', estimator.cv_results_)

#测试模型
def test_model(model_path : str = 'model.bin', img_path : str = 'img.png'):

    #读取并展示图片
    img = plt.imread(img_path) #此时img是28*28
    plt.imshow(img)
    plt.show()
    # 如果图片有3个维度 (例如 RGBA 或 RGB)，将其转换为灰度图
    if img.ndim == 3:
        # 如果是 RGBA (4通道) 或 RGB (3通道)，取平均值变为灰度
        # 这里取前三个通道(RGB)求平均，忽略 Alpha 通道(如果有)
        img = img[:, :, :3].mean(axis=2)
    #加载模型
    knn_model = joblib.load(model_path)
    #重塑
    img = img.reshape(1, -1)

    feature_names = data_df.columns[1:]
    img = pd.DataFrame(data = img, columns = feature_names)
    y_pre = knn_model.predict(img) #reshape之后是1 * 自适应
    print(f'图片绘制的数字是：{y_pre}')

if __name__ == '__main__':
    #绘图函数测试
    #show_num(23,img_path = 'img23.png',save = True)
    #train_model(save = True)
    #gridsearch_cv([1, 3, 5, 7])
    # test_model(img_path = 'img.png')
    # test_model(img_path = 'img99.png')
    # test_model(img_path = 'img11.png')
    # test_model(img_path = 'img23.png')
    test_model(img_path = 'img88.png')

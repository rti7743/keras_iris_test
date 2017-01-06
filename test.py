#
#とりあえず動くところまで作った iris分類.
#
import argparse
import numpy
import keras
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
numpy.random.seed(1192)  # for reproducibility

#分類した結果が文字列なので [0,1,2] という数字に変換する
def iris_name_to_index(s):
    if s == b'setosa':
        return 0   #正解データは0からスタートしないとダメ
    elif s == b'virginica':
        return 1
    elif s == b'versicolor':
        return 2
    else:
        print("Unknown Data:{}".format(s))
        raise()

#逆に、数字から、setosa/virginica/versicolor に変換する.
def iris_index_to_name(i):
    if i == 0:
        return 'setosa'
    elif i == 1:
        return 'virginica'
    elif i == 2:
        return 'versicolor'
    else:
        print("Unknown Label:{}".format(i))
        raise()


# 引数の処理
parser = argparse.ArgumentParser()
parser.add_argument('--mode',      type=str,   default='all') #train:学習だけ pred:分類だけ all:両方
parser.add_argument('--epoch',     type=int,   default=500) #何回学習を繰り返すか
parser.add_argument('--trainstart',type=int,   default=0)   #全データ150件のうち、何件から学習に使うか
parser.add_argument('--trainsize', type=int,   default=50)  #全データ150件のうち、何件を学習に使うか ディフォルト(150件中 0件から50件までを学習に使う)
parser.add_argument('--trainbatch',type=int,   default=50)  #学習するミニバッチに一度にかけるデータの個数
args = parser.parse_args()

#csv読込
#最初の4つが学習データ 最後の5番目が正解データ
csv = numpy.loadtxt("iris.csv",
    delimiter=",",                    #csvなので  , で、データは区切られている
    skiprows=1,                       #ヘッダーを読み飛ばす
    converters={4:iris_name_to_index} #4カラム目は分類がテキストでかかれているので 0から始まる数字ラベルに置き換える
    )

#学習データの定義
#学習データは float32の2次元配列
#例: [ [1,2,3,4],[5,6,7,8],[1,2,3,4],[5,6,7,8],[1,2,3,4] ].astype(numpy.float32)
data = csv[:,0:4].astype(numpy.float32)

#正解データの定義
#正解データは 0から始まる int32の1次元配列
#例: [ 0,1,2,1,0 ].astype(numpy.int32)
label= csv[:,4].astype(numpy.int32)

#学習
if args.mode in ['all','train']:

    model = keras.models.Sequential()
    model.add(Dense(100,input_dim=4))
    model.add(Activation('relu'))
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    x = data[args.trainstart:args.trainstart+args.trainsize]
    y = label[args.trainstart:args.trainstart+args.trainsize]

    model.fit(x, y ,batch_size=args.trainbatch,nb_epoch=args.epoch,verbose=1)

    #学習結果を保存.
    model.save("model.dat")

#分類
if args.mode in ['all','pred']:

    #学習結果を読み込む
    model = keras.models.load_model("model.dat")

    #すべてのデータに対して、1件ずつ、分類を答えさせて、正しいか採点する.

    #keras組み込みの evaluateメソッドで一発なんだけど、 表示をこりたいので、forを回すよ.
    #score = model.evaluate(data, label, verbose=1)
    #print(score)

    ok_count = 0
    for i in range(len(data)):
        x = data[i:i+1]        #このデータについて調べたい.

        y = model.predict(x, batch_size=1, verbose=0) #分類の結果を取得

        pred = numpy.argmax(y)    #3つの分類のうち、どれの確率が一番高いのかを返す
        if label[i] == pred:
           ok_count = ok_count + 1
           #print("OK i:{} pred:{}({}) data:{}".format(i,iris_index_to_name(pred),iris_index_to_name(label[i]),data[i:i+1] ))
        else:
           print("NG i:{} pred:{}({}) data:{}".format(i,iris_index_to_name(pred),iris_index_to_name(label[i]),data[i:i+1] ))

    print("total:{} OK:{} NG:{} rate:{}".format(len(data),ok_count,len(data)-ok_count,ok_count/len(data)) )


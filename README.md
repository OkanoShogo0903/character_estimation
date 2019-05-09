# JOJO立ち認識
OpenPoseで得られた姿勢から、キャラクターの推定を行う
![image](https://user-images.githubusercontent.com/25472671/56340597-543df200-61ed-11e9-96c7-532d0274f4ec.png)

![jojo](https://github.com/OkanoShogo0903/character_estimation/blob/master/etcs/jojo.jpg)

## Install

```
git clone https://github.com/OkanoShogo0903/character_estimation.git 
```

single.launchでデータセットを保存する場所などの設定をしてください。
デフォルトは設定されているURLを参考にすればいいと思います。
```xml
    <param name="dataset_url" value="/home/demulab/pose_dataset" type="string"/>
    <param name="dataset_filename" value="dataset.csv" type="string"/>
    <param name="model_url" value="/home/demulab/catkin_ws/src/character_estimation/" type="string"/>
    <param name="picture_url" value="/home/demulab/catkin_ws/src/character_estimation/etcs/" type="string"/>
```

## How to use
モードを指定して以下のコマンドで起動します。
- 0:データセット作成モード
学習ラベルを標準入力から受け取って100個分のデータを収集します。
収集したデータはcsvデータとしてlaunchファイルで指定したURLに保存します。
```
$ roslaunch character_estimation all.launch mode:=0 
```

- 1:学習モード
データセット作成モードで作ったデータからモデルを学習します。
```
$ roslaunch character_estimation all.launch mode:=1 
```

- 2:推論モード
学習モードで学習したモデルで推論を行います。
推論したときのクラス確率が最も高いクラスのラベルをTopic`/pose_label`に流します。
```
$ roslaunch character_estimation all.launch mode:=2 
```

結果を可視化したい場合は以下のコマンドで`character_img`どうぞ。
```
$ rqt_image_view
```

## TODO
[x] openposeで得られるのは18次元の関節位置なので、これを関節書くに変換する.  
[x] 欠損値の埋め方を決める  
[x] ハイパーパラメータの最適化  
[x] データ拡張の実装  
[x] 画像を表示して、人の近くにラベルを表示する  
[x] データ拡張が間違っていないか調べる  
[ ] アンサンブル学習にする  
[ ] それぞれのモデルの結果も画面に表示させる  
[ ] データセット増やす  
[ ] 特徴をプロットして確認する  


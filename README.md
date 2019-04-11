# JOJO立ち認識
OpenPoseから得られた姿勢から、キャラクターの姿勢推定を行う

![jojo](https://github.com/OkanoShogo0903/character_estimation/blob/master/etcs/jojo.jpg)

## Install
dataset urlをlaunchファイルで指定する。  
デフォルトはホームディレクトリ直下の/pose_dataset
```xml
      <param name="mode" value="estimate" type="string"/>
      <param name="is_test" value="1" type="bool"/>
      <param name="dataset_url" value="/home/demulab/pose_dataset" type="string"/>
```

## How to use
roslaunch character_estimation all.launch
rqt_image_view

## TODO
[x] openposeで得られるのは18次元の関節位置なので、これを関節書くに変換する.
[x] 欠損値の埋め方を決める

[x] データ拡張の実装
[x] 画像を表示して、人の近くにラベルを表示する
[ ] アンサンブル学習にする
[ ] それぞれのモデルの結果も画面に表示させる

[ ] データ拡張が間違っていないか調べる

[ ] データセット増やす
[ ] ハイパーパラメータの最適化
[ ] 仮面ライダーとかにも対応させる
[ ] 特徴をプロットして確認する


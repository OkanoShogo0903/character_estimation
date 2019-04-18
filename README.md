# JOJO立ち認識
OpenPoseで得られた姿勢から、キャラクターの推定を行う
![image](https://user-images.githubusercontent.com/25472671/56340597-543df200-61ed-11e9-96c7-532d0274f4ec.png)

![jojo](https://github.com/OkanoShogo0903/character_estimation/blob/master/etcs/jojo.jpg)

## Install
dataset url等をlaunchファイルで指定する。 

```xml
<launch>
  <arg name="mode" doc="MODE 0:CreateDataset ,1:Train ,2:Predict"/>
  <node pkg="character_estimation" name="character_estimation" type="main.py" output="screen">
    <param name="gui" value="1" type="bool"/>
    <param name="mode" value="$(arg mode)" type="int"/>
    <param name="dataset_url" value="/home/demulab/pose_dataset" type="string"/>
    <param name="dataset_filename" value="dataset.csv" type="string"/>
    <param name="model_url" value="/home/demulab/catkin_ws/src/character_estimation/" type="string"/>
    <param name="picture_url" value="/home/demulab/catkin_ws/src/character_estimation/etcs/" type="string"/>
  </node>
</launch>
```

## How to use
roslaunch character_estimation all.launch
rqt_image_view

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


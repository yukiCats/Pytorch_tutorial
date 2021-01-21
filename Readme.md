# Pytorch Tutorial

Pytorchを利用してDeeplearning初心者が画像認識や物体検出を実装するプロジェクト

## 1. 概要

「つくりながら学ぶ! PyTorchによる発展ディープラーニング」の内容を基に、転移学習の実装を学んでいます。書籍のコードをベースに新しいテーマを見つけて、モデルの実装を行います。

## 2. 構成

What things you need to install the software and how to install them

### 第0章. MINISTデータセットを利用したPytorchのキャッチアップ
- MINISTデータセット（0-9までの画像データ）を対象に、Pytorchの基本的な使い方を学んでいく。学習目標は①PytorchによるDeepLearningの実装と②Shapによる画像の特徴量分析とする。
    * 00_minist_deapleaning.ipynb
        * DeeplearningをPytorchにより実装して、MINISTデータセットの分類を実装するノートブック。
### 第1章. 画像分類と転移学習（VGG）
- MINISTデータセット（0-9までの画像データ）を対象に、Pytorchの基本的な使い方を学んでいく。学習目標は①PytorchによるDeepLearningの実装と②Shapによる画像の特徴量分析とする。
 * 130081_310927_bundle_archive
     * ヨーロッパで活躍した画家たちの作品が、画像データとして含まれる。
 * data
     * 生データから格納されている作品が多い順に作品を抽出し、学習・テストデータとして格納する。対象とするアーティストは下記の通り。
         - Vincent_van_Gogh : 0
         - Edgar_Degas : 1
         - Pablo_Picasso : 2
         - Pierre-Auguste_Renoir :3
         - Albrecht_Durer : 4
         - Paul_Gauguin :5
         - Francisco_Goya : 6
         - Rembrandt : 7
         - Alfred_Sisley :8
         - Titian : 9
 * 00_make_datadir.ipynb
     * 学習に利用するデータを格納する`data`ディレクトリを作成する。
 * 01_dataloader.ipynb
     * 加工されたデータをもとに、vgg-16をファインチューニングする。
 * 10_vgg16.ipynb
     * 使っていない。
### 第2章 物体認識（SSD）
- write about
  * 2-1_Dataset_DataLoader.ipynb
    * make_datapath_list
       * 学習、検証の画像データとアノテーションデータへのファイルパスリストを作成する
    * Anno_xml2list
      * 「XML形式のアノテーション」を、リスト形式に変換するクラス
    * DataTransform
      * 入力画像の前処理をするクラス
    * VOCDataset(data.Dataset)
      * VOC2012のDatasetを作成する
    * od_collate_fn(batch)
      * 画像ごとにアノテーションデータのサイズが異なるため、カスタイマイズしたcollate_fnを作成します。
  * 2-2_SSD_model_forward.ipynb
    * make_vgg
      * 35層にわたる、vggモジュールを作成
    *  make_extras
       * 8層にわたる、extrasモジュールを作成
    * make_loc_conf
      * デフォルトボックスのオフセットを出力するloc_layers、各クラスの信頼度confidenceを出力するconf_layersを作成
    * L2Norm
      * convC4_3からの出力をscale=20のL2Normで正規化する層
    * DBox
      * デフォルトボックスを出力するクラス
    * SSD(nn.Module):
      * SSDクラスを作成する
    * decode(loc, dbox_list):
      * オフセット情報を使い、DBoxをBBoxに変換する関数
    * nm_suppression
      * Non-Maximum Suppressionを行う関数
    * Detect
      * SSDの推論時にconfとlocの出力から、被りを除去したBBoxを出力する
    * SSD
      * SSDクラスを作成する
  * 2-3_loss_function
    * MultiBoxLoss
      * SSDの損失関数のクラスです。
  * 2-4_SSD_training
    * utilsからDataset/DataLoaderをロードし、モデルの訓練を行う。結果を.pth形式で訓練済パラメータを保存する。
  * 2-5_SSD_inference
    * 学習済パラメータを基に、推論を行う。

* 第3章 セマンティックセグメンテーション（PSPNet）
* 第4章 姿勢推定（OpenPose）
* 第5章 GANによる画像生成（DCGAN、Self-Attention GAN）
* 第6章 GANによる異常検知（AnoGAN、Efficient GAN)
* 第7章 自然言語処理による感情分析（Transformer）
* 第8章 自然言語処理による感情分析（BERT）
* 第9章 動画分類（3DCNN、ECO）

## 3. 環境構築



```
conda install -c peterjc123 pytorch
conda install pytorch torchvision -c pytorch
pip install shap
```

And repeat

```
until finished
```

End with an example of getting some data out of the system or using it for a little demo

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```
Give an example
```

### And coding style tests

Explain what these tests test and why

```
conda install -c peterjc123 pytorch
conda install pytorch torchvision -c pytorch
```

## 2. Deployment

Add additional notes about how to deploy this on a live system

## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)

See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Hat tip to anyone whose code was used
* Inspiration
* etc

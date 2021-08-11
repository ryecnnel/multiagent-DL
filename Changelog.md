# Change Log

### 現状書いてない変更がたくさんあるのでアテにしないように．

## [1.4.1] - 2020/02/03
### Changed
 - networkgraph.py 内，connected_directed_networkgraph method のコメントを修正しました．


## [1.4.0] - 2020/02/03
### Added
 - 今更ですが，Changelod.md を追加しました．
   - それに伴い，バージョンの数え方を変更しました．

### Changed (Major)
 - networkgraph.py を更新しました．
   - 各 method の説明を追加しました．
   - 平衡グラフならびに完全グラフを生成する method を追加しました．
   - 有向（不平衡）グラフを生成する方法を変更しました．
     - 「ランダムに辺をつくる」方式から， 「平衡グラフから確率に従って辺を削除する」方式へと変更しました．
     - それに伴い，引数に削除確率 p を追加しました．（デフォルト値は 0.35）
   - Watts-Strogatz グラフを作る method について，method 名のスペルミスを修正しました．
     - 誤：connected_wattzstrogatz_networkgraph
     - 正：connected_wattsstrogatz_networkgraph
  
 - common/multi_layer_net_extend.py を更新しました．
   - accuracy method について，精度と同時に損失を取得するように変更しました．
     - predict せずに損失のみを求める関数 __loss を追加しました． 
     - これによって学習評価処理の効率化が見込まれます．
   - accuracy method の返り値を，精度と loss のタプルに変更しました．
     - 更新前：return accuracy
     - 更新後：return accuracy, self.__loss(Y,T)
     - タプルが返されますため，変数への代入にご注意ください．

 - バグを発見しましたらお知らせください．
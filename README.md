# Dimensional reduction：t-SNE

將高維資料樣本間的相關性映射至低維空間，盡可能保留原始資料的局部相似關係

=>希望在低維空間中，保留高維資料樣本間的鄰近關係

## t-SNE 簡介

t-SNE(t-Distributed Stochastic Neighbor Embedding)是一種常見的非線性降維

透過建立高維資料中的局部相似關係，將高維資料樣本映射至低維空間，持續調整低維空間樣本位置，使其盡可能保留原始資料的局部近鄰關係

### t-分布 在此的應用

t-分布具有Heavy Tail特性，使距離較遠的樣本仍保有較大的權重

=>讓不相似樣本在低維空間中能夠充分分離，降低Crowding Problem

![t分布與常態分佈差異](t-disrtibution.png)

## t-SNE流程

![t-SNE演算法流程](process.png)

## 特色公式

在 t-SNE 中，主要比較兩個相似度機率分布：

- **P**：高維空間中的相似度分布，代表希望保留的原始資料局部關係
- **Q**：低維空間中的相似度分布，代表目前低維資料所呈現的局部關係

t-SNE 的目的就是透過持續調整低維資料的位置，使 **Q 盡可能接近 P**

### 1、Perplexity

$$perplexity(p_i)=2^{H(P_i)}$$

$$H(P_i) = -\sum_{j\neq i}{} P_{j|i} log_2 P_{j|i} $$

Perplexity 用來控制高維空間中的局部鄰域尺度，透過調整 $\sigma_i$，決定每個樣本應該以多大的範圍建立相似關係，進而影響高維相似度分布 **P** 的建立

### 2、KL Divergence 

$$KL(P||Q)=\sum_{j\neq i}{}P_{ij} ln(\frac{P_{ij}}{Q_{ij}}) $$

KL Divergence 用來衡量高維相似度分布P與低維相似度分布Q之間的差異

透過最小化 KL Divergence，使低維空間的相似關係Q盡可能接近高維空間的目標關係P

### 3、KL Gradient

$$\frac{\partial KL(P||Q)}{\partial y_i} = 4\sum_{j\neq i}{} (P_{ij}-Q_{ij}) (y_i-y_j)(1+||y_i-y_j||^{2})^{-1}$$

KL Divergence 只能判斷目前P與Q差多少，但無法直接告訴低維樣本應該往哪裡移動

因此透過 KL Gradient 計算 KL Divergence 對低維座標 $y_i$ 的梯度，決定樣本在低維空間中的調整方向，再持續更新位置，使Q逐漸接近P

## t-SNE演算法完整流程與實驗解釋

演算法主要可拆成三個部分

1、目標高維空間建立

2、目前狀態低維空間

3、優化與迭代

### 目標高維空間建立

#### 1、計算 $P(j|i)$

計算 $P(j|i)$，以 $x_i$ 為基準，建立 $x_i$ 與其他樣本 $x_j$ 的相似關係

$P_{j|i}$ 表示：

以 $x_i$ 為基準時，$x_j$ 在高維空間中與 $x_i$ 的相似程度

其條件機率公式為：

$$
P_{j|i}=
\frac{
\exp(-||x_i-x_j||^2/2\sigma_i^2)
}{
\sum_{k\neq i}
\exp(-||x_i-x_k||^2/2\sigma_i^2)
}
$$

其中分子：

$$
\exp
\left(
-\frac{||x_i-x_j||^2}{2\sigma_i^2}
\right)
$$

是以 $x_i$ 為基準，將 $x_i$ 與 $x_j$ 之間的距離透過 Gaussian 分布轉換為相似權重

距離越近，Gaussian 權重越大
距離越遠，Gaussian 權重越小

而分母：

$$
\sum_{k\neq i}
\exp(-||x_i-x_k||^2/2\sigma_i^2)
$$

則將 $x_i$ 與其他所有樣本的相似權重加總，作為共同的比較基準

=>將每個樣本的相似權重除以所有樣本的權重總和後，
即可將相似權重轉換為條件機率

=> 對固定的基準樣本 $x_i$ 而言：

$$
\sum_{j\neq i}P_{j|i}=1
$$

因此 $P_{j|i}$ 並不是單純表示距離，
而是表示：

**在以 $x_i$ 為基準的情況下，$x_j$ 在整體相似關係中的相對重要程度**

#### 2、以 Perplexity 做為目標進行調整

Perplexity 用來控制高維空間中的**局部鄰域尺度**

由條件機率分布計算 Entropy：

$$
H(P_i)
=-\sum_{j\neq i}
P_{j|i}\log_2P_{j|i}
$$

再將 Entropy 轉換為 Perplexity：

$$
Perplexity(P_i)
=2^{H(P_i)}
$$

Perplexity 可視為條件機率分布中「有效鄰居數量」的尺度。

因此不是直接選擇固定數量的鄰居，
而是透過調整 $\sigma_i$，改變 Gaussian 分布的寬度，
進而改變相似權重的分布範圍

$$
\sigma_i \uparrow
\Rightarrow
\text{分布變寬}
\Rightarrow
\text{更多樣本具有較高權重}
$$

$$
\sigma_i \downarrow
\Rightarrow
\text{分布變窄}
\Rightarrow
\text{權重集中於較近的樣本}
$$

持續調整 $\sigma_i$，直到：

$$
Perplexity(P_i)
\approx
Perplexity_{target}
$$

因此，每個基準樣本都會根據自身的資料分布，
找到適合自己的 $\sigma_i$

#### 3、將 $P(j|i)$ 與 $P(i|j)$ 進行對稱化

以不同樣本作為基準樣本時，
所得到的 $\sigma_i$ 與 $\sigma_j$ 可能不同

因此，即使是同一組樣本：

$$
P_{j|i}\neq P_{i|j}
$$

例如：

以 $x_i$ 為基準時，$x_j$ 可能具有較高的相似機率；

但以 $x_j$ 為基準時，由於其局部資料分布不同，
$x_i$ 的相似機率可能不同

=> 條件機率具有方向性

因此需要將兩個方向的條件機率進行對稱化：

$$
P_{ij}
=\frac{P_{j|i}+P_{i|j}}{2N}
$$

其中 $N$ 為樣本數。

經過對稱化後：

  消除 $P_{j|i}$ 與 $P_{i|j}$ 的方向性
  將兩個方向的相似關係整合
  重新正規化整體機率

最終建立高維空間的相似度分布 **P**

**P 即為 t-SNE 希望低維空間盡可能保留的目標關係**

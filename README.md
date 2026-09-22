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

t-SNE 的目的就是透過持續調整低維資料的位置，使 **Q 盡可能接近 P**。

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

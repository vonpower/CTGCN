# CTGCN
## Introduction
With the widespread application of blockchain technology, the traditional cyberspace security issue of phishing has also appeared in the emerging blockchain cryptocurrency ecosystem. 
As phishing fraud in cryptocurrency transactions has its own characteristics compared to traditional phishing, many existing phishing detection algorithms are not applicable.
And the cryptocurrency transaction graph has the characteristics of directed graph, sideband features and complex structure. But most traditional deep learning models are applicable to undirected graphs, and the more complex the constructed graph, the longer the running time of the model. 
Our work shares phishing account information from Ethereum and the code for how to crawl it. In addition, the CT-GCN algorithm for detection was also shared. 
## Dataset
### Labeled phishing account
We crawled the phishing accounts before July 15, 2021 and their first-order nodes from http://cn.etherscan.com/. There were 4,948 accounts flagged as phishing accounts, including 166 smart contract accounts.
### Labeled non-phishing account
We first randomly select 100 block numbers from blocks 1997275-12892110, and then extract all transaction records in the block number, and then randomly extract the same number of non-phishing accounts as the phishing accounts from the accounts in the transaction records, to solve the class imbalance problem between phishing accounts and non-phishing accounts in Ethereum.
##### After processing the collected data, we obtained 1928 phishing accounts as well as 1901 non-phishing accounts with a total of 336,500 transaction records.
## Code
### The crawler code
We provide the code to crawl information from http://cn.etherscan.com/.
### CT-GCN
We propose a cryptocurrency phishing fraud detection model that considers both direction and characteristics of the edges, expands the method of phishing detection in the Blockchain.

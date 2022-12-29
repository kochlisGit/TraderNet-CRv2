# TraderNet-CRv2
TraderNet-CRv2 - Combining Deep Reinforcement Learning with Technical Analysis and Trend Monitoring on Cryptocurrency Markets

# Description
This system architecture is an extended version of the original TraderNet-CR architecture, which is described by this paper: https://link.springer.com/chapter/10.1007/978-3-031-08333-4_25. In this work, we combine Proximal Policy Optimization algorithm (PPO), which is a DRL learning algorithm, with 2 rule-based safety mechanisms: N-Consecutive & Smurfing. Our experiments on 5 popular cryptocurrencies show very promising results.

![TraderNet-CRv2 Architecture](https://github.com/kochlisGit/TraderNet-CRv2/blob/main/tradernetcr-v2.png)


# Technical Indicators
Technical analysis has been applied on market data in order to train TraderNet. The following popular technical indicators have been used:

* EMA (Exponential Moving Average)
* DEMA (Double-Exponential Moving Average)
* MACD (Moving Average Convergence/Divergenc)
* AROON
* CCI (Commodity Channel Inde)
* ADX (Average Directional Inde)
* STOCH (Stochastic Oscillator)
* RSI (Relative Strength Index)
* OBV (On-Balance Volume)
* BBANDS (Bolliger Bands)
* VWAP (Volume-Weighted Average Pric)
* ADL (Accumulation/Distribution Line)

# Requirements
To run and evaluate our agent, You need to download the following libraries/packages:

* Python $\geq 3.6$ (https://www.python.org/)
* Numpy (https://numpy.org/)
* Pandas (https://pandas.pydata.org/)
* Matplotlib (https://matplotlib.org/)
* Tensorflow (https://www.tensorflow.org/)
* TF-Agents (https://www.tensorflow.org/agents)
* TA (https://technical-analysis-library-in-python.readthedocs.io/en/latest/)
* PyTrends (https://pypi.org/project/pytrends/)
* Scikit-Learn (https://scikit-learn.org/stable/)

# Instructions
Download Python 3.6 or higher and the libraries that are described on requirements using `pip` installer (e.g. `pip install numpy)`. Then:

1. Run `download_datasets.py` to download the datasets from CoinAPI platform (https://www.coinapi.io/).
1. Use `train_tradernet.ipynb` to train TraderNet module.
1. Use `train_smurf.ipynb` to train Smurf module.
1. Use `integrated.ipynb` to evaluate the Integrated agent.

# Supported Cryptocurrencies
1. Bitcoin (BTC)
1. Ethereum (ETH)
1. Cardano (ADA)
1. Litecoin (LTC)
1. XRP

# Paper

# Cite us

# Important Note
This AI is not a commercial product and is intended for research purposes only.

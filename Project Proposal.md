### Problem Statement

Sentiment analysis plays a crucial role in understanding public opinion and market trends. Analyzing sentiments in news articles and social media posts, particularly those related to the financial markets, can provide valuable insights for investors and financial institutions. However, the vast amount of textual data available makes manual sentiment analysis impractical. Therefore, there is a need to develop a large language model capable of accurately analyzing sentiments in news articles and tweets related to the financial markets.

### Deliverables

1. **Large Language Model**: Develop a state-of-the-art language model capable of sentiment analysis for news articles and tweets related to the stock markets. The chosen algorithm is Alpaca Lora, which is a LLM designed to be lightweight and efficient, making it suitable for running on local laptops without requiring substantial computational resources. Alpaca Lora will be fine-tuned and optimized for exceptional accuracy, speed, and scalability, ensuring reliable sentiment analysis results.

2. **API and User Interface**: Build a user-friendly API and web-based interface that allows users to input text (news articles or tweets) and receive sentiment analysis results in real-time. The interface should provide visualizations and summary statistics to facilitate easy interpretation of the sentiment analysis.

3. **Documentation and Deployment Guide**: Prepare comprehensive documentation that outlines the technical details of the developed language model, API usage instructions, and deployment guidelines. This documentation will enable users to integrate the sentiment analysis capabilities into their own applications or systems.

### Approach

### Dataset Description

#### News

1. [Stock News Sentiment Analysis(Massive Dataset)](https://www.kaggle.com/datasets/avisheksood/stock-news-sentiment-analysismassive-dataset)
    - Saved at "data/raw_datasets/news/Sentiment_Stock_data.csv"

2. [Stock-Market Sentiment Dataset](https://www.kaggle.com/datasets/yash612/stockmarket-sentiment-dataset)
    - Saved at "data/raw_datasets/news/stock_data.csv"

#### Tweets

1. [Bitcoin tweets - 16M tweets With Sentiment Tagged](https://www.kaggle.com/datasets/gauravduttakiit/bitcoin-tweets-16m-tweets-with-sentiment-tagged)
    - Sampled 40% of dataset
    - Saved at "data/raw_datasets/tweets/mbsa_sample.csv"
    

2. [Stock Market Tweet | Sentiment Analysis lexicon](https://www.kaggle.com/datasets/utkarshxy/stock-markettweets-lexicon-data)
    - Saved at "data/raw_datasets/tweets/tweets_labelled.csv"

3. [Stock Sentiment](https://www.kaggle.com/datasets/purvitsharma/stock-sentiment)
    - Saved at "data/raw_datasets/tweets/stock_sentiment.csv"

### Challenges

### Reference

1. [Introduction to Git LFS (Large File Storage)](https://www.youtube.com/watch?v=xPFLAAhuGy0&ab_channel=DanGitschooldude)
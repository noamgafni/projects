# Projects

A collection of data analysis and statistics projects completed during my undergraduate studies in Statistics at UC Davis.

## Cryptocurrency Volatility Forecasting
**[Live demo →](https://cryptoforecasting-g17.streamlit.app/)** *(hosted on Streamlit Community Cloud — the app may be asleep if inactive; click the link and give it ~30 seconds to spin back up)*

Team project building a time-series forecasting pipeline across 18 cryptocurrencies. Originally aimed at predicting tail-risk crash events, but pivoted after diagnosing that models achieved high precision but only 6% recall, missing 94% of actual crashes. Rebuilt around ARIMA-based forecasting of daily price ranges, achieving 14% median MAPE with well-calibrated 95% prediction intervals. My contribution: built the ARIMA modeling pipeline and developed the interactive Streamlit dashboard.

📁 [`/crypto_forecasting`](./crypto_forecasting)

## NBA Player Archetype Clustering
Independent contribution to a team project analyzing ~10,000 NBA player-seasons of play-by-play data. Engineered advanced performance metrics and built a PCA + K-means clustering pipeline applied separately by decade (1970s–2010s), using silhouette analysis to determine the optimal number of player archetypes for each era.

📁 [`/nba_dataset_project`](./nba_dataset_project)

## Job Posting Analysis Pipeline
Built a web scraper combining API requests and Selenium browser automation to collect job listing data, including automated handling of cookie-consent popups. Parsed HTML to extract and standardize job posting section headers, then applied text-mining techniques (stopword removal, n-gram extraction) to identify common language across job requirements and qualifications sections.

📁 [`/webscraping`](./webscraping)

---
**Contact:** [noam.gafni@gmail.com](mailto:noam.gafni@gmail.com) | [LinkedIn](https://www.linkedin.com/in/noam-gafni-14341b255/)

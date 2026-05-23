Project Title: Topic Modeling Analysis of Ghanaian Media Coverage During the 2024 Elections

This project examines how leading Ghanaian media organizations—including Daily Graphic, Ghanaian Times, Daily Guide, MyJoyOnline, and Citi Newsroom—framed and reported on key national issues during the 2024 presidential and parliamentary elections.

The study applies Natural Language Processing (NLP) and topic modeling techniques to uncover dominant themes in election-related news coverage. It further compares these media-driven narratives with voter priority data obtained from Global Info Analytics to identify alignment or gaps between media focus and public concerns.

Interactive dashboard:
https://2024elections-gzvwqour9wuhxiyrx4kecq.streamlit.app/

Objectives
To identify dominant themes in Ghanaian election-related media coverage
To analyze how different media outlets framed political and socio-economic issues
To compare media narratives with voter priorities
To uncover potential gaps between public concerns and media emphasis
Methodology
1. Data Collection
Scraped and compiled election-related articles published in 2024
Sources included major Ghanaian print and online news platforms
Focused on political, economic, and electoral content
2. Data Preprocessing
Cleaned raw text data by removing punctuation, special characters, and stopwords
Applied normalization (lowercasing and standard formatting)
Tokenized text and transformed it into numerical representations using vectorization techniques (e.g., TF-IDF / Count Vectorizer)
3. Topic Modeling
Implemented Latent Dirichlet Allocation (LDA) and/or BERTopic for unsupervised topic discovery
Extracted clusters of frequently co-occurring words to identify hidden thematic structures in the dataset
4. Topic Interpretation
Manually labeled generated clusters into meaningful real-world themes
Example:
(“cedi”, “fuel”, “prices”, “inflation”) → Economic Conditions & Cost of Living
(“EC”, “results”, “vote”, “collation”) → Electoral Process & Results Management
(“campaign”, “NDC”, “NPP”, “rally”) → Political Campaign Activities
Tools & Technologies
Python (Pandas, NumPy, Scikit-learn)
NLP Libraries (NLTK / SpaCy)
Topic Modeling (LDA / BERTopic)
Data Visualization (Matplotlib, Seaborn, Plotly)
Streamlit for interactive dashboard development
Expected Impact
Provides insight into how media shapes public perception during elections
Helps identify discrepancies between media coverage and voter concerns
Supports evidence-based discussions on media bias and political communication in Ghana
Demonstrates the use of NLP in real-world socio-political analysis

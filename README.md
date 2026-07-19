<img width="1860" height="846" alt="Screenshot 2026-07-01 111709" src="https://github.com/user-attachments/assets/39cb7fb3-2235-4fe3-935c-c72740472d02" />

 <img width="1851" height="851" alt="Screenshot 2026-07-01 111806" src="https://github.com/user-attachments/assets/fbec9478-545e-449b-9952-8aff9212077d" />

 <img width="1838" height="835" alt="Screenshot 2026-07-01 111926" src="https://github.com/user-attachments/assets/0c7ff862-6549-4390-9b6e-5a1b626b460d" />

 <img width="1867" height="857" alt="Screenshot 2026-07-01 112048" src="https://github.com/user-attachments/assets/9af1a61f-fbca-452e-8c8b-d71bdbbfddf4" />

 
 
 MindPulse
 
 Multi-Dimensional Sentiment, Emotion & Sarcasm Analysis System
 
📌 Overview
SentimentalPulse is an advanced analytical platform designed to decode the complex layers of human communication within an organization. Beyond basic sentiment scoring, this system integrates emotional state detection and sarcasm identification to provide a nuanced view of workplace culture. By processing communication records, the system identifies shifts in morale, ranks employee engagement, and utilizes predictive modeling to flag potential flight risks, enabling HR and management to foster a healthier and more proactive workplace.

✨ Key Features

🧠 Multi-Layered Analysis: Goes beyond polarity (positive/negative) to detect discrete emotional states and sarcastic intent.

⚡ Sarcasm Detection Engine: Specifically calibrated to identify ironic or contradictory language that traditional sentiment tools often misinterpret.

📈 Engagement Analytics: Performs exploratory data analysis to generate monthly sentiment scores and employee rankings.

🔮 Predictive Modeling: Implements a Linear Regression model to forecast future engagement trends based on communication behavior.

⚠️ Risk Identification: Proactively identifies potential "flight risk" employees based on historical engagement patterns and emotional volatility.

Predictive Methodology: Linear Regression

To forecast future employee engagement, we implemented a Linear Regression model that analyzes the relationship between communication behavior and multi-dimensional sentiment trends.

The Logic

We treat the Monthly Sentiment Score as our dependent variable ($y$). The model predicts this score based on behavioral features ($X$), adjusted by emotional and sarcasm weightings:

Message Frequency: The volume of interactions.

Character/Word Density: The depth and complexity of contributions.

Emotional Valence: The intensity of detected emotions.

Sarcasm Index: A modifier representing the presence of ironic intent.

Tech Stack

Layer                 Technology

Language               Python

NLP Pipeline          TextBlob, NLTK, Scikit-learn

Modeling               Linear Regression

Data Handling          Pandas, NumPy

Visualization      Matplotlib, Seaborn

Setup Instructions

Clone the Repository
git clone <repository-url>

cd SentimentalPulse

Create Virtual Environment

python -m venv venv
Activate it:

Windows: venv\Scripts\activate

Mac/Linux: source venv/bin/activate

Install Dependencies

pip install -r requirements.txt

Prepare Data

Future Scope

🤖 Deep Learning Transition: Incorporate transformer-based models (e.g., RoBERTa) for higher-accuracy sarcasm detection.

🌐 Real-Time Dashboard: Build a web-based UI for live sentiment, emotion, and sarcasm heatmaps.

📧 Automated Alerting: Integrate notifications when detected "emotional volatility" or high sarcasm levels cross specific thresholds.

Ensure your test.csv file is placed in the data/ directory.

Run Analysis
Launch the Jupyter notebook to execute the workflow:
jupyter notebook notebooks/sentiment_analysis.ipynb

Mathematical Formulation

The model follows the linear equation:

<img width="437" height="45" alt="image" src="https://github.com/user-attachments/assets/a1d78b10-dfc1-4dec-b26a-f7494fbc044e" />

Implementation Workflow

Preprocessing: Feature scaling is applied to normalize the communication metrics.

Training: The model is trained on historical data from test.csv using scikit-learn.

Validation: We measure performance using Mean Squared Error (MSE) to ensure accuracy in sentiment forecasting.

Forecasting: By feeding current month engagement metrics into the trained model, the system generates a projected sentiment score, allowing us to identify downward trends before they become critical retention risks.

⭐ Why This Project Stands Out

Moves beyond binary sentiment to understand intent.

Combines HR Tech + Data Science.

Employs robust statistical forecasting to predict retention risks.

Modular and scalable architecture for enterprise-level communication analysis.

📜 License

This project is licensed under the MIT License. See the LICENSE file for details.

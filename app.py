import gradio as gr
import sqlite3
import numpy as np
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import csv
import os


import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

# Language translations
    
ADMIN_KEY = "1234"

    # Load Data
@st.cache_data
def load_data():
        return pd.read_csv(r"E:\\Vineeth_Final (1)\\Vineeth_Final\\Vineeth_Copy\\Vineeth_Copy\\health_alert_bangalore_v2 (1).csv")

df = load_data()

    # Streamlit UI
st.sidebar.title("⚙️ Settings")
user_type = st.sidebar.radio("Select User Type:", ["User", "Admin" , "Chat bots"])

if user_type == "User":
    translations = {
        "English": {
            "title": "🌍 Bangalore Health Alert System",
            "subtitle": "Stay updated on disease outbreaks in Bangalore!",
            "upload_text": "Upload Health Alert Dataset (CSV)",
            "select_location": "Select your current location:",
            "subscribe_header": "📩 Subscribe for Outbreak Alerts",
            "email_input": "Enter your email to receive health alerts:",
            "subscribe_button": "Subscribe",
            "subscription_success": "✅ Subscription successful! You will receive alerts for severe outbreaks in your area.",
            "email_warning": "⚠️ Please enter a valid email address.",
            "health_alert": "🚨 Health Alert in",
            "common_disease": "Most Common Disease:",
            "precautions_header": "🛡️ Precautionary Measures",
            "trend_analysis": "📊 Disease Trend Analysis",
            "no_outbreak": "✅ No disease outbreaks reported in your area.",
            "email_sent": "📧 Alert sent to",
            "email_failed": "❌ Failed to send email:",
            "checking_alerts": "Checking for alerts...",
            "high_risk_alert": "⚠️ High-Risk Alert!",
            "high_risk_message": "is experiencing a severe outbreak of",
            "take_precautions": "Take precautions!"
        },
        "ಕನ್ನಡ": {  # Kannada
            "title": "🌍 ಬೆಂಗಳೂರು ಆರೋಗ್ಯ ಎಚ್ಚರಿಕೆ ವ್ಯವಸ್ಥೆ",
            "subtitle": "ಬೆಂಗಳೂರಿನಲ್ಲಿ ರೋಗ ಹರಡುವಿಕೆಯ ಬಗ್ಗೆ ನವೀಕರಿಸಿದ ಮಾಹಿತಿಯನ್ನು ಪಡೆಯಿರಿ!",
            "upload_text": "ಆರೋಗ್ಯ ಎಚ್ಚರಿಕೆ ಡೇಟಾಸೆಟ್ ಅಪ್‌ಲೋಡ್ ಮಾಡಿ (CSV)",
            "select_location": "ನಿಮ್ಮ ಪ್ರಸ್ತುತ ಸ್ಥಳವನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
            "subscribe_header": "📩 ಹರಡುವಿಕೆ ಎಚ್ಚರಿಕೆಗಳಿಗೆ ಚಂದಾದಾರರಾಗಿ",
            "email_input": "ಆರೋಗ್ಯ ಎಚ್ಚರಿಕೆಗಳನ್ನು ಪಡೆಯಲು ನಿಮ್ಮ ಇಮೇಲ್ ನಮೂದಿಸಿ:",
            "subscribe_button": "ಚಂದಾದಾರರಾಗಿ",
            "subscription_success": "✅ ಚಂದಾದಾರಿಕೆ ಯಶಸ್ವಿಯಾಗಿದೆ! ನಿಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ತೀವ್ರ ಹರಡುವಿಕೆಗಳ ಬಗ್ಗೆ ನೀವು ಎಚ್ಚರಿಕೆಗಳನ್ನು ಪಡೆಯುತ್ತೀರಿ.",
            "email_warning": "⚠️ ದಯವಿಟ್ಟು ಮಾನ್ಯವಾದ ಇಮೇಲ್ ವಿಳಾಸವನ್ನು ನಮೂದಿಸಿ.",
            "health_alert": "🚨 ಆರೋಗ್ಯ ಎಚ್ಚರಿಕೆ",
            "common_disease": "ಹೆಚ್ಚು ಕಂಡುಬರುವ ರೋಗ:",
            "precautions_header": "🛡️ ಮುನ್ನೆಚ್ಚರಿಕೆ ಕ್ರಮಗಳು",
            "trend_analysis": "📊 ರೋಗ ಪ್ರವೃತ್ತಿ ವಿಶ್ಲೇಷಣೆ",
            "no_outbreak": "✅ ನಿಮ್ಮ ಪ್ರದೇಶದಲ್ಲಿ ಯಾವುದೇ ರೋಗ ಹರಡುವಿಕೆಯ ವರದಿಯಾಗಿಲ್ಲ.",
            "email_sent": "📧 ಎಚ್ಚರಿಕೆಯನ್ನು ಕಳುಹಿಸಲಾಗಿದೆ:",
            "email_failed": "❌ ಇಮೇಲ್ ಕಳುಹಿಸಲು ವಿಫಲವಾಗಿದೆ:",
            "checking_alerts": "ಎಚ್ಚರಿಕೆಗಳನ್ನು ಪರಿಶೀಲಿಸಲಾಗುತ್ತಿದೆ...",
            "high_risk_alert": "⚠️ ಅಧಿಕ-ಅಪಾಯದ ಎಚ್ಚರಿಕೆ!",
            "high_risk_message": "ನಲ್ಲಿ ತೀವ್ರವಾದ ಹರಡುವಿಕೆ ಕಂಡುಬಂದಿದೆ",
            "take_precautions": "ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ!"
        }
    }

    # Disease translations
    disease_translations = {
        "English": {
            "Dengue": "Dengue",
            "Malaria": "Malaria",
            "COVID-19": "COVID-19",
            "Cholera": "Cholera",
            "Swine Flu": "Swine Flu"
        },
        "ಕನ್ನಡ": {  # Kannada
            "Dengue": "ಡೆಂಗ್ಯೂ",
            "Malaria": "ಮಲೇರಿಯಾ",
            "COVID-19": "ಕೋವಿಡ್-19",
            "Cholera": "ಕಾಲರಾ",
            "Swine Flu": "ಹಂದಿ ಜ್ವರ"
        }
    }

    # Precautions translations
    precautions_translations = {
        "English": {
            "Dengue": "Use mosquito repellents, avoid stagnant water, and sleep under mosquito nets.",
            "Malaria": "Use insecticide-treated nets, wear long sleeves, and eliminate standing water.",
            "COVID-19": "Wear masks, sanitize hands regularly, and avoid crowded areas.",
            "Cholera": "Drink boiled water, maintain hygiene, and avoid contaminated food.",
            "Swine Flu": "Cover mouth while sneezing, avoid close contact with infected people, and get vaccinated.",
            "default": "Follow general hygiene and stay updated on local health advisories."
        },
        "ಕನ್ನಡ": {  # Kannada
            "Dengue": "ಸೊಳ್ಳೆ ನಿವಾರಕಗಳನ್ನು ಬಳಸಿ, ನಿಂತ ನೀರನ್ನು ತಪ್ಪಿಸಿ, ಮತ್ತು ಸೊಳ್ಳೆ ಬಲೆಗಳ ಅಡಿಯಲ್ಲಿ ಮಲಗಿ.",
            "Malaria": "ಕೀಟನಾಶಕ ಚಿಕಿತ್ಸೆ ಮಾಡಿದ ಬಲೆಗಳನ್ನು ಬಳಸಿ, ದೀರ್ಘ ತೋಳುಗಳನ್ನು ಧರಿಸಿ, ಮತ್ತು ನಿಂತಿರುವ ನೀರನ್ನು ತೆಗೆದುಹಾಕಿ.",
            "COVID-19": "ಮುಖಗವಸುಗಳನ್ನು ಧರಿಸಿ, ಕೈಗಳನ್ನು ನಿಯಮಿತವಾಗಿ ಸ್ವಚ್ಛಗೊಳಿಸಿ, ಮತ್ತು ಜನದಟ್ಟಣೆಯ ಪ್ರದೇಶಗಳನ್ನು ತಪ್ಪಿಸಿ.",
            "Cholera": "ಕುದಿಸಿದ ನೀರನ್ನು ಕುಡಿಯಿರಿ, ನೈರ್ಮಲ್ಯವನ್ನು ಕಾಪಾಡಿಕೊಳ್ಳಿ, ಮತ್ತು ಕಲುಷಿತ ಆಹಾರವನ್ನು ತಪ್ಪಿಸಿ.",
            "Swine Flu": "ಸೀನುವಾಗ ಬಾಯಿ ಮುಚ್ಚಿ, ಸೋಂಕಿತ ಜನರೊಂದಿಗೆ ನಿಕಟ ಸಂಪರ್ಕವನ್ನು ತಪ್ಪಿಸಿ, ಮತ್ತು ಲಸಿಕೆ ಹಾಕಿಸಿಕೊಳ್ಳಿ.",
            "default": "ಸಾಮಾನ್ಯ ಸ್ವಚ್ಛತೆಯನ್ನು ಅನುಸರಿಸಿ ಮತ್ತು ಸ್ಥಳೀಯ ಆರೋಗ್ಯ ಸಲಹೆಗಳ ಬಗ್ಗೆ ನವೀಕರಿಸಿದ ಮಾಹಿತಿಯನ್ನು ಪಡೆಯಿರಿ."
        }
    }

    # Column translations for DataFrame display
    column_translations = {
        "English": {
            "Disease": "Disease",
            "Reported Cases": "Reported Cases",
            "Severity": "Severity",
            "Date": "Date",
            "High": "High",
            "Medium": "Medium",
            "Low": "Low"
        },
        "ಕನ್ನಡ": {
            "Disease": "ರೋಗ",
            "Reported Cases": "ವರದಿಯಾದ ಪ್ರಕರಣಗಳು",
            "Severity": "ತೀವ್ರತೆ",
            "Date": "ದಿನಾಂಕ",
            "High": "ಅಧಿಕ",
            "Medium": "ಮಧ್ಯಮ",
            "Low": "ಕಡಿಮೆ"
        }
    }

    SUBSCRIBERS_FILE = "subscribers.csv"
    if not os.path.exists(SUBSCRIBERS_FILE):
        pd.DataFrame(columns=["Email", "Location"]).to_csv(SUBSCRIBERS_FILE, index=False)

    # Load subscribers
    subscribers_df = pd.read_csv(SUBSCRIBERS_FILE)

    # Streamlit UI
    st.sidebar.title("⚙️ Settings")
    language = st.sidebar.selectbox("Language / ಭಾಷೆ", ["English", "ಕನ್ನಡ"])
    t = translations[language]  # Get translations for selected language

    # Title and intro
    st.title(t["title"])
    st.markdown(t["subtitle"])

    # File uploader
    uploaded_file = st.file_uploader(t["upload_text"], type=["csv"])



    # Translate disease names in dataframe if needed
    if language == "ಕನ್ನಡ":
        df = df.copy()
        df["Disease"] = df["Disease"].apply(lambda x: disease_translations["ಕನ್ನಡ"].get(x, x))
        df["Severity"] = df["Severity"].replace({"High": "ಅಧಿಕ", "Medium": "ಮಧ್ಯಮ", "Low": "ಕಡಿಮೆ"})

    # Subscription File
    SUBSCRIBERS_FILE = "subscribers.csv"
    if not os.path.exists(SUBSCRIBERS_FILE):
        pd.DataFrame(columns=["Email", "Location"]).to_csv(SUBSCRIBERS_FILE, index=False)

    # Load subscribers
    subscribers_df = pd.read_csv(SUBSCRIBERS_FILE)

    # User selects current location
    selected_location = st.selectbox(t["select_location"], df["Locality"].unique())
    location_data = df[df["Locality"] == selected_location]

    # Email Subscription Feature
    st.subheader(t["subscribe_header"])
    user_email = st.text_input(t["email_input"])
    if st.button(t["subscribe_button"]):
        if user_email:
            new_subscriber = pd.DataFrame([[user_email, selected_location]], columns=["Email", "Location"])
            new_subscriber.to_csv(SUBSCRIBERS_FILE, mode='a', header=False, index=False)
            st.success(t["subscription_success"])
        else:
            st.warning(t["email_warning"])

    # Function to Send Email Alert (with language support)
    def send_email_alert(recipient_email, location, disease, lang="English"):
        sender_email = "vineethb11@gmail.com"  # Replace with your email
        sender_password = "mjxh vqsi vzoz hijo"  # Replace with your email password
        
        # Translate disease name for email subject
        disease_translated = disease_translations[lang].get(disease, disease) if disease in disease_translations.get("English", {}) else disease
        
        if lang == "English":
            subject = f"⚠️ Health Alert: {disease_translated} Outbreak in {location}"
            body = f"Dear User,\n\nA severe outbreak of {disease_translated} has been detected in {location}. Please take necessary precautions.\n\nStay Safe,\nBangalore Health Alert System"
        else:  # Kannada
            subject = f"⚠️ ಆರೋಗ್ಯ ಎಚ್ಚರಿಕೆ: {location} ನಲ್ಲಿ {disease_translated} ಹರಡುವಿಕೆ"
            body = f"ಪ್ರಿಯ ಬಳಕೆದಾರರೇ,\n\n{location} ನಲ್ಲಿ {disease_translated} ನ ತೀವ್ರ ಹರಡುವಿಕೆಯು ಪತ್ತೆಯಾಗಿದೆ. ದಯವಿಟ್ಟು ಅಗತ್ಯ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳನ್ನು ತೆಗೆದುಕೊಳ್ಳಿ.\n\nಸುರಕ್ಷಿತವಾಗಿರಿ,\nಬೆಂಗಳೂರು ಆರೋಗ್ಯ ಎಚ್ಚರಿಕೆ ವ್ಯವಸ್ಥೆ"
        
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = recipient_email
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain"))
        
        try:
            server = smtplib.SMTP("smtp.gmail.com", 587)
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            server.quit()
            st.success(f"{t['email_sent']} {recipient_email}!")
        except Exception as e:
            st.error(f"{t['email_failed']} {str(e)}")

    if not location_data.empty:
        # Expand the date range to last 5 months for better outbreak detection
        latest_data = location_data[location_data["Date"] >= (pd.to_datetime("today") - pd.Timedelta(days=150)).strftime("%Y-%m-%d")]
        
        if not latest_data.empty:
            most_common_disease = latest_data["Disease"].value_counts().idxmax()
        else:
            most_common_disease = location_data["Disease"].value_counts().idxmax()
        
        # Get original disease name before translation for system use
        original_disease_name = most_common_disease
        if language == "ಕನ್ನಡ":
            # Reverse lookup to get English disease name
            for eng_name, kan_name in disease_translations["ಕನ್ನಡ"].items():
                if kan_name == most_common_disease:
                    original_disease_name = eng_name
                    break
        
        st.subheader(f"{t['health_alert']} {selected_location}")
        st.markdown(f"**{t['common_disease']}** {most_common_disease}")
        
        # Display data with translated column names if in Kannada
        if language == "ಕನ್ನಡ":
            display_df = location_data.copy()
            display_df.columns = [column_translations["ಕನ್ನಡ"].get(col, col) for col in display_df.columns]
            st.write(display_df[[column_translations["ಕನ್ನಡ"]["Disease"], 
                                column_translations["ಕನ್ನಡ"]["Reported Cases"], 
                                column_translations["ಕನ್ನಡ"]["Severity"], 
                                column_translations["ಕನ್ನಡ"]["Date"]]])
        else:
            st.write(location_data[["Disease", "Reported Cases", "Severity", "Date"]])
        
        # Show Map with Red Boundary instead of markers
        map_center = [location_data.iloc[0]["Latitude"], location_data.iloc[0]["Longitude"]]
        m = folium.Map(location=map_center, zoom_start=14)
        folium.Circle(
            location=map_center,
            radius=1000,  # Radius in meters
            color="red",
            fill=True,
            fill_color="red",
            fill_opacity=0.4,
            popup=f"{selected_location} - {most_common_disease} Alert"
        ).add_to(m)
        
        folium_static(m)
        
        # Real-Time Notification if High-Risk Area
        severity_col = "Severity"
        severity_check = "High"
        if language == "ಕನ್ನಡ":
            severity_check = "ಅಧಿಕ"
        
        if latest_data[severity_col].str.contains(severity_check).any():
            with st.spinner(t["checking_alerts"]):
                time.sleep(2)
            st.error(f"{t['high_risk_alert']} {selected_location} {t['high_risk_message']} {most_common_disease}. {t['take_precautions']}")
            
            # Send Email Alert to Users Subscribed to This Location
            affected_users = subscribers_df[subscribers_df["Location"] == selected_location]["Email"]
            for email in affected_users:
                send_email_alert(email, selected_location, original_disease_name, language)
        
        # Health suggestions and precautions based on language
        st.subheader(t["precautions_header"])
        st.markdown(precautions_translations[language].get(original_disease_name, precautions_translations[language]["default"]))
        
        # Data Analysis: Bar plot for cases over time
        st.subheader(t["trend_analysis"])
        
        # Create a copy for plotting to prevent modifying the original
        plot_data = latest_data.copy()
        
        # Customize plot based on language
        if language == "ಕನ್ನಡ":
            plot_title = f"{selected_location} ನಲ್ಲಿ ರೋಗಗಳ ಪ್ರವೃತ್ತಿಗಳು (ಕಳೆದ 5 ತಿಂಗಳುಗಳು)"
            x_label = "ದಿನಾಂಕ"
            y_label = "ವರದಿಯಾದ ಪ್ರಕರಣಗಳು"
        else:
            plot_title = f"Disease Trends in {selected_location} (Last 5 Months)"
            x_label = "Date"
            y_label = "Reported Cases"
        
        fig = px.bar(
            plot_data, 
            x="Date", 
            y="Reported Cases", 
            color="Disease", 
            title=plot_title
        )
        
        fig.update_layout(
            xaxis_title=x_label,
            yaxis_title=y_label
        )
        
        st.plotly_chart(fig)
    else:
        st.success(t["no_outbreak"])
        

elif user_type == "Admin":
    
    admin_key = st.sidebar.text_input("Enter Admin Key:", type="password")
    if admin_key != ADMIN_KEY:
        st.sidebar.warning("⚠️ Incorrect Key! Access Denied.")
        st.stop()
    else:
        st.sidebar.success("✅ Access Granted. Viewing Admin Panel.")

        # Admin Dashboard
        st.title("📊 Admin Dashboard - Monitoring & Insights")
        
        # Full Dataset
        st.subheader("📂 Full Dataset Overview")
        st.dataframe(df)
        
        # Disease Frequency Analysis
        st.subheader("📈 Disease Frequency Analysis")
        disease_counts = df["Disease"].value_counts()
        st.bar_chart(disease_counts)
        
        # Severity Distribution Analysis
        st.subheader("⚠️ Severity Distribution")
        severity_counts = df["Severity"].value_counts()
        st.bar_chart(severity_counts)
        
        # Cases Over Time
        st.subheader("📆 Cases Over Time")
        fig = px.line(df, x="Date", y="Reported Cases", color="Disease", title="Disease Cases Over Time")
        st.plotly_chart(fig)
        
        # Most Affected Areas
        
        
        # View Subscribers List
        st.subheader("📜 Subscribers List")
        subscribers_df = pd.read_csv("subscribers.csv")
        st.dataframe(subscribers_df)
        
        
        # Stop execution to prevent access to user UI
        st.stop()
elif user_type == "Chat bots":

    # Configure Gemini API
    genai.configure(api_key="************************")
    generation_config = {"temperature": 0.9, "top_p": 1, "top_k": 1, "max_output_tokens": 2048}
    model = genai.GenerativeModel("gemini-2.0-flash", generation_config=generation_config)

    # Database and CSV paths
    db_path = r'vector_database.sqlite3'
    csv_file_path = "chat_history.csv"

    # Ensure chat history CSV exists
    if not os.path.exists(csv_file_path):
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["User Query", "Bot Response"])

    def log_chat_to_csv(user_query, bot_response):
        """Append each query-response pair to a CSV file."""
        with open(csv_file_path, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([user_query, bot_response])

    def generate_queries_gemini(original_query):
        """Generate refined queries using Gemini API."""
        content_prompts = [f"Generate a single refined query related to: {original_query} with a proper description in 4-5 lines."]
        response = model.generate_content(content_prompts)
        return response.text.strip()

    def vector_search(query):
        """Perform vector similarity search in SQLite database."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT filename, vector FROM documents")
            rows = cursor.fetchall()
            conn.close()
        except sqlite3.Error as e:
            print(f"SQLite error: {e}")
            return {}

        query_vector = np.random.rand(1, 384).astype(np.float32)
        scores = {}
        for filename, vector_blob in rows:
            vector = np.frombuffer(vector_blob, dtype=np.float32).reshape(1, -1)
            similarity_score = cosine_similarity(query_vector, vector)[0][0]
            scores[filename] = similarity_score

        return {doc: score for doc, score in sorted(scores.items(), key=lambda x: x[1], reverse=True) if score > 0}

    def reciprocal_rank_fusion(search_results_dict, k=60):
        """Perform Reciprocal Rank Fusion (RRF) for search results."""
        fused_scores = {}
        for query, doc_scores in search_results_dict.items():
            for rank, (doc, score) in enumerate(sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)):
                fused_scores[doc] = fused_scores.get(doc, 0) + 1 / (rank + k)
        return {doc: score for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)}

    def generate_content_google(reranked_results, queries):
        """Summarize key information from retrieved documents."""
        
        prompt = (f"Imagine you are a doctor, and a patient is describing their symptoms to you. "
                f"Show empathy for their suffering, analyze the symptoms carefully, and determine "
                f"the most likely disease they might have. Then, recommend appropriate medicines. "
                f"Make sure to format the response in separate lines as follows: \n"
                f"**Name of the disease:** - [Predicted Disease]\n"
                f"**Prescription:** - [Suggested Medicines]\n"
                f"Also, add a disclaimer advising the patient to consult a doctor for more information.\n"
                f"Symptoms: {queries}. Consider these relevant documents: {list(reranked_results.keys())}.")
        
        response = model.generate_content([prompt])
        
        # Clean and structure response properly
        response_text = response.text.strip()
        
        return response_text  # Ensure the full formatted response is returned

    def translate_text(text, target_language):
        """Translate text into the selected language."""
        try:
            return GoogleTranslator(source='auto', target=target_language).translate(text)
        except Exception as e:
            print(f"Translation error: {e}")
            return text  

    def chatbot_interface(message, chat_history, language):
        """Handle user input and generate chatbot response."""
        if chat_history is None:
            chat_history = []  
        
        chat_history.append({"role": "user", "content": message})
        generated_queries = generate_queries_gemini(message)
        all_results = {query: vector_search(query) for query in generated_queries.split('\n')}
        reranked_results = reciprocal_rank_fusion(all_results)
        response = generate_content_google(reranked_results, generated_queries)
        
        if language == "Hindi": 
            response = translate_text(response, "hi")
        elif language == "Tamil":  
            response = translate_text(response, "ta")
        log_chat_to_csv(message, response)
        chat_history.append({"role": "assistant", "content": response})
        return chat_history, chat_history

    # Streamlit UI
    st.title("RAG-Gemini Chatbot")
    user_input = st.text_input("Enter your query:")
    language = st.selectbox("Select Language", ["English", "Hindi", "Tamil"])

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if st.button("Submit"):
        st.session_state.chat_history, displayed_chat = chatbot_interface(user_input, st.session_state.chat_history, language)

        for chat in displayed_chat:
            role = "User" if chat["role"] == "user" else "Bot"
            st.text(f"{role}: {chat['content']}")

# Subscription File







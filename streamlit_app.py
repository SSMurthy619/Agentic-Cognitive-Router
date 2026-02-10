import os
import json
import pandas as pd
import streamlit as st
from intent_picker import IntentPicker

# Initialize picker
picker = IntentPicker(debug=True)

st.set_page_config(page_title="Insurance AI Assistant", layout="wide")

st.title("📊 Insurance AI Assistant")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# User input
user_input = st.chat_input("Ask me anything about insurance policies, data, or forecasts...")

if user_input:
    # Call intent picker
    try:
        result = picker.route(user_input)

        # Decide how to display
        if "table" in result:
            table = result["table"]
            sql_query = result.get("sql", {}).get("query", result.get("sql", ""))
            st.markdown(f"**SQL Query:** `{sql_query}`")
            df = pd.DataFrame(result["table"]["rows"], columns=result["table"]["columns"])
            st.dataframe(df)

        elif result.get("ModuleUsed") == "Forecast":
            sql_query = result.get("sql", "")
            metrics = result.get("metrics", {})
            forecast_head = result.get("forecast_head", [])

            if sql_query:
                st.markdown(f"**SQL Query:** `{sql_query}`")
            if metrics:
                st.json(metrics)
            if forecast_head:
                df = pd.DataFrame(forecast_head)
                st.dataframe(df.head(10))
            else:
                st.info("No forecast data available")

        elif "AnswerText" in result:
            st.markdown(result["AnswerText"])
        elif "final_text" in result:
            st.markdown(result["final_text"])
        else:
            st.json(result)

        # Add to history
        st.session_state.chat_history.append(("You", user_input))
        st.session_state.chat_history.append(("Bot", str(result)))

    except Exception as e:
        st.error(f"Error: {str(e)}")

# Show chat history
st.markdown("### Chat History")
for sender, message in st.session_state.chat_history[-10:]:
    if sender == "You":
        st.markdown(f"**🧑 You:** {message}")
    else:
        st.markdown(f"**🤖 Bot:** {message}")

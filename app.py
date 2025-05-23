import pandas as pd
import streamlit as st
import ast
import plotly.graph_objects as go
import google.generativeai as genai
import os

# Load CSV
df = pd.read_csv("data/tsla.csv")

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Create tabs
tab1, tab2 = st.tabs(["📊 Chart", "🤖 Gemini Chatbot"])

# Tab 1: Chart
with tab1:
    st.title("TSLA Dashboard")

    # Let the user choose the date range
    date_range_option = st.selectbox(
        "Select Date Range for Chart",
        ["August 2022 (Sample)", "Full Range (2022 to 2025)"],
        index=0
    )

    # Filter data based on the selected range
    if date_range_option == "August 2022 (Sample)":
        start_date = "2022-08-25"
        end_date = "2022-08-31"
        filtered_df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
        yaxis_range = [0, 350]
    else:
        filtered_df = df
        yaxis_range = [0, 400]

    # Display the table
    st.dataframe(df[['timestamp', 'direction', 'Support', 'Resistance', 'open', 'high', 'low', 'close', 'volume']].head())

    # Convert Support and Resistance columns to lists
    filtered_df["Support"] = filtered_df["Support"].apply(ast.literal_eval)
    filtered_df["Resistance"] = filtered_df["Resistance"].apply(ast.literal_eval)

    # Prepare support and resistance bands
    support_lower = []
    support_upper = []
    resistance_lower = []
    resistance_upper = []

    for _, row in filtered_df.iterrows():
        time = row["timestamp"].strftime('%Y-%m-%d')
        if row["Support"] and isinstance(row["Support"], list) and len(row["Support"]) > 0:
            support_lower.append((time, min(row["Support"])))
            support_upper.append((time, max(row["Support"])))
        if row["Resistance"] and isinstance(row["Resistance"], list) and len(row["Resistance"]) > 0:
            resistance_lower.append((time, min(row["Resistance"])))
            resistance_upper.append((time, max(row["Resistance"])))

    # Create the candlestick chart
    fig = go.Figure()

    # Add candlestick
    fig.add_trace(go.Candlestick(
        x=filtered_df['timestamp'],
        open=filtered_df['open'],
        high=filtered_df['high'],
        low=filtered_df['low'],
        close=filtered_df['close'],
        name='OHLC'
    ))

    # Add support band
    fig.add_trace(go.Scatter(
        x=[x[0] for x in support_lower],
        y=[x[1] for x in support_lower],
        mode='lines',
        name='Support',
        line=dict(color='green', width=1),
        opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=[x[0] for x in support_upper],
        y=[x[1] for x in support_upper],
        mode='lines',
        name='Support Band',
        fill='tonexty',
        fillcolor='rgba(0, 255, 0, 0.1)',
        line=dict(color='green', width=1),
        opacity=0.3
    ))

    # Add resistance band
    fig.add_trace(go.Scatter(
        x=[x[0] for x in resistance_lower],
        y=[x[1] for x in resistance_lower],
        mode='lines',
        name='Resistance',
        line=dict(color='red', width=1),
        opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=[x[0] for x in resistance_upper],
        y=[x[1] for x in resistance_upper],
        mode='lines',
        name='Resistance Band',
        fill='tonexty',
        fillcolor='rgba(255, 0, 0, 0.1)',
        line=dict(color='red', width=1),
        opacity=0.3
    ))

    # Add markers
    for _, row in filtered_df.iterrows():
        direction = row["direction"]
        if pd.isna(direction):
            fig.add_annotation(
                x=row["timestamp"],
                y=(row["high"] + row["low"]) / 2,
                text="None",
                showarrow=True,
                arrowhead=0,
                arrowcolor="yellow",
                font=dict(color="yellow")
            )
        elif direction == "SHORT":
            fig.add_annotation(
                x=row["timestamp"],
                y=row["high"] + 5,
                text="SHORT",
                showarrow=True,
                arrowhead=1,
                arrowcolor="red",
                font=dict(color="red")
            )
        elif direction == "LONG":
            fig.add_annotation(
                x=row["timestamp"],
                y=row["low"] - 5,
                text="LONG",
                showarrow=True,
                arrowhead=1,
                arrowcolor="green",
                font=dict(color="green")
            )

    # Update layout with a date range slider
    fig.update_layout(
        title="TSLA Candlestick Chart",
        yaxis_title="Price",
        xaxis_title="Date",
        plot_bgcolor="#222",
        paper_bgcolor="#222",
        font=dict(color="#DDD"),
        showlegend=True,
        yaxis=dict(range=yaxis_range),
        xaxis=dict(
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    # Add instructions for the slider
    st.write("Use the slider below the chart to zoom in on specific date ranges.")

    # Render the chart
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Gemini Chatbot
with tab2:
    st.title("Gemini Chatbot for TSLA Data")

    # Configure Gemini API
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel("gemini-1.5-pro")
    except Exception as e:
        st.error(f"Error setting up Gemini API: {str(e)}")
        st.write("Please ensure your GEMINI_API_KEY is set correctly in your environment.")
        st.stop()

    # Input for user prompt
    prompt = st.text_input("Ask about TSLA data (e.g., 'How many days in 2023 was TSLA bullish?'):")

    # Suggested questions for brownie points
    st.write("**Suggested Questions to Try:**")
    st.write("- What was the lowest closing price of TSLA in 2024?")
    st.write("- How many days in 2022 did TSLA have a volume above 150?")
    st.write("- What was the biggest daily price change in 2023?")
    st.write("- On which days in 2024 did TSLA have a closing price above 300?")
    st.write("- Summarize TSLA's performance in 2022.")

    if prompt:
        # Preprocess the prompt to extract relevant data
        relevant_df = df  # Default to the full dataset
        data_summary = ""

        # Check if the prompt mentions a specific year
        years = [str(year) for year in range(2022, 2026)]
        mentioned_year = None
        for year in years:
            if year in prompt:
                mentioned_year = year
                break

        if mentioned_year:
            # Filter data for the mentioned year
            relevant_df = df[df['timestamp'].dt.year == int(mentioned_year)]
            st.write(f"Filtering data for {mentioned_year}...")

        # Optimize the data summary to reduce token count
        # Only include relevant columns based on the question
        columns_to_include = ['timestamp', 'direction', 'open', 'close', 'high', 'low', 'volume']
        
        # If the question is about specific columns, include only those
        if "bullish" in prompt.lower() or "closing price" in prompt.lower():
            columns_to_include = ['timestamp', 'open', 'close']
        elif "volume" in prompt.lower():
            columns_to_include = ['timestamp', 'volume']
        elif "short" in prompt.lower() or "long" in prompt.lower():
            columns_to_include = ['timestamp', 'direction']
        elif "price change" in prompt.lower():
            columns_to_include = ['timestamp', 'high', 'low']
        elif "summarize" in prompt.lower():
            columns_to_include = ['timestamp', 'open', 'close', 'volume']

        # Convert the relevant data to a summarized text format
        if not relevant_df.empty:
            data_summary = f"TSLA stock data for {mentioned_year if mentioned_year else 'all years'}:\n"
            for _, row in relevant_df.iterrows():
                row_summary = []
                for col in columns_to_include:
                    if col == 'timestamp':
                        row_summary.append(f"Date: {row['timestamp'].strftime('%Y-%m-%d')}")
                    elif col == 'direction':
                        row_summary.append(f"Direction: {row['direction'] if not pd.isna(row['direction']) else 'None'}")
                    else:
                        row_summary.append(f"{col.capitalize()}: {row[col]}")
                data_summary += ", ".join(row_summary) + "\n"
        else:
            data_summary = "No data available for the requested period."

        # Combine the user prompt with the data summary
        full_prompt = f"The following is TSLA stock data:\n{data_summary}\n\nUser question: {prompt}\nAnswer the question based on the data."

        # Generate response from Gemini
        try:
            response = model.generate_content(full_prompt)
            st.write("**Gemini Response:**")
            st.write(response.text)
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.write("Please try rephrasing your question or check your API key.")
            
             #python -m streamlit run app.py
# Intelligent Function-Calling Assistant  

This project is an intelligent assistant powered by the **Llama 2 (7B)** language model and **LangChain** framework. It dynamically interprets user queries, executes relevant functions, and returns precise results in real time. The assistant is designed to handle weather queries, stock price retrieval, and arithmetic calculations through natural language interactions.  

## Features  

- **Dynamic Function Execution**  
  The assistant processes user queries, identifies the appropriate function, and invokes it to retrieve accurate results.  

- **Real-Time Weather Retrieval**  
  Provides up-to-date weather information for any location using the `python-weather` API.  

- **Stock Price Queries**  
  Fetches live stock prices for specified ticker symbols using the `yfinance` library.  

- **Mathematical Calculations**  
  Performs basic arithmetic operations such as addition, subtraction, multiplication, and division.  

- **Flexible Interaction Modes**  
  Supports three modes for user queries:
  - **Invoke**: Handles single queries and provides immediate responses.  
  - **Stream**: Processes and streams results in real time.  
  - **Batch**: Handles multiple queries simultaneously and processes them efficiently.  

- **Optimized Performance**  
  Uses GPU acceleration via PyTorch for fast model inference, with fallback to CPU when necessary.  

## Technologies Used  

- **Language Model**: Llama 2 (7B) by Meta, hosted via Hugging Face Transformers.  
- **Framework**: LangChain for prompt engineering and function orchestration.  
- **APIs**:  
  - `python-weather` for weather data.  
  - `yfinance` for stock market data.  
- **Python Libraries**: PyTorch, Transformers, asyncio, JSON handling.  

## How It Works  

1. **User Query Parsing**:  
   The user enters a natural language query, e.g., *"What is the weather in New York?"* or *"Add 15 and 7"*.  

2. **Prompt Engineering**:  
   The query is embedded into a structured prompt and passed to the Llama 2 model via LangChain.  

3. **Response Processing**:  
   The model generates a JSON-like response containing the function to call and its arguments.  

4. **Function Execution**:  
   The script identifies and invokes the relevant function to compute or fetch the result.  

5. **Result Delivery**:  
   The final result is presented to the user in a human-readable format, e.g., *"The weather in New York is cloudy with a temperature of 65°F."*.  

## Getting Started  

### Prerequisites  

- Python 3.9 or higher  
- GPU with CUDA support (optional, but recommended for performance)  
- Install required libraries:
  ```bash
  pip install -r requirements.txt
  ```

### Running the Script  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repo/intelligent-assistant.git
   cd intelligent-assistant
   ```

2. Set up a virtual environment:  
   ```bash
   python -m venv venv
   source venv/bin/activate   # For Linux/Mac
   .\venv\Scripts\activate    # For Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:  
   ```bash
   python main.py
   ```

5. Example queries:  
   - *"What is the weather in New York?"*  
   - *"What is the stock price of AAPL?"*  
   - *"Add 15 and 7"*  

## Example Outputs  

### Weather Query  
Input: *"What is the weather in New York?"*  
Output: *"The weather in New York is cloudy with a temperature of 65°F."*  

### Stock Price Query  
Input: *"What is the stock price of AAPL?"*  
Output: *"The stock price of AAPL is $155.23."*  

### Mathematical Calculation  
Input: *"Add 15 and 7"*  
Output: *"The result of the addition is 22."*  

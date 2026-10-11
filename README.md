# KGMiner
 Scientific Knowledge Mining App
This app helps you search PubMed for scientific articles and uses AI to extract key biological relationships, engineering strategies, and performance metrics. It builds a Knowledge Graph that you can explore and ask questions about.
Below is a detailed, simple guide on how to set up and run the application.
## Prerequisites
- **Python**: Make sure you have Python installed on your computer.
---
## 1. Setup the Environment
To keep everything clean, we will create a virtual environment (`venv`).
### A. Create the Virtual Environment
Open your terminal or command prompt in the project folder and run:
```bash
python -m venv venv
```
### B. Activate the Virtual Environment
- **On Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **On Mac / Linux**:
  ```bash
  source venv/bin/activate
  ```
*(You will know it is activated when you see `(venv)` at the start of your terminal line.)*
### C. Install Dependencies
With the virtual environment activated, install all required packages:
```bash
pip install -r requirements.txt
```
## 2. Create the `.env` File

The app needs API keys to use the AI models (Groq and Cerebras), as well as an email address.
1. Create a new file in the project folder and name it EXACTLY `.env`
2. Open it and paste your API keys and email in this format:
```env
GROQ_API_KEY1=your_groq_api_key_here
GROQ_API_KEY2=your_groq_key_here
HELIUS_API_KEY1=your_helius_key_here
HELIUS_API_KEY2=your_helius_key_here
HELIUS_API_KEY3=your_helius_key_here
HELIUS_API_KEY4=your_helius_key_here
EMAIL=your_email@example.com
```
*(Replace `your_..._here` with actual keys and email. If you don't have all 5 Cerebras keys, just provide the ones you have. The app will work with whatever is available.)*
---
## 3. How to Start the App
Both the **Backend API** and the **HTML Frontend** are served by the same single application.
1. Open a terminal in the project folder.
2. Activate your virtual environment:
   - **On Windows**: `venv\Scripts\activate`
   - **On Mac / Linux**: `source venv/bin/activate`
3. Run this command to start the server:
   ```bash
   uvicorn app:app --reload --port 5678
   ```
4. Open your web browser and go to:
   **[http://localhost:5678](http://localhost:5678)**
*(Leave this terminal window open and running while you use the app.)*
---
## 4. How to Use the App
1. **Enter a Goal**: Once the frontend is open, type in your research goal (for example, "production of beta-carotene").
2. **Process**: The system will automatically search PubMed for relevant scientific abstracts.
3. **Extraction**: The AI models will process the text, extract relationships, and build a structured "Knowledge Graph".
4. **Chat**: Once the pipeline is complete, you can use the chat interface to ask specific questions about the data it found.
To stop the app when you are done, go to both terminal windows and press `Ctrl + C`. Then type `deactivate` to exit the virtual environment.

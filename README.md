# Hotel Analytics RAG 🏨

Hotel Analytics RAG is a powerful analytical platform and Q&A system that transforms hotel booking data into meaningful insights using Retrieval-Augmented Generation (RAG) with FAISS and LLaMA 2.

## 🌟 Features

- **📊 Advanced Analytics Dashboard**: Visualize revenue trends, cancellation patterns, booking geography, and customer segmentation
- **🔍 Natural Language Q&A System**: Ask questions about your hotel data in plain English
- **⚡ High-Performance Backend**: GPU-accelerated embeddings and inference with CUDA support
- **🧠 LLaMA 2 Integration**: Leverage the power of 7B parameter LLM for generating accurate responses
- **📈 Comprehensive Insights**: From lead time analysis to geographical patterns and customer behavior

## 📋 System Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)
- **Hugging Face account with access to LLaMA 2 (7B Chat)** - You must accept the model terms at [LLaMA 2 Model Card](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf)
- CUDA configured for GPU acceleration

## ⚙️ Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/Hotel-Analytics-RAG.git
cd Hotel-Analytics-RAG
```

2. **Create and activate virtual environment**
```bash
# Create virtual environment
python -m venv venv

# Activate Virtual Environment
cd D:\Projects\Hotel-Analytics-RAG
venv\Scripts\activate  # On Windows
source venv/bin/activate  # On Linux/Mac
```

3. **Install dependencies**
```bash
# Install Required Libraries
pip install -r requirements.txt
```

4. **Set up Hugging Face token for LLaMA 2 access**
```python
# Add to your environment or use in code
from huggingface_hub import login
login("your_hf_token")
```

5. **Build the RAG system**
```bash
# Build RAG: Generate embeddings and FAISS index
python pipeline/main.py
```

6. **Launch the applications**
```bash
# Run Analytics Dashboard
streamlit run analytics\dashboard.py

# Run RAG API (Backend)
uvicorn app.main:app --reload
```

## 🗂️ Project Structure & File Details

```
Hotel-Analytics-RAG/
│
├── analytics/               # Analytics module
│   ├── dashboard.py         # Main analytics dashboard that integrates all visualizations
│   ├── visuals1.py          # Revenue trend visualizations
│   ├── visuals2.py          # Cancellation analysis charts
│   ├── visuals3.py          # Geographical distribution maps
│   ├── visuals4.py          # Lead time distributions
│   └── visuals5.py          # Additional segmentation analytics
│
├── pipeline/                # RAG pipeline
│   ├── main.py              # Main pipeline orchestration script
│   ├── embeddings.py        # Sentence embedding generation using transformers
│   ├── faiss_index.py       # Vector database creation and similarity search
│   ├── db_operations.py     # SQLite CRUD operations for analytics
│   ├── utils.py             # Helper functions and preprocessing utilities
│   └── country_mapping.json # Standardization map for country names
│
├── app/                     # API module
│   ├── main.py              # FastAPI application with routes for analytics and Q&A
│   └── templates/           # Frontend templates
│       └── index.html       # UI interface for user interaction
│
├── dataset/                 # Data storage
│   ├── hotel_bookings_dataset.csv  # Primary dataset
│   ├── analytics.db         # SQLite database with precomputed insights
│   ├── sources.txt          # Documentation on data sources
│   └── details.txt          # Field explanations and metadata
│
├── preprocessing/           # Data preparation
│   ├── preprocess.ipynb     # Data cleaning and transformation notebook
│   └── backup.csv           # Backup of preprocessed data
│
└── requirements.txt         # Project dependencies
```

## 🔍 API Endpoints

- **POST /analytics**: Returns precomputed analytical insights including charts and textual data
- **POST /ask**: Accepts natural language queries and returns RAG-generated answers by combining vector search with LLaMA 2 generation

## 💡 Example Queries

| Query | Sample Response |
|-------|-----------------|
| "What was the revenue in August 2017?" | $24,520.34 |
| "Which countries canceled the most bookings?" | Portugal, United Kingdom, Brazil |
| "Average length of stay?" | 3.5 nights (on average) |
| "Show me booking trends by market segment" | Corporate bookings constitute 30.5%, while leisure travel accounts for 45.2%... |
| "What's the relationship between lead time and cancellation?" | Bookings made 60+ days in advance have a 24% higher cancellation rate... |

## 🧠 Project Deliverables (Step-Wise)

### 1️⃣ Data Collection & Preprocessing
- **Dataset**: Used the hotel_bookings_dataset.csv from Kaggle
- **Tasks Completed**:
  - Merged and deduplicated booking records from multiple source files
  - Handled missing values using appropriate strategies (mean/median/mode)
  - Standardized formatting for dates, currencies, and country names
  - Implemented data type conversions and normalization
  - Created a structured SQLite database (analytics.db) for efficient querying

### 2️⃣ Analytics & Reporting
- **Implemented in analytics/dashboard.py and visuals modules**:
  - 📈 **Revenue trends over time**: Monthly and seasonal revenue analysis with year-over-year comparisons
  - ❌ **Cancellation rate analysis**: Percentage cancellations by time period, lead time, customer segment
  - 🌍 **Booking geography**: Interactive maps showing booking distribution and revenue by country
  - ⏳ **Lead time distribution**: Analysis of booking patterns by days before arrival
  - 🧩 **Additional analytics**: 
    - Customer segmentation analysis
    - Agent and company booking patterns
    - Wait list performance metrics
    - ADR (Average Daily Rate) analysis by room type and season

### 3️⃣ Retrieval-Augmented Generation (RAG)
- **Vector Database**:
  - Implemented FAISS for efficient similarity search
  - Created sentence embeddings of analytical insights using sentence-transformers
  - Built a custom chunking strategy for optimal retrieval
- **LLM Integration**:
  - Integrated LLaMA 2 7B Chat model via Hugging Face for question answering
  - Implemented context window management for efficient token usage
  - Created prompt templates optimized for analytical question answering

### 4️⃣ API Development
- **FastAPI Implementation (app/main.py)**:
  - Built RESTful API with asynchronous processing
  - Created two main endpoints:
    - POST /analytics: Returns comprehensive analytical insights with visualizations
    - POST /ask: Natural language interface to the RAG pipeline
  - Implemented proper error handling and response formatting
  - Created HTML frontend for easy user interaction

### 5️⃣ Performance Evaluation
- **Accuracy Assessment**:
  - Manually validated responses against known dataset values
  - Created test suite with expected answers for key analytical questions
- **Performance Optimization**:
  - Measured and optimized response latency
  - Implemented CUDA acceleration for embedding generation and model inference
  - Optimized FAISS indexing with GPU support
  - Created caching mechanisms for frequent queries

### 6️⃣ Deployment & Submission
- **Packaging**:
  - Created comprehensive documentation
  - Defined clear setup instructions and requirements
  - Provided example queries and expected answers
- **Deployment**:
  - Configured for local deployment via uvicorn
  - Made ready for cloud deployment options (Azure, Vercel, Streamlit sharing)
  - Created containerization support

## 📊 Analytics Capabilities

- Revenue analysis by time period, room type, and customer segment
- Cancellation patterns and predictors with associated revenue impact
- Geographic distribution of bookings with interactive mapping
- Lead time analysis and correlation with other booking factors
- Customer segmentation and behavior analysis
- Agent and company booking pattern analysis
- Seasonal trends and occupancy patterns
- Pricing strategy insights and optimization opportunities

## 🛠️ Technologies Used

- **FastAPI**: High-performance REST API framework
- **FAISS**: Efficient similarity search and vector database
- **Sentence Transformers**: Text embedding generation for semantic search
- **LLaMA 2 (7B Chat)**: Large language model for question understanding and response generation
- **SQLite**: Lightweight relational database for analytics storage
- **Streamlit**: Interactive dashboard creation
- **Pandas/Matplotlib/Seaborn**: Data manipulation and visualization
- **CUDA**: GPU acceleration for model inference and embedding generation

## 🔧 Troubleshooting

- **CUDA Issues**: Ensure you have compatible NVIDIA drivers installed for your GPU
- **LLaMA 2 Access**: Make sure you've accepted the model terms on Hugging Face
- **Memory Errors**: For large datasets, consider increasing chunk size or using CPU fallback

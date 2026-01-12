# Green AI | Pune 2027 Smart Urban Heat Forecast

**AI-powered interactive dashboard forecasting Urban Heat Island (UHI) effect across Pune in 2027**  
Recommends strategic **Neem tree plantation** to fight rising temperatures using machine learning and real-time weather data.

<p align="center">
  <img src="https://via.placeholder.com/1280x720/0ea5e9/ffffff?text=Green+AI+Pune+2027+Dashboard+Preview" alt="Dashboard Preview" width="80%"/>
  <br><br>
  <em>Glassmorphic UI • Interactive Leaflet map • Scientific yet beautiful climate forecast</em>
</p>

## ✨ Key Features

- 2027 temperature prediction for 11 major zones in Pune  
- Urban Heat Island (UHI) intensity index & visual classification  
- Estimated future air density under warming scenarios  
- Data-driven calculation of **additional Neem trees needed** for cooling  
- Sleek glass-effect cards + responsive design  
- Real-time weather integration (EOS API with realistic fallback)  
- Random Forest models for temperature & cooling effect prediction  

## 🛠️ Tech Stack

- **Backend** — Flask (Python)  
- **Machine Learning** — scikit-learn RandomForestRegressor  
- **Frontend** — HTML5, Bootstrap 5, custom glassmorphism CSS  
- **Maps** — Leaflet.js + CartoDB Positron tiles  
- **Weather** — EOS Data Analytics Weather API  
- **Icons** — Font Awesome  

## Quick Start

```bash
# Clone repository
git clone https://github.com/yourusername/green-ai-pune-2027.git
cd green-ai-pune-2027

# Create & activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate          # Linux/macOS
# or
venv\Scripts\activate             # Windows

# Install dependencies
pip install flask requests scikit-learn pandas numpy

# Run the application
python app.py
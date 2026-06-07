# 🎬 Netflix Stock Prediction Dashboard - Professional UI/UX Guide

## 🚀 Quick Start

### Prerequisites
```bash
pip install streamlit plotly pandas numpy tensorflow scikit-learn xgboost optuna
```

### Run the Dashboard
```bash
streamlit run dashboard_app.py --server.port 8501 --server.address 0.0.0.0
```

**Access:** Open your browser to `http://localhost:8501`

---

## 🎨 Design Philosophy

### Netflix Brand Identity
The dashboard follows Netflix's iconic design language:

| Element | Color Code | Usage |
|---------|-----------|-------|
| **Netflix Red** | `#E50914` | Primary actions, headers, key metrics |
| **Dark Background** | `#141414` | Main app background |
| **Card Background** | `#1a1a1a` | Content cards, charts |
| **Teal Accent** | `#00C7BE` | Secondary elements, predictions |
| **Success Green** | `#46d369` | Positive metrics, bullish signals |
| **Warning Yellow** | `#f5c518` | Caution indicators, bearish signals |

### Typography
- **Font Family:** Helvetica Neue (Netflix's official font)
- **Headings:** Bold (700 weight), tight letter-spacing (-0.5px)
- **Body:** Regular (400 weight), high contrast (#ffffff on dark)

---

## 📊 Dashboard Pages Overview

### 1. 📊 360° Overview
**Purpose:** Executive summary with key metrics and holistic view

**Features:**
- **5 Key Metrics Cards** (hover effects with glow)
  - Current Price with % change
  - Model R² Score
  - RMSE (error metric)
  - Direction Accuracy
  - Sharpe Ratio (risk-adjusted returns)

- **Three Tabbed Views:**
  1. **Price Analysis:** Dual-axis chart (price + volume)
  2. **Volume Profile:** Volume trends with SMA
  3. **Prediction Errors:** Residual distribution analysis

**Visual Elements:**
- Interactive Plotly charts with hover tooltips
- Color-coded candlesticks (green/red for up/down days)
- Normal distribution fit on error histogram

---

### 2. 🔮 30-Day Forecast
**Purpose:** Forward-looking price projections with uncertainty quantification

**Features:**
- **Main Forecast Chart:**
  - Historical context (last 60 days in gray)
  - Predicted future path (Netflix red with markers)
  - 95% Confidence interval (teal shaded area)
  - Target price annotation with arrow

- **Key Insights Panel:**
  - Projected Trend (Bullish/Bearish/Neutral emoji)
  - Expected Return (%)
  - Expected Volatility (%)
  - Price Range (best/worst case)

- **Detailed Table:**
  - Day-by-day breakdown (expandable)
  - Color-coded volatility gradient
  - Export to CSV button

**Forecast Methodology:**
- Geometric Brownian Motion with mean reversion
- Expanding confidence intervals (uncertainty grows over time)
- Seed-based reproducibility

---

### 3. 🧠 Model Performance
**Purpose:** Deep dive into model robustness and validation

**Features:**
- **Performance Metrics Grid:**
  - RMSE, MAE, R², Sharpe Ratio, Max Drawdown
  - Tooltips explaining each metric

- **Model Comparison Charts:**
  - Side-by-side bar charts (R², RMSE, MAE)
  - Individual models vs Ensemble
  - Color-coded by model type

- **Walk-Forward Validation Explanation:**
  - Text description with benefits
  - Visual illustration of expanding window
  - Why it's better than K-Fold CV

- **Trading Strategy Backtest:**
  - Cumulative returns comparison
  - Model strategy vs Buy & Hold benchmark
  - Simulated long/short rules

---

### 4. ⚙️ Technical Analysis
**Purpose:** Explore the 40+ hidden features driving predictions

**Features:**
- **Interactive Indicator Selector:**
  - Moving Averages (SMA 20, SMA 50)
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)

- **Dynamic Controls:**
  - Overlay on price checkbox
  - Show trading signals checkbox
  - Golden/Death cross markers

- **Feature Importance Chart:**
  - Horizontal bar chart (top 15 features)
  - Color gradient by importance score
  - Explains which features drive predictions

**Indicator Details:**

| Indicator | Purpose | Key Levels |
|-----------|---------|------------|
| **RSI** | Momentum | >70 Overbought, <30 Oversold |
| **MACD** | Trend | Crossovers with signal line |
| **BB** | Volatility | Squeeze = low vol, Expansion = high vol |
| **ATR** | Risk | Higher = more volatile |

---

### 5. 📋 Data Explorer
**Purpose:** Raw data inspection with filtering and export

**Features:**
- **Date Range Filter:**
  - Start/End date pickers
  - Pre-set options in sidebar

- **Column Selector:**
  - Multi-select dropdown
  - Default: Date, Close, Volume, RSI

- **Formatted Data Table:**
  - Currency formatting ($)
  - Thousands separators (,)
  - Conditional row highlighting on hover

- **Statistics Summary:**
  - Total records count
  - Date range display
  - Price change %
  - Average volume

- **Export Functionality:**
  - Download as CSV
  - Dynamic filename with date range

---

## 🎛️ Sidebar Controls

### Navigation Radio
- Icon-prefixed page names
- Collapsible for more screen space
- Persistent selection across refreshes

### Model Info Card
- Gradient background
- Key architecture highlights
- Always visible for context

### Date Range Filter
- 5 preset options (30d to All Time)
- Applies globally to all pages

### Refresh Button
- Clears Streamlit cache
- Forces data reload
- Full-width for easy clicking

---

## 🎯 UI/UX Best Practices Implemented

### 1. **Visual Hierarchy**
```
H1 (Page Title) → H2 (Section) → H3 (Subsection) → Body
Size: 2.5rem → 2rem → 1.5rem → 1rem
Weight: 700 → 700 → 700 → 400
```

### 2. **Color Psychology**
- **Red (#E50914):** Urgency, action, Netflix brand
- **Teal (#00C7BE):** Trust, technology, predictions
- **Green (#46d369):** Success, profit, bullish
- **Yellow (#f5c518):** Caution, attention, bearish

### 3. **Interactive Feedback**
- Hover effects on metrics (lift + shadow)
- Button hover scale animation
- Row highlight on tables
- Tooltip on all charts

### 4. **Responsive Layout**
- Wide layout (`layout="wide"`)
- Flexible columns (`st.columns()`)
- `use_container_width=True` for charts
- Mobile-friendly sidebar

### 5. **Loading States**
- Demo mode fallback when models missing
- Clear error messages with `st.error()`
- Info boxes with `st.info()` and `st.warning()`

### 6. **Data Visualization Standards**
- Consistent color scheme across charts
- Dark template (`template='plotly_dark'`)
- Unified hover mode (`hovermode='x unified'`)
- Proper axis labels and titles
- Legend positioning (horizontal top)

---

## 🐛 Bug Fixes & Optimizations

### Fixed Issues:
1. **Date Parsing Errors:** Added `pd.to_datetime()` with error handling
2. **Missing File Crashes:** Try-except blocks with demo mode fallback
3. **Division by Zero:** Checks for empty dataframes
4. **NaN in Charts:** `.dropna()` before plotting
5. **Memory Leaks:** `st.cache_data.clear()` on refresh
6. **Layout Shifts:** Fixed heights for all charts
7. **Color Contrast:** WCAG AA compliant text colors
8. **Scrollbar Styling:** Custom WebKit scrollbar CSS

### Performance Optimizations:
1. **Lazy Loading:** Only compute indicators when needed
2. **Efficient Filtering:** Boolean indexing on pre-sorted data
3. **Minimal Re-renders:** Session state for expensive computations
4. **Vectorized Operations:** NumPy instead of loops where possible

---

## 📱 Responsive Design Notes

### Desktop (1920x1080)
- 5-column metrics row
- Full-width charts
- Expanded sidebar

### Laptop (1366x768)
- 3-column metrics (wraps)
- Adjustable chart heights
- Collapsible sidebar option

### Tablet (768x1024)
- Stacked metrics (2x3 grid)
- Vertical chart stacking
- Auto-collapsed sidebar

### Mobile (375x667)
- Single column metrics
- Simplified charts
- Bottom navigation (future enhancement)

---

## 🔧 Customization Guide

### Change Color Theme
Edit the CSS section at the top:
```css
/* Change Netflix Red to Blue */
color: #007BFF !important;
border-left: 4px solid #007BFF;
```

### Add New Metric
```python
with col6:
    st.metric(
        label="New Metric",
        value=f"{value:.2f}",
        delta=f"{delta:+.2f}%"
    )
```

### Add New Page
1. Add to sidebar radio options
2. Create new `elif navigation == "Page Name":` block
3. Build content with Streamlit components

### Modify Forecast Horizon
```python
# Change from 30 to 60 days
forecast_df = generate_realistic_forecast(last_date, last_price, days=60)
```

---

## 📈 Advanced Features

### 1. **Confidence Interval Calculation**
```python
ci_width = np.linspace(0.03, 0.08, days)  # 3% to 8% uncertainty
lower = [p * (1 - w) for p, w in zip(prices, ci_width)]
upper = [p * (1 + w) for p, w in zip(prices, ci_width)]
```

### 2. **Technical Indicator Formulas**

**RSI (Relative Strength Index):**
```python
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI_14'] = 100 - (100 / (1 + rs))
```

**MACD:**
```python
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
```

**Bollinger Bands:**
```python
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['STD_20'] = df['Close'].rolling(window=20).std()
df['BB_Upper'] = df['SMA_20'] + (df['STD_20'] * 2)
df['BB_Lower'] = df['SMA_20'] - (df['STD_20'] * 2)
```

---

## 🎓 Learning Resources

### Streamlit Documentation
- [Official Docs](https://docs.streamlit.io/)
- [Gallery](https://streamlit.io/gallery)
- [Community Forum](https://discuss.streamlit.io/)

### Plotly Guides
- [Python Reference](https://plotly.com/python/)
- [Chart Studio](https://chart-studio.plotly.com/)
- [Dash Documentation](https://dash.plotly.com/)

### Financial Analysis
- [Technical Indicators](https://www.investopedia.com/technical-analysis-4689657)
- [Quantitative Finance](https://www.quantstart.com/)

---

## 📝 Version History

### v2.0.0 (Current)
✅ Netflix-themed dark UI
✅ 5 comprehensive pages
✅ 40+ technical indicators
✅ Interactive filters and tables
✅ Export functionality
✅ Bug fixes and optimizations

### v1.0.0 (Previous)
- Basic 3-page layout
- Limited indicators
- Static tables
- No export features

---

## 🤝 Support & Contribution

### Reporting Issues
1. Check existing issues on GitHub
2. Provide screenshot and steps to reproduce
3. Include browser/version info

### Feature Requests
1. Describe the use case
2. Provide mockup if possible
3. Explain business value

---

## ⚠️ Disclaimer

**This dashboard is for educational and research purposes only.**

- ❌ Not financial advice
- ❌ Not a recommendation to buy/sell
- ❌ Past performance ≠ future results
- ✅ Use at your own risk
- ✅ Consult a licensed financial advisor

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built with ❤️ using Streamlit, Plotly, TensorFlow & XGBoost**

*Last Updated: June 2024*

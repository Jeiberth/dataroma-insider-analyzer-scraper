
<h1 align="center">Dataroma Insider Analyzer & Scraper</h1>
<h3 align="center">High-conviction insider trading analysis with Python, Playwright, and Polygon.io</h3>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Playwright-2EAD33?style=for-the-badge&logo=playwright&logoColor=white" alt="Playwright"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Polygon.io-5A33D6?style=for-the-badge&logo=polygon&logoColor=white" alt="Polygon.io"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge" alt="License"/>
</p>

---

### ✨ Key Features

This script identifies "high-conviction" stocks—those with purchases from *multiple* different insiders—and then analyzes the historical price performance following those buys.

* ✅ **Scrapes Dataroma** for all recent insider transactions using Playwright.
* ✅ **Filters for "High-Conviction" Symbols:** Identifies stocks purchased by **two or more different insiders**.
* ✅ **Fetches Complete History:** Scrapes the full insider transaction history for all qualifying symbols.
* ✅ **Performance Analysis:** Analyzes historical price performance at **3, 8, 15, 30, and 90 days** after a purchase.
* ✅ **Average % Performance:** Calculates the average percentage return for each time window, not just a binary success/fail.
* ✅ **Per-Purchaser Analysis:** Groups analysis by the individual purchaser to see which insiders have the best track records.
* ✅ **Intelligent Clustering:** Groups purchases by the *same reporter* within a 10-day window into a single "buy event".
* ✅ **Consolidated Output:** Exports all findings into a single, comprehensive JSON file.

---

### 🛠️ Technology Stack

<table width="100%">
  <tr>
    <td align="center" width="25%">
      <strong>Core</strong>
    </td>
    <td align="center" width="25%">
      <strong>Web Scraping</strong>
    </td>
    <td align="center" width="25%">
      <strong>Data & API</strong>
    </td>
    <td align="center" width="25%">
      <strong>Utilities</strong>
    </td>
  </tr>
  <tr>
    <td valign="top">
      <p align="center">
        <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
        <img src="https://img.shields.io/badge/Asyncio-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Asyncio" />
      </p>
    </td>
    <td valign="top">
      <p align="center">
        <img src="https://img.shields.io/badge/Playwright-2EAD33?style=for-the-badge&logo=playwright&logoColor=white" alt="Playwright" />
      </p>
    </td>
    <td valign="top">
      <p align="center">
        <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
        <img src="https://img.shields.io/badge/Polygon.io-5A33D6?style=for-the-badge&logo=polygon&logoColor=white" alt="Polygon.io" />
        <img src="https://img.shields.io/badge/AIOHTTP-2C528B?style=for-the-badge&logo=aiohttp&logoColor=white" alt="AIOHTTP" />
      </p>
    </td>
    <td valign="top">
      <p align="center">
        <img src="https://img.shields.io/badge/dotenv-ECD53F?style=for-the-badge&logo=dotenv&logoColor=black" alt="python-dotenv" />
      </p>
    </td>
  </tr>
</table>

---

### ⚙️ How It Works

1.  **Initial Scan:** Scrapes the main Dataroma insider page (`ins.php`) to get a broad list of all recent transactions.
2.  **Filtering:** Parses the initial list to find symbols that have been purchased by at least two *different* reporters.
3.  **Deep Dive:** For each qualifying symbol, the scraper navigates to its specific history page.
4.  **Purchase Identification:** It scans the history for all *purchases* made by those specific, qualifying reporters.
5.  **Clustering:** Groups purchases made by the *same reporter* within a 10-day window, treating them as a single event.
6.  **Price Fetching:** Uses the **Polygon.io API** to get historical OHLC price data for the symbol.
7.  **Analysis:** For each "buy event," it calculates the stock's price change after 3, 8, 15, 30, and 90 days.
8.  **Aggregation:** Calculates the average performance for the symbol as a whole and for each individual purchaser.
9.  **Output:** Saves all data (recent buys, analysis, summaries) into `insider_trades_analysis.json`.

---

### 🚀 Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/dataroma-insider-analyzer-scraper.git](https://github.com/your-username/dataroma-insider-analyzer-scraper.git)
    cd dataroma-insider-analyzer-scraper
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required Python packages:**
    ```bash
    pip install playwright pandas aiohttp python-dotenv
    ```

4.  **Install Playwright's browser binaries:**
    ```bash
    playwright install
    ```

---

### 🔑 Configuration

This script requires an API key from **[Polygon.io](https://polygon.io/)** to fetch historical price data.

1.  Create a file named `.env` in the root directory of the project.
2.  Add your Polygon.io API key to this file:

    ```ini
    POLYGON_API_KEY=YOUR_API_KEY_HERE
    ```

---

### ⚡ Running the Scraper

To run the scraper, you'll need a main script (e.g., `main.py`) to import and execute the `DataromaScraper` class.

**Example `main.py`:**

```python
import asyncio
import json
import pandas as pd
from dataroma_scraper import DataromaScraper # Assuming your class is in dataroma_scraper.py

async def run_analysis():
    # Max pages to scrape from the main Dataroma list.
    # Set to 1 or 2 for testing, or None to scrape all.
    INITIAL_PAGES_TO_SCRAPE = 2 

    # Max history pages to check per symbol.
    MAX_PAGES_PER_SYMBOL = 5 
    
    print("Starting scraper...")
    # The API key is loaded from .env automatically
    scraper = DataromaScraper()
    
    # 1. Scrape initial pages
    initial_data = await scraper.scrape_all_pages(max_pages=INITIAL_PAGES_TO_SCRAPE)
    
    # 2. Filter for symbols with multiple reporters
    filtered_data, symbol_reporters = scraper.filter_multiple_reporters(initial_data)
    
    if not symbol_reporters:
        print("No symbols found with multiple reporters in the scanned pages.")
        return

    # 3. Analyze all qualifying symbols
    analysis_data, global_results = await scraper.analyze_all_symbols(
        symbol_reporters, 
        max_pages_per_symbol=MAX_PAGES_PER_SYMBOL
    )
    
    # 4. Save consolidated JSON
    scraper.save_to_json(analysis_data, "insider_trades_analysis.json")

    print("\n--- DETAILED CONSOLE OUTPUT ---")
    for symbol, data in analysis_data.items():
        print(scraper.format_symbol_output(
            symbol,
            data.get("recent_purchases", []),
            data.get("price_performance_analysis", []),
            data.get("average_performance", {}),
            data.get("purchaser_performance_summary", {})
        ))

if __name__ == "__main__":
    asyncio.run(run_analysis())
````

Then, run the main file from your terminal:

```bash
python main.py
```

-----

### 📊 Example Output

The script generates a single `insider_trades_analysis.json` file with a nested structure.

```json
{
  "SYMBOL_A": {
    "recent_purchases": [
      {
        "f_date": "10/20/2025",
        "rep_name": "CEO Jane Doe",
        "amt": "$1,000,000",
        "parsed_date": "2025-10-20T00:00:00"
      }
    ],
    "price_performance_analysis": [
      {
        "Purchaser": "CEO Jane Doe",
        "Symbol": "SYMBOL_A",
        "Purchase Date": "08/15/2025",
        "Cluster Size": 1,
        "+3 Days": "1.50%",
        "+8 Days": "2.10%",
        "+15 Days": "-0.50%",
        "+1 Month": "4.20%",
        "+3 Months": "10.80%"
      }
    ],
    "average_performance": {
      "avg_3_days": 1.5,
      "avg_8_days": 2.1,
      "avg_total": 3.62
    },
    "purchaser_performance_summary": {
      "CEO Jane Doe": {
        "avg_3_days": 1.5,
        "avg_total": 3.62
      }
    },
    "polygon_api_response": {
      "ticker": "SYMBOL_A",
      "resultsCount": 180,
      "results": [ ... ]
    }
  },
  "SYMBOL_B": { ... }
}
```

-----

### ⚠️ Disclaimer

\<p align="center"\>
This tool is for educational and informational purposes only. \<strong\>It is not financial advice.\</strong\><br>
The data is scraped from a third-party website and is not guaranteed to be accurate, complete, or timely.<br>
Always do your own research before making any investment decisions.
\</p\>

```

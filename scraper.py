"""
Dataroma Insider Trading Web Scraper with Price Analysis
Scrapes insider trading data, filters for symbols with multiple reporters,
and analyzes price performance after insider purchases.

MODIFIED (v3):
- Calculates average % performance instead of success rate.
- Clusters purchases by *purchaser* first, then by date.
- Adds performance analysis per individual purchaser.
- Outputs a single, consolidated JSON file with all data.
"""

import asyncio
import json
import csv
from typing import List, Dict, Optional, Set, Tuple
from playwright.async_api import async_playwright, Page
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta
import aiohttp
import os
from dotenv import load_dotenv
import random
import time
import sys

load_dotenv()

class DataromaScraper:
    def __init__(self, base_url: str = "https://www.dataroma.com/m/ins/ins.php", polygon_api_key: str = None):
        self.base_url = base_url
        self.base_params = "?t=m&po=1&am=100000&sym=&o=is&d=d"

        if polygon_api_key is None:
            self.polygon_api_key = os.getenv('POLYGON_API_KEY')
        else:
            self.polygon_api_key = polygon_api_key

    async def scrape_page(self, page: Page, page_num: int = 1) -> List[Dict]:
        """Scrape a single page and extract all row data"""
        url = f"{self.base_url}{self.base_params}&L={page_num}"
        print(f"Scraping page {page_num}: {url}")

        await page.goto(url, wait_until="networkidle")

        # Wait for the table to load
        await page.wait_for_selector("tbody tr", timeout=10000)

        # Extract data from all rows
        rows = await page.query_selector_all("tbody tr")

        data = []
        for row in rows:
            try:
                # Extract all cells
                cells = await row.query_selector_all("td")

                if len(cells) < 11:  # Ensure we have all expected columns
                    continue

                # Extract f_date with link and time
                f_date_cell = await cells[0].query_selector("a")
                f_date_link = await f_date_cell.get_attribute("href") if f_date_cell else ""
                f_date_text = await cells[0].inner_text()

                # Extract iss_sym
                iss_sym_cell = await cells[1].query_selector("a")
                iss_sym = await iss_sym_cell.inner_text() if iss_sym_cell else ""
                iss_sym = iss_sym.strip()

                # Extract iss_name
                iss_name = await cells[2].inner_text()
                iss_name = iss_name.strip()

                # Extract rep_name
                rep_name_cell = await cells[3].query_selector("a")
                rep_name = await rep_name_cell.inner_text() if rep_name_cell else ""
                rep_name = rep_name.strip()

                # Extract remaining fields
                rel = (await cells[4].inner_text()).strip()
                t_date = (await cells[5].inner_text()).strip()
                tran_code = (await cells[6].inner_text()).strip()
                sh = (await cells[7].inner_text()).strip()
                pr = (await cells[8].inner_text()).strip()
                amt = (await cells[9].inner_text()).strip()
                dir_ind = (await cells[10].inner_text()).strip()

                row_data = {
                    "f_date": f_date_text.strip(),
                    "f_date_link": f_date_link,
                    "iss_sym": iss_sym,
                    "iss_name": iss_name,
                    "rep_name": rep_name,
                    "rel": rel,
                    "t_date": t_date,
                    "tran_code": tran_code,
                    "sh": sh,
                    "pr": pr,
                    "amt": amt,
                    "dir_ind": dir_ind
                }

                data.append(row_data)

            except Exception as e:
                print(f"Error extracting row: {e}")
                continue

        return data

    async def scrape_all_pages(self, max_pages: Optional[int] = None) -> List[Dict]:
        """Scrape all pages until no more data is found"""
        all_data = []
        page_num = 1

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                while True:
                    if max_pages and page_num > max_pages:
                        print(f"Reached maximum page limit: {max_pages}")
                        break

                    page_data = await self.scrape_page(page, page_num)

                    if not page_data:
                        print(f"No data found on page {page_num}. Stopping.")
                        break

                    all_data.extend(page_data)
                    print(f"Extracted {len(page_data)} rows from page {page_num}")

                    page_num += 1

                    # Small delay to be respectful to the server
                    await asyncio.sleep(1)

            finally:
                await browser.close()

        return all_data

    def filter_multiple_reporters(self, data: List[Dict]) -> tuple[List[Dict], Dict[str, Set[str]]]:
        """
        Filter data to only include symbols that appear at least twice 
        with different reporting names.
        Returns: (filtered_data, symbol_to_reporters_mapping)
        """
        # Group by symbol and collect unique reporter names
        symbol_reporters = defaultdict(set)

        for row in data:
            symbol = row.get("iss_sym", "").strip()
            reporter = row.get("rep_name", "").strip()

            if symbol and reporter:  # Only count non-empty values
                symbol_reporters[symbol].add(reporter)

        # Find symbols with at least 2 different reporters
        qualifying_symbols = {
            symbol for symbol, reporters in symbol_reporters.items() 
            if len(reporters) >= 2
        }

        print(f"\nFound {len(qualifying_symbols)} symbols with multiple reporters:")
        for symbol in sorted(qualifying_symbols):
            reporters = symbol_reporters[symbol]
            print(f"  {symbol}: {len(reporters)} different reporters")

        # Filter the original data
        filtered_data = [
            row for row in data 
            if row.get("iss_sym", "").strip() in qualifying_symbols
        ]

        # Return only the qualifying symbols' reporters
        qualifying_symbol_reporters = {
            symbol: reporters 
            for symbol, reporters in symbol_reporters.items() 
            if symbol in qualifying_symbols
        }

        return filtered_data, qualifying_symbol_reporters

    async def scrape_symbol_history(self, page: Page, symbol: str, page_num: int = 1) -> List[Dict]:
        """Scrape a single page of a symbol's history"""
        url = f"{self.base_url}?t=y2&po=1&am=0&sym={symbol}&o=td&d=d&L={page_num}"

        try:
            # Use domcontentloaded instead of networkidle for faster loading
            await page.goto(url, wait_until="domcontentloaded", timeout=60000)

            # Wait for the table to load
            await page.wait_for_selector("tbody tr", timeout=10000)
        except Exception as e:
            print(f"  Warning: Could not load page {page_num} for {symbol}: {str(e)[:100]}")
            return []  # No data on this page

        # Extract data from all rows
        rows = await page.query_selector_all("tbody tr")

        data = []
        for row in rows:
            try:
                cells = await row.query_selector_all("td")

                if len(cells) < 11:
                    continue

                # Extract rep_name
                rep_name_cell = await cells[3].query_selector("a")
                rep_name = await rep_name_cell.inner_text() if rep_name_cell else ""
                rep_name = rep_name.strip()

                # Extract transaction date and code
                t_date = (await cells[5].inner_text()).strip()
                tran_code = (await cells[6].inner_text()).strip()

                # Extract other fields for completeness
                f_date_text = await cells[0].inner_text()
                iss_sym_cell = await cells[1].query_selector("a")
                iss_sym = await iss_sym_cell.inner_text() if iss_sym_cell else ""
                iss_name = (await cells[2].inner_text()).strip()
                rel = (await cells[4].inner_text()).strip()
                sh = (await cells[7].inner_text()).strip()
                pr = (await cells[8].inner_text()).strip()
                amt = (await cells[9].inner_text()).strip()
                dir_ind = (await cells[10].inner_text()).strip()

                row_data = {
                    "f_date": f_date_text.strip(),
                    "iss_sym": iss_sym.strip(),
                    "iss_name": iss_name,
                    "rep_name": rep_name,
                    "rel": rel,
                    "t_date": t_date,
                    "tran_code": tran_code,
                    "sh": sh,
                    "pr": pr,
                    "amt": amt,
                    "dir_ind": dir_ind
                }

                data.append(row_data)

            except Exception as e:
                print(f"Error extracting row: {e}")
                continue
        
        return data
    
    async def find_purchases_by_reporters(self, symbol: str, reporters: Set[str], max_pages: int = 10) -> List[Dict]:
        """
        For a given symbol, find all purchases made by the specified reporters
        """
        print(f"\n{'='*60}")
        print(f"Searching purchases for symbol: {symbol}")
        print(f"Looking for reporters: {', '.join(sorted(reporters))}")

        all_purchases = []
        page_num = 1
        consecutive_failures = 0

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            try:
                while page_num <= max_pages:
                    page_data = await self.scrape_symbol_history(page, symbol, page_num)

                    if not page_data:
                        consecutive_failures += 1
                        if consecutive_failures >= 2:  # Stop after 2 consecutive empty pages
                            break
                        page_num += 1
                        continue

                    consecutive_failures = 0  # Reset counter on success

                    # Filter for purchases by matching reporters
                    for row in page_data:
                        rep_name = row.get("rep_name", "").strip()
                        tran_code = row.get("tran_code", "").strip().upper()

                        # Check if this is a purchase by one of our reporters
                        if rep_name in reporters and "PURCHASE" in tran_code:
                            all_purchases.append(row)

                    page_num += 1
                    await asyncio.sleep(0.5)  # Be respectful to the server

            except Exception as e:
                print(f"  Error processing {symbol}: {str(e)[:100]}")
            finally:
                await browser.close()

        return all_purchases

    def parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string and return datetime object"""
        try:
            # Try common date formats
            for fmt in ["%m/%d/%Y", "%m/%d/%y", "%Y-%m-%d", "%d-%b-%Y", "%d %b %Y"]:
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None
        except:
            return None

    async def fetch_polygon_data_with_retry(self, symbol: str, start_date: str, end_date: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Fetch price data from Polygon.io API with retry logic and exponential backoff
        """
        if not self.polygon_api_key:
            print(f"  Warning: No Polygon API key provided for {symbol}")
            return None

        url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
        params = {
            "adjusted": "true",
            "sort": "asc",
            "apiKey": self.polygon_api_key
        }

        for attempt in range(max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, params=params, timeout=30) as response:
                        
                        if response.status == 200:
                            data = await response.json()
                            print(f"  Fetched {data.get('resultsCount', 0)} price points for {symbol}")
                            return data
                        
                        elif response.status == 429:
                            # Rate limited - implement exponential backoff
                            wait_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                            print(f"  Rate limited (429) for {symbol}. Attempt {attempt + 1}/{max_retries}. Waiting {wait_time:.2f} seconds...")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        else:
                            print(f"  API error for {symbol}: Status {response.status}")
                            if attempt < max_retries - 1:
                                wait_time = 1 + random.uniform(0.1, 0.5)
                                await asyncio.sleep(wait_time)
                            else:
                                return None

            except asyncio.TimeoutError:
                print(f"  Timeout for {symbol} on attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(wait_time)
                continue
                
            except Exception as e:
                print(f"  Error fetching data for {symbol} on attempt {attempt + 1}: {str(e)[:100]}")
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                    await asyncio.sleep(wait_time)
                else:
                    return None

        print(f"  Failed to fetch data for {symbol} after {max_retries} attempts")
        return None

    def calculate_price_changes(self, purchase_date: datetime, price_data: Dict) -> Dict[str, Optional[float]]:
        """Calculate percentage price changes at 3 days, 8 days, 15 days, and 1 month, 3 months"""
        if not price_data or "results" not in price_data:
            return {"3_days": None, "8_days": None, "15_days": None, "1_month": None, "3_months": None}

        results = price_data["results"]

        # Convert results to a dict keyed by date
        price_by_date = {}
        for bar in results:
            # Polygon timestamps are in milliseconds
            bar_date = datetime.fromtimestamp(bar["t"] / 1000)
            price_by_date[bar_date.date()] = bar["c"]  # closing price

        purchase_date_obj = purchase_date.date()

        # Find the purchase price (or closest available date)
        purchase_price = None
        for i in range(10):  # Look up to 10 days forward
            check_date = purchase_date_obj + timedelta(days=i)
            if check_date in price_by_date:
                purchase_price = price_by_date[check_date]
                break

        if not purchase_price:
            return {"3_days": None, "8_days": None, "15_days": None, "1_month": None, "3_months": None}

        # Calculate changes
        changes = {}

        # 3 days after
        target_3 = purchase_date_obj + timedelta(days=3)
        price_3 = self._find_nearest_price(target_3, price_by_date, days_tolerance=2)
        changes["3_days"] = ((price_3 - purchase_price) / purchase_price * 100) if price_3 else None

        # 8 days after
        target_8 = purchase_date_obj + timedelta(days=8)
        price_8 = self._find_nearest_price(target_8, price_by_date, days_tolerance=5)
        changes["8_days"] = ((price_8 - purchase_price) / purchase_price * 100) if price_8 else None
        
        # 15 days after
        target_15 = purchase_date_obj + timedelta(days=15)
        price_15 = self._find_nearest_price(target_15, price_by_date, days_tolerance=5)
        changes["15_days"] = ((price_15 - purchase_price) / purchase_price * 100) if price_15 else None
        
        # 1 month after
        target_30 = purchase_date_obj + timedelta(days=30)
        price_30 = self._find_nearest_price(target_30, price_by_date, days_tolerance=5)
        changes["1_month"] = ((price_30 - purchase_price) / purchase_price * 100) if price_30 else None

        # 3 months after
        target_90 = purchase_date_obj + timedelta(days=90)
        price_90 = self._find_nearest_price(target_90, price_by_date, days_tolerance=10)
        changes["3_months"] = ((price_90 - purchase_price) / purchase_price * 100) if price_90 else None
        
        return changes
    
    def _find_nearest_price(self, target_date, price_by_date: Dict, days_tolerance: int = 5) -> Optional[float]:
        """Find the nearest available price within tolerance"""
        for i in range(days_tolerance + 1):
            # Check forward
            forward_date = target_date + timedelta(days=i)
            if forward_date in price_by_date:
                return price_by_date[forward_date]
            
            # Check backward
            if i > 0:
                backward_date = target_date - timedelta(days=i)
                if backward_date in price_by_date:
                    return price_by_date[backward_date]
        
        return None
    
    def calculate_average_performance(self, analysis_list: List[Dict]) -> Dict:
        """
        Calculates the average performance (% change) for a list of analysis results.
        "Total Avg" is the average of all individual time window averages.
        """
        if not analysis_list:
            return {}

        time_windows = ["3_days", "8_days", "15_days", "1_month", "3_months"]
        sums = {k: 0 for k in time_windows}
        counts = {k: 0 for k in time_windows}
        
        for analysis in analysis_list:
            raw_changes = analysis.get("_raw_changes", {})
            
            for window in time_windows:
                change = raw_changes.get(window)
                if change is not None:
                    sums[window] += change
                    counts[window] += 1
        
        averages = {}
        window_averages = []
        
        for window in time_windows:
            if counts[window] > 0:
                avg = sums[window] / counts[window]
                averages[f"avg_{window}"] = avg
                window_averages.append(avg)
            else:
                averages[f"avg_{window}"] = None
        
        if window_averages:
            averages["avg_total"] = sum(window_averages) / len(window_averages)
        else:
            averages["avg_total"] = None
        
        return averages

    async def analyze_all_symbols(self, symbol_reporters: Dict[str, Set[str]], max_pages_per_symbol: int = 10) -> Tuple[Dict, List[Dict]]:
        print("\n" + "="*60)
        print("ANALYZING PURCHASES FOR ALL QUALIFYING SYMBOLS")
        print("="*60)
        
        final_json_data = {}
        global_analysis_results = []
        
        current_date = datetime.now()
        one_month_ago = current_date - timedelta(days=30)
        recent_cutoff_date = current_date - timedelta(days=60)
        
        symbol_count = len(symbol_reporters)
        
        for i, symbol in enumerate(sorted(symbol_reporters.keys())):
            reporters = symbol_reporters[symbol]
            purchases = await self.find_purchases_by_reporters(symbol, reporters, max_pages_per_symbol)
            
            if not purchases:
                print(f"\nNo purchases found for {symbol}")
                continue

            print(f"\nFound {len(purchases)} total purchases for {symbol}")
            
            # --- Parse all dates and filter ---
            all_purchases_with_dates = []
            for p in purchases:
                parsed_date = self.parse_date(p.get("t_date"))
                if parsed_date:
                    all_purchases_with_dates.append({**p, "parsed_date": parsed_date})
            
            # Sort by date, most recent first
            all_purchases_with_dates.sort(key=lambda x: x["parsed_date"], reverse=True)
            
            # Filter for recent purchases (for display)
            recent_purchases = [
                p for p in all_purchases_with_dates 
                if p["parsed_date"] >= recent_cutoff_date
            ]
            
            # Filter for analysis purchases (older than 1 month)
            purchases_for_analysis = [
                p for p in all_purchases_with_dates 
                if p["parsed_date"] < one_month_ago
            ]
            
            # Initialize the data structure for this symbol
            final_json_data[symbol] = {
                "recent_purchases": recent_purchases,
                "price_performance_analysis": [],
                "average_performance": {},
                "purchaser_performance_summary": {},
                "polygon_api_response": None
            }
            
            if not purchases_for_analysis:
                print(f"  No purchases older than 1 month for {symbol} to analyze.")
                continue
            
            # --- Perform Analysis ---
            print(f"  {len(purchases_for_analysis)} purchases older than 1 month for analysis")
            
            # --- NEW: Group by reporter *before* clustering ---
            purchases_by_reporter = defaultdict(list)
            for p in purchases_for_analysis:
                purchases_by_reporter[p['rep_name']].append(p)
            
            print(f"  Grouped into {len(purchases_by_reporter)} unique purchasers for analysis.")

            earliest_date = min(p["parsed_date"] for p in purchases_for_analysis)
            one_year_ago = current_date - timedelta(days=365)
            start_date = min(earliest_date, one_year_ago)
            
            start_date_str = start_date.strftime("%Y-%m-%d")
            end_date_str = current_date.strftime("%Y-%m-%d")
            
            print(f"  Fetching price data from {start_date_str} to {end_date_str}")
            price_data = await self.fetch_polygon_data_with_retry(symbol, start_date_str, end_date_str, max_retries=3)
            
            # Store API response
            final_json_data[symbol]["polygon_api_response"] = price_data
            
            if price_data:
                symbol_analysis_results = []
                purchaser_performance_summary = {}
                
                # Loop through each reporter
                for rep_name, rep_purchases in purchases_by_reporter.items():
                    # Now cluster by date, *only for this reporter*
                    clustered_purchases = self.cluster_purchases_by_date(rep_purchases)
                    print(f"    Analyzing {len(clustered_purchases)} purchase cluster(s) for {rep_name}")
                    
                    reporter_analysis_list = [] # To store results for this reporter

                    for j, cluster in enumerate(clustered_purchases):
                        if len(cluster) > 1:
                            print(f"    Cluster {j+1}: {len(cluster)} purchases within 10 days by {rep_name}")
                        
                        middle_idx = len(cluster) // 2
                        selected_purchase = cluster[middle_idx]
                        
                        cluster_info = f" (cluster {j+1})" if len(clustered_purchases) > 1 else ""
                        print(f"    Analyzing purchase{cluster_info}: {selected_purchase['rep_name']} on {selected_purchase['t_date']}")
                        
                        changes = self.calculate_price_changes(selected_purchase["parsed_date"], price_data)
                        
                        change_3 = f"{changes['3_days']:.2f}%" if changes['3_days'] is not None else "N/A"
                        change_8 = f"{changes['8_days']:.2f}%" if changes['8_days'] is not None else "N/A"
                        change_15 = f"{changes['15_days']:.2f}%" if changes['15_days'] is not None else "N/A"
                        change_30 = f"{changes['1_month']:.2f}%" if changes['1_month'] is not None else "N/A"
                        change_90 = f"{changes['3_months']:.2f}%" if changes['3_months'] is not None else "N/A"
                        
                        result = {
                            "Purchaser": selected_purchase["rep_name"],
                            "Symbol": symbol,
                            "Purchase Date": selected_purchase["t_date"],
                            "Cluster Size": len(cluster),
                            "+3 Days": change_3,
                            "+8 Days": change_8,
                            "+15 Days": change_15,
                            "+1Month": change_30,
                            "+3 Months": change_90,
                            "_raw_changes": changes  # Store raw data for avg calculation
                        }
                        
                        reporter_analysis_list.append(result) # For this reporter
                        symbol_analysis_results.append(result) # For the whole symbol
                        global_analysis_results.append(result) # For the global summary
                        
                        print(f"    Price changes: 3d={change_3}, 8d={change_8}, 15d={change_15}, 1m={change_30}, 3m={change_90}")

                    # --- NEW: Calculate performance for this specific purchaser ---
                    reporter_avg_perf = self.calculate_average_performance(reporter_analysis_list)
                    purchaser_performance_summary[rep_name] = reporter_avg_perf

                # Store symbol-specific analysis
                final_json_data[symbol]["price_performance_analysis"] = symbol_analysis_results
                final_json_data[symbol]["purchaser_performance_summary"] = purchaser_performance_summary
                
                # Calculate and store average performance for this *entire symbol*
                symbol_avg_perf = self.calculate_average_performance(symbol_analysis_results)
                final_json_data[symbol]["average_performance"] = symbol_avg_perf
                
            # Rate limiting between symbols
            if i < symbol_count - 1:
                delay = 1.5 + (i * 0.1)
                print(f"  Waiting {delay:.2f} seconds before next symbol...")
                await asyncio.sleep(delay)
        
        return final_json_data, global_analysis_results

    def cluster_purchases_by_date(self, purchases_with_dates: List[Dict], days_threshold: int = 10) -> List[List[Dict]]:
        """
        Group purchases that are within days_threshold of each other.
        Note: Assumes input is already filtered for a single reporter.
        """
        if not purchases_with_dates:
            return []
        
        # Sort by date, ascending (oldest first) for clustering
        sorted_purchases = sorted(purchases_with_dates, key=lambda x: x["parsed_date"])
        
        clusters = []
        current_cluster = [sorted_purchases[0]]
        
        for i in range(1, len(sorted_purchases)):
            current_date = sorted_purchases[i]["parsed_date"]
            # Use the start of the current cluster for comparison
            cluster_start_date = current_cluster[0]["parsed_date"]
            
            days_diff = (current_date - cluster_start_date).days
            
            if days_diff <= days_threshold:
                current_cluster.append(sorted_purchases[i])
            else:
                clusters.append(current_cluster)
                current_cluster = [sorted_purchases[i]]
        
        clusters.append(current_cluster)
        
        return clusters

    def save_to_json(self, data, filename: str = "insider_trades.json"):
        """Save data to JSON file"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"\nSaved data to {filename}")
    
    def to_dataframe(self, data: List[Dict]) -> pd.DataFrame:
        """Convert data to Pandas DataFrame"""
        return pd.DataFrame(data)

    def _format_perf(self, val: Optional[float]) -> str:
        """Helper to format performance percentages for display"""
        return f"{val:.2f}%" if val is not None else "N/A"

    def format_symbol_output(
        self, 
        symbol: str, 
        recent_purchases: List[Dict], 
        analysis_summary: List[Dict], 
        average_performance: Dict,
        purchaser_performance_summary: Dict
    ) -> str:
        """Format output for a single symbol with all analysis"""
        output = []
        
        avg_total = average_performance.get('avg_total')
        avg_str = f" (Avg Total Return: {self._format_perf(avg_total)})" if avg_total is not None else ""
        
        # Symbol header
        output.append(f"\n{'='*80}")
        output.append(f"SYMBOL: {symbol}{avg_str}")
        output.append(f"{'='*80}")
        
        # Recent Purchases section
        output.append(f"\n1. RECENT PURCHASES (Last 60 Days)")
        output.append(f"{'-'*40}")
        
        if recent_purchases:
            # Take first 5 recent purchases for display (already sorted)
            recent_purchases_top5 = recent_purchases[:5]
            
            output.append(f"{'Date':<12} {'Amount':<15} {'Purchaser':<25} {'Relationship':<15}")
            output.append(f"{'-'*12} {'-'*15} {'-'*25} {'-'*15}")
            
            for purchase in recent_purchases_top5:
                date = purchase.get('t_date', 'N/A')
                amount = purchase.get('amt', 'N/A')
                purchaser = purchase.get('rep_name', 'N/A')[:23]  # Truncate
                relationship = purchase.get('rel', 'N/A')[:13]  # Truncate
                
                output.append(f"{date:<12} {amount:<15} {purchaser:<25} {relationship:<15}")
        else:
            output.append("No purchases found in the last 60 days.")
        
        # Price Performance Analysis Summary section
        output.append(f"\n2. PRICE PERFORMANCE ANALYSIS (Purchases > 1 Month Old)")
        output.append(f"{'-'*40}")
        
        if analysis_summary:
            output.append(f"{'Purchaser':<25} {'Purchase Date':<12} {'+3 Days':<10} {'+8 Days':<10} {'+15 Days':<10} {'+1 Month':<10} {'+3 Months':<10}")
            output.append(f"{'-'*25} {'-'*12} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
            
            for analysis in analysis_summary:
                purchaser = analysis['Purchaser'][:23]  # Truncate
                purchase_date = analysis['Purchase Date']
                change_3 = analysis['+3 Days']
                change_8 = analysis['+8 Days']
                change_15 = analysis['+15 Days']
                change_30 = analysis['+1Month']
                change_90 = analysis['+3 Months']
                
                output.append(f"{purchaser:<25} {purchase_date:<12} {change_3:<10} {change_8:<10} {change_15:<10} {change_30:<10} {change_90:<10}")
        else:
            output.append("No purchases older than 1 month to analyze.")
        
        # Symbol Average Performance section
        output.append(f"\n3. SYMBOL AVERAGE PERFORMANCE (Based on Analysis)")
        output.append(f"{'-'*40}")
        
        if average_performance:
            output.append(f"  Total Avg Return:   {self._format_perf(average_performance.get('avg_total'))}")
            output.append(f"  +3 Days Avg:        {self._format_perf(average_performance.get('avg_3_days'))}")
            output.append(f"  +8 Days Avg:        {self._format_perf(average_performance.get('avg_8_days'))}")
            output.append(f"  +15 Days Avg:       {self._format_perf(average_performance.get('avg_15_days'))}")
            output.append(f"  +1 Month Avg:       {self._format_perf(average_performance.get('avg_1_month'))}")
            output.append(f"  +3 Months Avg:      {self._format_perf(average_performance.get('avg_3_months'))}")
        else:
            output.append("No analysis data to calculate average performance.")

        # --- NEW: Purchaser Performance Summary section ---
        output.append(f"\n4. PURCHASER PERFORMANCE SUMMARY (Based on Analysis)")
        output.append(f"{'-'*40}")

        if purchaser_performance_summary:
            # Create table header
            output.append(f"{'Purchaser':<25} {'Total Avg':<10} {'+3d Avg':<10} {'+8d Avg':<10} {'+15d Avg':<10} {'+1m Avg':<10} {'+3m Avg':<10}")
            output.append(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

            for rep_name, perf in purchaser_performance_summary.items():
                purchaser = rep_name[:23] # Truncate
                total_avg = self._format_perf(perf.get('avg_total'))
                avg_3d = self._format_perf(perf.get('avg_3_days'))
                avg_8d = self._format_perf(perf.get('avg_8_days'))
                avg_15d = self._format_perf(perf.get('avg_15_days'))
                avg_1m = self._format_perf(perf.get('avg_1_month'))
                avg_3m = self._format_perf(perf.get('avg_3_months'))

                output.append(f"{purchaser:<25} {total_avg:<10} {avg_3d:<10} {avg_8d:<10} {avg_15d:<10} {avg_1m:<10} {avg_3m:<10}")
        else:
            output.append("No analysis data to calculate purchaser performance.")

        return "\n".join(output)

async def main():
    """Main execution function"""

    scraper = DataromaScraper()
    
    print("Starting scraper...")
    print("=" * 60)
    
    # Scrape all pages (set max_pages to limit, or None for all)
    all_data = await scraper.scrape_all_pages(max_pages=5)  # Adjust as needed
    
    print(f"\nTotal rows scraped: {len(all_data)}")
    
    # Filter for symbols with multiple reporters
    print("\n" + "=" * 60)
    print("Filtering for symbols with multiple reporters...")
    filtered_data, symbol_reporters = scraper.filter_multiple_reporters(all_data)
    
    print(f"\nFiltered rows: {len(filtered_data)}")
    
    # Now analyze each symbol for purchases and price performance
    if symbol_reporters:
        final_json_data, global_analysis_results = await scraper.analyze_all_symbols(
            symbol_reporters, 
            max_pages_per_symbol=10
        )
        
        # Save the new single JSON file
        if final_json_data:
            scraper.save_to_json(final_json_data, "dataroma_analysis_output.json")
        
        # Display separate output for each symbol
        print("\n" + "="*80)
        print("INDIVIDUAL SYMBOL ANALYSIS")
        print("="*80)
        
        # Open output.txt for writing (overwrite each run)
        with open("output.txt", "w", encoding="utf-8") as f:
            # Helper that prints to both console and file
            def dual_print(*args, **kwargs):
                print(*args, **kwargs)
                print(*args, **kwargs, file=f)

            # Start writing the report
            dual_print("\n" + "="*80)
            dual_print("INDIVIDUAL SYMBOL ANALYSIS")
            dual_print("="*80)

            for symbol in sorted(final_json_data.keys()):
                symbol_data = final_json_data[symbol]
                symbol_output = scraper.format_symbol_output(
                    symbol,
                    symbol_data["recent_purchases"],
                    symbol_data["price_performance_analysis"],
                    symbol_data["average_performance"],
                    symbol_data["purchaser_performance_summary"]
                )
                dual_print(symbol_output)

            # Also display the summary table for completeness
            if global_analysis_results:
                dual_print("\n" + "="*80)
                dual_print("OVERALL PRICE PERFORMANCE ANALYSIS SUMMARY (ALL SYMBOLS)")
                dual_print("="*80)
                
                display_analysis = [
                    {k: v for k, v in r.items() if k != "_raw_changes"} 
                    for r in global_analysis_results
                ]
                df = pd.DataFrame(display_analysis)
                dual_print("\n" + df.to_string(index=False))
                
                overall_avg_perf = scraper.calculate_average_performance(global_analysis_results)
                dual_print("\n" + "="*80)
                dual_print("OVERALL AVERAGE PERFORMANCE (ALL SYMBOLS)")
                dual_print("="*80)
                
                dual_print(f"  Total Avg Return:   {scraper._format_perf(overall_avg_perf.get('avg_total'))}")
                dual_print(f"  +3 Days Avg:        {scraper._format_perf(overall_avg_perf.get('avg_3_days'))}")
                dual_print(f"  +8 Days Avg:        {scraper._format_perf(overall_avg_perf.get('avg_8_days'))}")
                dual_print(f"  +15 Days Avg:       {scraper._format_perf(overall_avg_perf.get('avg_15_days'))}")
                dual_print(f"  +1 Month Avg:       {scraper._format_perf(overall_avg_perf.get('avg_1_month'))}")
                dual_print(f"  +3 Months Avg:      {scraper._format_perf(overall_avg_perf.get('avg_3_months'))}")
                
                dual_print(f"\n✓ Analyzed {len(global_analysis_results)} insider purchase clusters")
        # === END FILE + CONSOLE OUTPUT SECTION ===
    
    print("\n" + "=" * 60)
    print("Scraping complete!")

if __name__ == "__main__":
    asyncio.run(main())
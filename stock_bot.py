import discord
from discord.ext import commands
import yfinance as yf
import pandas as pd
import numpy as np
from openai import OpenAI
import os
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import socket
from duckduckgo_search import DDGS
from scipy.stats import norm
import datetime
import io
import re
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

# ================= 配置区域 =================
# 建议使用环境变量，或者直接在此处填入 Key
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')


# 配置 DeepSeek AI
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL_ID = 'deepseek-reasoner'

# 配置 Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ================= 健康检查模块 (用于部署) =================

class HealthCheckHandler(BaseHTTPRequestHandler):
    """A simple handler for the health check server."""
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"OK")

def run_health_check_server():
    """Runs a simple HTTP server for health checks in a background thread."""
    port = int(os.getenv('PORT', 8000)) # Koyeb provides the port to listen on via the PORT env var
    server_address = ('', port)
    httpd = HTTPServer(server_address, HealthCheckHandler)
    print(f"✅ Health check server running on port {port}...")
    httpd.serve_forever()

# ================= 核心逻辑模块 =================

class StockAnalyzer:
    @staticmethod
    def get_data(ticker_symbol):
        """获取历史数据和更全面的基本面信息"""
        try:
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="1y")
            
            if df.empty:
                return None, None, None

            info = stock.info
            fundamentals = {
                "name": info.get('longName', ticker_symbol),
                "sector": info.get('sector', 'Unknown'),
                "price": info.get('currentPrice', df['Close'].iloc[-1]),
                "currency": info.get('currency', 'USD'),
                "market_cap": info.get('marketCap', 'N/A'),
                "pe": info.get('trailingPE', 'N/A'),
                "pb": info.get('priceToBook', 'N/A'),
                "eps": info.get('trailingEps', 'N/A'),
                "roe": info.get('returnOnEquity', 'N/A'),
                "debt_to_equity": info.get('debtToEquity', 'N/A'),
                "forward_pe": info.get('forwardPE', 'N/A'),
                "beta": info.get('beta', 'N/A'),
                "peg_ratio": info.get('pegRatio', 'N/A'),
                "profit_margins": info.get('profitMargins', 'N/A'),
                "short_percent": info.get('shortPercentOfFloat', 'N/A'),
                "business_summary": info.get('longBusinessSummary', '暂无详细业务描述'),
            }

            # === 新增: 财务报表数据 (10-Q/10-K) ===
            financials_data = {}
            try:
                # 获取季度报表
                q_income = stock.quarterly_financials
                q_balance = stock.quarterly_balance_sheet
                q_cashflow = stock.quarterly_cashflow

                if not q_income.empty:
                    latest_q = q_income.iloc[:, 0] # 最近一个季度
                    financials_data['date'] = str(latest_q.name).split(' ')[0]
                    financials_data['revenue'] = latest_q.get('Total Revenue', 'N/A')
                    financials_data['net_income'] = latest_q.get('Net Income', 'N/A')
                    financials_data['gross_profit'] = latest_q.get('Gross Profit', 'N/A')
                
                if not q_balance.empty:
                    latest_b = q_balance.iloc[:, 0]
                    financials_data['total_cash'] = latest_b.get('Cash And Cash Equivalents', 'N/A')
                    financials_data['total_debt'] = latest_b.get('Total Debt', 'N/A')
                
                if not q_cashflow.empty:
                    latest_c = q_cashflow.iloc[:, 0]
                    financials_data['op_cashflow'] = latest_c.get('Operating Cash Flow', 'N/A')
            except Exception as e:
                print(f"Financials Error: {e}")
            
            fundamentals['financials'] = financials_data

            # === 新增: 分析师数据 ===
            analyst_data = {
                'target_mean': info.get('targetMeanPrice', 'N/A'),
                'target_high': info.get('targetHighPrice', 'N/A'),
                'target_low': info.get('targetLowPrice', 'N/A'),
                'recommendation': info.get('recommendationKey', 'N/A'),
                'num_analysts': info.get('numberOfAnalystOpinions', 'N/A'),
                'recent_ratings': []
            }
            try:
                upgrades = stock.upgrades_downgrades
                if upgrades is not None and not upgrades.empty:
                    latest_upgrades = upgrades.sort_index(ascending=False).head(3)
                    for index, row in latest_upgrades.iterrows():
                        analyst_data['recent_ratings'].append(f"{str(index).split(' ')[0]}: {row['Firm']} -> {row['ToGrade']}")
            except Exception: pass
            fundamentals['analyst'] = analyst_data

            # === 新增: 关键事件日历 (Earnings & Events) ===
            try:
                cal = stock.calendar
                # yfinance calendar 可能是 dict 或 DataFrame
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    dates = cal['Earnings Date']
                    if dates:
                        next_date = dates[0] # 通常是最近的一个
                        fundamentals['next_earnings'] = str(next_date)
                        # 计算天数
                        today = datetime.date.today()
                        if isinstance(next_date, datetime.datetime):
                            next_date = next_date.date()
                        fundamentals['days_to_earnings'] = (next_date - today).days
                else:
                    fundamentals['next_earnings'] = 'N/A'
                    fundamentals['days_to_earnings'] = 'N/A'
            except Exception:
                fundamentals['next_earnings'] = 'N/A'
                fundamentals['days_to_earnings'] = 'N/A'

            # === 获取期权数据 (Put/Call Ratio) ===
            try:
                exps = stock.options
                if exps:
                    # 获取最近的一个到期日
                    nearest_exp = exps[0]
                    opt = stock.option_chain(nearest_exp)
                    
                    # 计算总成交量和持仓量
                    c_vol = opt.calls['volume'].sum() if not opt.calls.empty else 0
                    p_vol = opt.puts['volume'].sum() if not opt.puts.empty else 0
                    c_oi = opt.calls['openInterest'].sum() if not opt.calls.empty else 0
                    p_oi = opt.puts['openInterest'].sum() if not opt.puts.empty else 0

                    fundamentals['pc_ratio_vol'] = round(p_vol / c_vol, 2) if c_vol > 0 else 'N/A'
                    fundamentals['pc_ratio_oi'] = round(p_oi / c_oi, 2) if c_oi > 0 else 'N/A'
                    fundamentals['options_expiry'] = nearest_exp
                else:
                    raise ValueError("No options")
            except Exception:
                fundamentals['pc_ratio_vol'] = 'N/A'
                fundamentals['pc_ratio_oi'] = 'N/A'
                fundamentals['options_expiry'] = 'N/A'
            
            news = stock.news
            return df, fundamentals, news
        except Exception as e:
            print(f"Data Error: {e}")
            return None, None, None

    @staticmethod
    def calculate_indicators(df):
        """计算更多技术和量化指标"""
        df = df.copy()
        
        # 1. 移动平均线 (SMA)
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # 2. RSI (相对强弱指数)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # 3. 布林带
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (2 * df['BB_Std'])
        df['BB_Lower'] = df['BB_Middle'] - (2 * df['BB_Std'])
        
        # 4. MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 5. 波动率 (30日历史波动率)
        df['Log_Ret'] = df['Close'].apply(lambda x: np.log(x)).diff()
        df['Volatility'] = df['Log_Ret'].rolling(window=30).std() * np.sqrt(252) # 年化

        return df

    @staticmethod
    def get_web_search(ticker):
        """使用 DuckDuckGo 搜索最新的市场新闻、事件、管理层指引以及社交媒体情绪"""
        results = []
        try:
            with DDGS() as ddgs:
                # 1. 核心催化剂与未来事件 (Event-Driven Focus)
                query_event = f"{ticker} stock upcoming catalyst events earnings date fda approval product launch"
                results.extend(list(ddgs.text(query_event, max_results=3)))

                # 2. 隐含波动率与期权异动 (Market Pricing of Events)
                query_iv = f"{ticker} stock implied volatility rank option flow unusual activity"
                results.extend(list(ddgs.text(query_iv, max_results=2)))
                
                # 3. 10-Q/10-K 管理层指引
                query_guidance = f"{ticker} stock earnings guidance management discussion 10-Q highlights"
                results.extend(list(ddgs.text(query_guidance, max_results=2)))
                
                # 4. X (Twitter) 交易员情绪
                query_social = f"site:twitter.com OR site:x.com {ticker} stock analysis sentiment discussion"
                social_results = list(ddgs.text(query_social, max_results=2))
                for r in social_results:
                    r['title'] = f"[X/Twitter] {r['title']}"
                results.extend(social_results)

                return results
        except Exception as e:
            print(f"Web Search Error: {e}")
            return results

    @staticmethod
    def black_scholes_gamma(S, K, T, r, sigma):
        """计算 Black-Scholes Gamma"""
        try:
            if T <= 0 or sigma <= 0:
                return 0
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            return gamma
        except:
            return 0

    @staticmethod
    def get_gamma_exposure(stock, current_price):
        """计算 Gamma Exposure (GEX) 和关键挤压位置"""
        try:
            exps = stock.options
            if not exps:
                return None
            
            # 使用最近的到期日 (Gamma 风险最大)
            expiry_date_str = exps[0]
            expiry_date = datetime.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
            today = datetime.date.today()
            T = (expiry_date - today).days / 365.0
            if T <= 1e-5: T = 1/365.0 # 防止除以零

            opt = stock.option_chain(expiry_date_str)
            calls = opt.calls.copy()
            puts = opt.puts.copy()
            
            r = 0.045 # 假设无风险利率 4.5%
            
            # 计算 Gamma
            calls['gamma'] = calls.apply(lambda x: StockAnalyzer.black_scholes_gamma(current_price, x['strike'], T, r, x['impliedVolatility']), axis=1)
            puts['gamma'] = puts.apply(lambda x: StockAnalyzer.black_scholes_gamma(current_price, x['strike'], T, r, x['impliedVolatility']), axis=1)

            # 计算 GEX (名义价值) = Gamma * OI * 100 * Price
            # Call GEX 通常视为正向 (Dealer Short Call -> Long Stock to hedge)
            # Put GEX 通常视为负向 (Dealer Short Put -> Short Stock to hedge)
            calls['gex'] = calls['gamma'] * calls['openInterest'] * 100 * current_price
            puts['gex'] = puts['gamma'] * puts['openInterest'] * 100 * current_price * -1

            # 寻找关键墙 (Walls)
            call_wall = calls.loc[calls['gex'].idxmax()]['strike'] if not calls.empty else 0
            put_wall = puts.loc[puts['gex'].abs().idxmax()]['strike'] if not puts.empty else 0
            net_gex = calls['gex'].sum() + puts['gex'].sum()

            return {
                "expiry": expiry_date_str,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "net_gex": net_gex
            }
        except Exception as e:
            print(f"GEX Error: {e}")
            return None

    @staticmethod
    def get_option_flow(stock, current_price):
        """分析期权资金流，寻找异常大单和聪明钱布局 (Volume > Open Interest)"""
        try:
            exps = stock.options
            if not exps:
                return []
            
            flow_data = []
            today = datetime.date.today()
            cutoff_date = today + datetime.timedelta(days=180)

            # 扫描未来半年内的到期日
            for date in exps:
                try:
                    if datetime.datetime.strptime(date, "%Y-%m-%d").date() > cutoff_date:
                        continue
                    opt = stock.option_chain(date)
                    
                    # 筛选逻辑: 成交量 > 500 且 成交量 > 持仓量 * 1.1 (疑似主力主动开仓)
                    # Calls
                    calls = opt.calls
                    if not calls.empty:
                        active_calls = calls[
                            (calls['volume'] > 500) & 
                            (calls['volume'] > calls['openInterest'] * 1.1)
                        ].copy()
                        for _, row in active_calls.iterrows():
                            flow_data.append({
                                'type': 'CALL 🐂',
                                'expiry': date,
                                'strike': row['strike'],
                                'volume': int(row['volume']),
                                'oi': int(row['openInterest']),
                                'ratio': round(row['volume'] / (row['openInterest'] if row['openInterest'] > 0 else 1), 1)
                            })

                    # Puts
                    puts = opt.puts
                    if not puts.empty:
                        active_puts = puts[
                            (puts['volume'] > 500) & 
                            (puts['volume'] > puts['openInterest'] * 1.1)
                        ].copy()
                        for _, row in active_puts.iterrows():
                            flow_data.append({
                                'type': 'PUT 🐻',
                                'expiry': date,
                                'strike': row['strike'],
                                'volume': int(row['volume']),
                                'oi': int(row['openInterest']),
                                'ratio': round(row['volume'] / (row['openInterest'] if row['openInterest'] > 0 else 1), 1)
                            })
                except Exception: continue
            
            # 按成交量降序排序，取前 5 大异动
            flow_data.sort(key=lambda x: x['volume'], reverse=True)
            return flow_data[:5]
        except Exception as e:
            print(f"Flow Error: {e}")
            return []

    @staticmethod
    def create_pdf_report(ticker, report_text, fund_data):
        """生成 PDF 报告"""
        try:
            buffer = io.BytesIO()
            # 调整页边距，增加内容容纳空间
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=50, leftMargin=50, topMargin=50, bottomMargin=50)
            styles = getSampleStyleSheet()
            
            # 注册中文字体 (STSong-Light 是 Adobe 预定义的简体中文字体，无需额外字体文件)
            pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
            
            # === 自定义样式优化 ===
            title_style = ParagraphStyle(
                'CustomTitle', parent=styles['Title'], fontName='STSong-Light', fontSize=22, leading=26, spaceAfter=20, alignment=1, textColor=colors.HexColor("#1a73e8")
            )
            heading_style = ParagraphStyle(
                'CustomHeading', parent=styles['Heading2'], fontName='STSong-Light', fontSize=14, leading=18, spaceBefore=12, spaceAfter=8, textColor=colors.HexColor("#202124")
            )
            normal_style = ParagraphStyle(
                'CustomNormal', parent=styles['Normal'], fontName='STSong-Light', fontSize=10.5, leading=15, spaceAfter=6, textColor=colors.HexColor("#3c4043")
            )
            bullet_style = ParagraphStyle(
                'CustomBullet', parent=normal_style, leftIndent=15, firstLineIndent=0, spaceAfter=4
            )
            
            story = []
            
            # 1. 报告标题
            story.append(Paragraph(f"{ticker} 深度投资分析报告", title_style))
            
            # 2. 核心数据表格 (比纯文本更美观)
            def fmt_num(n):
                if isinstance(n, (int, float)):
                    if n > 1e12: return f"{n/1e12:.2f}T"
                    if n > 1e9: return f"{n/1e9:.2f}B"
                    return f"{n:,.2f}"
                return str(n)

            data = [
                ['标的名称', f"{fund_data.get('name', ticker)}", '当前价格', f"{fund_data.get('price', 'N/A')} {fund_data.get('currency', '')}"],
                ['所属行业', fund_data.get('sector', 'Unknown'), '生成日期', datetime.datetime.now().strftime("%Y-%m-%d")],
                ['P/E (TTM)', str(fund_data.get('pe', 'N/A')), 'P/B', str(fund_data.get('pb', 'N/A'))],
                ['ROE', str(fund_data.get('roe', 'N/A')), '市值', fmt_num(fund_data.get('market_cap', 'N/A'))]
            ]
            
            t = Table(data, colWidths=[70, 180, 70, 120])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'STSong-Light'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor("#3c4043")),
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#f1f3f4")), # 标签列背景色
                ('BACKGROUND', (2, 0), (2, -1), colors.HexColor("#f1f3f4")),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")), # 网格线
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))
            
            # 3. 解析 Markdown 文本并转换为 PDF 元素
            lines = report_text.split('\n')
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # 简单 Markdown 转换: 加粗
                line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
                # 处理代码块标记 (移除)
                line = line.replace('```', '')
                
                if line.startswith('###'):
                    story.append(Paragraph(line.replace('###', '').strip(), heading_style))
                elif line.startswith('##'):
                    story.append(Paragraph(line.replace('##', '').strip(), heading_style))
                elif line.startswith('#'):
                    story.append(Paragraph(line.replace('#', '').strip(), title_style))
                elif line.startswith('- '):
                    # 列表项优化
                    story.append(Paragraph(f"• {line[2:]}", bullet_style))
                else:
                    story.append(Paragraph(line, normal_style))
            
            # 添加页脚
            def add_footer(canvas, doc):
                canvas.saveState()
                canvas.setFont('STSong-Light', 9)
                canvas.setFillColor(colors.grey)
                canvas.drawCentredString(letter[0]/2.0, 30, "Generated by DeepSeek AI Stock Bot | Not Financial Advice")
                canvas.restoreState()

            doc.build(story, onFirstPage=add_footer, onLaterPages=add_footer)
            buffer.seek(0)
            return buffer
        except Exception as e:
            print(f"PDF Generation Error: {e}")
            return None

    @staticmethod
    async def get_ai_analysis(ticker, fund, tech_data, news_data, web_search_data, gex_data, flow_data):
        """调用 LLM 生成更深度的自然语言报告"""
        latest = tech_data.iloc[-1]
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Safely extract news headlines, skipping items that might not have a 'title' key.
        news_headlines = "\n".join([f"- {n['title']}" for n in news_data[:5] if 'title' in n])
        
        # 格式化网络搜索结果
        web_content = "\n".join([f"- [Web] {r['title']}: {r['body']}" for r in web_search_data])

        # 格式化 GEX 数据
        gex_info = "- 暂无期权 Gamma 数据"
        if gex_data:
            gex_info = f"""- 到期日: {gex_data['expiry']}
            - Net GEX (净伽马敞口): ${gex_data['net_gex']:,.0f}
            - Call Wall (最大阻力/做市商做空点): {gex_data['call_wall']}
            - Put Wall (最大支撑/做市商回补点): {gex_data['put_wall']}"""

        # 格式化资金流数据
        flow_info = "- 暂无显著期权异动"
        if flow_data:
            flow_info = "\n".join([f"- {f['type']} | 到期: {f['expiry']} | 行权: {f['strike']} | Vol: {f['volume']} (OI: {f['oi']}, 倍数: {f['ratio']}x)" for f in flow_data])

        # 格式化分析师评级
        analyst_ratings_str = "- 暂无近期评级变动"
        if fund['analyst']['recent_ratings']:
            analyst_ratings_str = "\n".join([f"  - {r}" for r in fund['analyst']['recent_ratings']])

        # 构建更强大的提示词 (Prompt)
        prompt = f"""
            # Role
            你是一位拥有20年深厚资历的华尔街量化与宏观对冲基金首席投资官 (CIO)。你擅长将自上而下的宏观逻辑（Top-Down）与自下而上的量化因子（Bottom-Up）相结合，挖掘市场尚未完全定价的“预期差”。

            # Input Data Panel
            - **当前分析日期**: {current_date}

            ## 1. 标的基本面与质量 (Quality & Value)
            - 标的: {ticker} ({fund['name']}) | 行业: {fund['sector']}
            - 业务概览 (10-K): {fund['business_summary'][:400]}...
            - 核心估值: P/E: {fund['pe']} | Fwd P/E: {fund['forward_pe']} | PEG: {fund['peg_ratio']} | P/B: {fund['pb']}
            - 盈利质量: ROE: {fund['roe']} | 净利率: {fund['profit_margins']} | EPS: {fund['eps']}
            - 财务杠杆: 负债权益比: {fund['debt_to_equity']} | Beta: {fund['beta']}

            ## 2. 量化与技术面 (Quant & Technicals)
            - 趋势指标: 50D SMA: {latest['SMA_50']:.2f} | 200D SMA: {latest['SMA_200']:.2f}
            - 动能指标: RSI: {latest['RSI']:.2f} | MACD: {latest['MACD']:.2f} (Signal: {latest['MACD_Signal']:.2f})
            - 波动率: 30日年化波动率: {latest['Volatility']:.2%}
            - 布林带位置: Upper: {latest['BB_Upper']:.2f} | Lower: {latest['BB_Lower']:.2f} | Close: {latest['Close']:.2f}

            ## 3. 衍生品与情绪 (Derivatives & Sentiment)
            - 期权 Put/Call Ratio (Volume): {fund['pc_ratio_vol']} (基于最近到期日 {fund['options_expiry']})
            - 期权 Put/Call Ratio (Open Interest): {fund['pc_ratio_oi']}
            - 空头流通占比 (Short Float): {fund['short_percent']}
            {gex_info}
            
            ## 4. 资金流向与聪明钱 (Smart Money Flow)
            - 异常期权异动 (Unusual Whales - Vol > OI):
            {flow_info}

            ## 4. 市场催化剂、管理层指引与交易员情绪 (Catalysts, Guidance & Sentiment)
            ## 5. 市场催化剂、管理层指引与交易员情绪 (Catalysts, Guidance & Sentiment)
            - 下次财报日期: {fund.get('next_earnings', 'N/A')} (距离现在 {fund.get('days_to_earnings', 'N/A')} 天)
            - 实时网络搜索 (含未来事件、IV分析、X/Twitter讨论):
            {web_content if web_content else "- 暂无网络搜索结果"}
            - 交易所新闻 (Exchange News): 
            {news_headlines if news_headlines else "- 暂无交易所新闻"}

            ## 5. 财务报表透视 (Financials - Latest Quarter)
            ## 6. 财务报表透视 (Financials - Latest Quarter)
            - 报告日期: {fund['financials'].get('date', 'N/A')}
            - 总营收: {fund['financials'].get('revenue', 'N/A')} | 净利润: {fund['financials'].get('net_income', 'N/A')}
            - 毛利润: {fund['financials'].get('gross_profit', 'N/A')} | 经营现金流: {fund['financials'].get('op_cashflow', 'N/A')}
            - 资产负债: 现金储备 {fund['financials'].get('total_cash', 'N/A')} vs 总债务 {fund['financials'].get('total_debt', 'N/A')}

            ## 6. 华尔街分析师共识 (Analyst Consensus)
            ## 7. 华尔街分析师共识 (Analyst Consensus)
            - 综合评级: {fund['analyst']['recommendation']} (基于 {fund['analyst']['num_analysts']} 位分析师)
            - 目标价: Mean: {fund['analyst']['target_mean']} | High: {fund['analyst']['target_high']} | Low: {fund['analyst']['target_low']}
            - 近期机构评级变动:
            {analyst_ratings_str}

            # Analysis Requirements
            请基于以上数据，生成一份逻辑严密、具备实战指导意义的分析报告。要求：

            ### 1. 🏛️ 宏观叙事与行业定性
            分析当前宏观环境对该行业及公司的边际影响。判断标的处于周期的哪个阶段。

            ### 2. 📊 因子深度分析
            - **估值与预期**: 结合 P/E 和 Forward P/E，判断市场当前的预期是否过高或过低。
            - **财务健康度 (10-Q)**: 结合最新财报数据，分析营收/利润增长趋势及现金流状况。
            - **业务指引 (Guidance)**: 结合管理层在 10-Q/10-K 中的描述及最新指引，评估未来增长的可持续性。
            - **交易员情绪 (Sentiment)**: 结合 X (Twitter) 上的讨论内容，分析市场情绪（FOMO/恐慌/分歧），并判断是否与基本面出现背离。
            - **基本面质量**: 评估 ROE 和负债水平，判断公司的护城河与抗风险能力。
            - **资金流分析 (Smart Money)**: 解读期权异动数据。是否有大资金在 OTM 位置通过 Call 扫货博取反弹？或者大量 Put 正在对冲下行风险？识别主力资金的布局点位。
            - **期权博弈与 Gamma Squeeze**: 
                1. 分析 P/C Ratio 判断情绪。
                2. **重点分析 Gamma 数据**: 
                   - 如果当前价格接近 **Call Wall**，是否存在向上突破引发 Gamma Squeeze (逼空) 的可能？
                   - 如果 Net GEX 为负，说明做市商处于 Short Gamma 状态，市场波动率是否会放大？
                   - Put Wall 是否提供了有效支撑？

            ### 4. 📅 事件驱动与变盘点 (Event-Driven)
            - **关键节点**: 识别未来30-90天内的核心催化剂（财报、产品发布、监管决议）。
            - **市场定价**: 分析隐含波动率（IV）或期权异动是否暗示了即将到来的剧烈波动？
            - **博弈策略**: 针对即将到来的事件，是应该提前埋伏（Run-up），还是防范“利好出尽”（Sell the news）？

            ### 5. 📈 技术面共振
            - 分析 50D/200D 均线的排列关系（金叉/死叉）。
            - 结合 RSI 和布林带位置，判断当前是否超买或超卖。

            ### 6. 🛠️ 组合构建建议 (Portfolio Construction)
            - **投资评级**: (强力买入 / 逢低买入 / 持股观望 / 卖出)
            - **操作逻辑**: 给出基于“预期差”的核心逻辑。
            - **风控参数**: 
            - 入场区间 (Entry): [精确到价格范围]
            - 目标止盈 (TP): [基于历史波动率或压力位]
            - 硬性止损 (SL): [基于 $ATR$ 或关键支撑位]
            - 建议仓位权重: (如：2% 试验仓 / 5% 标准仓 / 8% 进攻仓)

            请使用专业、简洁、富有洞察力的语言输出。
        """
        try:
            loop = asyncio.get_running_loop()
            
            def call_deepseek():
                response = client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
                return response.choices[0].message.content

            return await loop.run_in_executor(None, call_deepseek)
        except Exception as e:
            return f"AI 分析生成失败: {str(e)}"

# ================= Discord 命令处理 =================

@bot.event
async def on_ready():
    print(f'✅ Bot 已登录: {bot.user}')
    print('DeepSeek 模式就绪。尝试输入: !a TSLA')

@bot.command(name='a', aliases=['analyze', 'stock', 'gp'])
async def analyze(ctx, ticker: str):
    """
    分析股票命令。用法: !a TSLA 或 !a 600519
    """
    ticker = ticker.upper()
    
    # === A股代码自动后缀补全 ===
    if ticker.isdigit() and len(ticker) == 6:
        if ticker.startswith('6'):
            ticker = f"{ticker}.SS" # 上海证券交易所
        elif ticker.startswith(('0', '3')):
            ticker = f"{ticker}.SZ" # 深圳证券交易所
        elif ticker.startswith(('4', '8')):
            ticker = f"{ticker}.BJ" # 北京证券交易所

    status_msg = await ctx.send(f"🔍 正在分析 **{ticker}**，请稍候...")
    
    try:
        # 1. 获取数据
        await status_msg.edit(content=f"🧠 正在获取 **{ticker}** 的基本面、新闻和历史数据...")
        df, fund, news = StockAnalyzer.get_data(ticker)
        
        if df is None:
            await status_msg.edit(content=f"❌ 找不到股票代码 **{ticker}**，请检查拼写或重试。")
            return

        # 2. 计算指标
        await status_msg.edit(content=f"📈 正在计算 **{ticker}** 的技术指标与量化信号...")
        df_tech = StockAnalyzer.calculate_indicators(df)
        
        # 3. 执行网络搜索 (在后台线程运行以防阻塞)
        loop = asyncio.get_running_loop()
        web_results = await loop.run_in_executor(None, lambda: StockAnalyzer.get_web_search(ticker))

        # 4. 初始化 Ticker 对象 (复用以提高效率)
        stock_obj = yf.Ticker(ticker)

        # 5. 计算 Gamma Exposure (GEX)
        await status_msg.edit(content=f"🧮 正在计算 **{ticker}** 的 Gamma Exposure (GEX) 与挤压风险...")
        gex_data = await loop.run_in_executor(None, lambda: StockAnalyzer.get_gamma_exposure(stock_obj, fund['price']))

        # 6. 扫描期权资金流 (Option Flow)
        await status_msg.edit(content=f"💸 正在扫描 **{ticker}** 的期权资金流与聪明钱布局...")
        flow_data = await loop.run_in_executor(None, lambda: StockAnalyzer.get_option_flow(stock_obj, fund['price']))

        # 7. 获取 AI 报告
        await status_msg.edit(content=f"🤖 DeepSeek R1 (深度思考模式) 正在生成分析报告...")
        report = await StockAnalyzer.get_ai_analysis(ticker, fund, df_tech, news, web_results, gex_data, flow_data)

        # 8. 构建 Embed 消息
        embed = discord.Embed(
            title=f"📑 {ticker} 深度投资分析报告",
            description=report,
            color=0x1a73e8 # Google Blue
        )
        
        latest = df_tech.iloc[-1]
        embed.add_field(name="当前价格", value=f"{fund['price']}", inline=True)
        embed.add_field(name="P/E 估值", value=f"{fund['pe']}", inline=True)
        embed.add_field(name="P/B 估值", value=f"{fund['pb']}", inline=True)
        embed.add_field(name="RSI (14)", value=f"{latest['RSI']:.1f}", inline=True)
        embed.add_field(name="波动率", value=f"{latest['Volatility']:.2%}", inline=True)
        embed.add_field(name="P/C Ratio (Vol)", value=f"{fund['pc_ratio_vol']}", inline=True)
        if gex_data:
            embed.add_field(name="Call Wall (阻力)", value=f"{gex_data['call_wall']}", inline=True)
            embed.add_field(name="Put Wall (支撑)", value=f"{gex_data['put_wall']}", inline=True)
        if flow_data:
            top_flow = flow_data[0]
            embed.add_field(name="最大异动", value=f"{top_flow['type']} {top_flow['strike']} (Vol:{top_flow['volume']})", inline=True)
        embed.add_field(name="趋势 (50/200)", value=f'{"金叉" if latest["SMA_50"] > latest["SMA_200"] else "死叉"}', inline=True)

        embed.set_footer(text=f"分析对象: {fund['name']} | Host: {socket.gethostname()} | 由 DeepSeek AI 强力驱动")
        embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/8569/8569731.png") # 一个中性的图表icon

        # 9. 生成 PDF 并发送
        pdf_file = None
        pdf_buffer = StockAnalyzer.create_pdf_report(ticker, report, fund)
        if pdf_buffer:
            pdf_file = discord.File(pdf_buffer, filename=f"{ticker}_Analysis.pdf")

        # 5. 发送结果
        await status_msg.edit(content="", embed=embed, attachments=[pdf_file] if pdf_file else [])

    except Exception as e:
        error_message = f"❌ 处理 **{ticker}** 时发生严重错误: {str(e)}\n"
        error_message += "这可能是由于数据源问题或内部计算错误。请稍后再试。"
        await status_msg.edit(content=error_message)

# 启动 Bot
if __name__ == "__main__":
    if not DISCORD_TOKEN or not DEEPSEEK_API_KEY:
        print("⚠️ 请设置 DISCORD_TOKEN 和 DEEPSEEK_API_KEY 环境变量")
    else:
        # Start the health check server in a background thread for deployment platforms
        health_check_thread = threading.Thread(target=run_health_check_server)
        health_check_thread.daemon = True  # Allows main thread to exit even if this thread is running
        health_check_thread.start()

        # Start the bot
        bot.run(DISCORD_TOKEN)

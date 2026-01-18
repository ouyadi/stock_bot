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
            }

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
        """使用 DuckDuckGo 搜索最新的市场新闻和事件"""
        try:
            with DDGS() as ddgs:
                query = f"{ticker} stock latest news catalyst analysis"
                # 获取前 3 条结果，包含标题和摘要
                results = list(ddgs.text(query, max_results=3))
                return results
        except Exception as e:
            print(f"Web Search Error: {e}")
            return []

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
    async def get_ai_analysis(ticker, fund, tech_data, news_data, web_search_data, gex_data):
        """调用 LLM 生成更深度的自然语言报告"""
        latest = tech_data.iloc[-1]

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

        # 构建更强大的提示词 (Prompt)
        prompt = f"""
            # Role
            你是一位拥有20年深厚资历的华尔街量化与宏观对冲基金首席投资官 (CIO)。你擅长将自上而下的宏观逻辑（Top-Down）与自下而上的量化因子（Bottom-Up）相结合，挖掘市场尚未完全定价的“预期差”。

            # Input Data Panel
            ## 1. 标的基本面与质量 (Quality & Value)
            - 标的: {ticker} ({fund['name']}) | 行业: {fund['sector']}
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

            ## 4. 市场催化剂 (Catalysts)
            - 实时网络搜索 (Web Search):
            {web_content if web_content else "- 暂无网络搜索结果"}
            - 交易所新闻 (Exchange News): 
            {news_headlines if news_headlines else "- 暂无交易所新闻"}

            # Analysis Requirements
            请基于以上数据，生成一份逻辑严密、具备实战指导意义的分析报告。要求：

            ### 1. 🏛️ 宏观叙事与行业定性
            分析当前宏观环境对该行业及公司的边际影响。判断标的处于周期的哪个阶段。

            ### 2. 📊 因子深度分析
            - **估值与预期**: 结合 P/E 和 Forward P/E，判断市场当前的预期是否过高或过低。
            - **基本面质量**: 评估 ROE 和负债水平，判断公司的护城河与抗风险能力。
            - **期权博弈与 Gamma Squeeze**: 
                1. 分析 P/C Ratio 判断情绪。
                2. **重点分析 Gamma 数据**: 
                   - 如果当前价格接近 **Call Wall**，是否存在向上突破引发 Gamma Squeeze (逼空) 的可能？
                   - 如果 Net GEX 为负，说明做市商处于 Short Gamma 状态，市场波动率是否会放大？
                   - Put Wall 是否提供了有效支撑？

            ### 3. 📈 技术面共振
            - 分析 50D/200D 均线的排列关系（金叉/死叉）。
            - 结合 RSI 和布林带位置，判断当前是否超买或超卖。

            ### 4. 🛠️ 组合构建建议 (Portfolio Construction)
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
    分析股票命令。用法: !a TSLA
    """
    ticker = ticker.upper()
    
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

        # 4. 计算 Gamma Exposure (GEX)
        await status_msg.edit(content=f"🧮 正在计算 **{ticker}** 的 Gamma Exposure (GEX) 与挤压风险...")
        gex_data = await loop.run_in_executor(None, lambda: StockAnalyzer.get_gamma_exposure(StockAnalyzer.get_data(ticker)[0].parent if hasattr(StockAnalyzer.get_data(ticker)[0], 'parent') else yf.Ticker(ticker), fund['price']))

        # 5. 获取 AI 报告
        await status_msg.edit(content=f"🤖 DeepSeek R1 (深度思考模式) 正在生成分析报告...")
        report = await StockAnalyzer.get_ai_analysis(ticker, fund, df_tech, news, web_results, gex_data)

        # 6. 构建 Embed 消息
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
        embed.add_field(name="趋势 (50/200)", value=f'{"金叉" if latest["SMA_50"] > latest["SMA_200"] else "死叉"}', inline=True)

        embed.set_footer(text=f"分析对象: {fund['name']} | Host: {socket.gethostname()} | 由 DeepSeek AI 强力驱动")
        embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/8569/8569731.png") # 一个中性的图表icon

        # 5. 发送结果
        await status_msg.edit(content="", embed=embed)

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

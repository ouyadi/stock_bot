import discord
from discord.ext import commands
import yfinance as yf
import pandas as pd
import numpy as np
from google import genai
import os
import asyncio
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import socket

# ================= 配置区域 =================
# 建议使用环境变量，或者直接在此处填入 Key
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# 配置 Gemini AI
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_ID = 'gemini-2.0-flash'

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
            }
            
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
    async def get_ai_analysis(ticker, fund, tech_data, news_data):
        """调用 LLM 生成更深度的自然语言报告"""
        latest = tech_data.iloc[-1]

        # Safely extract news headlines, skipping items that might not have a 'title' key.
        news_headlines = "\n".join([f"- {n['title']}" for n in news_data[:5] if 'title' in n])

        # 构建更强大的提示词 (Prompt)
        prompt = f"""
        # Role 定位
        你是一位拥有20年经验的华尔街量化与宏观对冲基金首席投资官 (CIO)。你擅长将自上而下的宏观逻辑与自下而上的量化多因子分析相结合。

        # 核心数据面板
        【标的信息】
        - 股票: {ticker} ({fund['name']}) | 行业: {fund['sector']}
        - 价格/市值: {fund['price']} {fund['currency']} / {fund['market_cap']}

        【多因子基本面】
        - 估值维度: P/E (TTM): {fund['pe']} | Forward P/E: {fund['forward_pe']} | P/B: {fund['pb']}
        - 质量维度: ROE: {fund['roe']} | EPS: {fund['eps']} | 负债权益比: {fund['debt_to_equity']}
        - 增长维度: [请根据行业背景评估其营收与利润增长动能]

        【量化与波动特征】
        - 30日年化波动率: {latest['Volatility']:.2%}
        - 贝塔系数 (Beta): {fund['beta']}

        【技术面共振】
        - 动能指标: RSI(14): {latest['RSI']:.2f} | MACD: {latest['MACD']:.2f} (信号线: {latest['MACD_Signal']:.2f})
        - 均线结构: 50D SMA: {latest['SMA_50']:.2f} | 200D SMA: {latest['SMA_200']:.2f} (当前价格{"偏离" if abs(latest['Close']-latest['SMA_200'])/latest['SMA_200'] > 0.1 else "贴近"}长周期成本线)
        - 波动区间: 布林带 ({latest['BB_Lower']:.2f} - {latest['BB_Upper']:.2f})

        【市场情绪与驱动力】
        - 近期新闻摘要: {news_headlines if news_headlines else "- 暂无显著负面/正面催化剂"}
        - 宏观环境背景: [当前利率环境、行业监管政策、汇率变动]

        # 任务要求：撰写深度投资分析报告
        请生成一份严谨、具备实战指导意义的 Markdown 格式报告，包含：

        ## 1. 💎 核心投资逻辑 (Investment Thesis)
        不要罗列数据，请给出“一针见血”的判断。目前是估值修复、动能追涨还是价值陷阱？是否存在宏观叙事支持？

        ## 2. 📊 财务质量与估值分位
        - 对比行业平均水平，评估 {ticker} 的基本面防御性。
        - 结合 ROE 和债务结构，分析其在当前高利率/低增长环境下的生存能力。

        ## 3. 📉 量化特征与技术面博弈
        - **趋势强度**: 分析均线系统是“多头排列”还是“均线缠绕”。
        - **超买/超卖与背离**: RSI 是否与价格走势背离？MACD 金叉/死叉的含金量如何？
        - **波动率挤压**: 根据布林带开口情况判断是否面临爆发性的方向选择。

        ## 4. ⚡ 催化剂与风险溢价
        - 深入分析近期新闻对资金流向的实际影响。
        - 识别潜在的“黑天鹅”风险（如政策变动、财报暴雷点）。

        ## 5. 🛠 机构级交易执行建议
        - **评级**: (强力买入 / 逢低买入 / 持股观望 / 卖出)
        - **策略结构**: 给出具体的 Entry (入场)、Target (目标价)、Stop-loss (止损位)。
        - **仓位管理**: 建议配置权重 (如：轻仓试探、标准配置、进攻性配置)。

        请直接输出报告内容，语言风格要求：专业、客观、不带情绪色彩，多使用金融专业术语。
        """        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: client.models.generate_content(model=MODEL_ID, contents=prompt))
            return response.text
        except Exception as e:
            return f"AI 分析生成失败: {str(e)}"

# ================= Discord 命令处理 =================

@bot.event
async def on_ready():
    print(f'✅ Bot 已登录: {bot.user}')
    print('Gemini 模式就绪。尝试输入: !a TSLA')

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
        
        # 3. 获取 AI 报告
        await status_msg.edit(content=f"🤖 Gemini AI 正在生成深度分析报告...")
        report = await StockAnalyzer.get_ai_analysis(ticker, fund, df_tech, news)

        # 4. 构建 Embed 消息
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
        embed.add_field(name="趋势 (50/200)", value=f'{"金叉" if latest["SMA_50"] > latest["SMA_200"] else "死叉"}', inline=True)

        embed.set_footer(text=f"分析对象: {fund['name']} | Host: {socket.gethostname()} | 由 Gemini AI 强力驱动")
        embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/8569/8569731.png") # 一个中性的图表icon

        # 5. 发送结果
        await status_msg.edit(content="", embed=embed)

    except Exception as e:
        error_message = f"❌ 处理 **{ticker}** 时发生严重错误: {str(e)}\n"
        error_message += "这可能是由于数据源问题或内部计算错误。请稍后再试。"
        await status_msg.edit(content=error_message)

# 启动 Bot
if __name__ == "__main__":
    if not DISCORD_TOKEN or not GEMINI_API_KEY:
        print("⚠️ 请设置 DISCORD_TOKEN 和 GEMINI_API_KEY 环境变量")
    else:
        # Start the health check server in a background thread for deployment platforms
        health_check_thread = threading.Thread(target=run_health_check_server)
        health_check_thread.daemon = True  # Allows main thread to exit even if this thread is running
        health_check_thread.start()

        # Start the bot
        bot.run(DISCORD_TOKEN)

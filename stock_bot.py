import discord
from discord.ext import commands
import yfinance as yf
import pandas as pd
import numpy as np
from google import genai
import os
import asyncio

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
        你是一位专业的华尔街量化与宏观对冲基金经理。请根据以下综合数据，深度分析股票 {ticker} ({fund['name']})。

        【基本面数据】
        - 行业: {fund['sector']}
        - 当前价格: {fund['price']} {fund['currency']}
        - 市值: {fund['market_cap']}
        - 市盈率 (P/E): {fund['pe']}
        - 市净率 (P/B): {fund['pb']}
        - 每股收益 (EPS): {fund['eps']}
        - 净资产收益率 (ROE): {fund['roe']}
        - 负债权益比: {fund['debt_to_equity']}

        【量化分析】
        - 30日年化波动率: {latest['Volatility']:.2%} (越高代表价格变动越剧烈)

        【技术指标 (最新收盘)】
        - RSI (14): {latest['RSI']:.2f} (RSI>70超买, <30超卖)
        - 50日均线: {latest['SMA_50']:.2f}
        - 200日均线: {latest['SMA_200']:.2f}
        - MACD: {latest['MACD']:.2f} (信号线: {latest['MACD_Signal']:.2f})
        - 布林带: 上轨 {latest['BB_Upper']:.2f}, 下轨 {latest['BB_Lower']:.2f}
        - 长期趋势: 当前价格 {"高于" if latest['Close'] > latest['SMA_200'] else "低于"} 200日均线，呈{"上升" if latest['SMA_50'] > latest['SMA_200'] else "下降"}趋势。

        【事件驱动 (近期新闻)】
        {news_headlines if news_headlines else "- 暂无重要新闻"}

        请生成一份专业、深刻的 Markdown 格式投资分析报告，包含以下部分：
        1. **📈 综合评估与核心观点**: 结合基本面、技术面、量化指标和新闻，给出核心投资逻辑。
        2. **🏢 基本面健康度**: 评估公司财务状况、估值是否合理，有无增长潜力。
        3. **📉 量化与技术面分析**: 结合波动率、RSI、MACD和均线，判断市场情绪和趋势，给出关键技术位。
        4. **📰 事件驱动因素**: 分析近期新闻可能对股价造成的影响。
        5. **🎯 交易策略与风险**: 给出明确的操作建议（长线持有/波段做多/保持观望/逢高做空），并阐述主要风险点。

        请直接输出报告内容，展现你的专业性。
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
    try:
        print("正在检查可用模型列表...")
        for m in client.models.list():
            print(f" - {m.name}")
    except Exception as e:
        print(f"⚠️ 无法列出模型: {e}")
    print('纯文字模式就绪。尝试输入: !a TSLA')

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

        embed.set_footer(text=f"分析对象: {fund['name']} | 由 Gemini AI 强力驱动")
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
        bot.run(DISCORD_TOKEN)

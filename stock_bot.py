import discord
from discord.ext import commands
import yfinance as yf
import pandas as pd
import google.generativeai as genai
import os
import asyncio

# ================= 配置区域 =================
# 建议使用环境变量，或者直接在此处填入 Key
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# 配置 Gemini AI
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.5-flash')

# 配置 Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# ================= 核心逻辑模块 =================

class StockAnalyzer:
    @staticmethod
    def get_data(ticker_symbol):
        """获取历史数据和基本面信息"""
        try:
            stock = yf.Ticker(ticker_symbol)
            # 获取1年数据用于计算指标
            df = stock.history(period="1y")
            
            if df.empty:
                return None, None

            info = stock.info
            fundamentals = {
                "name": info.get('longName', ticker_symbol),
                "sector": info.get('sector', 'Unknown'),
                "pe": info.get('trailingPE', 'N/A'),
                "market_cap": info.get('marketCap', 'N/A'),
                "price": info.get('currentPrice', df['Close'].iloc[-1]),
                "currency": info.get('currency', 'USD')
            }
            return df, fundamentals
        except Exception as e:
            print(f"Data Error: {e}")
            return None, None

    @staticmethod
    def calculate_indicators(df):
        """计算技术指标 (用于喂给 AI)"""
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

        return df

    @staticmethod
    async def get_ai_analysis(ticker, fund, tech_data):
        """调用 LLM 生成自然语言报告"""
        latest = tech_data.iloc[-1]
        
        # 构建提示词 (Prompt)
        prompt = f"""
        你是一位专业的华尔街量化交易员。请根据以下数据分析股票 {ticker} ({fund['name']})。
        
        【基本面数据】
        - 行业: {fund['sector']}
        - 当前价格: {fund['price']} {fund['currency']}
        - 市盈率 (P/E): {fund['pe']}
        - 市值: {fund['market_cap']}
        
        【技术指标 (最新收盘)】
        - RSI (14): {latest['RSI']:.2f} (RSI>70超买, <30超卖)
        - 50日均线: {latest['SMA_50']:.2f}
        - 200日均线: {latest['SMA_200']:.2f}
        - 布林带: 上轨 {latest['BB_Upper']:.2f}, 下轨 {latest['BB_Lower']:.2f}
        - 趋势判断: 当前价格 {"高于" if latest['Close'] > latest['SMA_200'] else "低于"} 200日均线
        
        请生成一份简短、犀利的 Markdown 格式报告，包含以下部分：
        1. **📊 市场情绪**：基于RSI和布林带位置，判断当前是贪婪还是恐慌。
        2. **🏢 基本面概览**：简评估值水平。
        3. **🎯 交易策略**：给出明确的操作建议（做多/做空/观望），并给出支撑位和阻力位的参考。
        4. **⚠️ 风险提示**：简述潜在风险。
        
        请直接输出报告内容，不要包含寒暄。
        """
        
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: model.generate_content(prompt))
            return response.text
        except Exception as e:
            return f"AI 分析生成失败: {str(e)}"

# ================= Discord 命令处理 =================

@bot.event
async def on_ready():
    print(f'✅ Bot 已登录: {bot.user}')
    # 打印可用模型列表以方便调试
    try:
        print("正在检查可用模型列表...")
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
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
    
    # 1. 发送简单的加载状态
    status_msg = await ctx.send(f"🔍 正在分析 **{ticker}** 的基本面与技术面数据...")
    
    # 2. 获取数据
    df, fund = StockAnalyzer.get_data(ticker)
    
    if df is None:
        await status_msg.edit(content=f"❌ 找不到股票代码 **{ticker}**，请检查拼写。")
        return

    try:
        # 3. 计算指标 (虽然不画图，但AI需要这些数字)
        df_tech = StockAnalyzer.calculate_indicators(df)
        
        # 4. 获取 AI 报告
        report = await StockAnalyzer.get_ai_analysis(ticker, fund, df_tech)

        # 5. 构建 Embed 消息
        embed = discord.Embed(
            title=f"📑 {ticker} 投资分析报告",
            description=report,
            color=0x3498db # 蓝色
        )
        
        # 添加一些关键数据字段作为摘要
        latest = df_tech.iloc[-1]
        embed.add_field(name="当前价格", value=f"{fund['price']}", inline=True)
        embed.add_field(name="RSI (14)", value=f"{latest['RSI']:.1f}", inline=True)
        embed.add_field(name="P/E 估值", value=f"{fund['pe']}", inline=True)
        
        embed.set_footer(text=f"分析对象: {fund['name']} | 由 Gemini AI 驱动")

        # 6. 发送结果
        await status_msg.edit(content="", embed=embed)

    except Exception as e:
        await status_msg.edit(content=f"❌ 处理过程中发生错误: {str(e)}")

# 启动 Bot
if __name__ == "__main__":
    if not DISCORD_TOKEN or not GEMINI_API_KEY:
        print("⚠️ 请设置 DISCORD_TOKEN 和 GEMINI_API_KEY 环境变量")
    else:
        bot.run(DISCORD_TOKEN)

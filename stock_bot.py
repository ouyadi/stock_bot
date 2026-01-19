import warnings
# 忽略 duckduckgo_search 的包更名警告
warnings.filterwarnings("ignore", category=RuntimeWarning, message="This package.*duckduckgo_search.*")
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.graphics.shapes import Drawing, Line
from reportlab.lib import colors
import matplotlib
matplotlib.use('Agg') # 设置后端为 Agg，适用于无头服务器环境
import matplotlib.pyplot as plt
import json
import time
from supabase import create_client, Client
import uvicorn
from fastapi import FastAPI, Form, File, UploadFile, HTTPException, Body, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import base64
from lxml import html as lxml_html
import PyPDF2
try:
    from PIL import Image as PILImage, ImageEnhance
    import pytesseract
except ImportError:
    PILImage = None
    ImageEnhance = None
    pytesseract = None

# ================= 配置区域 =================
# 建议使用环境变量，或者直接在此处填入 Key
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')
DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY')
DISCORD_AI_REPORT_CHANNEL_ID = os.getenv('DISCORD_AI_REPORT_CHANNEL_ID') # 指定频道 ID
INSTITUTION_REPORT_CHANNEL_ID = '1434770162573250560' # 投研机构带飞频道
SUPABASE_URL = os.getenv('SUPABASE_URL')
SUPABASE_KEY = os.getenv('SUPABASE_KEY')
SUPABASE_BUCKET = os.getenv('SUPABASE_BUCKET', 'reports') # 默认 bucket 名为 reports


# 配置 DeepSeek AI
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
MODEL_ID = 'deepseek-reasoner'

# 配置 Discord Bot
intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix='!', intents=intents)

# 配置 Supabase
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# 配置 FastAPI
app = FastAPI()

# 配置 CORS (允许前端跨域调用)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境建议限制为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 投研报告处理模块 =================

class ResearchAnalyzer:
    @staticmethod
    async def summarize_content(content: str, subject: str) -> str:
        """使用 DeepSeek API 对投研报告内容进行总结"""
        prompt = f"""
        # Role
        你是一名顶尖的金融分析师，你的任务是阅读并总结一份来自投研机构的电子邮件报告。

        # Task
        请根据报告内容，生成一份精炼、专业的摘要。
        **注意：请忽略报告末尾或文中出现的法律免责声明 (Disclaimer)、风险披露 (Risk Disclosure) 等合规性文本，专注于实质性的投资分析内容。**
        摘要应包含以下几点：
        1.  **核心观点 (Core Thesis)**: 报告最关键的结论是什么？(例如: 看多/看空某资产、市场趋势预测等)
        2.  **关键论据 (Key Arguments)**: 支撑核心观点的三到五个最重要的数据、事件或逻辑是什么？
        3.  **潜在风险 (Potential Risks)**: 报告中提及了哪些可能导致结论失效的风险因素？
        4.  **目标价与评级 (Target & Rating)**: 如果报告中明确给出了目标价或投资评级(如买入/持有/卖出)，请明确指出。

        请使用中文撰写，语言风格要专业、客观、条理清晰。

        # Input Data
        - **邮件主题**: {subject}
        - **报告内容**:
        ---
        {content[:50000]} 
        ---
        """
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None,
                lambda: client.chat.completions.create(
                    model=MODEL_ID,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False
                )
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"DeepSeek Error: {e}")
            return f"AI 摘要生成失败: {e}"

    @staticmethod
    def create_summary_pdf(summary_text: str, subject: str) -> io.BytesIO:
        """将 AI 生成的摘要转换为 PDF"""
        try:
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=72)
            styles = getSampleStyleSheet()
            pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))

            title_style = ParagraphStyle('Title', fontName='STSong-Light', fontSize=18, alignment=1, spaceAfter=20, textColor=colors.navy)
            normal_style = ParagraphStyle('Normal', fontName='STSong-Light', fontSize=11, leading=14, spaceAfter=6)
            bullet_style = ParagraphStyle('Bullet', parent=normal_style, leftIndent=10)
            
            story = []
            story.append(Paragraph(f"投研报告摘要: {subject}", title_style))
            
            # 简单的 Markdown 解析
            for line in summary_text.split('\n'):
                line = line.strip()
                if line.startswith('#'):
                    story.append(Paragraph(line.lstrip('#').strip(), ParagraphStyle('h2', parent=normal_style, fontSize=14, spaceBefore=10)))
                elif line.startswith('- ') or line.startswith('* '):
                    story.append(Paragraph(f"• {line[2:]}", bullet_style))
                elif line:
                    story.append(Paragraph(line, normal_style))

            doc.build(story)
            buffer.seek(0)
            return buffer
        except Exception as e:
            print(f"PDF Creation Error: {e}")
            return None

    @staticmethod
    async def send_discord_notification(summary: str, subject: str, pdf_url: str, status_msg: Optional[discord.Message] = None):
        """发送通知到指定的 Discord 频道"""
        channel_id = int(INSTITUTION_REPORT_CHANNEL_ID) # 投研机构带飞频道
        channel = bot.get_channel(channel_id)
        if not channel:
            print(f"错误: 找不到频道 ID {channel_id}")
            return

        embed = discord.Embed(
            title=f"📬 新投研报告摘要: {subject}",
            description=summary,
            color=discord.Color.blue()
        )
        embed.add_field(name="下载完整 PDF 报告", value=f"[点击这里]({pdf_url})", inline=False)
        embed.set_footer(text="由 CloudMailIn -> DeepSeek -> Supabase 驱动")
        
        if status_msg:
            await status_msg.edit(content="", embed=embed)
        else:
            await channel.send(embed=embed)


# 定义 CloudMailIn 的数据模型
class CloudmailinAttachment(BaseModel):
    file_name: str
    content_type: str
    content: str  # Base64 encoded content
    size: int

class CloudmailinPayload(BaseModel):
    plain: Optional[str] = None
    html: Optional[str] = None
    subject: Optional[str] = "无主题"
    attachments: List[CloudmailinAttachment] = []

async def process_email_task(payload: CloudmailinPayload):
    """后台异步处理邮件任务"""
    subject = payload.subject
    print(f"🔄 后台任务启动: 处理邮件 '{subject}'")
    
    # === 发送初始状态消息 ===
    status_msg = None
    try:
        channel_id = int(INSTITUTION_REPORT_CHANNEL_ID)
        channel = bot.get_channel(channel_id)
        if channel:
            status_msg = await channel.send(f"📧 收到新邮件: **{subject}**\n⏳ 正在解析附件与正文...")
    except Exception as e:
        print(f"Discord status update failed: {e}")

    analysis_content = ""
    source = ""

    # 1. 提取内容 (聚合所有来源: 正文 + PDF + 图片提示)
    parts = []
    sources = []

    try:
        # --- 处理邮件正文 ---
        body_text = ""
        if payload.html:
            try:
                # 使用 lxml 清理 HTML 标签
                doc = lxml_html.fromstring(payload.html)
                # 移除脚本和样式
                for bad in doc.xpath("//script | //style"):
                    bad.getparent().remove(bad)
                body_text = doc.text_content().strip()
                if body_text:
                    sources.append("HTML正文")
            except Exception as e:
                print(f"HTML parsing warning: {e}")
        
        # 如果 HTML 解析失败或为空，尝试纯文本
        if not body_text and payload.plain:
            body_text = payload.plain.strip()
            if body_text:
                sources.append("纯文本正文")
        
        if body_text:
            parts.append(f"=== 邮件正文 ===\n{body_text}")

        # --- 处理 PDF 附件 ---
        pdf_attachments = [a for a in payload.attachments if "pdf" in a.content_type]
        for pdf in pdf_attachments:
            try:
                print(f"📄 处理 PDF 附件: {pdf.file_name}")
                pdf_content = base64.b64decode(pdf.content)
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
                pdf_text = ""
                for page in pdf_reader.pages:
                    pdf_text += page.extract_text() or ""
                
                if pdf_text.strip():
                    parts.append(f"=== PDF附件: {pdf.file_name} ===\n{pdf_text}")
                    sources.append(f"PDF:{pdf.file_name}")
            except Exception as e:
                print(f"PDF reading error ({pdf.file_name}): {e}")

        # --- 处理图片附件 ---
        image_attachments = []
        for a in payload.attachments:
            # 增强判断：如果 Content-Type 丢失或为 octet-stream，尝试通过后缀名识别
            if "image" in a.content_type.lower() or \
               any(a.file_name.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp', '.heic']):
                image_attachments.append(a)

        for img in image_attachments:
            if PILImage and pytesseract:
                try:
                    print(f"🖼️ 正在OCR识别图片: {img.file_name}")
                    img_content = base64.b64decode(img.content)
                    image = PILImage.open(io.BytesIO(img_content))
                    
                    # === OCR 预处理优化 ===
                    # 1. 转为灰度图 (消除色彩干扰)
                    image = image.convert('L')
                    
                    # 2. 增强对比度 (让文字更清晰)
                    if ImageEnhance:
                        enhancer = ImageEnhance.Contrast(image)
                        image = enhancer.enhance(2.0) # 提高对比度

                    # 3. 放大图片 (Tesseract 对小字号识别较差，放大有助于识别)
                    width, height = image.size
                    if width < 1000:
                        image = image.resize((width * 2, height * 2), PILImage.Resampling.LANCZOS)

                    # === Tesseract 配置 ===
                    # --psm 6: 假设是一个统一的文本块。这对表格特别有效，因为它会按行读取，而不是试图分栏。
                    custom_config = r'--oem 3 --psm 6'

                    # 尝试识别中文和英文，如果失败则回退到默认语言
                    try:
                        text = pytesseract.image_to_string(image, lang='chi_sim+eng', config=custom_config)
                    except Exception:
                        text = pytesseract.image_to_string(image, config=custom_config)
                    
                    if text.strip():
                        parts.append(f"=== 图片附件 ({img.file_name}) OCR内容 ===\n{text}")
                        sources.append(f"OCR:{img.file_name}")
                except Exception as e:
                    print(f"OCR Error ({img.file_name}): {e}")
            else:
                parts.append(f"=== 图片附件 ({img.file_name}) ===\n[服务器未安装 OCR 库，无法提取文字]")
                if "图片(未OCR)" not in sources: sources.append("图片(未OCR)")

        if not parts:
            print("❌ 邮件内容为空")
            if status_msg:
                try: await status_msg.edit(content=f"❌ 处理邮件 **{subject}** 失败: 邮件内容为空")
                except: pass
            return

        analysis_content = "\n\n".join(parts)
        source = ", ".join(sources)
        print(f"📝 汇总内容来源: {source}")
        
        if status_msg:
            try: await status_msg.edit(content=f"📧 收到新邮件: **{subject}**\n📝 内容提取完成 ({source})，正在调用 DeepSeek 进行深度分析...")
            except: pass

        # 2. 调用 AI 进行总结
        print("🤖 正在发送内容到 DeepSeek 进行总结...")
        if not analysis_content.strip():
             summary_text = "报告内容为空或无法解析。"
        else:
             summary_text = await ResearchAnalyzer.summarize_content(analysis_content, payload.subject)
        
        if status_msg:
            try: await status_msg.edit(content=f"📧 收到新邮件: **{subject}**\n🤖 AI 分析完成，正在生成 PDF 报告...")
            except: pass

        # 3. 生成 PDF
        print("📑 正在生成摘要 PDF...")
        pdf_buffer = ResearchAnalyzer.create_summary_pdf(summary_text, payload.subject)
        
        if not pdf_buffer:
            print("❌ 无法生成 PDF")
            if status_msg:
                try: await status_msg.edit(content=f"❌ 处理邮件 **{subject}** 失败: 无法生成 PDF")
                except: pass
            return
            
        if status_msg:
            try: await status_msg.edit(content=f"📧 收到新邮件: **{subject}**\n☁️ PDF 生成完毕，正在上传至 Supabase...")
            except: pass

        # 4. 上传到 Supabase
        print("☁️ 正在上传 PDF 到 Supabase...")
        pdf_filename = f"report_summary_{int(time.time())}.pdf"
        public_url = "Supabase not configured"
        if supabase:
            res = supabase.storage.from_(SUPABASE_BUCKET).upload(
                file=pdf_buffer.getvalue(), 
                path=pdf_filename, 
                file_options={"content-type": "application/pdf"}
            )
            public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(pdf_filename)

        # 5. 发送到 Discord
        print("💬 正在发送通知到 Discord...")
        await ResearchAnalyzer.send_discord_notification(summary_text, payload.subject, public_url, status_msg)

        print("✅ 投研报告处理流程完成!")

    except Exception as e:
        print(f"处理邮件时发生严重错误: {e}")
        if status_msg:
            try: await status_msg.edit(content=f"❌ 处理邮件 **{subject}** 时发生错误: {str(e)}")
            except: pass

@app.post("/email-report")
async def handle_email_report(request: Request, background_tasks: BackgroundTasks):
    """
    接收来自 CloudMailIn 的邮件 POST 请求，进行处理和转发。
    """
    print("📧 收到新邮件请求...")
    content_type = request.headers.get("content-type", "")
    payload = None
    subject = "未知主题"

    # === 1. 解析请求数据 (支持 JSON 和 Multipart) ===
    if "application/json" in content_type:
        # 处理 Google Script 发送的 JSON
        try:
            data = await request.json()
            print("🔍 解析 JSON Payload (Google Script)")
            
            subject = data.get("subject", "无主题")
            plain = data.get("body")
            html = None # Google Script 通常只发送 getPlainBody
            
            attachments_list = []
            for att in data.get("attachments", []):
                content_b64 = att.get("content", "")
                attachments_list.append(CloudmailinAttachment(
                    file_name=att.get("fileName", "unknown"),
                    content_type=att.get("mimeType", "application/octet-stream"),
                    content=content_b64,
                    size=len(content_b64)
                ))
            
            payload = CloudmailinPayload(
                plain=plain,
                html=html,
                subject=subject,
                attachments=attachments_list
            )
        except Exception as e:
            print(f"JSON Parse Error: {e}")
            raise HTTPException(status_code=400, detail=f"JSON parsing error: {e}")
    else:
        # 处理 CloudMailIn 发送的 Multipart Form Data
        try:
            form = await request.form()
            print(f"🔍 Form Keys: {list(form.keys())}")
            
            plain = form.get("plain")
            html = form.get("html")
            subject = form.get("headers[subject]") or form.get("subject") or "无主题"
            
            attachments_list = []
            for key, value in form.multi_items():
                if isinstance(value, UploadFile) or (hasattr(value, "filename") and value.filename):
                    print(f"📂 收到附件: {value.filename} (Key: {key}, Content-Type: {value.content_type})")
                    try:
                        content = await value.read()
                        if content:
                            b64_content = base64.b64encode(content).decode('utf-8')
                            attachments_list.append(CloudmailinAttachment(
                                file_name=value.filename or "unknown",
                                content_type=value.content_type or "application/octet-stream",
                                content=b64_content,
                                size=len(content)
                            ))
                    finally:
                        await value.close()
                elif "attachment" in key:
                        print(f"⚠️ 发现疑似附件字段 '{key}' 但未被识别为文件对象 (Type: {type(value)})")
            
            payload = CloudmailinPayload(
                plain=str(plain) if plain else None,
                html=str(html) if html else None,
                subject=str(subject),
                attachments=attachments_list
            )
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Form parsing error: {e}")

    # 2. 添加到后台任务
    background_tasks.add_task(process_email_task, payload)
    
    # 3. 立即返回响应
    return {"status": "received", "message": "Processing started in background"}

class AnalyzeRequest(BaseModel):
    ticker: str

@app.post("/analyze")
async def api_analyze(request: AnalyzeRequest):
    """Web API: 分析股票接口"""
    print(f"🌍 收到 Web API 分析请求: {request.ticker}")
    
    loop = asyncio.get_running_loop()
    # 在线程池中运行同步的分析流程，避免阻塞主线程
    pdf_buffer, report, oi_chart_url = await loop.run_in_executor(
        None, 
        lambda: StockAnalyzer.run_full_analysis_pipeline(request.ticker)
    )
    
    if not report:
        raise HTTPException(status_code=404, detail="Analysis failed or ticker not found")

    # 上传 PDF 到 Supabase (如果配置了)
    pdf_url = None
    if pdf_buffer and supabase:
        pdf_filename = f"{request.ticker}_{int(time.time())}.pdf"
        pdf_url = await loop.run_in_executor(None, lambda: StockAnalyzer.upload_file_to_supabase(pdf_filename, pdf_buffer, "application/pdf"))

    return {"status": "success", "ticker": request.ticker, "report": report, "pdf_url": pdf_url, "oi_chart_url": oi_chart_url}

@app.get("/")
def health_check():
    return {"status": "ok", "bot_status": "logged_in" if bot.is_ready() else "connecting"}

# ================= 核心逻辑模块 (保持不变) =================
class StockAnalyzer:
    @staticmethod
    def run_full_analysis_pipeline(ticker):
        """执行完整的分析流程 (同步方法，供 Web Service 调用)"""
        try:
            ticker = ticker.upper()
            # A股后缀处理
            if ticker.isdigit() and len(ticker) == 6:
                if ticker.startswith('6'): ticker = f"{ticker}.SS" # 上海证券交易所
                elif ticker.startswith(('0', '3')): ticker = f"{ticker}.SZ" # 深圳证券交易所
                elif ticker.startswith(('4', '8')): ticker = f"{ticker}.BJ" # 北京证券交易所

            # 1. 获取数据
            df, fund, news, macro_data = StockAnalyzer.get_data(ticker)
            if df is None: return None, None, None

            # 2. 计算指标
            df_tech = StockAnalyzer.calculate_indicators(df)
            latest = df_tech.iloc[-1]
            price_change = 0
            if len(df) >= 2:
                price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]

            # 3. 外部数据获取
            web_results = StockAnalyzer.get_web_search(ticker)
            stock_obj = yf.Ticker(ticker)
            gex_data = StockAnalyzer.get_gamma_exposure(stock_obj, fund['price'])
            flow_data = StockAnalyzer.get_option_flow(stock_obj, fund['price'])
            oi_chart_buffer = StockAnalyzer.get_option_open_interest_chart(stock_obj, fund['price'])
            
            # 上传 OI 图表
            oi_chart_url = None
            if oi_chart_buffer and supabase:
                oi_chart_filename = f"{ticker}_oi_chart_{int(time.time())}.png"
                oi_chart_url = StockAnalyzer.upload_file_to_supabase(oi_chart_filename, oi_chart_buffer, "image/png")

            # 4. AI 生成
            report = StockAnalyzer._generate_ai_report_sync(ticker, fund, df_tech, news, web_results, gex_data, flow_data, macro_data)

            # 5. PDF 生成
            pdf_buffer = StockAnalyzer.create_pdf_report(ticker, report, fund, latest, price_change, oi_chart_buffer)
            return pdf_buffer, report, oi_chart_url
        except Exception as e:
            print(f"Pipeline Error: {e}")
            return None, None, None

    @staticmethod
    def upload_file_to_supabase(filename: str, buffer: io.BytesIO, content_type: str) -> Optional[str]:
        """通用文件上传到 Supabase Storage 并返回公开链接"""
        if not supabase:
            print("Supabase not configured, skipping upload.")
            return None
        try:
            path = f"{filename}"
            # 使用 getvalue() 来获取全部内容，避免移动 buffer 的指针
            supabase.storage.from_(SUPABASE_BUCKET).upload(
                file=buffer.getvalue(),
                path=path,
                file_options={"content-type": content_type}
            )
            return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(path)
        except Exception as e:
            print(f"Supabase upload error for {filename}: {e}")
            return None

    @staticmethod
    def get_data(ticker_symbol):
        """获取历史数据和更全面的基本面信息"""
        try:
            stock = yf.Ticker(ticker_symbol)
            df = stock.history(period="1y")
            
            if df.empty:
                return None, None, None, None

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
                "turnover_rate": "N/A"
            }
            
            # === 计算换手率 (Turnover Rate) ===
            # 优先使用 floatShares (流通股), 其次使用 sharesOutstanding (总股本)
            shares_base = info.get('floatShares') or info.get('sharesOutstanding')
            avg_vol_10d = info.get('averageVolume10days') or info.get('averageVolume')
            if shares_base and avg_vol_10d:
                fundamentals['turnover_rate'] = f"{(avg_vol_10d / shares_base):.2%}"

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
            
            # === 获取宏观市场数据 (Macro Data) ===
            macro_data = {}
            try:
                market_symbol = "^GSPC" # 默认标普500
                vix_symbol = "^VIX"
                
                if ticker_symbol.endswith(('.SS', '.SZ', '.BJ')):
                    market_symbol = "000001.SS" # 上证指数
                    vix_symbol = None # A股暂不获取VIX (或使用 510050 等替代，此处简化)
                
                market_ticker = yf.Ticker(market_symbol)
                market_hist = market_ticker.history(period="5d")
                if not market_hist.empty:
                    macro_data['market_index'] = market_symbol
                    macro_data['market_price'] = market_hist['Close'].iloc[-1]
                    macro_data['market_change'] = (market_hist['Close'].iloc[-1] - market_hist['Close'].iloc[-2]) / market_hist['Close'].iloc[-2]
                
                if vix_symbol:
                    vix_ticker = yf.Ticker(vix_symbol)
                    vix_hist = vix_ticker.history(period="5d")
                    if not vix_hist.empty:
                        macro_data['vix'] = vix_hist['Close'].iloc[-1]
                        macro_data['vix_change'] = (vix_hist['Close'].iloc[-1] - vix_hist['Close'].iloc[-2]) / vix_hist['Close'].iloc[-2]
            except Exception as e:
                print(f"Macro Data Error: {e}")

            return df, fundamentals, news, macro_data
        except Exception as e:
            print(f"Data Error: {e}")
            return None, None, None, None

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

        # 5. KDJ (随机指标 - A股常用)
        # RSV = (Close - Low_9) / (High_9 - Low_9) * 100
        low_list = df['Low'].rolling(window=9, min_periods=9).min()
        high_list = df['High'].rolling(window=9, min_periods=9).max()
        rsv = (df['Close'] - low_list) / (high_list - low_list) * 100
        df['K'] = rsv.ewm(com=2, adjust=False).mean() # com=2 等同于 alpha=1/3
        df['D'] = df['K'].ewm(com=2, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']

        # 6. ATR (平均真实波幅 - 波动率替代指标)
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR'] = true_range.rolling(window=14).mean()

        # 5. 波动率 (30日历史波动率)
        df['Log_Ret'] = df['Close'].apply(lambda x: np.log(x)).diff()
        df['Volatility'] = df['Log_Ret'].rolling(window=30).std() * np.sqrt(252) # 年化

        return df

    @staticmethod
    def get_web_search(ticker):
        """使用 DuckDuckGo 搜索最新的市场新闻、事件、管理层指引以及社交媒体情绪"""
        results = []
        try:
            is_ashare = ticker.endswith(('.SS', '.SZ', '.BJ'))
            with DDGS() as ddgs:
                # 1. 核心催化剂与未来事件 (Event-Driven Focus)
                query_event = f"{ticker} stock upcoming catalyst events earnings date fda approval product launch"
                if is_ashare:
                    query_event = f"{ticker} 股票 重大利好 业绩预告 资产重组 政策驱动"
                results.extend(list(ddgs.text(query_event, max_results=3)))

                # 2. 隐含波动率与期权异动 (Market Pricing of Events)
                query_iv = f"{ticker} stock implied volatility rank option flow unusual activity"
                if is_ashare:
                    # A股替代搜索: 北向资金、龙虎榜、主力资金
                    query_iv = f"{ticker} 北向资金流向 龙虎榜数据 主力资金 融资融券"
                results.extend(list(ddgs.text(query_iv, max_results=2)))
                
                # 3. 10-Q/10-K 管理层指引
                query_guidance = f"{ticker} stock earnings guidance management discussion 10-Q highlights"
                results.extend(list(ddgs.text(query_guidance, max_results=2)))
                
                # 4. 社交媒体情绪 - 分开搜索以提高覆盖率
                # 4.1 Reddit 深度讨论 (r/stocks, r/investing, r/wallstreetbets)
                query_reddit = f"site:reddit.com {ticker} stock due diligence discussion analysis"
                reddit_results = list(ddgs.text(query_reddit, max_results=3))
                for r in reddit_results:
                    r['title'] = f"[Reddit] {r['title']}"
                results.extend(reddit_results)

                # 4.2 Stocktwits 情绪 (散户大本营)
                query_st = f"site:stocktwits.com {ticker} sentiment bullish bearish"
                st_results = list(ddgs.text(query_st, max_results=2))
                for r in st_results:
                    r['title'] = f"[Stocktwits] {r['title']}"
                results.extend(st_results)

                # 5. 所属板块趋势 (Sector Trends)
                query_sector = f"{ticker} sector industry trends performance outlook"
                results.extend(list(ddgs.text(query_sector, max_results=2)))

                return results
        except Exception as e:
            print(f"Web Search Error: {e}")
            return results

    @staticmethod
    def get_risk_free_rate():
        """获取当前无风险利率 (基于 10年期美债收益率 ^TNX)"""
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            if not hist.empty:
                return hist['Close'].iloc[-1] / 100.0
        except Exception as e:
            print(f"Risk-Free Rate Error: {e}")
        return 0.045 # 默认 4.5%

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
            
            r = StockAnalyzer.get_risk_free_rate()
            
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
    def get_option_open_interest_chart(stock, current_price):
        """生成期权持仓量 (Open Interest) 分布图"""
        try:
            exps = stock.options
            if not exps: return None
            
            # 使用最近的到期日
            expiry = exps[0]
            opt = stock.option_chain(expiry)
            
            calls = opt.calls
            puts = opt.puts
            
            if calls.empty and puts.empty: return None
            
            # 筛选当前价格附近 +/- 15% 的行权价，避免图表过宽
            lower_bound = current_price * 0.85
            upper_bound = current_price * 1.15
            
            calls = calls[(calls['strike'] >= lower_bound) & (calls['strike'] <= upper_bound)]
            puts = puts[(puts['strike'] >= lower_bound) & (puts['strike'] <= upper_bound)]
            
            if calls.empty and puts.empty: return None

            # 绘图
            plt.style.use('ggplot')
            fig, ax = plt.subplots(figsize=(8, 3))
            
            # 提取数据
            # 为了简化，我们只画出有数据的 Strike
            all_strikes = sorted(list(set(calls['strike'].tolist() + puts['strike'].tolist())))
            
            call_oi = [calls[calls['strike'] == k]['openInterest'].sum() for k in all_strikes]
            put_oi = [puts[puts['strike'] == k]['openInterest'].sum() for k in all_strikes]
            
            indices = np.arange(len(all_strikes))
            width = 0.35
            
            ax.bar(indices - width/2, call_oi, width, label='Call OI', color='#2ca02c', alpha=0.8)
            ax.bar(indices + width/2, put_oi, width, label='Put OI', color='#d62728', alpha=0.8)
            
            ax.set_xticks(indices)
            ax.set_xticklabels([str(int(s)) for s in all_strikes], rotation=45, fontsize=7)
            ax.set_title(f'Open Interest Distribution (Expiry: {expiry})', fontsize=10)
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # 标记当前价格
            curr_idx = np.interp(current_price, all_strikes, indices)
            ax.axvline(x=curr_idx, color='blue', linestyle='--', alpha=0.6, label='Current Price')
            
            plt.tight_layout()
            
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            plt.close(fig)
            return buf
        except Exception as e:
            print(f"Chart Error: {e}")
            return None

    @staticmethod
    def create_pdf_report(ticker, report_text, fund_data, tech_latest, price_change, oi_chart_buffer):
        """生成 PDF 报告"""
        try:
            buffer = io.BytesIO()
            # 调整页边距，增加内容容纳空间
            doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
            styles = getSampleStyleSheet()
            
            # 注册中文字体 (STSong-Light 是 Adobe 预定义的简体中文字体，无需额外字体文件)
            pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
            
            # === 自定义样式优化 ===
            title_style = ParagraphStyle(
                'CustomTitle', parent=styles['Title'], fontName='STSong-Light', fontSize=24, leading=28, spaceAfter=10, alignment=1, textColor=colors.HexColor("#1a73e8"),
                keepWithNext=True # 确保标题不与后续内容分页
            )
            heading_style = ParagraphStyle(
                'CustomHeading', parent=styles['Heading2'], fontName='STSong-Light', fontSize=15, leading=18, spaceBefore=15, spaceAfter=8, textColor=colors.HexColor("#202124"),
                keepWithNext=True # 确保标题不与后续内容分页
            )
            normal_style = ParagraphStyle(
                'CustomNormal', parent=styles['Normal'], fontName='STSong-Light', fontSize=10.5, leading=15, spaceAfter=6, textColor=colors.HexColor("#3c4043")
            )
            bullet_style = ParagraphStyle(
                'CustomBullet', parent=normal_style, leftIndent=15, firstLineIndent=0, spaceAfter=4, bulletFontName='STSong-Light'
            )
            sub_bullet_style = ParagraphStyle(
                'CustomSubBullet', parent=normal_style, leftIndent=35, firstLineIndent=0, spaceAfter=4, bulletFontName='STSong-Light'
            )
            
            story = []
            
            # 1. 报告标题
            story.append(Paragraph(f"{ticker} 深度投资分析报告", title_style))
            story.append(Spacer(1, 15))
            
            # 2. 顶部仪表盘 (Dashboard)
            def fmt_num(n):
                if isinstance(n, (int, float)):
                    if n > 1e12: return f"{n/1e12:.2f}T"
                    if n > 1e9: return f"{n/1e9:.2f}B"
                    return f"{n:,.2f}"
                return str(n)
            
            # 涨跌幅颜色
            change_color = colors.green if price_change >= 0 else colors.red
            change_str = f"{price_change:+.2%}"

            # 仪表盘数据
            dash_data = [
                [f"{ticker}", f"{fund_data.get('price', 'N/A')}", change_str, fmt_num(fund_data.get('market_cap', 'N/A'))],
                ["TICKER", "PRICE", "24H CHANGE", "MARKET CAP"]
            ]
            dash_table = Table(dash_data, colWidths=[120, 120, 120, 120])
            dash_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTSIZE', (0, 0), (-1, 0), 16), # 第一行大字
                ('FONTSIZE', (0, 1), (-1, 1), 8),  # 第二行标签小字
                ('TEXTCOLOR', (0, 1), (-1, 1), colors.grey),
                ('TEXTCOLOR', (2, 0), (2, 0), change_color), # 涨跌幅颜色
                ('BOTTOMPADDING', (0, 0), (-1, 0), 6),
                ('TOPPADDING', (0, 1), (-1, 1), 4),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#e0e0e0")),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8f9fa")),
            ]))
            story.append(dash_table)
            story.append(Spacer(1, 15))

            # 3. 关键指标红绿灯 (Key Indicators)
            # 逻辑: RSI < 30 (超卖/绿), > 70 (超买/红); P/E 仅展示
            rsi_val = tech_latest.get('RSI', 50)
            rsi_color = colors.green if rsi_val < 30 else (colors.red if rsi_val > 70 else colors.black)
            
            ind_data = [
                ['P/E (TTM)', 'RSI (14)', 'Volatility', 'P/C Ratio'],
                [str(fund_data.get('pe', 'N/A')), f"{rsi_val:.2f}", f"{tech_latest.get('Volatility', 0):.2%}", str(fund_data.get('pc_ratio_vol', 'N/A'))]
            ]
            t = Table(ind_data, colWidths=[120, 120, 120, 120])
            t.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e8f0fe")), # 表头背景
                ('TEXTCOLOR', (1, 1), (1, 1), rsi_color), # RSI 颜色
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e0e0e0")),
            ]))
            story.append(t)
            story.append(Spacer(1, 20))

            # 4. 插入期权 OI 图表
            if oi_chart_buffer:
                img = Image(oi_chart_buffer, width=480, height=180)
                story.append(img)
                # 添加图表说明
                oi_desc = "<b>图表说明:</b> 绿色=Call持仓(潜在阻力), 红色=Put持仓(潜在支撑), 蓝色虚线=当前股价. 最高的柱子通常代表关键的市场博弈点位(Walls)."
                story.append(Paragraph(oi_desc, ParagraphStyle('OIDesc', parent=normal_style, fontSize=9, textColor=colors.grey, alignment=1, spaceBefore=5)))
                story.append(Spacer(1, 20))
            
            # 5. 解析 Markdown 文本并转换为 PDF 元素
            def clean_text(text):
                # 1. 替换会导致乱码的特殊符号 (Smart Quotes, Dashes, Bullets)
                replacements = {
                    '\u2014': '-',  # Em Dash (—) -> 篳
                    '\u2013': '-',  # En Dash (–)
                    '\u2018': "'",  # Left Single Quote (‘)
                    '\u2019': "'",  # Right Single Quote (’) -> 篳
                    '\u201c': '"',  # Left Double Quote (“)
                    '\u201d': '"',  # Right Double Quote (”)
                    '\u2022': '-',  # Bullet (•)
                    '\u25e6': '-',  # White Bullet (◦)
                    '\u27a2': '->', # Arrow (➢) -> 縴
                    '\u2026': '...', # Ellipsis (…)
                }
                for k, v in replacements.items():
                    text = text.replace(k, v)
                
                # 2. 仅保留安全字符 (ASCII + 中文 + 常用标点)
                # 过滤掉 Emoji 和其他生僻符号
                return "".join(c for c in text if 
                               (0x20 <= ord(c) <= 0x7E) or  # ASCII
                               (0x4E00 <= ord(c) <= 0x9FFF) or # CJK Unified Ideographs
                               (0x3000 <= ord(c) <= 0x303F) or # CJK Punctuation
                               (0xFF00 <= ord(c) <= 0xFFEF) or # Fullwidth ASCII
                               c in '\n\r\t')

            def format_content(text):
                # 1. 中英文之间增加空格 (Pangu Spacing)
                text = re.sub(r'([\u4e00-\u9fa5])([A-Za-z0-9])', r'\1 \2', text)
                text = re.sub(r'([A-Za-z0-9])([\u4e00-\u9fa5])', r'\1 \2', text)
                
                # 2. 将 ASCII 字符 (英文/数字/标点) 包裹在 Helvetica 字体中，解决 STSong-Light 英文挤压问题
                # 排除 * (用于加粗) 和 < > (用于标签)
                def repl(match):
                    return f'<font name="Helvetica">{match.group(1)}</font>'
                text = re.sub(r'([A-Za-z0-9\.\,\%\$\-\+\:\/\=\(\)\?\!]+)', repl, text)
                
                # 3. 处理加粗 (**Text** -> <b>Text</b>)
                text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
                return text

            lines = report_text.split('\n')
            for line in lines:
                line = clean_text(line)
                # 检测缩进 (用于判断嵌套列表)
                is_indented = line.startswith('  ') or line.startswith('\t')
                
                line = line.strip()
                if not line: continue
                line = line.replace('```', '')
                
                if line.startswith('#'):
                    # 标题处理：优化编号与标题之间的间距
                    level = line.count('#')
                    text = line.lstrip('#').strip()
                    # 使用 Regex 在数字编号(如 "1.")后添加不换行空格，增加间距
                    # 使用 \u00A0 防止被 format_content 破坏
                    text = re.sub(r'^(\d+\.)\s*', lambda m: f"{m.group(1)}\u00A0\u00A0", text)
                    
                    # 格式化内容
                    text = format_content(text)
                    
                    if level == 1:
                        story.append(Paragraph(text, title_style))
                        # === 优化: 在一级标题下添加分割线 ===
                        d = Drawing(512, 1) # 宽度匹配页边距 (612 - 50 - 50 = 512)
                        d.add(Line(0, 0, 512, 0, strokeColor=colors.HexColor("#1a73e8"), strokeWidth=1))
                        d.keepWithNext = True # 确保分割线紧贴下一元素
                        story.append(d)
                        s = Spacer(1, 8)
                        s.keepWithNext = True # 确保间隔紧贴下一元素 (正文)
                        story.append(s)
                    else:
                        story.append(Paragraph(text, heading_style))
                        
                elif line.startswith('- ') or line.startswith('* '):
                    content = line[2:]
                    content = format_content(content)
                    # 策略部分优化：如果是风控参数相关的行，强制缩进
                    # 注意：content 现在可能包含 <font> 标签，正则需要适配
                    is_strategy_param = re.search(r'(入场|止盈|止损|仓位|Entry|TP|SL)', content)
                    
                    if is_indented or is_strategy_param:
                        story.append(Paragraph(f"  {content}", sub_bullet_style)) # 移除特殊符号 ◦
                    else:
                        story.append(Paragraph(f"-&nbsp; {content}", bullet_style)) # 将 • 替换为安全的 -
                else:
                    line = format_content(line)
                    story.append(Paragraph(line, normal_style))
            
            # 4. 添加文末免责声明板块
            story.append(Spacer(1, 20))
            disclaimer = "<b>免责声明 (Disclaimer):</b> 本报告由 AI 系统基于公开数据自动生成，仅供信息参考，不构成任何投资建议。市场有风险，投资需谨慎。请务必结合独立思考与专业顾问意见进行决策。"
            story.append(Paragraph(disclaimer, ParagraphStyle('Disclaimer', parent=normal_style, fontSize=8, textColor=colors.grey, alignment=0)))

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
    def _generate_ai_report_sync(ticker, fund, tech_data, news_data, web_search_data, gex_data, flow_data, macro_data):
        """生成 AI 报告内容的同步核心方法"""
        latest = tech_data.iloc[-1]
        current_date = datetime.datetime.now().strftime("%Y-%m-%d")

        # Safely extract news headlines, skipping items that might not have a 'title' key.
        news_headlines = "\n".join([f"- {n['title']}" for n in news_data[:5] if 'title' in n])
        
        # 格式化网络搜索结果
        web_content = "\n".join([f"- [Web] {r['title']}: {r['body']}" for r in web_search_data])

        # 格式化 GEX 数据
        gex_info = "- 暂无期权 Gamma 数据"
        if gex_data:
            gex_info = f"""
            - 到期日: {gex_data['expiry']}
            - Net GEX (净伽马敞口): ${gex_data['net_gex']:,.0f}
            - Call Wall (最大阻力/做市商做空点): {gex_data['call_wall']}
            - Put Wall (最大支撑/做市商回补点): {gex_data['put_wall']} """

        # 格式化资金流数据
        flow_info = "- 暂无显著期权异动"
        if flow_data:
            flow_info = "\n".join([f"- {f['type']} | 到期: {f['expiry']} | 行权: {f['strike']} | Vol: {f['volume']} (OI: {f['oi']}, 倍数: {f['ratio']}x)" for f in flow_data])

        # 格式化分析师评级
        analyst_ratings_str = "- 暂无近期评级变动"
        if fund['analyst']['recent_ratings']:
            analyst_ratings_str = "\n".join([f"  - {r}" for r in fund['analyst']['recent_ratings']])

        # 格式化宏观数据
        market_price = macro_data.get('market_price')
        market_price_str = f"{market_price:.2f}" if isinstance(market_price, (int, float)) else "N/A"
        market_change = macro_data.get('market_change')
        market_change_str = f"{market_change:+.2%}" if isinstance(market_change, (int, float)) else "N/A"
        
        vix_val = macro_data.get('vix')
        vix_str = f"{vix_val:.2f}" if isinstance(vix_val, (int, float)) else "N/A"
        vix_change = macro_data.get('vix_change')
        vix_change_str = f"{vix_change:+.2%}" if isinstance(vix_change, (int, float)) else "N/A"

        # 构建更强大的提示词 (Prompt)
        prompt = f"""
            # Role
            你是一位拥有20年深厚资历的华尔街量化与宏观对冲基金首席投资官 (CIO)。你擅长将自上而下的宏观逻辑（Top-Down）与自下而上的量化因子（Bottom-Up）相结合，挖掘市场尚未完全定价的“预期差”。

            # Analysis Requirements
            请基于以下数据，生成一份逻辑严密、具备实战指导意义的分析报告。
            **请直接开始报告内容，不要包含任何自我介绍或开场白。**
            
            结构要求如下：

            ### 1.  核心结论与交易驱动 (Executive Summary & Driver)
            - **交易驱动类型**: [基本面驱动 / 事件驱动 / 量化驱动 / 技术面驱动] (请根据分析判定主导因素)
            - **投资评级**: (强力买入 / 买入 / 增持 / 中性 / 减持 / 卖出)
            - **操作时间框架**: (例如: 短线波段 / 中期趋势 / 长线配置)
            - **AI 置信度**: (例如: 极高置信度 >90% / 高置信度 75-90% / 中等置信度 60-75% / 低置信度 <60%)
            - **操作计划**:
              - 入场区间 (Entry): [具体价格]
              - 目标止盈 (TP): [具体价格]
              - 硬性止损 (SL): [具体价格]
            - **核心逻辑摘要**: 一句话概括为何做此交易。

            ### 2. 🏛️ 宏观叙事与基本面 (Macro & Fundamentals)
            - **宏观环境**: 结合大盘走势 ({macro_data.get('market_index', 'Market')}) 和 VIX 恐慌指数，判断当前市场是 Risk-On 还是 Risk-Off。
            - **板块趋势**: 分析所属板块 ({fund['sector']}) 的整体表现。
            - **AI/FSD/增长故事**: 结合业务指引和行业趋势，分析核心增长逻辑。
            - **估值逻辑**: P/E 是否合理？结合 PEG 和历史分位判断。

            ### 3. 🔬 微观筹码与期权博弈 (Micro & Chips)
            - **Gamma Squeeze 风险**: 分析 Call Wall/Put Wall 位置，判断是否存在逼空或杀跌动能。
            - **资金流向 (Smart Money)**: 解读期权异动 (Option Flow)，主力是在布局反弹还是对冲风险？
            - **交易员情绪**: 结合社交媒体情绪，判断市场是否过热或恐慌。

            ### 4. 📈 技术面共振 (Technicals)
            - **关键均线**: 50D/200D SMA 的支撑与阻力。
            - **指标信号**: RSI 是否超买/超卖？MACD 是否背离？
            
            请使用专业、简洁、富有洞察力的语言输出。

            # Input Data Panel
            - **当前分析日期**: {current_date}

            ## 0. 宏观市场环境 (Macro Context)
            - 大盘指数 ({macro_data.get('market_index', 'N/A')}): {market_price_str} (Change: {market_change_str})
            - 市场恐慌指数 (VIX): {vix_str} (Change: {vix_change_str})

            ## 1. 标的基本面与质量 (Quality & Value)
            - 标的: {ticker} ({fund['name']}) | 行业: {fund['sector']}
            - 业务概览 (10-K): {fund['business_summary'][:400]}...
            - 核心估值: P/E: {fund['pe']} | Fwd P/E: {fund['forward_pe']} | PEG: {fund['peg_ratio']} | P/B: {fund['pb']}
            - 盈利质量: ROE: {fund['roe']} | 净利率: {fund['profit_margins']} | EPS: {fund['eps']}
            - 财务杠杆: 负债权益比: {fund['debt_to_equity']} | Beta: {fund['beta']}

            ## 2. 量化与技术面 (Quant & Technicals)
            - 趋势指标: 50D SMA: {latest['SMA_50']:.2f} | 200D SMA: {latest['SMA_200']:.2f}
            - 动能指标: RSI: {latest['RSI']:.2f} | MACD: {latest['MACD']:.2f} (Signal: {latest['MACD_Signal']:.2f})
            - A股特色指标: KDJ: K={latest['K']:.1f} D={latest['D']:.1f} J={latest['J']:.1f}
            - 风险与活跃度: ATR(14): {latest['ATR']:.2f} | 换手率: {fund.get('turnover_rate', 'N/A')}
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

            ## 5. 市场催化剂、管理层指引与交易员情绪 (Catalysts, Guidance & Sentiment)
            - 下次财报日期: {fund.get('next_earnings', 'N/A')} (距离现在 {fund.get('days_to_earnings', 'N/A')} 天)
            - 实时网络搜索 (含未来事件、IV分析、X/Twitter讨论):
            {web_content if web_content else "- 暂无网络搜索结果"}
            - 交易所新闻 (Exchange News): 
            {news_headlines if news_headlines else "- 暂无交易所新闻"}

            ## 6. 财务报表透视 (Financials - Latest Quarter)
            - 报告日期: {fund['financials'].get('date', 'N/A')}
            - 总营收: {fund['financials'].get('revenue', 'N/A')} | 净利润: {fund['financials'].get('net_income', 'N/A')}
            - 毛利润: {fund['financials'].get('gross_profit', 'N/A')} | 经营现金流: {fund['financials'].get('op_cashflow', 'N/A')}
            - 资产负债: 现金储备 {fund['financials'].get('total_cash', 'N/A')} vs 总债务 {fund['financials'].get('total_debt', 'N/A')}

            ## 7. 华尔街分析师共识 (Analyst Consensus)
            - 综合评级: {fund['analyst']['recommendation']} (基于 {fund['analyst']['num_analysts']} 位分析师)
            - 目标价: Mean: {fund['analyst']['target_mean']} | High: {fund['analyst']['target_high']} | Low: {fund['analyst']['target_low']}
            - 近期机构评级变动:
            {analyst_ratings_str}
            """
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        return response.choices[0].message.content
    @staticmethod
    async def get_ai_analysis(ticker, fund, tech_data, news_data, web_search_data, gex_data, flow_data, macro_data):
        """调用 LLM 生成更深度的自然语言报告 (Async Wrapper)"""
        try:
            loop = asyncio.get_running_loop()
            # 复用同步生成方法
            return await loop.run_in_executor(
                None, 
                lambda: StockAnalyzer._generate_ai_report_sync(ticker, fund, tech_data, news_data, web_search_data, gex_data, flow_data, macro_data)
            )
        except Exception as e:
            return f"AI 分析生成失败: {str(e)}"

# ================= Discord 命令处理 =================

@bot.event
async def on_ready():
    print(f'✅ Bot 已登录: {bot.user}')
    if DISCORD_AI_REPORT_CHANNEL_ID:
        print(f'🔒 频道限制已启用: 仅在频道 ID {DISCORD_AI_REPORT_CHANNEL_ID} 响应')
    print('DeepSeek 模式就绪。尝试输入: !a TSLA')

@bot.command(name='a', aliases=['analyze', 'stock', 'gp'])
async def analyze(ctx, ticker: str):
    """
    分析股票命令。用法: !a TSLA 或 !a 600519
    """
    # === 频道限制检查 ===
    if DISCORD_AI_REPORT_CHANNEL_ID and str(ctx.channel.id) != str(DISCORD_AI_REPORT_CHANNEL_ID):
        target_channel = bot.get_channel(int(DISCORD_AI_REPORT_CHANNEL_ID))
        channel_name = target_channel.name if target_channel else "指定频道"
        await ctx.send(f"⚠️ 请在指定频道 #{channel_name} 使用此命令。", delete_after=10)
        return

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
        df, fund, news, macro_data = StockAnalyzer.get_data(ticker)
        
        if df is None:
            await status_msg.edit(content=f"❌ 找不到股票代码 **{ticker}**，请检查拼写或重试。")
            return

        # 计算涨跌幅
        price_change = 0
        if len(df) >= 2:
            price_change = (df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]

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

        # 7. 生成期权 OI 图表
        oi_chart_buffer = await loop.run_in_executor(None, lambda: StockAnalyzer.get_option_open_interest_chart(stock_obj, fund['price']))

        # 新增: 上传图表到 Supabase 以获取 URL
        oi_chart_url = None
        if oi_chart_buffer and supabase:
            oi_chart_filename = f"{ticker}_oi_chart_{int(time.time())}.png"
            oi_chart_url = await loop.run_in_executor(None, lambda: StockAnalyzer.upload_file_to_supabase(oi_chart_filename, oi_chart_buffer, "image/png"))

        # 8. 获取 AI 报告
        await status_msg.edit(content=f"🤖 DeepSeek R1 (深度思考模式) 正在生成分析报告...")
        report = await StockAnalyzer.get_ai_analysis(ticker, fund, df_tech, news, web_results, gex_data, flow_data, macro_data)

        # 9. 构建 Embed 消息
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

        # 将 OI 图表直接嵌入消息
        if oi_chart_url:
            embed.set_image(url=oi_chart_url)

        embed.set_footer(text=f"分析对象: {fund['name']} | Host: {socket.gethostname()} | 由 DeepSeek AI 强力驱动")
        embed.set_thumbnail(url="https://cdn-icons-png.flaticon.com/512/8569/8569731.png") # 一个中性的图表icon

        # 10. 生成 PDF 并发送
        pdf_file = None
        pdf_buffer = StockAnalyzer.create_pdf_report(ticker, report, fund, latest, price_change, oi_chart_buffer)
        if pdf_buffer:
            pdf_file = discord.File(pdf_buffer, filename=f"{ticker}_Analysis.pdf")

        # 11. 发送结果
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
        # 使用 asyncio 同时运行 discord bot 和 fastapi server
        async def main():
            # 启动 discord bot 作为后台任务
            bot_task = asyncio.create_task(bot.start(DISCORD_TOKEN))
            
            # 配置 uvicorn
            port = int(os.getenv('PORT', 8000))
            config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
            server = uvicorn.Server(config)
            
            # 启动 fastapi server
            server_task = asyncio.create_task(server.serve())
            
            # 等待两个任务完成
            await asyncio.gather(
                bot_task,
                server_task
            )

        asyncio.run(main())
import streamlit as st
import os
import google.generativeai as genai
from dotenv import load_dotenv
import pandas as pd
from pandasai import PandasAI, Config
from pandasai.llm import OpenAI
import customtkinter as ctk
import numpy as np
from tkinter import filedialog

# 載入環境變數
load_dotenv()
# 設定API金鑰

PANDAS_AI_KEY = '$2a$10$4/R.Hy6jBx.w6gMPbCRB.uxobMzC8dtMsYTlk3NYaNycjhTFjbyS'

# 新增 API 金鑰和本地資料夾選擇
api_key = st.text_input('請輸入 API 金鑰', value="")

# 選擇平台
platform = st.selectbox('選擇平台', ['GOOGLE', 'AZURE', 'OPENAI'])

# 根據選擇的平臺顯示模型選項
if platform == 'GOOGLE':
    models = [
        'gemini-2.0-flash-001（穩定版）',
        'gemini-2.0-flash（自動更新版）',
        'gemini-2.0-flash-lite-preview-02-05（預覽版）',
        'gemini-2.0-pro-exp-02-05（實驗版）',
        'gemini-1.5-flash-002（穩定版）',
        'gemini-1.5-flash-001（穩定版）',
        'gemini-1.5-flash（自動更新版）',
        'gemini-1.5-pro-002（穩定版）',
        'gemini-1.5-pro-001（穩定版）',
        'gemini-1.5-pro（自動更新版）'
    ]
elif platform == 'AZURE':
    models = ['gpt-3.5-turbo', 'gpt-4']
else:
    models = ['gpt-3.5-turbo', 'gpt-4']

selected_model = st.selectbox('選擇模型', models)

# 設定模型名稱
if platform == 'AZURE':
    API_CONFIG = {
        'MODELS': {
            'chat': 'gpt-3.5-turbo',
            'vision': 'gpt-4-vision-preview',
            'embedding': 'text-embedding-ada-002'
        }
    }
    MODEL_MAPPINGS = {
        'gpt-4': 'gpt-4',
        'gpt-3.5-turbo': 'gpt-3.5-turbo',
        'azure-gpt4': 'gpt-4'
    }
else:
    # 預設為 GOOGLE 模型
    model = genai.GenerativeModel(selected_model)

# 新增本地資料夾選擇
local_folder = st.text_input('請輸入本地資料夾路徑')

if os.path.isdir(local_folder):
    model_files = [f for f in os.listdir(local_folder) if f.endswith(('.gguf', '.bin'))]
else:
    st.warning('請確認資料夾路徑是否正確')

genai.configure(api_key=api_key)

# 定義生成回應的函數
def get_gemini_response(prompt):
    try:
        # 生成回應
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f'發生錯誤: {str(e)}'

# 用戶輸入
user_input = st.text_area('請輸入您的問題：', height=150)

if st.button('提交問題'):
    if user_input:
        # 獲取模型回應
        model_response = get_gemini_response(user_input)
        # 顯示模型回應
        st.write(model_response)
    else:
        st.warning('請輸入問題。')

# 新增使用 PandasAI 進行數據分析的範例
llm = OpenAI(api_token='你的API金鑰')
config = Config(
    llm=llm,
    enable_cache=True,
    custom_prompts=True,
    save_charts=True,
    export_format='png'
)
pandas_ai = PandasAI(config)

# 讀取數據
# df = pd.read_csv('你的數據.csv')

# 常用查詢範例
# 基本數據分析

# 描述性統計
response = pandas_ai(df, '計算這個數據集的平均值和標準差')

# 數據摘要
response = pandas_ai(df, '幫我總結一下這個數據集的主要特點')

# 找出特定條件的數據
response = pandas_ai(df, '找出銷售額超過1000的所有記錄')

# 數據視覺化
# 繪製數據集內容
response = pandas_ai(df, '請顯示數據集的內容')

# 繪製數據分布
response = pandas_ai(df, '請顯示數據的分布情況')

# 繪製數據關係
response = pandas_ai(df, '請顯示數據之間的關係')

# 多維度分析
response = pandas_ai(df, '請用散點圖分析數據之間的關係')

# 進階功能使用
# 自定義配置
from pandasai import Config

config = Config(
    llm=llm,
    enable_cache=True,
    custom_prompts=True,
    save_charts=True,
    export_format='png'
)

pandas_ai = PandasAI(config)

# 多表分析
# 讀取多個數據表
df1 = pd.read_csv('銷售數據.csv')
df2 = pd.read_csv('客戶數據.csv')

# 進行跨表分析
response = pandas_ai([df1, df2], '分析不同客戶群的銷售表現')

# 特徵工程
# 自動生成新特徵
response = pandas_ai(df, '請基於現有數據創建一個新的特徵來預測客戶流失風險')

# 數據轉換
response = pandas_ai(df, '將類別變量轉換為數值型特徵')

# 實用技巧
# 錯誤處理
try:
    response = pandas_ai(df, '你的查詢')
    print(response)
except Exception as e:
    print(f'發生錯誤: {str(e)}')

# 快取管理
# 清除快取
pandas_ai.clear_cache()

# 設置快取目錄
pandas_ai.set_cache_dir('path/to/cache')

# 導出結果
# 保存分析結果
response = pandas_ai(df, '分析銷售趨勢並將結果保存為報告')
response.to_csv('分析報告.csv')

# 導出圖表
response = pandas_ai(df, '生成銷售報告圖表')
response.save_chart('銷售報告.png')

# Langchain方式
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 載入本地模型
llm = LlamaCpp(
    model_path='D:/model/你的模型檔案.bin',
    temperature=0.7,
    max_tokens=100
)

# 建立提示模板
prompt = PromptTemplate(
    input_variables=['question'],
    template='{question}'
)

# 建立chain
chain = LLMChain(llm=llm, prompt=prompt)
response = chain.run('你的問題')

# Streamlit網頁介面
import streamlit as st

def init_llm():
    return LlamaCpp(model_path='你的模型路徑')


def chat_interface():
    st.title('本地LLM聊天機器人')
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).write(message['content'])
        
    if prompt := st.chat_input('請輸入訊息'):
        st.chat_message('user').write(prompt)
        response = llm(prompt)
        st.chat_message('assistant').write(response)

if __name__ == '__main__':
    llm = init_llm()
    chat_interface()

# PicoLLM方式
from picollm import PicoLLM

# 初始化模型
llm = PicoLLM(
    access_key='你的金鑰',
    model_path='模型路徑'
)

# 生成回應
response = llm.complete('你的問題')
print(response)

# 使用環境變數
import os
from pandasai import PandasAI
from pandasai.llm import OpenAI
import pandas as pd
from dotenv import load_dotenv

# 載入.env檔案
load_dotenv()

# 從環境變數獲取API金鑰
api_key = os.getenv('PANDASAI_API_KEY')

# 初始化LLM
llm = OpenAI(api_token=api_key)
pandas_ai = PandasAI(llm)

# 讀取數據
df = pd.read_csv('your_data.csv')

# 使用PandasAI
response = pandas_ai(df, '分析這個數據集的趨勢')

# 直接設定API金鑰
llm = OpenAI(api_token='你的API金鑰')

# 配置PandasAI
config = {
    'llm': llm,
    'enable_cache': True,
    'save_charts': True,
    'verbose': True
}

pandas_ai = PandasAI(**config)

# 使用範例
df = pd.read_csv('your_data.csv')
response = pandas_ai(df, '計算銷售總額')

# 使用配置檔案
import json

# 讀取配置檔案
with open('config.json', 'r') as f:
    config = json.load(f)

# 初始化LLM
llm = OpenAI(api_token=config['api_key'])
pandas_ai = PandasAI(llm)

# 使用PandasAI
df = pd.read_csv('your_data.csv')
response = pandas_ai(df, '生成銷售報表')

# 安全性建議
# 創建.env檔案：
# PANDASAI_API_KEY=你的API金鑰
# 將.env加入.gitignore：
# .env
# config.json
# 使用虛擬環境：
# python -m venv venv
# source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate     # Windows

class 分析器(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title('TF-AI 數據分析助手')
        self.geometry('1200x800')

        # 設置 API Key
        self.api_key = ''  # 預設金鑰為空
        self.api_key_entry = ctk.CTkEntry(self, placeholder_text='輸入 API 金鑰')
        self.api_key_entry.pack(fill='x', padx=10, pady=10)
        
        self.save_api_key_btn = ctk.CTkButton(self, text='儲存 API 金鑰', command=self.save_api_key)
        self.save_api_key_btn.pack(fill='x', padx=10, pady=5)
        
        self.load_api_key_btn = ctk.CTkButton(self, text='加載 API 金鑰', command=self.load_api_key)
        self.load_api_key_btn.pack(fill='x', padx=10, pady=5)
        
        self.load_api_key()  # 在初始化時加載 API 金鑰
        
        openai.api_key = self.api_key
        self.current_model = ''  # 當前模型
        self.selected_directory = ''  # 選擇的資料夾
        self.setup_ui()
        
    def setup_ui(self):
        # 主分割框架
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # 工具列
        self.toolbar = ctk.CTkFrame(self.main_frame)
        self.toolbar.pack(fill='x', padx=5, pady=5)
        
        self.load_btn = ctk.CTkButton(
            self.toolbar,
            text='載入數據',
            command=self.load_data
        )
        self.load_btn.pack(side='left', padx=5)
        
        self.analyze_btn = ctk.CTkButton(
            self.toolbar,
            text='分析數據',
            command=self.analyze_data
        )
        self.analyze_btn.pack(side='left', padx=5)

        self.select_model_btn = ctk.CTkButton(
            self.toolbar,
            text='選擇模型資料夾',
            command=self.select_directory
        )
        self.select_model_btn.pack(side='left', padx=5)

        # 模型名稱顯示
        self.model_label = ctk.CTkLabel(self.toolbar, text='當前模型: ')
        self.model_label.pack(side='right', padx=5)

        # 左右分割視窗
        self.paned = ctk.CTkFrame(self.main_frame)
        self.paned.pack(fill='both', expand=True)

        # 左側：聊天區
        self.chat_frame = ctk.CTkFrame(self.paned)
        self.chat_frame.pack(side='left', fill='both', expand=True, padx=(0,5))

        self.chat_display = ctk.CTkTextbox(self.chat_frame)
        self.chat_display.pack(fill='both', expand=True, pady=(0,5))

        # 右側：數據顯示區
        self.data_frame = ctk.CTkFrame(self.paned)
        self.data_frame.pack(side='right', fill='both', expand=True, padx=(5,0))

        self.data_display = ctk.CTkTextbox(self.data_frame)
        self.data_display.pack(fill='both', expand=True)

        # 輸入區
        self.input_frame = ctk.CTkFrame(self.main_frame)
        self.input_frame.pack(fill='x', pady=(5,0))
        
        self.input_entry = ctk.CTkEntry(
            self.input_frame,
            placeholder_text='輸入分析請求...'
        )
        self.input_entry.pack(side='left', fill='x', expand=True, padx=5)

        self.send_btn = ctk.CTkButton(
            self.input_frame,
            text='發送',
            command=self.send_message
        )
        self.send_btn.pack(side='right', padx=5)

    def save_api_key(self):
        self.api_key = self.api_key_entry.get()
        with open('W:\AI_Agent\api\api_key.txt', 'w') as file:
            file.write(self.api_key)
        openai.api_key = self.api_key  # 更新 API 金鑰

    def load_api_key(self):
        try:
            with open('W:\AI_Agent\api\api_key.txt', 'r') as file:
                self.api_key = file.read().strip()
                self.api_key_entry.insert(0, self.api_key)  # 將金鑰填入輸入框
                openai.api_key = self.api_key  # 更新 API 金鑰
        except FileNotFoundError:
            pass  # 檔案不存在，無需處理

    def select_directory(self):
        self.selected_directory = filedialog.askdirectory(title='選擇模型資料夾')
        if self.selected_directory:
            model_list = self.get_model_list(self.selected_directory)
            if model_list:
                self.current_model = model_list[0]  # 預設選擇第一個模型
                self.model_label.configure(text=f'當前模型: {self.current_model}')
            else:
                self.chat_display.insert('end', '該資料夾中沒有可用的模型！\n\n')

    def get_model_list(self, model_directory):
        return [f for f in os.listdir(model_directory) if f.endswith('.gguf')]

    def load_data(self):
        file_path = filedialog.askopenfilename(
            filetypes=[
                ('CSV 檔案', '*.csv'),
                ('Excel 檔案', '*.xlsx;*.xls'),
                ('所有檔案', '*.*')
            ]
        )
        
        if file_path:
            try:
                if file_path.endswith('.csv'):
                    self.data = pd.read_csv(file_path)
                else:
                    self.data = pd.read_excel(file_path)
                
                self.data_display.delete('1.0', 'end')
                self.data_display.insert('end', '數據預覽：\n\n')
                self.data_display.insert('end', self.data.head().to_string())
                self.chat_display.insert('end', '數據載入成功！\n\n')
            except Exception as e:
                self.chat_display.insert('end', f'數據載入錯誤：{str(e)}\n\n')

    def analyze_data(self):
        if hasattr(self, 'data'):
            analysis = f"""
基本統計：
{self.data.describe().to_string()}

數據類型：
{self.data.dtypes.to_string()}
"""
            # 使用 OpenAI API 分析
            try:
                response = openai.ChatCompletion.create(
                    model=self.current_model,
                    messages=[
                        {'role': 'system', 'content': '你是一個數據分析專家。'},
                        {'role': 'user', 'content': f'請分析以下數據並給出見解：\n{analysis}'}
                    ]
                )
                ai_analysis = response.choices[0].message.content
                self.chat_display.insert('end', f'AI 分析：\n{ai_analysis}\n\n')
            except Exception as e:
                self.chat_display.insert('end', f'API 錯誤：{str(e)}\n\n')
        else:
            self.chat_display.insert('end', '請先載入數據！\n\n')

    def send_message(self):
        message = self.input_entry.get()
        if message:
            self.chat_display.insert('end', f'You: {message}\n')
            try:
                response = openai.ChatCompletion.create(
                    model=self.current_model,
                    messages=[
                        {'role': 'user', 'content': message}
                    ]
                )
                ai_response = response.choices[0].message.content
                self.chat_display.insert('end', f'AI: {ai_response}\n\n')
            except Exception as e:
                self.chat_display.insert('end', f'API 錯誤：{str(e)}\n\n')

            self.input_entry.delete(0, 'end')
            self.chat_display.see('end')

    def calculate_response(self, X, Y):
        return X**2 + Y**2 - 2*X*Y + np.random.normal(0, 0.1, X.shape)

    def analyze_response(self, Z):
        return {
            'mean': float(np.mean(Z)),
            'std': float(np.std(Z)),
            'min': float(np.min(Z)),
            'max': float(np.max(Z)),
            'range': float(np.max(Z) - np.min(Z))
        }

if __name__ == '__main__':
    app = 分析器()
    app.mainloop()

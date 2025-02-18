import os
import customtkinter as ctk
import pandas as pd
import openai
import numpy as np
from tkinter import filedialog
from dotenv import load_dotenv
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

# 載入環境變數
load_dotenv()

# 從環境變數獲取API金鑰
api_key = os.getenv('PANDASAI_API_KEY')

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
        
        # Langchain方式
        self.llm = LlamaCpp(
            model_path='D:/model/你的模型檔案.bin',
            temperature=0.7,
            max_tokens=100
        )
        self.prompt = PromptTemplate(
            input_variables=['question'],
            template='{question}'
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
        
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
                response = self.chain.run(message)
                ai_response = response
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

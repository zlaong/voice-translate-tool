"""
voice_translate.py - 支持设备选择的实时语音翻译工具
更新内容：
1. 音频设备自动枚举
2. 输入源选择功能
3. macOS兼容性增强
"""

import os
import sys
import json
import pyaudio
from vosk import Model, KaldiRecognizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 配置区 =======================================================
SAMPLE_RATE = 16000    # 音频采样率
CHUNK_SIZE = 4096      # 音频块大小
SOURCE_LANG = "en"     # 源语言（中文）
TARGET_LANG = "zh"     # 目标语言（英文）

# 音频设备选择函数 =============================================
def select_audio_device():
    pa = pyaudio.PyAudio()
    
    print("\n可用音频输入设备：")
    devices = []
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        if info["maxInputChannels"] > 0:
            dev_str = f"{i}. {info['name']} (Channels: {info['maxInputChannels']})"
            # macOS设备名称美化
            if sys.platform == "darwin":
                dev_str = dev_str.replace("Built-in", "系统内置").replace("External", "外接")
            devices.append(dev_str)
            print(dev_str)

    while True:
        try:
            choice = int(input("\n请输入设备编号: "))
            if 0 <= choice < len(devices):
                print(f"已选择设备: {devices[choice].split(') ')[1]}")
                return choice
            print("编号无效，请重新输入")
        except ValueError:
            print("请输入有效数字")

# 主程序 ======================================================
def realtime_translation():
    # 初始化模型
    model_path = os.path.join(os.path.dirname(__file__), "vosk-model-cn-0.22")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到Vosk模型，请确认路径: {model_path}")
    
    recognizer = KaldiRecognizer(Model(model_path), SAMPLE_RATE)
    tokenizer, trans_model = init_translation_model()
    
    # 选择音频设备
    device_index = select_audio_device()
    
    # 配置音频流（macOS特殊参数）
    pa = pyaudio.PyAudio()
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=CHUNK_SIZE,
        # macOS需要以下参数
        start=False,
        stream_callback=None
    )
    
    print("\n实时翻译已启动（按CTRL+C停止）...")
    stream.start_stream()

    try:
        while stream.is_active():
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            if recognizer.AcceptWaveform(data):
                result = json.loads(recognizer.Result())
                process_translation(result, tokenizer, trans_model)

    except KeyboardInterrupt:
        print("\n终止翻译进程")
    finally:
        stream.stop_stream()
        stream.close()
        pa.terminate()

# 辅助函数 ====================================================
def init_translation_model():
    return (
        AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-zh-en"),
        AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-zh-en")
    )

def process_translation(result, tokenizer, model):
    source_text = result.get('text', '').strip()
    if source_text:
        inputs = tokenizer(source_text, return_tensors="pt")
        translated = model.generate(**inputs)
        target_text = tokenizer.batch_decode(translated, skip_special_tokens=True)[0]
        print(f"\n[识别] {source_text}\n[翻译] {target_text}")

if __name__ == "__main__":
    realtime_translation()
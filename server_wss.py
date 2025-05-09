from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.status import HTTP_422_UNPROCESSABLE_ENTITY
from pydantic_settings import BaseSettings
from pydantic import BaseModel, Field
from funasr import AutoModel
import numpy as np
import soundfile as sf
import argparse
import uvicorn
from urllib.parse import parse_qs
import os
import re
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from loguru import logger
import sys
import json
import traceback
import time

logger.remove()
log_format = "{time:YYYY-MM-DD HH:mm:ss} [{level}] {file}:{line} - {message}"
logger.add(sys.stdout, format=log_format, level="DEBUG", filter=lambda record: record["level"].no < 40)
logger.add(sys.stderr, format=log_format, level="ERROR", filter=lambda record: record["level"].no >= 40)


class Config(BaseSettings):
    ##说话人验证的阈值，用于确定说话人验证结果的置信度
    sv_thr: float = Field(0.3, description="Speaker verification threshold")
    ##每个音频块的大小，单位为毫秒，用于分割音频数据 这里设置为300ms
    chunk_size_ms: int = Field(300, description="Chunk size in milliseconds")
    ##音频的采样率，单位为赫兹，用于对音频数据进行采样
    sample_rate: int = Field(16000, description="Sample rate in Hz")
    ##音频数据的位深度，确定每个音频样本的位数
    bit_depth: int = Field(16, description="Bit depth")
    ##音频通道数
    channels: int = Field(1, description="Number of audio channels")
    ##平均对数概率的阈值，确定自动语音识别结果的置信度
    avg_logprob_thr: float = Field(-0.25, description="average logprob threshold")

config = Config()

emo_dict = {
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
}

event_dict = {
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|Cry|>": "😭",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "🤧",
}

emoji_dict = {
	"<|nospeech|><|Event_UNK|>": "❓",
	"<|zh|>": "",
	"<|en|>": "",
	"<|yue|>": "",
	"<|ja|>": "",
	"<|ko|>": "",
	"<|nospeech|>": "",
	"<|HAPPY|>": "😊",
	"<|SAD|>": "😔",
	"<|ANGRY|>": "😡",
	"<|NEUTRAL|>": "",
	"<|BGM|>": "🎼",
	"<|Speech|>": "",
	"<|Applause|>": "👏",
	"<|Laughter|>": "😀",
	"<|FEARFUL|>": "😰",
	"<|DISGUSTED|>": "🤢",
	"<|SURPRISED|>": "😮",
	"<|Cry|>": "😭",
	"<|EMO_UNKNOWN|>": "",
	"<|Sneeze|>": "🤧",
	"<|Breath|>": "",
	"<|Cough|>": "😷",
	"<|Sing|>": "",
	"<|Speech_Noise|>": "",
	"<|withitn|>": "",
	"<|woitn|>": "",
	"<|GBG|>": "",
	"<|Event_UNK|>": "",
}

lang_dict =  {
    "<|zh|>": "<|lang|>",
    "<|en|>": "<|lang|>",
    "<|yue|>": "<|lang|>",
    "<|ja|>": "<|lang|>",
    "<|ko|>": "<|lang|>",
    "<|nospeech|>": "<|lang|>",
}

emo_set = {"😊", "😔", "😡", "😰", "🤢", "😮"}
event_set = {"🎼", "👏", "😀", "😭", "🤧", "😷",}

def format_str(s):
	for sptk in emoji_dict:
		s = s.replace(sptk, emoji_dict[sptk])
	return s

def format_str_v2(s):
	sptk_dict = {}
	for sptk in emoji_dict:
		sptk_dict[sptk] = s.count(sptk)
		s = s.replace(sptk, "")
	emo = "<|NEUTRAL|>"
	for e in emo_dict:
		if sptk_dict[e] > sptk_dict[emo]:
			emo = e
	for e in event_dict:
		if sptk_dict[e] > 0:
			s = event_dict[e] + s
	s = s + emo_dict[emo]

	for emoji in emo_set.union(event_set):
		s = s.replace(" " + emoji, emoji)
		s = s.replace(emoji + " ", emoji)
	return s.strip()

def format_str_v3(s):
	def get_emo(s):
		return s[-1] if s[-1] in emo_set else None
	def get_event(s):
		return s[0] if s[0] in event_set else None

	s = s.replace("<|nospeech|><|Event_UNK|>", "❓")
	for lang in lang_dict:
		s = s.replace(lang, "<|lang|>")
	s_list = [format_str_v2(s_i).strip(" ") for s_i in s.split("<|lang|>")]
	new_s = " " + s_list[0]
	cur_ent_event = get_event(new_s)
	for i in range(1, len(s_list)):
		if len(s_list[i]) == 0:
			continue
		if get_event(s_list[i]) == cur_ent_event and get_event(s_list[i]) != None:
			s_list[i] = s_list[i][1:]
		#else:
		cur_ent_event = get_event(s_list[i])
		if get_emo(s_list[i]) != None and get_emo(s_list[i]) == get_emo(new_s):
			new_s = new_s[:-1]
		new_s += s_list[i].strip().lstrip()
	new_s = new_s.replace("The.", " ")
	return new_s.strip()

def contains_chinese_english_number(s: str) -> bool:
    # Check if the string contains any Chinese character, English letter, or Arabic number
    return bool(re.search(r'[\u4e00-\u9fffA-Za-z0-9]', s))


##完成语音识别，验证说话人的身份
sv_pipeline = pipeline(
    task='speaker-verification',
    model='iic/speech_eres2net_large_sv_zh-cn_3dspeaker_16k',
    model_revision='v1.0.0'
)


asr_pipeline = pipeline(
    task=Tasks.auto_speech_recognition,
    model='iic/SenseVoiceSmall',
    model_revision="master",
    device="cuda:0",
    disable_update=True
)


##完成语音识别 实现语音转文字
model_asr = AutoModel(
    model="iic/SenseVoiceSmall",
    trust_remote_code=True,
    remote_code="./model.py",    
    device="cuda:0",
    disable_update=True
)

##完成语音活动检测
model_vad = AutoModel(
    model="fsmn-vad",
    model_revision="v2.0.4",
    disable_pbar = True,
    max_end_silence_time=500,
    # speech_noise_thres=0.6,
    disable_update=True,
)

reg_spks_files = [
    "speaker/speaker_me.wav"
]

##初始化说话人相关信息
def reg_spk_init(files):
    reg_spk = {}
    for f in files:
        ##data记录音频数据，sr记录采样率
        data, sr = sf.read(f, dtype="float32")
        k, _ = os.path.splitext(os.path.basename(f))
        reg_spk[k] = {
            "data": data,
            "sr":   sr,
        }
    return reg_spk

reg_spks = reg_spk_init(reg_spks_files)

"""
完成语音识别，验证说话人的身份
:param audio: 音频数据 
:param sv_thr: 说话人验证的阈值 
:returns: hit 是否验证通过 (Boolean) 
          k  验证通过的说话人名称 (String)
"""
def speaker_verify(audio, sv_thr):
    hit = False
    for k, v in reg_spks.items():
        res_sv = sv_pipeline([audio, v["data"]], sv_thr)
        if res_sv["score"] >= sv_thr:
           hit = True
        logger.info(f"[speaker_verify] audio_len: {len(audio)}; sv_thr: {sv_thr}; hit: {hit}; {k}: {res_sv}")
    return hit, k


'''
完成语音识别，实现语音转文字
:param audio: 音频数据 
:param lang: 语言类型 
:param cache: 缓存数据 
:param use_itn: 是否使用反标准化 默认false
:returns: result 语音识别结果 (List)
'''
def asr(audio, lang, cache, use_itn=False):
    # with open('test.pcm', 'ab') as f:
    #     logger.debug(f'write {f.write(audio)} bytes to `test.pcm`')
    # result = asr_pipeline(audio, lang)
    start_time = time.time()
    result = model_asr.generate(
        input           = audio,
        cache           = cache,
        language        = lang.strip(),
        use_itn         = use_itn,
        batch_size_s    = 60,
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.debug(f"asr elapsed: {elapsed_time * 1000:.2f} milliseconds")
    return result

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def custom_exception_handler(request: Request, exc: Exception):
    logger.error("Exception occurred", exc_info=True)
    if isinstance(exc, HTTPException):
        status_code = exc.status_code
        message = exc.detail
        data = ""
    elif isinstance(exc, RequestValidationError):
        status_code = HTTP_422_UNPROCESSABLE_ENTITY
        message = "Validation error: " + str(exc.errors())
        data = ""
    else:
        status_code = 500
        message = "Internal server error: " + str(exc)
        data = ""

    return JSONResponse(
        status_code=status_code,
        content=TranscriptionResponse(
            code=status_code,
            msg=message,
            data=data
        ).model_dump()
    )

# Define the response model
class TranscriptionResponse(BaseModel):
    code: int
    info: str
    data: str

@app.websocket("/ws/transcribe")
async def websocket_endpoint(websocket: WebSocket):
    try:
        query_params = parse_qs(websocket.scope['query_string'].decode())
        ##身份验证，默认false不开启 小写后存在列表中返回true，否则返回false
        sv = query_params.get('sv', ['false'])[0].lower() in ['true', '1', 't', 'y', 'yes']
        ##语言类型，默认自动识别
        lang = query_params.get('lang', ['auto'])[0].lower()
        
        await websocket.accept()
        ##音频块的大小
        chunk_size = int(config.chunk_size_ms * config.sample_rate / 1000)
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)

        cache = {}
        cache_asr = {}
        last_vad_beg = last_vad_end = -1
        offset = 0
        hit = False
        ##初始化一个空的字节缓冲区，用于存储接收到的音频数据。
        buffer = b""
        while True:
            data = await websocket.receive_bytes()
            # logger.info(f"received {len(data)} bytes")

            buffer += data
            if len(buffer) < 2:
                continue
            
            ##转化音频数据格式
            audio_buffer = np.append(
                audio_buffer, 
                np.frombuffer(buffer[:len(buffer) - (len(buffer) % 2)], dtype=np.int16).astype(np.float32) / 32767.0
            )
            
            # with open('buffer.pcm', 'ab') as f:
            #     logger.debug(f'write {f.write(buffer[:len(buffer) - (len(buffer) % 2)])} bytes to `buffer.pcm`')
                
            buffer = buffer[len(buffer) - (len(buffer) % 2):]
   
            while len(audio_buffer) >= chunk_size:
                chunk = audio_buffer[:chunk_size]
                audio_buffer = audio_buffer[chunk_size:]
                audio_vad = np.append(audio_vad, chunk)
                
                # with open('chunk.pcm', 'ab') as f:
                #     logger.debug(f'write {f.write(chunk)} bytes to `chunk.pcm`')
                    
                if last_vad_beg > 1:
                    if sv:
                        # speaker verify
                        # If no hit is detected, continue accumulating audio data and check again until a hit is detected
                        # `hit` will reset after `asr`.
                        if not hit:
                            hit, speaker = speaker_verify(audio_vad[int((last_vad_beg - offset) * config.sample_rate / 1000):], config.sv_thr)
                            if hit:
                                response = TranscriptionResponse(
                                    code=2,
                                    info="detect speaker",
                                    data=speaker
                                )
                                await websocket.send_json(response.model_dump())
                    else:
                        response = TranscriptionResponse(
                            code=2,
                            info="detect speech",
                            data=''
                        )
                        await websocket.send_json(response.model_dump())


                '''
                流程：实现语音检测-->语音识别转文字
                '''
                res = model_vad.generate(input=chunk, cache=cache, is_final=False, chunk_size=config.chunk_size_ms)
                # logger.info(f"vad inference: {res}")
                if len(res[0]["value"]):
                    vad_segments = res[0]["value"]
                    for segment in vad_segments:
                        if segment[0] > -1: # speech begin
                            last_vad_beg = segment[0]                           
                        if segment[1] > -1: # speech end
                            last_vad_end = segment[1]
                        if last_vad_beg > -1 and last_vad_end > -1:
                            last_vad_beg -= offset
                            last_vad_end -= offset
                            offset += last_vad_end
                            beg = int(last_vad_beg * config.sample_rate / 1000)
                            end = int(last_vad_end * config.sample_rate / 1000)
                            logger.info(f"[vad segment] audio_len: {end - beg}")
                            result = None if sv and not hit else asr(audio_vad[beg:end], lang.strip(), cache_asr, True)
                            logger.info(f"asr response: {result}")
                            audio_vad = audio_vad[end:]
                            last_vad_beg = last_vad_end = -1
                            hit = False
                            
                            if  result is not None:
                                response = TranscriptionResponse(
                                    code=0,
                                    info=json.dumps(result[0], ensure_ascii=False),
                                    data=format_str_v3(result[0]['text'])
                                )
                                await websocket.send_json(response.model_dump())
                                
                        # logger.debug(f'last_vad_beg: {last_vad_beg}; last_vad_end: {last_vad_end} len(audio_vad): {len(audio_vad)}')

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Unexpected error: {e}\nCall stack:\n{traceback.format_exc()}")
        await websocket.close()
    finally:
        audio_buffer = np.array([], dtype=np.float32)
        audio_vad = np.array([], dtype=np.float32)
        cache.clear()
        logger.info("Cleaned up resources after WebSocket disconnect")


##配置和运行FastAPI应用
##如果直接运行，则默认监听端口为27000，否则跳过配置
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the FastAPI app with a specified port.")
    parser.add_argument('--port', type=int, default=27000, help='Port number to run the FastAPI app on.')
    # parser.add_argument('--certfile', type=str, default='path_to_your_SSL_certificate_file.crt', help='SSL certificate file')
    # parser.add_argument('--keyfile', type=str, default='path_to_your_SSL_certificate_file.key', help='SSL key file')
    args = parser.parse_args()
    # uvicorn.run(app, host="0.0.0.0", port=args.port, ssl_certfile=args.certfile, ssl_keyfile=args.keyfile)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
